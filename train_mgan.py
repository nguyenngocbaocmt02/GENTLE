import torch
import argparse
import os
from torch.multiprocessing import Process
import torch.distributed as dist
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import pandas as pd
from datasets.datasets import data_preparation
import numpy as np
from models.torch_mlp import MLPM
import wandb
import math
from models.torch_mlp import MLPM

class FCFFNet(nn.Module):
    def __init__(self, layers, nonlinearity, nonlinearity_params=None, 
                 out_nonlinearity=None, out_nonlinearity_params=None, normalize=False):
        super(FCFFNet, self).__init__()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1

        self.layers = nn.ModuleList()
        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))
            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))
                if nonlinearity_params is not None:
                    self.layers.append(nonlinearity(*nonlinearity_params))
                else:
                    self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            if out_nonlinearity_params is not None:
                self.layers.append(out_nonlinearity(*out_nonlinearity_params))
            else:
                self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x


#%%
def train(rank, gpu, args):
    wandb.init(project="rl-df", 
            entity="ml-with-vibes", 
            config=vars(args), 
            sync_tensorboard=True,
            #mode="disabled",
            )
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))
    bsize = args.batch_size
    os.makedirs(args.checkpoint_folder, exist_ok=True)

    df = pd.read_csv(os.path.join("datasets", args.dataset, "train.csv"))
    y_data, x_data, _ = data_preparation(df, name=args.dataset, balance=int(args.balance))
    x_data = x_data.reshape(-1, 1)
    dataset = torch.utils.data.TensorDataset(y_data, x_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    ydata_loader = DataLoader(torch.utils.data.TensorDataset(y_data), batch_size=args.batch_size, shuffle=True, drop_last=True)

    
    dx = x_data.shape[1]
    dy = y_data.shape[1]

    
    #Define loss
    mse_loss = torch.nn.MSELoss()

    #Transport map and discriminator
    network_params = [dx+dy] + args.n_layers * [args.n_units]
    F = MLPM(input_dim=dx + dy).to(device)
    D = MLPM(input_dim=dx + dy).to(device)
    #argsimizers
    optimimizer_F = torch.optim.Adam(F.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimimizer_D = torch.optim.Adam(D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    # Schedulers
    sch_F = torch.optim.lr_scheduler.StepLR(optimimizer_F, step_size = len(dataloader), gamma=0.995)
    sch_D = torch.optim.lr_scheduler.StepLR(optimimizer_D, step_size = len(dataloader), gamma=0.995)

    # define arrays to store results
    monotonicity    = torch.zeros(args.n_epochs,)
    D_train         = torch.zeros(args.n_epochs,)
    F_train         = torch.zeros(args.n_epochs,)

    for ep in range(args.n_epochs):

        F.train()
        D.train()

        # define counters for inner epoch losses
        D_train_inner = 0.0
        F_train_inner = 0.0
        mon_percent = 0.0

        for y, x in dataloader:
            #Data batch
            y, x = y.to(device), x.to(device)
            ones = torch.ones(bsize, 1, device=device)
            zeros = torch.zeros(bsize, 1, device=device)

            ###Loss for transport map###

            optimimizer_F.zero_grad()

            #Draw from reference
            z1 = next(iter(ydata_loader))[0].to(device)
            z2 = torch.randn(bsize, dx, device=device)
            z = torch.cat((z2, z1), 1)

            #Transport reference to conditional x|y
            Fz = F(z)
            Fz = Fz.reshape(Fz.shape[0], -1)
            #Transport of reference z1 to y marginal is by identity map
            #Compute loss for generator
            D_tmp = D(torch.cat((z1, Fz), 1))
            D_tmp = D_tmp.reshape(D_tmp.shape[0], -1)
            F_loss = mse_loss(D_tmp, ones)
            F_train_inner += F_loss.item()

            #Draw new reference sample
            z1_prime = next(iter(ydata_loader))[0].to(device)
            z2_prime = torch.randn(bsize, dx, device=device)
            z_prime = torch.cat((z2_prime, z1_prime), 1)
            F_prime = F(z_prime)
            F_prime = F_prime.reshape(F_prime.shape[0], -1)
            #Monotonicity constraint
            mon_penalty = torch.sum(((Fz - F_prime).view(bsize,-1))*((z2 - z2_prime).view(bsize,-1)), 1)
            if args.monotone_param > 0.0:
                F_loss = F_loss - args.monotone_param*torch.mean(mon_penalty)

            # take step for F
            F_loss.backward()
            optimimizer_F.step()
            sch_F.step()

            #Percent of examples in batch with monotonicity satisfied
            mon_penalty = mon_penalty.detach() + torch.sum((z1.view(bsize,-1) - z1_prime.view(bsize,-1))**2, 1).detach()
            mon_percent += float((mon_penalty>=0).sum().item())/bsize

            ###Loss for discriminator###

            optimimizer_D.zero_grad()
            D_ones = D(torch.cat((y,x),1))
            D_zeros = D(torch.cat((z1, Fz.detach()), 1))
            D_ones = D_ones.reshape(D_ones.shape[0], -1)
            D_zeros = D_zeros.reshape(D_zeros.shape[0], -1)
            #Compute loss for discriminator
            D_loss = 0.5*(mse_loss(D_ones, ones) + mse_loss(D_zeros, zeros))
            D_train_inner += D_loss.item()

            # take step for D
            D_loss.backward()
            optimimizer_D.step()
            sch_D.step()


        F.eval()
        D.eval()

        #Average monotonicity percent over batches
        mon_percent = mon_percent/math.ceil(float(args.n_train)/bsize)
        monotonicity[ep] = mon_percent

        #Average generator and discriminator losses
        F_train[ep] = F_train_inner/math.ceil(float(args.n_train)/bsize)
        D_train[ep] = D_train_inner/math.ceil(float(args.n_train)/bsize)
        wandb.log({"D_loss": float(D_train[ep])})
        wandb.log({"G_loss": float(F_train[ep])})
        print('Epoch %3d, Monotonicity: %f, Generator loss: %f, Critic loss: %f' % \
            (ep, monotonicity[ep], F_train[ep], D_train[ep]))
        
        if ep % 10 == 0:
            torch.save(F.state_dict(), os.path.join(args.checkpoint_folder, 'generator.pth'))
            torch.save(D.state_dict(), os.path.join(args.checkpoint_folder, 'discriminator.pth'))

def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '12000'
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    #dist.barrier()
    cleanup()  

def cleanup():
    dist.destroy_process_group()   

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parameters')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed used for initialization')
    parser.add_argument('--dataset', type=str, default="cancer",
                        help='name of dataset, check the folder datasets')
    parser.add_argument('--balance', type=bool, default=1,
                        help='balance the data or not')
    parser.add_argument('--save_every', type=int, default=100,
                        help='print training information frequency')
    parser.add_argument('--checkpoint_folder', type=str, default="checkpoints/monogan/cancer",
                        help='folder to save checkpoints')
    parser.add_argument("--monotone_param", type=float, default=0.01, help="monotone penalty constant")
    parser.add_argument("--n_train", type=int, default=10000, help="number of training samples")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--n_layers", type=int, default=3, help="number of layers in network")
    parser.add_argument("--n_units", type=int, default=128, help="number of hidden units in each layer")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size (Should divide Ntest)")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="learning rate")

    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=1,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')

    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node
 
    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('starting in debug mode')
        init_processes(0, size, train, args)