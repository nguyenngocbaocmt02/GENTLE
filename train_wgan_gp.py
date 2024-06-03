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

def compute_gradient_penalty(D, real_samples, fake_samples, condition):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    interpolates_input = torch.cat([interpolates.view(interpolates.size(0), -1), condition], dim=1)
    d_interpolates = D(interpolates_input)
    fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).squeeze().to(real_samples.device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates_input,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


#%%
def train(rank, gpu, opt):
    wandb.init(project="rl-df", 
            entity="ml-with-vibes", 
            config=vars(args), 
            sync_tensorboard=True,
            #mode="disabled",
            )
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    torch.manual_seed(opt.seed + rank)
    torch.cuda.manual_seed(opt.seed + rank)
    torch.cuda.manual_seed_all(opt.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))
    batch_size = opt.batch_size
    lambda_gp = opt.lambda_gp
    img_shape = (opt.channels, opt.img_size, opt.img_size)
    os.makedirs(opt.checkpoint_folder, exist_ok=True)

    df = pd.read_csv(os.path.join("datasets", opt.dataset, "train.csv"))
    x_data, y_data, _ = data_preparation(df, name=opt.dataset, balance=int(opt.balance))
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    generator = MLPM(input_dim=x_data.shape[1]+1).to(device)
    discriminator = MLPM(x_data.shape[1] + 1).to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (condition, imgs) in enumerate(dataloader):
            batch_shape = (imgs.shape[0],) + img_shape
            imgs = imgs.reshape(batch_shape)
            # Configure input
            real_imgs = Variable(imgs.type(Tensor)).to(device)
            condition = condition.to(device)
            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Generate a batch of images
            z = torch.rand((batch_shape[0], 1), dtype=torch.float32).to(device)
            fake_imgs = generator(torch.cat([z, condition], dim=1)).reshape(batch_shape)

            # Real images
            real_validity = discriminator(torch.cat([real_imgs.view(real_imgs.size(0), -1), condition], dim=1))
            # Fake images
            fake_validity = discriminator(torch.cat([fake_imgs.view(real_imgs.size(0), -1), condition], dim=1))
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, condition)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs = generator(torch.cat([z, condition], dim=1))
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(torch.cat([fake_imgs.view(fake_imgs.size(0), -1), condition], dim=1))
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()
                wandb.log({"D_loss": float(d_loss)})
                wandb.log({"G_loss": float(g_loss)})
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )
                if epoch % 10 == 0:
                    torch.save(generator.state_dict(), os.path.join(opt.checkpoint_folder, 'generator.pth'))
                    torch.save(discriminator.state_dict(), os.path.join(opt.checkpoint_folder, 'discriminator.pth'))
                batches_done += opt.n_critic


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '12002'
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
    parser.add_argument('--checkpoint_folder', type=str, default="checkpoints/wgan_gp/cancer",
                        help='folder to save checkpoints')
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=5000, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=1, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    parser.add_argument("--lambda_gp", type=float, default=10.0, help="Loss weight for gradient")

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