import torch
import argparse
import os
from torch.multiprocessing import Process
import torch.distributed as dist
import torch
import pandas as pd
from datasets.datasets import data_preparation
import numpy as np
import wandb


from baselines.cdsb.bridge.runners.ipf import IPFAnalytic, IPFSequential
from baselines.cdsb.bridge.runners.accelerator import Accelerator

#%%
def train(rank, gpu, opt):
    # wandb.init(project="rl-df", 
    #         entity="ml-with-vibes", 
    #         config=vars(args), 
    #         sync_tensorboard=True,
    #         #mode="disabled",
    #         )
    
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    torch.manual_seed(opt.seed + rank)
    torch.cuda.manual_seed(opt.seed + rank)
    torch.cuda.manual_seed_all(opt.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))
    batch_size = opt.batch_size

    df = pd.read_csv(os.path.join("datasets", opt.dataset, "train.csv"))
    x_data, y_data, _ = data_preparation(df, name=opt.dataset, balance=int(opt.balance))
    dataset = torch.utils.data.TensorDataset(y_data.unsqueeze(1), x_data)
    args.x_dim = 1
    args.y_dim = x_data.shape[1]
    args.y_cond = [f'torch.randn({args.y_dim})', f'torch.randn({args.y_dim})', f'torch.randn({args.y_dim})']
    args.x_cond_true = None
    # test: sampled_x, sampled_y = [x for x in dataset][0]  -> x ([1]), y ([9])
    
    accelerator = Accelerator(train_batch_size=batch_size, cpu=False, split_batches=True)
    ipf = IPFSequential(dataset, final_ds=None, mean_final=torch.tensor([0.]), var_final=10.*torch.tensor([1.]), args=opt, accelerator=accelerator,
                          final_cond_model=None, valid_ds=None, test_ds=None)
    # ipf = IPFAnalytic(dataset, final_ds=None, mean_final=torch.tensor([0.]), var_final=10.*torch.tensor([1.]), args=opt, accelerator=accelerator,
    #                       final_cond_model=None, valid_ds=None, test_ds=None)
    accelerator.print(accelerator.state)
    accelerator.print(ipf.net['b'])
    accelerator.print('Number of parameters:', sum(p.numel() for p in ipf.net['b'].parameters() if p.requires_grad))
    ipf.train()
            


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6022'
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
    parser.add_argument('--balance', type=bool, default=0,
                        help='balance the data or not')
    parser.add_argument('--save_every', type=int, default=100,
                        help='print training information frequency')
    parser.add_argument('--checkpoint_folder', type=str, default="checkpoints/cdsb/cancer",
                        help='folder to save checkpoints')
    parser.add_argument("--batch_size", type=int, default=5000, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")

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
    
    
    ################################## CDSB args ####################################
    
    #### important params ####
    args.n_ipf = 10
    args.num_iter = 500
    # args.num_iter = 10000
    
    args.name = f"{args.dataset}"
    args.Dataset = args.dataset
    args.data = {'dataset': 'type1'}
    args.num_workers = 0
    args.pin_memory = False
    # args.model = {'deg': 2, 'basis': 'rbf', 'x_radius': None, 'y_radius': None, 'use_ridgecv': True, 'alphas': [1e-06, 0.001, 0.1, 1.0], 'use_fp16': False}
    args.model = {'encoder_layers': [16], 'temb_dim': 16, 'decoder_layers': [128, 128], 'temb_max_period': 1000, 'use_fp16': False}
    args.Model = 'BasicCond'
    args.transfer = False
    args.adaptive_mean = False 
    args.final_adaptive = False
    args.cond_final = False
    args.use_prev_net = False
    args.mean_match = False
    args.ema = False
    args.ema_rate = 0.999
    args.grad_clipping = False
    args.grad_clip = 1.0
    args.npar = 10000
    
    args.cache_npar = 10000
    args.lr = 0.0001
    args.num_steps =  10
    args.gamma_max = 0.1
    args.gamma_min = 0.1
    args.gamma_space = "linspace"
    args.weight_distrib = False
    args.weight_distrib_alpha = 100
    args.fast_sampling = True
    args.symmetric_gamma = False
    args.var_final_gamma_scale = False
    args.double_gamma_scale = True
    args.langevin_scale = "2*torch.sqrt(gamma)"
    args.loss_scale = 1.
    # checkpoint
    args.checkpoint_run = False
    args.checkpoint_it = 1
    args.checkpoint_pass = "b"  # b or f (skip b ipf run)
    args.checkpoint_iter = 0
    args.checkpoint_dir = None
    args.sample_checkpoint_f = None
    args.sample_checkpoint_b = f"{args.checkpoint_dir}/"
    args.checkpoint_f = None
    args.checkpoint_b = f"{args.checkpoint_dir}/"
    args.optimizer_checkpoint_f = None
    args.optimizer_checkpoint_b = f"{args.checkpoint_dir}/"
    args.LOGGER = "CSV"
    args.CSV_log_dir = "./"
    args.run = 0
    args.nosave = False
    args.plot_npar = 2000
    args.test_npar = 10000
    args.log_stride = 50
    args.gif_stride = args.num_iter
    
    args.optimizer = "Adam"
    args.cache_cpu = True
    args.num_cache_batches = 1
    args.cache_refresh_stride = 1000
    args.test_batch_size = 10000
    args.plot_level = 1
    args.paths = {'experiments_dir_name': 'experiments', 'data_dir_name': 'data'}
    #################################################################################
    
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