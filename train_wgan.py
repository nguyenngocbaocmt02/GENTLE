import torch
import argparse
import os
from torch.multiprocessing import Process
import torch.distributed as dist
import pandas as pd
import wandb
import wgan   

from datasets.datasets import data_preparation

#%%
def train(rank, gpu, args):
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))
    batch_size = args.batch_size
    critic_learning_rate = args.critic_learning_rate
    generator_learning_rate = args.generator_learning_rate
    num_epochs = args.num_epochs
    dataset = args.dataset
    checkout_freq = args.checkout_freq
    save_every = args.save_every

    checkpoint_path = args.checkpoint_path
    #os.makedirs(checkpoint_folder, exist_ok=True)

    df = pd.read_csv(os.path.join("datasets", dataset, "train.csv"))
    _, _, dw = data_preparation(df, name=dataset, balance=0)
    y, x = dw.preprocess(df)
    
    spec = wgan.Specifications(dw, batch_size=batch_size, max_epochs=num_epochs, critic_lr=critic_learning_rate, generator_lr=generator_learning_rate,
                                print_every=checkout_freq, save_checkpoint=checkpoint_path, save_every=save_every, device = device)
    generator = wgan.Generator(spec)
    critic = wgan.Critic(spec)
    wgan.train(generator, critic, y, x, spec)

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
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints/wgan/cancer.pth",
                        help='folder to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='batch size')
    parser.add_argument('--critic_learning_rate', type=float, default=1e-3,
                        help='learning_rate')
    parser.add_argument('--generator_learning_rate', type=float, default=1e-3,
                        help='learning_rate')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='num_epochs')
    parser.add_argument('--checkout_freq', type=int, default=100,
                        help='print training information frequency')
    parser.add_argument('--save_every', type=int, default=100,
                        help='print training information frequency')

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