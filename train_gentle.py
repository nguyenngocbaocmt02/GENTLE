import torch
import argparse
import os
import torch.distributed as dist
import pandas as pd
import wandb 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets.datasets import data_preparation
from models.torch_mlp import MLPM
from utils import prim, preprocess_dataset
import math
from scipy.stats import gaussian_kde
from scipy.stats import norm
from utils import LinearInterpolationModel
import copy
import time
#%%
def train(gpu, args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda:{}'.format(gpu))
    wandb.init(project="rl-df", 
            entity="ml-with-vibes", 
            config=vars(args), 
            sync_tensorboard=True,
            mode="disabled",
            )
    
    # Read the config
    batch_size = args.batch_size
    learning_rate_mapping = args.learning_rate_mapping
    learning_rate_potential = args.learning_rate_potential
    num_epochs = args.num_epochs
    dataset = args.dataset
    model_type = args.model_type
    ld = args.ld
    epsilon = args.epsilon
    checkpoint_folder = args.checkpoint_folder
    batch_node = args.batch_node
    bandwith = args.bandwith
    r1 = args.r1 
    r2 = args.r2
    beta = args.beta
    mu = args.mu
    balance = args.balance

    os.makedirs(checkpoint_folder, exist_ok=True)
    tb_dir = os.path.join(checkpoint_folder, "tensorboard")

    # Read the dataset
    df = pd.read_csv(os.path.join("datasets", dataset, "train.csv"))
    x_data, y_data, _ = data_preparation(df, name=dataset, balance=int(balance))
    x_data = x_data.numpy()
    y_data = y_data.numpy()
    input_dim = 1 + x_data.shape[1]
    # Create models
    if model_type == "mono_mlp":
        pass
    elif model_type == "mlp":
        mapping_model = MLPM(input_dim)
        potential_model = MLPM(input_dim)
    mapping_model = mapping_model.to(device)
    potential_model = potential_model.to(device)
    #mapping_model = torch.nn.DataParallel(mapping_model)
    optimizer_mapping = optim.Adam(mapping_model.parameters(), lr=learning_rate_mapping)
    optimizer_potential = optim.Adam(potential_model.parameters(), lr=learning_rate_potential)
    
    # Create the relationship between potentials
    grouped_by_x = {}
    for xi, yi in zip(x_data, y_data):
        current_x = tuple(xi)
        grouped_by_x.setdefault(current_x, []).append(yi)
    distinct_x = np.array([np.array(x) for x in grouped_by_x.keys()])
    y_asc = [np.array(y) for y in grouped_by_x.values()]
    cdfs = []
    for i in range(len(y_asc)):
        try:
            if np.var(y_asc[i]) > 1e-8:
                kde = gaussian_kde(y_asc[i], bw_method=bandwith)
                y_values = np.linspace(min(np.min(y_asc[i]), -10.0), max(np.max(y_asc[i]), 10.0), 1000)
                if np.sum(kde(y_values)) == 0.0:
                    cdf_values = np.cumsum(kde(y_values)) / (np.sum(kde(y_values)) + 1e-8)
                else:
                    cdf_values = np.cumsum(kde(y_values)) / (np.sum(kde(y_values)))
            else:
                cdf_fn = lambda x: norm.cdf(x=x, loc=y_asc[i][0], scale=bandwith)
                y_values = np.linspace(min(np.min(y_asc[i]), -10.0), max(np.max(y_asc[i]), 10.0), 1000)
                cdf_values = cdf_fn(y_values)
        except:
            cdf_fn = lambda x: norm.cdf(x=x, loc=y_asc[i][0], scale=bandwith)
            y_values = np.linspace(min(np.min(y_asc[i]), -10.0), max(np.max(y_asc[i]), 10.0), 1000)
            cdf_values = cdf_fn(y_values)

        cdf_function = LinearInterpolationModel(cdf_values=cdf_values, y_values = y_values).to(device)
        cdfs.append(cdf_function)
    if os.path.exists("relation/" + dataset + "_" + str(bandwith) + ".npy"):
        relation = np.load("relation/" + dataset + "_" + str(bandwith) + ".npy")
    else:
        t_begin = time.time()
        diff = distinct_x[:, np.newaxis, :] - distinct_x[np.newaxis, :, :]
        graph = np.linalg.norm(diff, axis=2)
        edges, _ = prim(graph)
        relation = [-1 for _ in range(distinct_x.shape[0])]
        for edge in edges:
            if relation[edge[1]] == -1:
                relation[edge[1]] = edge[0]
            elif relation[edge[0]] == -1:
                relation[edge[0]] = edge[1]
        relation = np.array(relation)
        np.save("relation/" + dataset + "_" + str(bandwith) + ".npy", relation)
    z_pre = copy.deepcopy(mapping_model.state_dict())
    v_pre = copy.deepcopy(potential_model.state_dict())

    # Training loop
    for epoch in range(num_epochs):
        indices = np.arange(len(distinct_x))
        np.random.shuffle(indices)

        log_l2 = 0.0
        log_reg = 0.0
        log_r1 = 0.0
        log_r2 = 0.0
        cnt = 0

        for begin_node in range(0, len(distinct_x), batch_node):
            cnt += 1
            end_node = min(begin_node + batch_node, len(distinct_x))
            
            # Training the mapping model
            mapping_model.train()
            for param in mapping_model.parameters():
                param.requires_grad = True

            potential_model.eval()
            for param in potential_model.parameters():
                param.requires_grad = False

            l2 = torch.tensor(0.).to(device)
            sum_reg = torch.tensor(0.).to(device)

            for idx in indices[begin_node:end_node]:
                U = torch.rand((batch_size, 1), dtype=torch.float32).to(device)
                X_i = torch.Tensor(distinct_x[idx]).view(1, -1).repeat(batch_size, 1).to(device)
                XU_i = torch.cat([U, X_i], dim=1)
                T_XU_i = mapping_model(XU_i)
                F_inv = cdfs[idx](T_XU_i)
                l2 += torch.mean((U.squeeze() - F_inv.squeeze())**2)
                
                related_node = relation[idx]
                if related_node == -1: 
                    Utm = torch.rand((batch_size, 1), dtype=torch.float32).to(device)
                    X_j = torch.Tensor(distinct_x[related_node]).view(1, -1).repeat(batch_size, 1).to(device)
                    XU_j = torch.cat([Utm, X_j], dim=1)
                    T_XU_j = mapping_model(XU_j).unsqueeze(1)

                    X_j_tmp = torch.cat([T_XU_j, X_j], dim=1)
                    v_values = potential_model(X_j_tmp).unsqueeze(1)

                    pwdis = torch.square(T_XU_i.squeeze()[:, None] - T_XU_j.squeeze()[None, :])
                    v_c = -epsilon * torch.log(torch.mean(torch.exp((v_values.squeeze()[None, :] - pwdis) / epsilon), dim=1))
                    reg = torch.mean(v_c) + torch.mean(v_values) - epsilon
                    sum_reg += reg
            
            l2 /= (end_node - begin_node)
            sum_reg /= (end_node - begin_node)
            term_r1 = torch.tensor(0.).to(device)
            for name, param in mapping_model.named_parameters():
                term_r1 += torch.sum((param - z_pre[name]) ** 2)
            loss = l2 + ld * sum_reg + 0.5 * r1 * term_r1
            optimizer_mapping.zero_grad()
            loss.backward()
            optimizer_mapping.step()
            log_l2 += l2.item()
            log_r1 += term_r1.item()



            # Training the potential model
            potential_model.train()
            for param in potential_model.parameters():
                param.requires_grad = True

            mapping_model.eval()
            for param in mapping_model.parameters():
                param.requires_grad = False

            sum_reg = torch.tensor(0.).to(device)
            for idx in indices[begin_node:end_node]:
                related_node = relation[idx]
                if related_node == -1:
                    continue
                U1 = torch.rand((batch_size, 1), dtype=torch.float32).to(device)

                X_i = torch.Tensor(distinct_x[idx]).view(1, -1).repeat(batch_size, 1).to(device)
                XU_i = torch.cat([U1, X_i], dim=1)
                T_XU_i = mapping_model(XU_i).unsqueeze(1)
                
                U2 = torch.rand((batch_size, 1), dtype=torch.float32).to(device)
                X_j = torch.Tensor(distinct_x[related_node]).view(1, -1).repeat(batch_size, 1).to(device)
                XU_j = torch.cat([U2, X_j], dim=1)
                T_XU_j = mapping_model(XU_j).unsqueeze(1)

                X_j_tmp = torch.cat([T_XU_j, X_j], dim=1)
                v_values = potential_model(X_j_tmp).unsqueeze(1)
                
                pwdis = torch.abs(T_XU_i.squeeze()[:, None] - T_XU_j.squeeze()[None, :])
                v_c = -epsilon * torch.log(torch.mean(torch.exp((v_values.squeeze()[None, :] - pwdis) / epsilon), dim=1))
                
                reg = torch.mean(v_c) + torch.mean(v_values) - epsilon
                #reg = loss_semi_dual_entropic(v_values.squeeze(), T_XU_i.reshape(T_XU_i.shape[0], -1), T_XU_j.reshape(T_XU_j.shape[0], -1), reg=epsilon, metric="sqeuclidean")
                sum_reg += reg
                
            sum_reg /= (end_node - begin_node)

            term_r2 = torch.tensor(0.).to(device)
            for name, param in potential_model.named_parameters():
                term_r2 += torch.sum((param - v_pre[name]) ** 2)
            sum_reg_r2 = ld * sum_reg - 0.5 * r2  * term_r2
            optimizer_potential.zero_grad()
            (-sum_reg_r2).backward()
            optimizer_potential.step()

            log_r2 += term_r2.item()
            log_reg += sum_reg.item()

            with torch.no_grad():
                # Update r1_term, r2_term
                z_new = mapping_model.state_dict()
                z_pre = {key: (1 - beta) * z_pre[key] + beta * z_new[key] for key in z_pre.keys()}

                v_new = potential_model.state_dict()
                v_pre = {key: (1 - mu) * v_pre[key] + mu * v_new[key] for key in v_pre.keys()}

        log_l2 /= cnt
        log_reg /= cnt
        log_r1 /= cnt
        log_r2 /= cnt
        if epoch % 1 == 0:
            torch.save(potential_model.state_dict(), os.path.join(checkpoint_folder, 'final_potential_model.pth'))
            torch.save(mapping_model.state_dict(), os.path.join(checkpoint_folder, 'final_mapping_model.pth'))
            torch.save(z_pre, os.path.join(checkpoint_folder, 'z_pre.pth'))
            torch.save(v_pre, os.path.join(checkpoint_folder, 'v_pre.pth'))
        print(f"Epoch {epoch + 1}/{num_epochs}, L2 Loss: {log_l2:.4f}, Regularization: {ld * log_reg:.4f}, R1: {0.5 * r1 * log_r1:.10f}, R2: {- 0.5 * r2  * log_r2:.10f}")

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
    parser.add_argument('--model_type', type=str, default="mlp",
                        help='model type')
    parser.add_argument('--dataset', type=str, default="cps",
                        help='name of dataset, check the folder datasets')
    parser.add_argument('--checkpoint_folder', type=str, default="checkpoints/cps",
                        help='folder to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('--batch_node', type=int, default=1000,
                        help='batch size for node loop')
    parser.add_argument('--learning_rate_potential', type=float, default=1e-3,
                        help='learning_rate')
    parser.add_argument('--learning_rate_mapping', type=float, default=1e-3,
                        help='learning_rate')
    parser.add_argument('--ld', type=float, default=0.3,
                        help='lambda')
    parser.add_argument('--epsilon', type=float, default=1.0,  
                        help='epsilon')
    parser.add_argument('--bandwith', type=float, default=0.3,
                        help='the bandwith for  KDE')
    parser.add_argument('--r1', type=float, default=3.0,
                        help='r1')
    parser.add_argument('--r2', type=float, default=2.0,
                        help='r2')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='gamma')
    parser.add_argument('--mu', type=float, default=0.7,
                        help='delta')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='num_epochs')
    parser.add_argument('--balance', type=bool, default=1,
                        help='balance the data or not')
    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    
    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node
    '''
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
    '''
    print('starting in debug mode')
    train(args.node_rank, args)