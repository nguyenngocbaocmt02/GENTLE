import argparse
import os
import  ml_collections

def get_arguments():
    args = ml_collections.ConfigDict()

    ################################## CDSB args ####################################
    args.seed = 42
    args.dataset = "cps"
    args.balance = True
    args.save_every = 100
    args.checkpoint_folder = "checkpoints/wgan_gp/cps"
    args.batch_size = 5000
    args.n_cpu = 8
    args.channels = 1
    args.num_proc_node = 1
    args.num_process_per_node = 1
    args.node_rank = 0
    args.local_rank = 1
    args.master_address = '127.0.0.1'
    args.infer = True
    
    # CHECKPOINT RUN  - FOR INFERENCE
    args.checkpoint_run = False
    args.checkpoint_it = 2
    args.checkpoint_pass = "b"  # b or f (skip b ipf run)
    args.checkpoint_iter = 0
    args.checkpoint_dir = "checkpoints/version_0"
    args.sample_checkpoint_f = None
    args.sample_checkpoint_b = None
    args.checkpoint_f = os.path.join(args.checkpoint_dir, "net_f_20_5000.ckpt")
    args.checkpoint_b = os.path.join(args.checkpoint_dir, "net_b_20_5000.ckpt")
    args.optimizer_checkpoint_f =  os.path.join(args.checkpoint_dir, "optimizer_f_20_5000.ckpt")
    args.optimizer_checkpoint_b = os.path.join(args.checkpoint_dir, "optimizer_b_20_5000.ckpt")

    #### important params ####
    args.n_ipf = 20
    args.num_iter = 5000
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
    args.num_steps =  20
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
    
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    return args
