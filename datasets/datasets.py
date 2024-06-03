import torch
import argparse
import os
from torch.multiprocessing import Process
import torch.distributed as dist
import pandas as pd
import wandb
import wgan 
import numpy as np

def data_preparation(df, name="cps", balance=0):
    if name == "cps" or name == "experimental" or name == "psid":
        if balance == 1:
            df = df.sample(2*len(df), weights=(1-df.t.mean())*df.t+df.t.mean()*(1-df.t), replace=True, random_state=1000) # balanced df for training
        continuous_vars_1 = ["re78"]
        continuous_lower_bounds_1 = {"re78": 0}
        categorical_vars_1 = []
        context_vars_1 = ["t", "age", "education", "re74", "re75", "black", "hispanic", "married", "nodegree"]

        # Initialize objects
        data_wrapper = wgan.DataWrapper(df, continuous_vars_1, categorical_vars_1, 
                                        context_vars_1, continuous_lower_bounds_1)
        y, x = data_wrapper.preprocess(df)
        return x, y.squeeze(), data_wrapper

    elif name == "toy1":
        continuous_vars_1 = ["y"]
        continuous_lower_bounds_1 = {}
        categorical_vars_1 = []
        context_vars_1 = ["x1", "x2", "x3"]

        # Initialize objects
        data_wrapper = wgan.DataWrapper(df, continuous_vars_1, categorical_vars_1, 
                                        context_vars_1, continuous_lower_bounds_1)
        y, x = data_wrapper.preprocess(df)
        return x, y.squeeze(), data_wrapper
    
    elif name == "toy2":
        continuous_vars_1 = ["y"]
        continuous_lower_bounds_1 = {}
        categorical_vars_1 = []
        context_vars_1 = ["x1", "x2"]

        # Initialize objects
        data_wrapper = wgan.DataWrapper(df, continuous_vars_1, categorical_vars_1, 
                                        context_vars_1, continuous_lower_bounds_1)
        y, x = data_wrapper.preprocess(df)
        return x, y.squeeze(), data_wrapper

    elif name == "cancer":
        continuous_vars_1 = ["QALY"]
        continuous_lower_bounds_1 = {}
        categorical_vars_1 = []
        context_vars_1 = ["Barrett", "aspirinEffect", 'statinEffect', 'drugIndex', 'initialAge']

        # Initialize objects
        data_wrapper = wgan.DataWrapper(df, continuous_vars_1, categorical_vars_1, 
                                        context_vars_1, continuous_lower_bounds_1)
        y, x = data_wrapper.preprocess(df)
        return x, y.squeeze(), data_wrapper
    else:
        raise NotImplementedError(
            f'Dataset {name} not yet supported.')
    
