# my_model.py

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from airt.keras.layers import MonoDense
import numpy as np
import tensorflow as tf

def build_mono_mlp(input_dim=3):
    model = Sequential()

    model.add(Input(shape=(input_dim,)))
    
    monotonicity_indicator = [1] + [0] * (input_dim - 1)

    model.add(MonoDense(256, activation="relu", monotonicity_indicator=monotonicity_indicator))
    model.add(MonoDense(256, activation="relu"))
    model.add(MonoDense(128, activation="relu"))
    model.add(MonoDense(128, activation="relu"))
    model.add(MonoDense(64, activation="relu"))
    model.add(MonoDense(64, activation="relu"))
    
    # Output layer
    model.add(MonoDense(1))

    return model
