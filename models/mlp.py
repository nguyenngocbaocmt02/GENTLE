# my_model.py

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2

def build_mlp(input_dim=3):
    model = Sequential()

    model.add(Input(shape=(input_dim,)))

    model.add(Dense(256, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    
    # Output layer
    model.add(Dense(1))

    return model
