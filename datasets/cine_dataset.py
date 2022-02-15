"""
Dataset Library
"""
import torch
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
from torch.utils.data import Dataset



def read_h5py(path):
  with h5py.File(path, "r") as f:
    print(f"Reading from {path} ====================================================")
    print("Keys in the h5py file : %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data1 = np.array((f[a_group_key]))
    print(f"Number of samples : {len(data1)}")
    print(f"Shape of each data : {data1.shape}")
    return data1