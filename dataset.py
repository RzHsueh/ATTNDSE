from random import sample
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import paretoFront
import random
from config import args

PE = args.position_encode
shuffle_indices = args.shuffle_indices

def shuffle_with_indices(origin_data, shuffle_indices):
    shuffled_list = [origin_data[i] for i in shuffle_indices]
    return shuffled_list


def TxTforCSV(filename=None):
    assert filename is not None, "file not found!"

    parameter, ipc, power, area = [], [], [], []

    print(f"Loading data...")
    with open(filename, 'r') as f:
        txt_content = f.read()

    for line in txt_content.split('\n'):   
        if not line.strip():
            continue
        columns = line.split(' ')
        parameter.append(columns[0])
        ipc.append(float(columns[1]))
        power.append(float(columns[2]))
        area.append(float(columns[3]))
        
    print(f"Finish data loading.")
    return parameter, ipc, power, area

# data/train_data/996.specrand_fs.txt
class DSEDataset(Dataset):
    def __init__(self, path: str = "data/train_data/996.specrand_fs.txt", mode: str = "train", target: str = "ipc"):
        super().__init__()
        assert path is not None, "data path must be specified."
        assert mode in ["train", "val", "test"]
        assert target in ["ipc", "power", "area"]

        self.target = target
        parameter, ipc, power, area = TxTforCSV(filename=path)
        self.ipc = torch.Tensor(ipc)
        self.power = torch.Tensor(power)
        self.area = torch.Tensor(area)
        self.parameters = []
        for i, item in enumerate(parameter):
            self.parameters.append([int(num) for num in item.split(',')])

        print(f"length of parameters is {len(self.parameters)}")
        if PE == "random":
            print(f"shuffle_indices is {shuffle_indices}")
            for i in range(len(self.parameters)):
                self.parameters[i] = shuffle_with_indices(self.parameters[i], shuffle_indices)

        train_indices = list(range(1000))
        if mode == "train":
            self.parameters = [self.parameters[i] for i in train_indices]
            self.ipc = torch.Tensor([self.ipc[i] for i in train_indices])
            self.power = torch.Tensor([self.power[i] for i in train_indices])
            self.area = torch.Tensor([self.area[i] for i in train_indices])
        elif mode == "val":
            self.parameters = [self.parameters[i] for i in range(len(self.parameters)) if i not in train_indices]
            self.ipc = torch.Tensor([self.ipc[i] for i in range(len(self.parameters)) if i not in train_indices])
            self.power = torch.Tensor([self.power[i] for i in range(len(self.parameters)) if i not in train_indices])
            self.area = torch.Tensor([self.area[i] for i in range(len(self.parameters)) if i not in train_indices])
        else:# test set
            pass

        self.parameters = torch.Tensor(self.parameters)


    def __len__(self):
        if self.target == "ipc":
            return self.ipc.shape[0]
        elif self.target == "power":
            return self.power.shape[0]
        elif self.target == "area":
            return self.area.shape[0]
    
    def __getitem__(self, index):

        arch_vec = self.parameters[index].to(torch.int32)
        ipc = self.ipc[index]
        power = self.power[index]
        area = self.area[index]

        if self.target == "ipc":
            return arch_vec, ipc
        elif self.target == "power":
            return arch_vec, power
        elif self.target == "area":
            return arch_vec, area



