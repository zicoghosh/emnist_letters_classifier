import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

# project_path = "/home/zico/MTechCS/courses/DPR/DPR_Assignment_4"
# DATA_DIR = project_path + '/Data/'

def read_dataset(root, split):
    
    training_data = datasets.EMNIST(root = root , split = split, train = True, download = True, transform = ToTensor())
    test_data = datasets.EMNIST(root = root , split = split, train = False, download = True, transform = ToTensor())
    
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    train_eval_loader = DataLoader(training_data, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
    
    return train_dataloader, train_eval_loader, test_dataloader

# train_dataloader, train_eval_loader, test_dataloader = read_dataset()

# print(len(train_dataloader.dataset))
# print(len(test_dataloader.dataset))
# print(len(train_eval_loader.dataset))