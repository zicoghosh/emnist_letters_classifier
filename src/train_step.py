"""
Model agnostic meta-learning and evaluation on arbitrary
datasets.
"""
import random
import itertools
import ntpath

import torch
import torch.nn.functional as F
from torch import optim
import numpy as np

import torch.optim as optim
from copy import deepcopy
from collections import OrderedDict

import torchvision.utils as vutils

class STEP:

    def __init__(self, model, device, update_lr):

        self.device = device
        self.net = model.to(self.device)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=update_lr)
        # self.update_lr = update_lr


    def train_step(self, dataloader):
                
        create_graph, retain_graph = True, True
        
        # if order == 1:
        #     create_graph, retain_graph = False, False

        new_grads = []

        self.meta_optim.zero_grad()

        weight = OrderedDict(self.net.named_parameters())

        for batch_X, batch_y in dataloader:
            inputs = (torch.stack(batch_X)).to(self.device)
            labels = (torch.tensor(batch_y)).to(self.device)

            logits = self.net.functional_forward(inputs, weight)
            loss = F.cross_entropy(logits, labels)
            loss.backward(retain_graph=retain_graph)

            self.meta_optim.step()
            self.meta_optim.zero_grad()


    def evaluate(self, dataset):

        weight = OrderedDict(self.net.named_parameters())

        inputs, labels = zip(*dataset)
        inputs = (torch.stack(inputs)).to(self.device)
        labels = (torch.tensor(labels)).to(self.device)

        logits = self.net.functional_forward(inputs, weight)
        test_preds = (F.softmax(logits, dim=1)).argmax(dim=1)

        num_correct = torch.eq(test_preds, labels).sum()

        return num_correct.item()