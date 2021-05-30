"""
Training helpers.
"""

import os
import random
import itertools
import ntpath

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import optim
import numpy as np

import torch.optim as optim
from copy import deepcopy
from collections import OrderedDict

import torchvision.utils as vutils


class MODEL_STEP:

    def __init__(self, model, device, update_lr):

        self.device = device
        self.net = model.to(self.device)
        self.optim = optim.Adam(self.net.parameters(), lr=update_lr)
        self.loss_fn = CrossEntropyLoss()


    def train_step(self, dataloader):
                
        create_graph, retain_graph = True, True
        
        # if order == 1:
        #     create_graph, retain_graph = False, False
        
        self.optim.zero_grad()

        weight = OrderedDict(self.net.named_parameters())

        for batch_X, batch_y in dataloader:
            inputs = batch_X.to(self.device)
            labels = batch_y.to(self.device)
            labels = labels - 1
            # labels = F.one_hot(labels, num_classes=26)

            try:
                logits = self.net.functional_forward(inputs, weight, train_mode=True)
            except:
                logits = self.net.forward(inputs)

            loss = self.loss_fn(logits, labels)
            loss.backward(retain_graph=retain_graph)

            self.optim.step()
            self.optim.zero_grad()


    def evaluate(self, dataloader):

        weight = OrderedDict(self.net.named_parameters())
        num_correct = 0

        for batch_X, batch_y in dataloader:
            inputs = batch_X.to(self.device)
            labels = batch_y.to(self.device)
            labels = labels - 1
            
            try:
                logits = self.net.functional_forward(inputs, weight)
            except:
                logits = self.net.forward(inputs)
      
            test_preds = (F.softmax(logits, dim=1)).argmax(dim=1)

            num_correct += torch.eq(test_preds, labels).sum()

        acc = num_correct.item()/len(dataloader.dataset)
        return acc


def train(model,
        train_dataloader,
        train_eval_loader,
        test_dataloader,
        model_save_path=None,
        epochs = 400,
        eval_interval = 50):
    """
    Train a model on a dataset.
    """
    train_accuracy, test_accuracy = [], []
    
    for i in range(epochs):

        model.train_step(train_dataloader)

        if i % eval_interval == 0:
            
            train_acc = model.evaluate(train_eval_loader)
            test_acc = model.evaluate(test_dataloader)

            train_accuracy.append(train_acc)
            test_accuracy.append(test_acc)

            print('Epoch %d: train=%f test=%f' % (i, train_accuracy[-1], test_accuracy[-1]))

            save_path = model_save_path + '/intermediate_' + str(i) + '_model.pt'
            torch.save({'model_state': model.net.state_dict(),
                        'optim_state': model.optim.state_dict()},save_path)

    final_save_path = model_save_path + '/final_model.pt'
    torch.save({'model_state': model.net.state_dict(),
                        'optim_state': model.optim.state_dict()},final_save_path)

    res_save_path = model_save_path + '/' + 'intermediate_accuracies.npz'

    np.savez(res_save_path, train_accuracy=np.array(train_accuracy),
        test_accuracy=np.array(test_accuracy))