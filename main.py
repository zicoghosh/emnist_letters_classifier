"""
Date : 29/05/2021

DPR Assignment 4 : Handwritten Upper-Case English Character Recognition using Convolution Neural Network
"""

import random
import os 
import sys
import numpy as np
import argparse
from datetime import datetime

import torch

project_path = "/home/sankha/Surjayan-archive/dpr/DPR_Assignment_4/classifier_emnist"
sys.path.insert(0,project_path)

DATA_DIR = project_path + '/Data/'

from process_dataset import read_dataset
from src.emnist_network3 import EMNIST
from src.train_model import train
from src.train_model import MODEL_STEP

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--classes", type=int, default=26)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument('--only-evaluation', help='Set for only evaluation', action='store_true', default=False)
    parser.add_argument('--checkpoint', help='Load saved checkpoint from path', default=None)
    return parser

def main():
    sys.dont_write_bytecode = True

    args = argument_parser().parse_args()
    torch.manual_seed(args.seed)

    dt = datetime.now()
    dt_str = dt.strftime("%d_%m_%Y_%H_%M_%S")

    filestart = 'ENGLISH_HW_UC_classfier_EMNIST_Net_3' + '_learning_rate_' + str(args.lr) + '_epochs_' + str(args.epochs) + '_datetime_' + dt_str
    print(filestart)

    if not os.path.exists('saved_model/EMNIST3/' + filestart):
        os.makedirs('saved_model/EMNIST3/' + filestart)

    device = torch.device('cuda')

    train_dataloader, train_eval_loader, test_dataloader = read_dataset(DATA_DIR, "letters")

    EMNIST_net = EMNIST(n_classes = args.classes)
    EMNIST_model = MODEL_STEP(EMNIST_net, device, update_lr = args.lr)

    if args.only_evaluation is True :
        checkpoint_model = torch.load(args.checkpoint)
        EMNIST_model.net.load_state_dict(checkpoint_model['model_state'])
        EMNIST_model.optim.load_state_dict(checkpoint_model['optim_state'])
    else:
        # starting_point = "/home/sankha/Surjayan-archive/dpr/DPR_Assignment_4/classifier_emnist/saved_model/EMNIST/ENGLISH_HW_UC_classfier_model_29_05_2021_06_01_41/intermediate_90_model.pt"
        # checkpoint_model = torch.load(starting_point)
        # EMNIST_model.net.load_state_dict(checkpoint_model['model_state'])
        # EMNIST_model.optim.load_state_dict(checkpoint_model['optim_state'])

        print("\nStarting Training...")
        train(EMNIST_model, train_dataloader, train_eval_loader, test_dataloader, model_save_path = 'saved_model/EMNIST3/' + filestart, epochs = args.epochs, eval_interval = args.eval_interval)
        print("\nTraining Complete...")
        model_save_path = 'saved_model/EMNIST3/' + filestart
        print("\nFinal Model saved at ",model_save_path)
    
    print("\nFinal Evaluation")
    train_acc = EMNIST_model.evaluate(train_eval_loader)
    test_acc = EMNIST_model.evaluate(test_dataloader)
    print('train=%f test=%f' % (train_acc, test_acc))

if __name__ == '__main__' :
    main()
