import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The few-shot classifier model as used in Reptile.
Repeat of block for 4 times:
    conv with 32 channels, a 3*3 kernel, and a stride of 1
    batchnorm 
    maxpool with 2*2 kernel and a stride of 2
    relu activation

a linear layer from flattened output of size 32*5*5 to number of ways.
'''

class EMNIST(nn.Module):

    def __init__(self, n_classes, p=0.10):

        super(EMNIST, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=(1, 1), padding_mode='zeros')
        self.batch1 = nn.BatchNorm2d(32, track_running_stats=False)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(1, 1), padding_mode='zeros')
        self.batch2 = nn.BatchNorm2d(32, track_running_stats=False)
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(1, 1), padding_mode='zeros')
        self.batch3 = nn.BatchNorm2d(32, track_running_stats=False)
        
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(1, 1), padding_mode='zeros')
        self.batch4 = nn.BatchNorm2d(32, track_running_stats=False)

        self.lin1 = nn.Linear(32*5*5, n_classes)

        self.dropout_percent = p
        self.drop_out = nn.Dropout(p)


    def forward(self, x):

        x = F.relu(F.max_pool2d(self.batch1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.batch2(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.batch3(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.batch4(self.conv4(x)), 2))
        x = x.view(-1, 32*5*5)
        x = self.lin1(x)
        x = self.drop_out(x)
        
        return x


    def functional_forward(self, x, weight_dict, train_mode = False):

        x = F.conv2d(x, weight_dict['conv1.weight'], weight_dict['conv1.bias'], padding=(1, 1))
        x = F.batch_norm(x, running_mean=None, running_var=None, 
            weight=weight_dict['batch1.weight'], bias=weight_dict['batch1.bias'], training=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        x = F.conv2d(x, weight_dict['conv2.weight'], weight_dict['conv2.bias'], padding=(1, 1))
        x = F.batch_norm(x, running_mean=None, running_var=None, 
            weight=weight_dict['batch2.weight'], bias=weight_dict['batch2.bias'], training=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        x = F.conv2d(x, weight_dict['conv3.weight'], weight_dict['conv3.bias'], padding=(1, 1))
        x = F.batch_norm(x, running_mean=None, running_var=None,
            weight=weight_dict['batch3.weight'], bias=weight_dict['batch3.bias'], training=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        x = F.conv2d(x, weight_dict['conv4.weight'], weight_dict['conv4.bias'], padding=(1, 1))
        x = F.batch_norm(x, running_mean=None, running_var=None,
            weight=weight_dict['batch4.weight'], bias=weight_dict['batch4.bias'], training=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        x = x.view(-1, 32*5*5)
        
        x = F.linear(x, weight=weight_dict['lin1.weight'], bias=weight_dict['lin1.bias'])
        
        if train_mode:
            x = F.dropout(x, p=self.dropout_percent, training=True, inplace=False)

        return x