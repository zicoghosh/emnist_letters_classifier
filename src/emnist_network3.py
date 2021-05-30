"""
The CNN Network
"""
import torch
import torch.nn as nn

class EMNIST(nn.Module):
    
    def __init__(self, n_classes):
        super(EMNIST, self).__init__()
        #Sequential() combines different functions into a module using:
        #Define the network framework:
        self.num_classes = n_classes
        self.Conv1 = nn.Sequential(
            #Convolution layer 1 (convolution core=16)
            nn.Conv2d(
 				in_channels = 1,   #Number of channels for input image, i.e. input height is 1
 				out_channels = 16, #Define 16 convolution cores, i.e. 16 output heights
 				kernel_size = 5,   #The convolution core size is (5,5)
 				stride = 1,        #step
 				padding = 2,       #The boundary filling is 0 (for example, if the step size is 1, to ensure that the output size image is consistent with the original size, the formula is: padding = (kernel_size-1)/2)
 			),
 			#Activation function layer
            nn.ReLU(),
            #Maximum pooled layer
            nn.MaxPool2d(kernel_size = 2)
        )
        self.Conv2 = nn.Sequential(
            #Convolution Layer 2
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.Dropout(p=0.2),
            #Activation function layer
            nn.ReLU(),
            #Maximum pooled layer
            nn.MaxPool2d(kernel_size = 2)
        )
        #Finally, connect three layers of full connection (to make the image one-dimensional)
        #Why 32*7*7:(1,28,28) -> (16,28,28) (conv1) -> (16,14,14) (pool1) -> (32,14,14) (conv2) -> (32,7,7) (pool2) ->output
        self.Linear = nn.Sequential(
            nn.Linear(32*7*7,400),
            #Dropout randomly discards some neurons by probability p
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(400,80),
            nn.ReLU(),
            nn.Linear(80,self.num_classes),
         )
    #Forward propagation:
    def forward(self, input):
        input = self.Conv1(input)
        input = self.Conv2(input)       #view can be understood as resize
        #input.size() = [100, 32, 7, 7], 100 is quantity per batch, 32 is thickness, picture size is 7*7
        #When a dimension is -1, its size is automatically calculated (the principle is that the total amount of data is constant):
        input = input.view(input.size(0), -1) #(batch=100, 1568), the end result is to compress a two-dimensional picture into one dimension (the amount of data remains constant)
        #Finally, connect to a full connection layer with output of 10:[100,1568]*[1568,10]=[100,10]
        output = self.Linear(input)
        return output
