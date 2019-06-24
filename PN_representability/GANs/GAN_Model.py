import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Generator(nn.Module):
    def __init__(self,input_size,output_size,drop_p=0.05):
        """  Builds autoencoder network with m as the code dimensionality.

        Arguments
        ----------
        input_size : integer, size of the input
        m: dimension on the middle layer
        drop_p: float in (0,1) , value of the dropout probability
        """
        super().__init__()
        # Add the first layer : input_size into the first hidden layer
        self.device  = device

        self.layers = nn.ModuleList([nn.Linear(input_size,1024).to(self.device),nn.Linear(1024,4096).to(self.device),nn.Linear(4096,4096).to(self.device)])
        self.output=nn.Linear(4096,output_size).to(self.device)

    def forward(self,x):
        # pass through each layers
        for layer in self.layers:
            x=F.relu(layer(x))
        return self.output(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class Discriminator(nn.Module):
    def __init__(self,input_size,output_size=1,drop_p=0.05):
        """  Builds autoencoder network with m as the code dimensionality.

        Arguments
        ----------
        input_size : integer, size of the input
        m: dimension on the middle layer
        drop_p: float in (0,1) , value of the dropout probability
        """
        super().__init__()
        # Add the first layer : input_size into the first hidden layer
        self.device  = device

        self.layers = nn.ModuleList([nn.Linear(input_size,2048).to(self.device),nn.Linear(2048,4096).to(self.device),nn.Linear(4096,4096).to(self.device)])
        self.output=nn.Linear(4096,output_size).to(self.device)

    def forward(self,x):
        # pass through each layers
        for layer in self.layers:
            x=F.relu(layer(x))
        return F.sigmoid (self.output(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
