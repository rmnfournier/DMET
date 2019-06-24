from GAN_Model import Generator
from Database import Database
from torch.nn.modules.loss import KLDivLoss, L1Loss, SmoothL1Loss
from torch.optim import Adam, Rprop, Adamax, RMSprop, SGD, LBFGS
from torch.utils.data import DataLoader
import torch
import csv
import numpy as np
import math
import pandas as pd
from collections import deque


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Import the desired architecture
m=6
filename = 'generator_weights.pth'
input_dim=21

print("Loading ACANN ")
## Define the size of the problem
coding_dim = 21
working_dim = 21

## Declare the generator and the discriminator
model = Generator(coding_dim,working_dim)
model.load_state_dict(torch.load(filename))
model.eval()


with torch.no_grad():
    data = model.forward(torch.rand([1000, coding_dim]).to(device)).to(device)
    answer = (pd.DataFrame(data.cpu().numpy()))
    answer.to_csv("L.csv",index=False,header=False)
