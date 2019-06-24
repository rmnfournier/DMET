'''
By Romain Fournier
example :
train_and_test_gan.py m nbdata
m is the coding dimension and nb_data the number of data to load from the dataset for the entire training

output :L_m_nb_data.csv Epoch, Loss generator, Loss discriminator, where loss is the cross entropy
        Evar_m_nbdata.csv 10 simulations to find the variational energy. Each column corresponds to 1 impurity
        generator/discriminator_weights_m_nb_data.pth save the neural networks
'''

import os,sys

dirname = os.path.dirname(__file__)
print(dirname)

sys.path.append(dirname+ "/../PN_representability/")

from RDM_DMET import RDM_DMET

from GAN_Model import Discriminator,Generator
from Database import Database
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import Adamax
import pandas as pd
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Necessary variables
database_folder = "../PN_representability/data_L6_N6_super_sym/"
nb_data=int(sys.argv[2])

data_to_load = 512

batch_per_epochs =int(nb_data/data_to_load)
epochs=int(sys.argv[3])
epochs_D = 250
epochs_G = 150

tol_d=0.35
tol_g = 0.2

history_loss_d=[]
history_loss_g=[]

loss = nn.BCELoss()
## Define the size of the problem
coding_dim = int(sys.argv[1])
working_dim = 2211

## Declare the generator and the discriminator
G = Generator(coding_dim,working_dim)
D = Discriminator(working_dim,1)
optimizer_G = Adamax(G.parameters())
optimizer_D = Adamax(D.parameters())
# Prepare the dataset
train_data = Database(csv_target=database_folder+"data.csv",csv_input=database_folder+"data.csv",nb_data=nb_data).get_loader()
trainloader = DataLoader(train_data, batch_size=int(data_to_load), shuffle=True)
# Start the training over epochs_G epochs
for e in range(epochs):
    print(str(e)+"/"+str(epochs))
    score=0


    # Train the discriminator
    G.eval()
    D.train()
    for e_d in range(epochs_D):
        # Generate a dataset
        data_d = next(iter(trainloader))[0]
        labels_d = torch.ones([data_to_load, 1]).to(device)
        # Create a fake dataset
        data_g = G.forward(torch.rand([data_to_load, coding_dim]).to(device)).to(device)
        labels_g = torch.zeros([data_to_load, 1]).to(device)

        # Train the discriminator using these datasets
        x_data_ = torch.cat((data_d, data_g), 0)
        y_data_ = torch.cat((labels_d, labels_g), 0)
        index = torch.randperm(2*data_to_load).long().to(device)

        x_data = x_data_.index_select(0,index).detach()
        y_data = y_data_.index_select(0,index).float().detach()
        optimizer_D.zero_grad()

        y_hat = D.forward(x_data)
        l = loss(y_hat,y_data)
        l.backward()

        optimizer_D.step()

        # Mean number of correctly sorted :
        correct=(y_hat.round()==y_data).double().mean()
        if (correct>0.5+tol_d):
            break

    # Validation score after the training
    # Generate a dataset
    data_d = next(iter(trainloader))[0]
    labels_d = torch.ones([data_to_load, 1]).to(device)
    # Create a fake dataset
    data_g = G.forward(torch.rand([data_to_load, coding_dim]).to(device)).to(device)
    labels_g = torch.zeros([data_to_load, 1]).to(device)
    x_data_ = torch.cat((data_d, data_g), 0)
    y_data_ = torch.cat((labels_d, labels_g), 0)
    index = torch.randperm(2 * data_to_load).long().to(device)

    x_data = x_data_.index_select(0, index).detach()
    y_data = y_data_.index_select(0, index).float().detach()
    with torch.no_grad():
        y_hat = D.forward(x_data)
        l = loss(y_hat, y_data)
        history_loss_d.append(l.item())
    #print("Discriminator) Percentage correct  : "+str(correct.item()))

    # train the generator
    G.train()
    D.eval()
    tmp_score = []
    for e_g in range(epochs_G):
        # Generate a dataset
        labels_d = torch.ones([data_to_load, 1]).to(device)

        optimizer_G.zero_grad()
        data_g = G.forward(torch.rand([data_to_load, coding_dim]).to(device)).to(device)
        y_hat = D.forward(data_g)
        l = loss(y_hat,labels_d)
        tmp_score.append(l.item())
        l.backward()
        optimizer_G.step()
        correct = (y_hat.round() == 0).double().mean()
        if(correct<0.5-tol_g):
            break

    # Validation score
    labels_d = torch.ones([data_to_load, 1]).to(device)
    data_g = G.forward(torch.rand([data_to_load, coding_dim]).to(device)).to(device)
    with torch.no_grad():
        y_hat = D.forward(data_g)
        l = loss(y_hat, labels_d)
        history_loss_g.append(l.item())
        correct = (y_hat.round() == 0).double().mean()

# Save the neural networks and the loss
suffix = "_"+str(coding_dim)+"_" + str(nb_data)+"_episodes_"+str(epochs)

torch.save(D.state_dict(), "discriminator_weights"+suffix+".pth")
torch.save(G.state_dict(), "generator_weights"+suffix+".pth")

Loss = pd.DataFrame(
    {
        'Epochs':np.arange(1,len(history_loss_d)+1),
        'Generator Loss':history_loss_g,
        'Discriminator Loss':history_loss_d
    }
)
Loss.to_csv("loss"+suffix+".csv",index=False)

#print("GAN trained, starting simulation :")
# Compute the energy
#dm_solver = RDM_DMET([2,2],2,1,0,[[1],[2]],coding_dim,"generator_weights"+suffix+".pth")

#energies = np.zeros([10,2])
#for i in range(10):
##    print("simulation nb " + str(i + 1) + "/10")
#    tmp_en = dm_solver.optimize_2rdm()
#    energies[i,0]=tmp_en[0]
#    energies[i,1]=tmp_en[1]
#energies = pd.DataFrame(
#    {
#        'Simulation':np.arange(10),
#        'imp 0 ':energies[:,0],
#        'imp 1':energies[:,1]
#    }
#)
#energies.to_csv("energies"+suffix+".csv",index=False)


