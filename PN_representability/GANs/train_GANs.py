from GAN_Model import Discriminator,Generator
from Database import Database
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Necessary variables
database_folder = "../data_L2_N2_sym/"
nb_data=1000000
data_to_load = 2048

batch_per_epochs =int(nb_data/data_to_load)
epochs=1000
epochs_D = 250
epochs_G = 150

tol_d=0.4
tol_g = 0.1

history_loss_d=[]
history_loss_g=[]

loss = nn.BCELoss()
## Define the size of the problem
coding_dim = 9
working_dim = 21

## Declare the generator and the discriminator
G = Generator(coding_dim,working_dim)
D = Discriminator(working_dim,1)
optimizer_G = Adam(G.parameters())
optimizer_D = Adam(D.parameters())
# Prepare the dataset
train_data = Database(csv_target=database_folder+"data.csv",csv_input=database_folder+"data.csv",nb_data=nb_data).get_loader()
trainloader = DataLoader(train_data, batch_size=int(data_to_load), shuffle=True)
# Start the training over epochs_G epochs
for e in range(epochs):
    print(str(e)+"/"+str(epochs))
    score=0
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
            #print("(Discriminator) Threshold reached after "+str(e_d)+" iterations")
            #print("... correct = "+str(np.around(correct.item(),2)))
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
            #print("(Generator) Threshold reached after "+str(e_g)+" iterations")
            #print("... fooled : " + str(1 - correct.item()))

            break
    # Generate a dataset
    labels_d = torch.ones([data_to_load, 1]).to(device)
    data_g = G.forward(torch.rand([data_to_load, coding_dim]).to(device)).to(device)
    with torch.no_grad():
        y_hat = D.forward(data_g)
        l = loss(y_hat, labels_d)
        history_loss_g.append(l.item())
        correct = (y_hat.round() == 0).double().mean()


torch.save(D.state_dict(), "discriminator_weights.pth")
torch.save(G.state_dict(), "generator_weights.pth")

loss_g = (pd.DataFrame(history_loss_g))
loss_d = (pd.DataFrame(history_loss_d))
loss_g.to_csv("Loss_g.csv",index=False,header=False)
loss_d.to_csv("Loss_d.csv",index=False,header=False)

G.eval()
with torch.no_grad():
    data = G.forward(torch.rand([1000, coding_dim]).to(device)).to(device)
    answer = (pd.DataFrame(data.cpu().numpy()))
    answer.to_csv("L.csv",index=False,header=False)
