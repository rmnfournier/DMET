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
from torch.optim import Adam
import pandas as pd
import numpy as np
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
coding_dim = int(sys.argv[1])
working_dim = 2211
nb_data_s=[5000]
prefix="saved_GANs/Generators/"
prefix="./"
us=[0,1,2,3,4,5]
nb_simulations=25
for nb_data in nb_data_s:
    suffix = "_" + str(coding_dim) + "_" + str(nb_data)
    for u in us:
        # Compute the energy2
        dm_solver = RDM_DMET([3,4],6,1,u,[[1,2,3],[4,5,6]],coding_dim,working_dim,prefix+"generator_weights"+suffix+".pth",is_ring=True)

        energies = np.zeros([nb_simulations,2])
        for i in range(nb_simulations):
            print("simulation nb " + str(i + 1) + "/"+str(nb_simulations))
            tmp_en = dm_solver.optimize_2rdm()
            energies[i,0]=tmp_en[0]
            energies[i,1]=tmp_en[1]
        energies = pd.DataFrame(
            {
                'Simulation':np.arange(nb_simulations),
                'imp 0 ':energies[:,0],
                'imp 1':energies[:,1]
            }
        )
        energies.to_csv("energies_u_"+str(u)+suffix+".csv",index=False)


