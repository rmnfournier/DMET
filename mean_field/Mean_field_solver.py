import sys

import numpy as np
import scipy.linalg

sys.path.insert(0, "../lattice/")
from HC_Lattice import HC_Lattice

'''
:author: Romain Fournier
:date: 28.03.2019 
:role: Given a lattice and a correlation potential, find the orbitals of the mean field solution of the problem
:todo: 
:contact: romain.fournier@epfl.ch
'''


class Mean_field_solver:
    def __init__(self, t, u, lattice_size):
        '''
        :param t: (double) hopping integral (appears as -t c+_i c_j)
        :param u: (2Nx2N double) correlation potential
        :param lattice_size: (2x1 int) number of rows and columns of the lattice
        '''
        self.t = t
        self.u = u

        self.lattice = HC_Lattice(height=lattice_size[0], length=lattice_size[1])
        self.H_cin = self.build_Hcin()

    def build_Hcin(self):
        '''
        Compute the hopping matrix
        :return: NxN matrix
        '''
        T = np.zeros([self.lattice.nb_sites, self.lattice.nb_sites])
        # Iterate over all sites
        for i in np.arange(self.lattice.nb_sites):
            for j in self.lattice.get_neighbors(i + 1):
                T[j - 1, i] = -self.t
        return T

    def solve_system(self,verbose=False):
        '''
        Give the eigen energies and the eigenvalues of the system
        :return: eig_val : 2Nx1 double corresponding to the eigenvalues (energies) of the system
                 eig_vec : 2Nx2N double, where the columns of eig_vec are the eigenstates of H
        '''
        # Define the Hamiltonian
        n = self.lattice.nb_sites
        H = np.zeros([2 * n, 2 * n],dtype=complex)
        # Add the kinetic part
        H[0:(n), 0:(n)] = self.H_cin
        H[(n):, (n):] = self.H_cin

        # Add the correlation potential
        H += self.u

        # Compute the Eigenstates and Eigen Energies
        [eig_val, eig_vec] = scipy.linalg.eigh(H)
        if(verbose):
            print("MF hamiltonian")
            print(np.around(H,2))
            print(np.shape(H))
        return eig_val, eig_vec
