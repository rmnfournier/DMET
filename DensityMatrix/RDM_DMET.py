'''
Class RDM_DMET
 Use reduced density matrix as impurity solver and neural network as MF potential fitter

 By Romain Fournier
'''
import os
import sys
dirname = os.path.dirname(__file__)
print(dirname)

sys.path.append(dirname+ "/../mean_field/")
from Mean_field_solver import Mean_field_solver

sys.path.append(dirname + "/../lattice/")
from HC_Lattice import HC_Lattice

sys.path.append(dirname + "/../PN_representability/GANs/")
from GAN_Model import Generator

import numpy as np
import scipy.linalg
import scipy.optimize
from numpy.core.umath_tests import inner1d
import torch
from scipy.optimize import Bounds,SR1

class RDM_DMET:
    def __init__(self,lattice_size,n,t,u,impurities,gan_input_dim,gan_output_dim,gan_weights,is_ring=False):
        '''
        :param lattice_size: [x,y], dimension of the lattice
        :param n: number of electrons
        :param t: hopping integral Hcin = -t ci+cj
        :param u : e-e repulsion
        :param impurities: [[impurity_1 sites],[imp_2 sites],etc]
        :param gan_input_dim : dimensionality of the input of the generator
        :param gan_weights : file where we can find the weigths of the nn.
        '''
        # Parameters
        self.t = t
        self.u = u
        self.n = n
        self.impurities = impurities
        self.lattice_size = lattice_size
        self.gan_input_dim = gan_input_dim

        # Declare the lattice
        self.lattice = HC_Lattice(height=lattice_size[0], length=lattice_size[1],ring=is_ring)
        self.filling = self.n / self.lattice.nb_sites / 2

        # Initialize the potential and the kinetic energy
        self.potential = self.initialize_potential()
        self.T = self.compute_T()

        # Describe the impurities
        self.L = len(impurities[0]) # number of sites on 1 impurity
        self.s = 4*self.L # number of spin-states for 1 impurity
        self.m = len(impurities) # number of impurities

        # Set up the generator
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.coding_dim = self.gan_input_dim
        self.param_dim = gan_output_dim

        # Declare the generator
        self.generator = Generator(self.coding_dim, self.param_dim)
        self.generator.load_state_dict(torch.load(gan_weights))
        self.generator.eval()

    def initialize_potential(self):
        '''
        Create the first correlation potential
        :return: 2*nb_sites x 2*nb_sites correlation potential
        '''
        # find the mean occupation number per spin_site
        nb_spin_states = 2 * self.lattice.nb_sites
        return self.u * self.filling * np.identity(nb_spin_states)

    def compute_T(self):
        '''
        Compute the hopping matrix
        :return: 2*2*L X 2*2*L matrix
        '''
        # compute the Hopping matrix
        T = np.zeros([2*self.lattice.nb_sites, 2*self.lattice.nb_sites])
        # Iterate over all sites
        for i in np.arange(self.lattice.nb_sites):
            for j in self.lattice.get_neighbors(i + 1):
                T[j - 1+self.lattice.nb_sites, i+self.lattice.nb_sites] = -self.t
                T[j - 1, i] = -self.t
        return T

    def solve_system(self):
        '''
        Solve the system
        '''
        # The goal is to find the correlation potential that generate a mean-field 1-RDM
        # that correspond to the ones of the impurities
        # The system is solved when either the potential does not move, or the 1-RDM fit

        # initialize the criteria with arbitrary values, in order to enter the loop
        potential_diff=1
        density_diff=1
        while potential_diff>1e-3 and density_diff>1e-3:
            # We start by optimizing the 2-RDM
            self.optimize_2rdm()
            # Then we optimize the correlation potential
            potential_diff,density_diff=self.optimize_mfpotential()

    def optimize_2rdm(self):
        '''
        Find the parameters p_gan, such that G(p_gan) gives the minimum variational energy
        return the energies of each impurity
        '''
        # we optimize each impurity
        energies=[]
        self.p_gan_list = [self.initialize_p_gan() for _ in self.impurities]
        orbitals = self.get_mean_field_orbitals(self.potential)
        for runner_impurity,impurity in enumerate(self.impurities):
            # We first compute the new basis
            B = self.get_schmidt_orbitals_from_orbitals(orbitals,impurity)
            # Then, we politely ask scipy to find the best parameters
            p_gan = np.asarray(self.p_gan_list[runner_impurity])
            bounds = Bounds(np.zeros([len(p_gan)]),np.ones([len(p_gan)])) # the parameters are between 0 and 1 by definition

            f = lambda p_gan : self.e_var(p_gan,B)
            # we save the result
            result = scipy.optimize.minimize(f,p_gan,bounds=bounds,options={"maxiter":1000,"eps":1/50.0,"gtol":0,"ftol":0})
            self.p_gan_list[runner_impurity] = result["x"]
            #print(result)
            energies.append(result["fun"])
        return energies

    def optimize_mfpotential(self):
        '''
        Find the mf potential that best suits the available RDM
        :returns Delta potential (norm of the difference) and Delta 1RDM
        '''
        # declare the parameters
        potential_parameters =np.random.uniform(0,1,[2*len(self.lattice.nb_sites)**2])
        # Then, we politely ask scipy to find the best parameters
        f = lambda potential_parameters : self.density_loss(potential_parameters)
        # we save the result
        old_pot = self.potential
        parameters = scipy.optimize.fmin(f,potential_parameters,maxiter=2500,maxfun=6000)
        self.potential=self.get_mf_pot_from_parameters(parameters)
        return scipy.linalg.norm(old_pot-self.potential), self.density_loss(parameters)

    def density_loss(self,parameters):
        '''
        Compoute the loss for the current parameters
        :param parameters: parameters defining the potential
        :return: score
        '''
        # We first compute the potential
        u = self.get_mf_pot_from_parameters(parameters)
        # Then, we compute the mf solution
        rho_mf = self.get_mf_density(u)
        # We initialize the score
        loss=0
        orbitals = self.get_mean_field_orbitals(u)

        for runner_impurity, impurity in enumerate(self.impurities):
            # Get schmidt orbitals
            B = self.get_schmidt_orbitals_from_orbitals(orbitals, impurity)
            rho_p = np.matmul(np.matmul(B,rho_mf),B.T)
            rho2_imp = self.get_2rdm(self.p_gan_list[runner_impurity])
            rho_imp = self.double2single(rho2_imp)
            loss+=scipy.linalg.norm(rho_p-rho_imp)
        return loss

    def get_mf_pot_from_parameters(self,c):
        '''
        :param c: coefficients that are put in the correlation potential
        :return: coorelation potential
        '''
        # declare the potential
        s = 2*self.lattice.nb_sites
        N=self.lattice.nb_sites
        u = np.zeros([s,s],dtype=complex)
        counter=0
        for i in np.arange(N):
            for j in np.arange(i,N):
                # diagonals of submatrices are real
                if i==j:
                    #up_up
                    u[i,j]=c[counter]
                    #down_down
                    u[i+N,j+N]=c[counter]
                    counter+=1
                    #up_down and down_up
                    u[i+N,j]=c[counter]
                    u[i,j+N]=c[counter]
                    counter+=1
                else:
                    #take care of the complex elements
                    #up_up and down_down
                    u[i,j]=c[counter]+1j*c[counter+1]
                    u[i+N, j+N] = c[counter] + 1j * c[counter + 1]
                    counter+=2
                    #cross spins
                    u[i + N, j] = c[counter]+1j*c[counter+1]
                    u[i, j + N] = c[counter]+1j*c[counter+1]
                    counter+=2
        u=u+np.conjugate(np.transpose(u))
        return u

    def get_mf_density(self,potential):
        '''
        Compute the 1-RDM for the mean field solution
        :param potential:
        :return:
        '''
        C = self.get_mean_field_orbitals(potential)
        rho = np.real(scipy.matmul(C[:, :self.n], np.conjugate(np.transpose(C[:, :self.n]))))
        return rho

    def get_schmidt_orbitals_from_orbitals(self,orbitals,impurity_index,verbose=False):
        '''
        Project the orbitals into the impurity
        :param orbitals: mf orbitals
        :param impurity: index of the sites
        :return: B
        '''
        C=orbitals
        lines = [x - 1 for x in impurity_index] + [x + self.lattice.nb_sites - 1 for x in impurity_index]
        if (verbose):
            print("Lines corresponding to the impurity")
            print(lines)
        O = C[lines, :].copy()
        if (verbose):
            print("Matrix corresponding to impurtiy")
            print(np.around(O, 2))
        # Performs the SVD
        U, s, Vh = scipy.linalg.svd(O)
        # rotate the occupation matrix
        C = np.matmul(C, np.conjugate(np.transpose(Vh)))
        if (verbose):
            print("Rotated Matrix")
            print(np.around(C, 2))
        # Create the matrix B, where C = ( A 0 ; B C) if the first lines correspond to the impurity
        mask = np.ones([2 * self.lattice.nb_sites], dtype=bool)
        mask[lines] = False
        B = C[mask, :len(s)]
        if (verbose):
            print("Entangled environment orbitals")
            print(np.around(B, 2))
        # Perform QR decomposition of B
        Q, _ = scipy.linalg.qr(B)
        if (verbose):
            print("orthogonal basis")
            print(np.around(Q[:, :len(s)], 2))

        # build the orbitals
        B= np.zeros([2 * self.lattice.nb_sites, 2 * len(s)], dtype=complex)

        B[mask, len(s):] = Q[:, :len(s)]
        B[~mask, :len(s)] = np.identity(len(lines))

        return B

    def get_schmidt_orbitals(self, u,impurity_index, verbose=False):
        '''
        Find the schmidt decomposition for the given impurity_index
        :param impurity_index: list of the index of the sites of interest
        :return: 2*nb_sites x 2*2*nb_impurity_sites x corresponding to the new orbitals
        '''
        # Get the MF orbitals
        C = self.get_mean_field_orbitals(u)
        return self.get_schmidt_orbitals_from_orbitals(C,impurity_index,verbose)

    def get_mean_field_orbitals(self, u, verbose=False):
        '''
        Use the current self.potential to get self.n first orbitals of the Mean Field solution
        :return: 2*nb_sites x self.n Orbital matrix
        '''
        mf_solver = Mean_field_solver(t=self.t, u=u, lattice_size=self.lattice_size)
        # Get the orbitals
        e, C = mf_solver.solve_system(verbose=verbose)
        # check if the ground state is degenerated
        index = np.argsort(e)
        C = C[:, index]
        if verbose:
            print("Energies:")
            print(e[index])
            print(np.shape(e))
        if (np.abs(e[index[self.n - 1]] - e[index[self.n]]) < 1e-6):
            spin_down_1 = np.sum(C[self.lattice.nb_sites:2 * self.lattice.nb_sites, self.n - 1])
            spin_down_2 = np.sum(C[self.lattice.nb_sites:2 * self.lattice.nb_sites, self.n])
            if (np.abs(spin_down_1) > 0.0001):
                c = spin_down_2 / spin_down_1
                beta = 1 / np.sqrt(1 + c ** 2)
                alpha = -beta * c
                C[:, self.n - 1] = alpha * C[:, self.n - 1] + beta * C[:, self.n]

        return C[:, :self.n]

    def initialize_p_gan(self):
        '''
        Give random p_gan
        :return:
        '''
        # During the training, the generator was trained with uniformly sampled data as input
        return np.random.uniform(0,1,[self.gan_input_dim])

    def e_var(self,p_gan,B):
        '''
        Compute the variational energy, given input parameters p_gan and basis B
        :param p_gan: input of the generator
        :param B: Basis
        :return: <psi|H|psi>
        '''
        # compute the 2-RDM matrix
        m_2rdm = self.get_2rdm(p_gan)
        # compute the 1-RDM matrix
        m_1rdm = self.double2single(m_2rdm)
        # Return the 2 contributions
        return np.real(self.H1(m_1rdm,B)+self.H2(m_2rdm,B))

    def double2single(self,m_2rdm):
        '''
        Get the 1-RDM from a 2-RDM
        :param m_2rdm: matrix of the 2-RDM
        :return: 1-RDM
        '''

        # initialize the matrix
        rho = np.zeros([self.s,self.s])
        # populate the matrix by looping over the coefficients a and b
        for alpha in np.arange(self.s,dtype=int):
            for beta in np.arange(self.s,dtype=int):
                # elements to trace out
                for nu in np.arange(self.s,dtype=int):
                    # 0 contribution if nu is alpha or beta
                    if nu != alpha and nu!=beta:
                        # take care of the sign
                        sgn = 1
                        if nu < alpha:
                            sgn*=-1
                        if nu < beta:
                            sgn*=-1
                        rho[alpha,beta]+=sgn*m_2rdm[self.get_matrix_index(min(alpha,nu),max(alpha,nu)),self.get_matrix_index(min(beta,nu),max(beta,nu))]

        return rho*2/(self.n-1)

    def get_matrix_index(self,k,l):
        '''
        The 2-RDM tensor is stored as a matrix.from 2 coefficients we can get the 1 of the matrix
        :param a: coef a
        :param b: coef b
        :return: index in the matrix
        '''
        # makre sure that the order is correct
        [k,l]=np.sort([l,k])
        return int(l - k + (2*self.s- k - 1) * k / 2-1)

    def H1(self,m_1rdm,B):
        '''
        Compute the single particle contribution
        :param m_1rdm: single particle reduced density matrix
        :param B: orbitals
        :return: sgle particle contribution
        '''

        T_tilda = np.matmul(np.matmul(B.T,self.T),np.conjugate(B))
        # return the trace of the matrix product
        return np.einsum('ij,ji->',m_1rdm,T_tilda.T)

    def H2(self,m_2rdm,B):
        '''
        Compute the 2 particles contribution

        :param m_2rdm: 2 particles reduced density matrix
        :param B: orbitals
        :return: 2 particles contributions
        '''
        U_tilda = self.compute_u_tilda(B)
        return np.einsum('ij,ji->',2*m_2rdm,U_tilda.T)

    def compute_u_tilda(self,B):
        '''
        transform the e-e repulsion in the new basis
        :param B: orbitals
        :return: new U
        '''
        # The tricks is that we can only consider the terms ck+cl+cmcn with k<l and m<n by symmetrizing U2 tilda conveniently
        # Therefore, we only need to store half of the (2spin*(n_impurity+N-bath))**2 - the diagonal

        n_param = int(self.s*(self.s-1)/2)
        # since we only care about k<l and m<n, we can store the 4th order as a matrix
        U2_tilda =  np.zeros([n_param,n_param],dtype=complex)

        for k in np.arange(self.s):
            for l in np.arange(k+1,self.s):
                # Compute the index x corresponding to kl
                id_x = self.get_matrix_index(k,l)

                for m in np.arange(self.s):
                    for n in np.arange(m+1,self.s):
                        # compute teh index y similaraly
                        id_y = self.get_matrix_index(m,n)
                        # Now, there will be 4 different contributions (ck cl -> -cl ck etc)
                        v1_up = (B[:2*self.L,k])
                        v1_down = (B[2*self.L:, k])

                        v2_up = (B[:2*self.L, l])
                        v2_down = (B[2*self.L:, l])

                        v3_up = np.conjugate(B[:2*self.L, m])
                        v3_down = np.conjugate(B[2*self.L:, m])

                        v4_up = np.conjugate(B[:2*self.L, n])
                        v4_down = np.conjugate(B[2*self.L:, n])

                        # compute the term (change the sign each time we do a permutation)
                        U2_tilda[id_x,id_y]=self.u*np.sum(v1_up*v2_down*v3_up*v4_down-v1_down*v2_up*v3_up*v4_down+v1_down*v2_up*v3_down*v4_up-v1_up*v2_down*v3_down*v4_up)

        return U2_tilda

    def get_2rdm(self,p_gan):
        '''
        Use the generator to get a 2-RDM from the parameters
        :param p_gan: parameters in the coding space
        :return: 2rdm
        '''
        # get the coefficients of the matrix L
        with torch.no_grad():
            p_gan_tensor= torch.from_numpy(p_gan).float()
            p_gan_tensor_cuda = p_gan_tensor.to(self.device)
            L=self.generator.forward(p_gan_tensor_cuda)
        # Rebuild L cholevski
        dim = int(self.s*(self.s-1)/2)
        L_chol = np.zeros([dim,dim])
        c = 0
        for j in np.arange(dim):
            for i in np.arange(dim):
                if j >= i:
                    L_chol[i,j] = L[c]
                    c +=1
        rho = np.matmul(L_chol.T,L_chol)
        rho_1 = self.double2single(rho)
        rho*=self.n/np.trace(rho_1)
        return rho


