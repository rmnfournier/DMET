import os
import sys

import numpy as np
import scipy.linalg
import scipy.optimize
import gmpy2
dirname = os.path.dirname(__file__)
path = os.path.join(dirname, "../mean_field/")
print(path)
sys.path.append(dirname + "/../lattice/")
from HC_Lattice import HC_Lattice

sys.path.append(dirname + "/../mean_field/")
from Mean_field_solver import Mean_field_solver

sys.path.append(dirname + "/../impurity/")
from Impurity_solver import Impurity_solver


'''
:author: Romain Fournier
:date: 01.04.2019
:role: Handle the communication between MF solution and Impurity solution
:contact: romain.fournier@epfl.ch
'''


class DMET_solver:
    def __init__(self, t, u, n, lattice_size):
        '''
        :param t: (double) hopping integral (appears as -t c+_i c_j)
        :param u: (double) on site e-e repulsion
        :param n: (int) number of electrons on the lattice
        :param lattice_size: (2x1 int) number of rows and columns of the lattice
        '''
        self.t = t
        self.u = u
        self.n = n
        self.lattice_size = lattice_size
        self.lattice = HC_Lattice(height=lattice_size[0], length=lattice_size[1])
        self.potential = self.initialize_potential()  # First potential
        self.mu =0 # chemical potential

        self.filling = self.n/self.lattice.nb_sites/2


    def get_mean_field_orbitals(self,u,verbose=False):
        '''
        Use the current self.potential to get self.n first orbitals of the Mean Field solution
        :return: 2*nb_sites x self.n Orbital matrix
        '''
        mf_solver = Mean_field_solver(t=self.t, u=u, lattice_size=self.lattice_size)
        # Get the orbitals
        e, C = mf_solver.solve_system(verbose=verbose)
        # check if the ground state is degenerated
        index = np.argsort(e)
        C=C[:,index]
        if verbose:
            print("Energies:")
            print(e[index])
            print(np.shape(e))
        if (np.abs(e[index[self.n-1]]-e[index[self.n]])<1e-6):
            spin_down_1 = np.sum(C[self.lattice.nb_sites:2*self.lattice.nb_sites,self.n-1])
            spin_down_2 = np.sum(C[self.lattice.nb_sites:2*self.lattice.nb_sites,self.n])
            if(np.abs(spin_down_1)>0.0001):
                c=spin_down_2/spin_down_1
                beta = 1/np.sqrt(1+c**2)
                alpha = -beta*c
                C[:,self.n-1]=alpha*C[:,self.n-1]+beta*C[:,self.n]

        return C[:,:self.n]



    def get_schmidt_orbitals(self, impurity_index, verbose=False):
        '''
        Find the schmidt decomposition for the given impurity_index
        :param impurity_index: list of the index of the sites of interest
        :return: 2*nb_sites x 2*2*nb_impurity_sites x corresponding to the new orbitals
        '''
        # Get the MF orbitals
        C = self.get_mean_field_orbitals(self.potential)
        # Build O, the matrix containing all lines related to the impurity
        lines = [x - 1 for x in impurity_index] + [x + self.lattice.nb_sites - 1 for x in impurity_index]
        if (verbose):
            print("Lines corresponding to the impurity")
            print(lines)
        O = C[lines, :].copy()
        if (verbose):
            print("Matrix corresponding to impurtiy")
            print(np.around(O,2))
        # Performs the SVD
        U, s, Vh = scipy.linalg.svd(O)
        # rotate the occupation matrix
        C = np.matmul(C, np.conjugate(np.transpose(Vh)))
        if (verbose):
            print("Rotated Matrix")
            print(np.around(C,2))
        # Create the matrix B, where C = ( A 0 ; B C) if the first lines correspond to the impurity
        mask = np.ones([2 * self.lattice.nb_sites], dtype=bool)
        mask[lines] = False
        B = C[mask, :len(s)]
        if (verbose):
            print("Entangled environment orbitals")
            print(np.around(B,2))
        # Perform QR decomposition of B
        Q, _ = scipy.linalg.qr(B)
        if (verbose):
            print("orthogonal basis")
            print(np.around(Q[:, :len(s)],2))

        # build the orbitals
        W = np.zeros([2 * self.lattice.nb_sites, 2 * len(s)],dtype=complex)
        if (verbose):
            print("W up")
            print(np.around(W[~mask,:len(s)],2))
            print("W down")
            print(np.around(W[mask,:len(s)],2))
            print("W")
            print(np.around(W,2))

        W[mask,len(s):]=Q[:, :len(s)]
        W[~mask,:len(s)]=np.identity(len(lines))

        return W

    def initialize_potential(self):
        '''
        Create the first correlation potential
        :return: 2*nb_sites x 2*nb_sites correlation potential
        '''
        # find the mean occupation number per spin_site
        nb_spin_states = 2 * self.lattice.nb_sites
        mean_occupancy = self.n / nb_spin_states

        return self.u * mean_occupancy * np.identity(nb_spin_states)

    def solve_system(self,impurity_indexes,verbose=False):
        '''
        find the potential that give a consistent solution between Mean Field and impurity solutions
        :param impurity_indexes: list of index of the impurity
        :param verbose:
        '''

        # Initialize potential
        self.potential=self.initialize_potential()
        if(verbose):
            print("Initial potential")
            print(self.potential)
        s=2*self.lattice.nb_sites
        n_parameters = 2*(self.lattice.nb_sites**2)
        # Initial density matrix
        rho_mf = self.get_rho_from_u(self.potential)
        # While the two results are different or the potential changes
        potential_diff=1
        lr=1
        # Compute the current density matrix and basis for the impurity
        B = []
        rho_impurity = []
        for index in impurity_indexes:
            B.append(self.get_schmidt_orbitals(index))
            rho_impurity.append(self.get_rho_impurity(index))
        coef=np.random.normal(0, 3, [n_parameters])
        while(self.density_norm(rho_mf,rho_impurity,B)>0.01 and potential_diff>0.05):
            #update the impurity potential (with the new potential)
            if(verbose):
                print("New orbitals")
                print(np.around(self.get_schmidt_orbitals([1]),1))

            # Compute the current density matrix and basis for the impurity
            B=[]
            rho_impurity=[]
            for index in impurity_indexes:
                B.append(self.get_schmidt_orbitals(index))
                rho_impurity.append(self.get_rho_impurity(index))
            #find the potential that make rho_mf like rho_im
            f = lambda coef : self.density_norm(self.get_rho_from_u(self.get_u_from_coef(coef)),rho_impurity,B_list=B)
            coef = scipy.optimize.fmin(f,coef,maxiter=250000,maxfun=500000)
            V=self.get_u_from_coef(coef)
            potential_diff=np.linalg.norm(V-self.potential)
            self.potential=(1-lr)*self.potential+lr*V
            rho_mf=self.get_rho_from_u(self.potential)
            print("Criteria : ")
            print(self.density_norm(rho_mf,rho_impurity,B_list=B))
            print(potential_diff)


    def density_norm(self,rho,rho_impurity_list,B_list):
        '''
        :param impurity_list : list of impurity sites
        :return:||rho-rho_imp||
        '''
        # Initialize the norm
        diff=0
        # For each impurity
        for rho_impurity,B in zip(rho_impurity_list,B_list):
            # compute the orbitals
            #B=np.conjugate(B)
            rho_projected = scipy.matmul(scipy.matmul(np.transpose(B),rho),np.conjugate(B))
            # update the difference
            diff += scipy.linalg.norm((rho_impurity)-(rho_projected))

        return diff

    def get_rho_from_u(self,V):
        '''
        :param V:correlation potential
        :param impurity_index: (list) location of the impurity
        :return:
        '''
        # We must first get the ground state
        C = self.get_mean_field_orbitals(V)
        rho= np.real(scipy.matmul(C[:,:self.n],np.conjugate(np.transpose(C[:,:self.n]))))
        return rho



    def get_rho_impurity(self,impurity_index,verbose=False):
        '''
        :param impurity_index: index of the impurity
        :return: single particle density matrix in the impurity basis
        '''
        # Get ground state (updates the chemical potential)
        vs = self.get_ground_state(impurity_index, verbose)
        v=np.zeros(np.shape(vs[0]),dtype=complex)
        if verbose:
            print(np.around(vs,2))
        # initialize the density matrix
        rho = np.zeros([4*len(impurity_index),4*len(impurity_index)],dtype=complex)

        # the coefs part ensures that the ground state has a defined spin on the impurity if it is degenerated
        coefs = np.zeros(len(vs),dtype=complex)
        for k,psi in enumerate(vs):
            for i in np.arange(len(psi)):
                impurity_down = 0
                for j in np.arange(len(impurity_index)):
                    if(gmpy2.bit_test(int(i),int(j+len(impurity_index)) ) and not gmpy2.bit_test(int(i),int(j))):
                        impurity_down+=1
                coefs[k]+=impurity_down*psi[i]
        if len(coefs)==2:
            if(np.abs(coefs[0])>0.0001):
                c = coefs[1]/coefs[0]
                beta = 1/np.sqrt(1+c**2)
                alpha=-beta*c
                v = alpha*vs[0]+beta*vs[1]
            else :
                v = vs[0]
        else :
            v=vs[0]
        v=v/np.linalg.norm(v)
        if verbose:
            print("new ground state")
            print(v)
        # populate the matrix
        # iterate over all impurity+bath positions i
        for i in np.arange(4*len(impurity_index),dtype=int):
            # iterate over all impurity+bath positions j
            for j in np.arange(4*len(impurity_index),dtype=int):
                #iterate over all elements in the basis of the Fock space
                for k in np.arange(2**(4*len(impurity_index)),dtype=int):
                    #check if a+i aj|k> is not null
                    if(gmpy2.bit_test(int(k),int(j))):
                        if(gmpy2.bit_test(int(k),int(i))==0 or i==j):
                            new_state=gmpy2.bit_flip(int(k),int(j))
                            new_state=gmpy2.bit_flip(new_state,int(i))
                            #and don't forget the sign !
                            occupancy_between=0
                            for l in np.arange(min(i,j)+1,max(i,j)):
                                occupancy_between+=gmpy2.bit_test(int(k),int(l))
                            rho[i,j]+=(-1)**occupancy_between*np.conjugate(v[new_state])*v[k]
        return rho


    def get_ground_state(self,impurity_index,verbose=False):
        W = self.get_schmidt_orbitals(impurity_index)
        up_mu = float("inf")
        down_mu = -float("inf")
        imp_solv = Impurity_solver(self.t, self.u, self.mu, W, self.lattice_size, impurity_index)

        n_el=-1
        if verbose :
            print("we desire "+str(self.filling*4*len(impurity_index))+" electrons on the impurity.")
        while np.abs(n_el-self.filling*4*len(impurity_index))>0.001:
            imp_solv.set_mu(self.mu)
            g = imp_solv.get_ground_state(verbose)
            index = np.where(np.abs(g[0])>0.01)
            one_state = index[0]
            one_state = int(one_state[0])
            n_el = gmpy2.popcount(one_state)
            if(verbose):
                print("mu ="+str(self.mu)+" gives "+str(n_el)+" electrons")
            # we need to increase mu to get more electrons
            if n_el<self.filling*4*len(impurity_index):
                down_mu=self.mu
                if(up_mu!=float("inf")):
                    self.mu=0.5*(down_mu+up_mu)
                else :
                    self.mu=(self.mu)*2**(np.sign(self.mu))+0.00000000001
            elif n_el>self.filling*4*len(impurity_index):
                up_mu=self.mu
                if(down_mu!=-float("inf")):
                    self.mu=0.5*(down_mu+up_mu)
                else :
                    self.mu=(self.mu)*2**(-np.sign(self.mu))-0.000000000001

        imp_solv.set_mu(self.mu)
        gs = imp_solv.get_ground_state()
        return gs


    def get_u_from_coef(self,c):
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
