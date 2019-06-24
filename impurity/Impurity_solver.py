import os
import sys

import numpy as np
import scipy.linalg

dirname = os.path.dirname(__file__)
path = os.path.join(dirname, "../mean_field/")
sys.path.append(dirname + "/../lattice/")
from HC_Lattice import HC_Lattice
import gmpy2
import math
'''
:author: Romain Fournier
:date: 01.04.2019
:role: Compute the ground state of a system Impurity+Bath given orbitals and correlation potential
:contact: romain.fournier@epfl.ch
'''

class Impurity_solver:
    def __init__(self,t,u,mu,W,lattice_size,impurity_index):
        '''

        :param t: (double) hopping parameter
        :param u: (double) on site e-e repulsion
        :param V: (2nx2n) correlation potential
        :param W: Schmidt orbitals
        :param lattice_size: 2x1
        :param impurity_index: list of indexes corresponding to the impurity site
        '''
        #save parameters
        self.t = t
        self.u = u
        self.mu = mu
        self.W = W
        self.problem_size = np.shape(W)

        self.impurity_index=impurity_index

        # create lattice
        self.lattice_size = lattice_size
        self.lattice = HC_Lattice(height=lattice_size[0], length=lattice_size[1])
        self.imp_bath_size=len(impurity_index)*4

        # compute H
        self.T_tilda=self.compute_T_tilda()
        self.H = self.build_H()


    def compute_T_tilda(self):
        '''
        Compute the hopping matrix in the b+ basis
        :return: 2*2*L X 2*2*L matrix
        '''
        # compute the Hopping matrix
        T = np.zeros([2*self.lattice.nb_sites, 2*self.lattice.nb_sites])
        # Iterate over all sites
        for i in np.arange(self.lattice.nb_sites):
            for j in self.lattice.get_neighbors(i + 1):
                T[j - 1+self.lattice.nb_sites, i+self.lattice.nb_sites] = -self.t
                T[j - 1, i] = -self.t
        # Not very intuitive, but T in the new basis is W*+ T W*
        return scipy.matmul(scipy.matmul(np.conjugate(np.transpose(self.W)), T), (self.W))

    def H_ij(self,i,j,verbose=False):
        '''
        Compute <i|H|j>
        :param i: int, binary representation of state |i>
        :param j: int, binary representation of state |j>
        :return: <i|H|j>
        '''
        i+=gmpy2.mpz(0)
        j+=gmpy2.mpz(0)

        L=int(len(self.impurity_index)*2) # L is the total number of site (bath + impurity)
        impurity_site=[x-1 for x in self.impurity_index]
        bath_site=[]
        #build the bath site
        for n in np.arange(L):
            if n not in impurity_site:
                bath_site.append(int(n))

        # if there is a different number of electron, return 0
        if(gmpy2.popcount(i)!=gmpy2.popcount(j)):
            if verbose:
                print("Not the same number of electron")
            return 0

        # otherwise prepare to sum
        hij=0

        # H(2) contribution, we need to distinguish between 3 cases, i.e. 0 jump, 1 jump and 2 jumps
        # jump has bit on on the sites that differs
        jump = i ^ j
        # first case 0 differences
        if(gmpy2.popcount(jump)==0):
            if(verbose):
                print("0 dif contr "+str(self.h_2_ii(i)))
            hij += self.h_2_ii(i)
            hij -= self.mu*gmpy2.popcount(i)
        #second case 2 differences (one jump)
        elif(gmpy2.popcount(jump)==2):
            if(verbose):
                print("2 dif contr "+str(self.h_2_cd(state_i=i,state_j=j)))
            hij+= self.h_2_cd(state_i=i,state_j=j)
        # last possibility : 2 jumps
        elif(gmpy2.popcount(jump)==4):
            if(verbose):
                print("4 dif contr "+str(self.h_2_cdef(i,j)))
            hij+=self.h_2_cdef(i,j)

        # H(1) contribution
        if (gmpy2.popcount((i^j))==2):
            if verbose:
                print("One jump")

            # jump has bit on on the 2 sites that differs
            jump=i^j
            # we now look for the index of the sites that were destroyed and created
            destruction_site = int(math.log2(j&jump) )
            creation_site = int(math.log2(i&jump))
            if verbose:
                print("Site of interest : cration at "+str(creation_site)+" and destruction at "+str(destruction_site))
            occupied=0
            for k in np.arange((min(creation_site,destruction_site)+1),max(creation_site,destruction_site)):
                occupied+=gmpy2.bit_test(j,int(k))

            sign = (-1)**(occupied)

            # we finally look at the element
            t = self.T_tilda[creation_site,destruction_site]
            if verbose:
                print("t between "+str(creation_site)+" and "+str(destruction_site)+" = "+str(t))
                print("occupation = "+str(occupied))
            hij+= sign*(t)
        # Kinetic diagonal elements may have appeared during the transformation
        elif i==j:
            for runner in np.arange(2*self.imp_bath_size):
                if(gmpy2.bit_test(int(i),int(runner))):
                    hij+=self.T_tilda[runner,runner]

        return hij

    def h_2_ii(self,state,verbose=False):
        '''
        :param state: (int) binary occupation number
        :return: The "on site" energy (site correspond to impurity or bath) of state i
        '''
        h=0
        state=int(state)
        if verbose:
            print("considering state "+str(state))
        for m in np.arange(self.imp_bath_size):
            for n in np.arange(self.imp_bath_size):
                m = int(m)
                n = int(n)
                occ_m = gmpy2.bit_test(state, m)
                occ_n = gmpy2.bit_test(state, n)
                if m!=n:
                    #direct term
                    h+=self.get_U_tilda(m,m,n,n)*occ_n*occ_m
                    # cross term
                    h+=self.get_U_tilda(m,n,n,m)*occ_m*(1-occ_n)
                else:
                    h+=self.get_U_tilda(m,m,m,m)*occ_m
        return h

    def h_2_cd(self,state_i,state_j):
        '''
        :param state_j: (int) state where the destruction happened
        :param state_i: (int) state where the creation happened
        :return: The "2 particles" energy for a jump from site d towards site c
        '''
        jump = state_i^state_j
        d = int(math.log2(state_j & jump))
        c = int(math.log2(state_i & jump))
        h=0
        state_i=int(state_i)
        state_j=int(state_j)
        for k in np.arange(self.imp_bath_size):
            k=int(k)
            occ_k_1=gmpy2.bit_test(state_i,k)
            occ_k_2=gmpy2.bit_test(state_j,k)
            occ_k_3=gmpy2.bit_test(state_i,k)*(k!=c)
            occ_k_4=gmpy2.bit_test(state_j,k)*(k!=d)
            if(occ_k_1==1):
                h+=self.get_U_tilda(k,k,c,d)
            if(k!=c and k!=d and occ_k_1==1):
                h+=-self.get_U_tilda(k,d,c,k)+self.get_U_tilda(c,k,k,d)
            h-=self.get_U_tilda(c,k,k,d)
            if(occ_k_2==1):
                h+=self.get_U_tilda(c,d,k,k)
            #if(occ_k_2==1):
            #    h+=self.get_U_tilda(c,d,k,k)
            #if(occ_k_3==1):
            #    h-=self.get_U_tilda(k,d,k,c)
            #if(occ_k_4==1):
            #    h-=self.get_U_tilda(c,k,d,k)
            #h+=self.get_U_tilda(c,k,d,k)
        #Check for the sign coming from c+i cj
        occupied = 0
        for k in np.arange((min(c, d) + 1), max(c, d)):
            occupied += gmpy2.bit_test(state_j, int(k))

        sign = (-1) ** (occupied)
        return sign*h

    def h_2_cdef(self,state_i,state_j):
        '''
        :param state_j: (int) state where the destruction happened
        :param state_i: (int) state where the creation happened
        :return: The "2 particles" energy for 2 jump from site d towards site c
        '''
        jumps = state_i^state_j
        # get the position of the 2 destruction and creation sites
        d_sites = int(state_j & jumps)
        c_sites = int((state_i & jumps))
        found_c1 = False
        found_d1 = False
        c1=0
        c2=0
        d1=0
        d2=0
        for i in np.arange(self.imp_bath_size):
            i=int(i)
            if(gmpy2.bit_test(d_sites,i)):
                if found_d1:
                    d2=i
                else:
                    found_d1=True
                    d1=i
            elif(gmpy2.bit_test(c_sites,i)):
                if found_c1:
                    c2=i
                else:
                    found_c1=True
                    c1=i

        # compute the term
        h=self.get_U_tilda(c1,d1,c2,d2)-self.get_U_tilda(c1,d2,c2,d1)-self.get_U_tilda(c2,d1,c1,d2)+self.get_U_tilda(c2,d2,c1,d1)

        # get the occupation between c2 d2
        occupied = 0
        for k in np.arange((min(c2, d2) + 1), max(c2, d2)):
            occupied += gmpy2.bit_test(state_j, int(k))

        s1 = (-1) ** (occupied)

        # compute the intermediary state
        tmp_state = gmpy2.bit_flip(state_j,c2)
        tmp_state = gmpy2.bit_flip(tmp_state,d2)
        # get the occupation berween c1 d1
        occupied = 0
        for k in np.arange((min(c1, d1) + 1), max(c1, d1)):
            occupied += gmpy2.bit_test(tmp_state, int(k))

        s2 = (-1) ** (occupied)

        return s1*s2*h

    def get_U_tilda(self,a,b,c,d):
        #get the required columns
        n=self.lattice.nb_sites
        O1=np.conjugate(self.W[:n,a])
        O2=(self.W[:n,b])
        O3=np.conjugate(self.W[n:,c])
        O4=(self.W[n:,d])
        #perform element wise multiplication
        p=0
        for i in np.arange(n):
            p+= O1[i]*O2[i]*O3[i]*O4[i]
        return self.u*p

    def build_H(self):
        '''
        Construct the hamiltonian in the Fock Space
        :return:
        '''
        # we have len(impurity_index) * 2 sites, each site has 4 possible occupancy
        n=2*len(self.impurity_index)
        fock_dim=2**(2*n)
        H=np.zeros([fock_dim,fock_dim],dtype=complex)

        for i in np.arange(fock_dim):
            for j in np.arange(fock_dim):
                H[i,j]=self.H_ij(i,j,  verbose=False)
        return H
    def set_mu(self,new_mu):
        '''
        update the value of mu in the Hamiltonian
        :param new_mu: new value of mu
        '''
        # Compute the modification on the diagonal of H
        delta_mu = self.mu-new_mu
        #compute the dimension of the hamiltonian
        n=2*len(self.impurity_index)
        fock_dim=2**(2*n)
        for i in np.arange(fock_dim):
            # update each diagonal element of H according to the number of electrons
            self.H[i,i]+=delta_mu*gmpy2.popcount(int(i))
        self.mu=new_mu

    def get_ground_state(self,verbose=False):
        '''
        :return: The ground state of the Hamiltonian
        '''
        e,v = scipy.linalg.eigh(self.H)
        if(verbose):
            print("energies : ")
            print(np.around(e,2))
        index=np.argsort(e)
        n_degenerate = 1
        continuer = True
        gs=[v[:,0]]
        #we also want to fix the number of electrons (avoid degeneracy with different numbers of electrons
        index_el = np.where(np.abs(v[:,0])>0.0001)[0]
        n_el = gmpy2.popcount(int(index_el[0]))
        while(continuer):
            if(np.abs(e[index[n_degenerate]]-e[index[0]])<0.00000001):
                index_bis = np.where(np.abs(v[:, index[n_degenerate]]) > 0.00001)[0]
                n_el_bis=0
                for runner in index_bis:
                    n_el_bis += gmpy2.popcount(int(runner))
                n_el_bis/=len(index_bis)
                if np.abs(n_el_bis-n_el)<0.001:
                    gs.append(v[:,index[n_degenerate]])
                n_degenerate+=1
            else:
                continuer=False
        return gs
