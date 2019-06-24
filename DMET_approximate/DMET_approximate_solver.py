import os
import sys

import numpy as np
import scipy.linalg
import scipy.optimize
import gmpy2
import math
import scipy.misc
dirname = os.path.dirname(__file__)
path = os.path.join(dirname, "../mean_field/")
sys.path.append(dirname + "/../lattice/")
from HC_Lattice import HC_Lattice

sys.path.append(dirname + "/../mean_field/")
from Mean_field_solver import Mean_field_solver



'''
:author: Romain Fournier
:date: 01.04.2019
:role: Handle the communication between MF solution and Impurity solution
:contact: romain.fournier@epfl.ch
'''


class DMET_approximate_solver:
    def __init__(self, t, u, n, lattice_size,impurities,influences,ring_config=False):
        '''
        :param t: (double) hopping integral (appears as -t c+_i c_j)
        :param u: (double) on site e-e repulsion
        :param n: (int) number of electrons on the lattice
        :param lattice_size: (2x1 int) number of rows and columns of the lattice
        '''
        self.t = t # transfer integral
        self.u = u # on-site repulsion
        self.n = n # total number of electrons

        self.impurities = self.lattice_site2impurity_site(impurities)
        self.influences = self.lattice_site2impurity_site(influences)
        # prepare the lattice
        self.lattice_size = lattice_size
        self.lattice = HC_Lattice(height=lattice_size[0], length=lattice_size[1],ring=ring_config)
        self.filling = self.n/self.lattice.nb_sites/2
        self.nb_sites = self.lattice.nb_sites

        # Initialize the potential
        self.potential = self.initialize_potential()
        self.lr = 1 # learning rate of the potential

        # prepare the impurity solver
        self.J,self.J_up,self.J_down = self.build_look_up_table(2*len(self.impurities[0]))
        self.H_1=[]
        self.H_2=[]

    def lattice_site2impurity_site(self,list_sites):
        '''
        substract 1 to all element of the list in all lists
        :param list_sites: list of list of sites
        :return: same list with all element minorated by 1
        '''
        for run_l,l in enumerate(list_sites):
            for run_x,x in enumerate(l):
                list_sites[run_l][run_x]=x-1
        return list_sites

    def build_look_up_table(self,nb_sites):
        '''
        Build a look up tables to index the states in occupation representation
        J(i) is the binary representation of the i-th state
        Starting with the i-th state, its index can be found by computing (J_up(int(i_up))+J_down(int(i_down)))
        :return: J J_up and J_down
        '''
        L = nb_sites
        N = nb_sites # todo : custum number of electrons

        n_states_per_spin = 2 ** (L)

        J_up = np.zeros([n_states_per_spin],dtype="int")
        J_down = np.zeros([n_states_per_spin],dtype="int")
        J = np.zeros([int(scipy.misc.comb(2 * L, N))],dtype="int")

        j_down = int(0) # First runner
        for n_down in np.arange(N+1):
            n_up=N-n_down
            n_states_up=scipy.misc.comb(L,n_up)
            j_up=int(0)
            for down_part in np.arange (n_states_per_spin):
                down_part=int(down_part)
                n_down_electrons = gmpy2.popcount(down_part)
                if n_down_electrons == n_down:
                    for upper_part in np.arange(n_states_per_spin):
                        upper_part=int(upper_part)
                        n_up_electrons = gmpy2.popcount(upper_part)
                        if (n_up_electrons==n_up):
                            J_up[upper_part]=j_up
                            J_down[down_part]=j_down
                            J[j_down+j_up]=n_states_per_spin*upper_part+down_part
                            j_up+=1
                    j_up=int(0)
                    j_down+=int(n_states_up)

        return J,J_up,J_down

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
        lines = [x for x in impurity_index] + [x + self.lattice.nb_sites for x in impurity_index]
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

    def solve_system(self,verbose=False):
        '''
        find the potential that give a consistent solution between Mean Field and impurity solutions
        :param impurity_indexes: list of index of the impurity
        :param verbose:
        '''
        old_potential = self.potential
        do = True
        while (scipy.linalg.norm(old_potential-self.potential)>1e-5 or do):
            do=False # do forces to go once in the loop
            old_potential=self.potential # save the potential
            for impurity,influence in zip(self.impurities,self.influences):
                self.optimize_potential(impurity,influence) # optimize the potential
                print("delta pot")
                print(scipy.linalg.norm(old_potential-self.potential))
            #self.optimize_potential_bis()

    def optimize_potential_bis(self):
        if (self.norm_bis(self.potential) > 1e-5):
            influence=self.influences[0]
            coef = np.random.normal(0, 1, [2 * len(influence) ** 2])  # Number of parameters
            # coef=np.zeros([2*len(influence)**2])
            f = lambda coef: self.norm_bis(self.get_u_from_coef(coef, influence))

            coef = scipy.optimize.fmin(f, coef, maxiter=250000,maxfun=500000, disp=True)
            V = self.get_u_from_coef(coef, influence)
            self.potential = (1 - self.lr) * self.potential + self.lr * V

    def norm_bis(self,V):
        total=0
        rho_mf = self.get_rho_from_u(V)
        for impurity,influence in zip(self.impurities,self.influences):
            #  get the orbitals corresponding to the bath and impurity
            B = self.get_schmidt_orbitals(impurity)
            #  get the density matrix
            rho_impurity = self.get_rho_impurity(B, impurity, influence)
            total+=self.density_norm(rho_mf,rho_impurity,B)
        return total

    def optimize_potential(self,impurity,influence):
        '''
        tune the values of u present in influence to make the mean field solution looks like the impurity solution
        '''
        # get the orbitals corresponding to the bath and impurity
        B = self.get_schmidt_orbitals(impurity)
        # get the density matrix
        rho_impurity = self.get_rho_impurity(B,impurity,influence)
        print("delta rho")
        print(np.around(self.density_norm(rho_imp=rho_impurity,rho_mf=self.get_rho_from_u(self.potential),B=B),2))
        if(self.density_norm(rho_imp=rho_impurity,rho_mf=self.get_rho_from_u(self.potential),B=B)>1e-5):
            # optimize the values of u
            coef = np.random.normal(0,1,[2*len(influence)**2]) # Number of parameters
            #coef=np.zeros([2*len(influence)**2])
            f = lambda coef: self.density_norm(rho_mf=self.get_rho_from_u(self.get_u_from_coef(coef,influence)),rho_imp=rho_impurity, B=B)
            coef = scipy.optimize.fmin(f, coef, maxiter=25000, maxfun=400000, disp=True)
            V = self.get_u_from_coef(coef,influence)
            self.potential = (1 - self.lr) * self.potential + self.lr *V
            self.potential-=np.mean(np.diag(self.potential))*np.eye(2*self.nb_sites)

    def get_u_from_coef(self,c,influence):
        '''
        :param c: coefficients that are put in the correlation potential
        :param influence: site that can be modified
        :return: coorelation potential
        '''
        # declare the potential
        s = 2*self.lattice.nb_sites
        N=self.lattice.nb_sites
        u = np.zeros([s,s],dtype=complex)  # the modifications we make to the potential
        counter=0
        for i in influence:
            for j in influence:
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
                elif j>i:
                    # take care of the complex elements
                    # up_up and down_down
                    u[i,j]=c[counter]+1j*c[counter+1]
                    u[i+N, j+N] = c[counter] + 1j * c[counter + 1]
                    counter+=2
                    #cross spins
                    u[i + N, j] = c[counter]+1j*c[counter+1]
                    u[i, j + N] = c[counter]+1j*c[counter+1]
                    counter+=2
        u=u+np.conjugate(np.transpose(u))
        u = u-1/s*np.trace(u)*np.eye(s) # chemical potential to 0
        return u

    def density_norm(self,rho_mf,rho_imp,B):
        '''
        :param impurity_list : list of impurity sites
        :return:||rho-rho_imp||
        '''
        B=np.conjugate(B)
        rho_projected = scipy.matmul(scipy.matmul(np.transpose(B),rho_mf),np.conjugate(B))
        # update the difference
        diff = scipy.linalg.norm(np.abs(rho_imp)-np.abs(rho_projected))
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

    def get_rho_impurity(self,B,impurity,influence,verbose=False):
        '''
        :param impurity: index of the impurity
        :param influence : index of the "influence sites"
        :param B schmidt orbitals
        :return: single particle density matrix in the impurity basis
        '''
        v = self.get_ground_state(impurity,influence,B, verbose)
        rho = self.get_rho_from_v(v,len(impurity))
        return rho

    def get_rho_from_v(self,v,l):
        '''
        :param v: wave vector
        :param l: number of sites
        :return: <psi|cidci|psi>
        '''
        # initialize the density matrix
        rho = np.zeros([4*l,4*l],dtype=complex)

        # populate the matrix

        # iterate over all impurity+bath positions i
        for i in np.arange(4*l,dtype=int):
            # iterate over all impurity+bath positions j
            for j in np.arange(4*l,dtype=int):
                #iterate over all elements in the basis of the look-up table
                for run_k,k in enumerate(self.J):
                    #check if a+i aj|k> is not null
                    if(gmpy2.bit_test(int(k),int(j))):
                        if(gmpy2.bit_test(int(k),int(i))==0 or i==j):
                            new_state=gmpy2.bit_flip(int(k),int(j))
                            new_state=gmpy2.bit_flip(new_state,int(i))
                            run_new = self.state2index(new_state)
                            #and don't forget the sign !
                            occupancy_between=0
                            for l in np.arange(min(i,j)+1,max(i,j)):
                                occupancy_between+=gmpy2.bit_test(int(k),int(l))
                            rho[i,j]+=(-1)**occupancy_between*np.conjugate(v[run_new])*v[run_k]
        return rho

    def get_ground_state(self,impurity,influence,B,verbose=False):
        '''
        Get the ground state of the impurity + bath system
        :param influence: list of the influence sites
        :param B: schmidt orbitals
        :param verbose:
        :return: ground state
        '''
        H = self.build_H(B,impurity,influence) # build the hamiltonian
        ev,v = scipy.linalg.eigh(H) # compute only the smallest eigenvector
        arg = np.argmin(ev)
        print(ev[arg])
        return v[:,arg] #only return the vector, not the eigenvalue

    def build_H(self,B,impurity,influence):
        # Initialize the 1 and 2 particle(s) operators
        self.H_1 = self.build_H1(B,influence) # single particle part (Hcin + Hpot outside influence)
        self.H_2 = self.build_U2_tilda(B,impurity,influence) # 2 particles part (Hpot inside influence)

        # declare H
        fock_dim = np.shape(self.J)[0]
        H=np.zeros([fock_dim,fock_dim],dtype=complex)

        # popuplate H
        for run_i,i in enumerate(self.J):
            # one particle operator
            for k in np.arange(4*len(impurity)):
                for m in np.arange(4*len(impurity)):
                    # apply ck^+ cm|i>
                    sgn,tmp_state = self.apply_c(i,m)
                    sgn2,tmp_state = self.apply_c_dagger(tmp_state,k)
                    # compute the total change of sign
                    sgn = sgn2*sgn
                    # if the operators did not destroy the state
                    if abs(sgn)>0.1:
                        # get the index of the final state
                        index_j = self.state2index(tmp_state)
                        # add the contribution
                        H[run_i,index_j]+=sgn*self.H_1[k,m]
            # two particle operators
            for k in np.arange(4*len(impurity)):
                for l in np.arange(k+1,4*len(impurity)):
                    for m in np.arange(4*len(impurity)):
                        for n in np.arange(m+1,4*len(impurity)):
                            # U^{k,l}_{m,n}c_k^+ c_l^+ c_m c_n |i>
                            sgn, tmp_state = self.apply_c(i, n)
                            sgn2, tmp_state = self.apply_c(tmp_state, m)
                            sgn=sgn*sgn2
                            sgn2, tmp_state = self.apply_c_dagger(tmp_state, l)
                            sgn = sgn * sgn2
                            sgn2, tmp_state = self.apply_c_dagger(tmp_state, k)
                            sgn = sgn * sgn2
                            # if we arrived somewhere
                            if abs(sgn)>0.1 :
                                # get the index of the final state
                                index_j = self.state2index(tmp_state)
                                # add the contribution
                                H[run_i, index_j] += sgn * self.H_2[self.get_matrix_index(k,l,int(4 * len(impurity))),self.get_matrix_index(m,n,int(4 * len(impurity)))]
        return H

    def state2index(self,ii):
        n =2* len(self.impurities[0])
        i_up = int(ii / 2 ** n)
        i_down = ii % (2 ** n)

        i = self.J_up[i_up] + self.J_down[i_down]
        return i

    def get_matrix_index(self,k,l,r):
        return int(l-k-1+(2*r-k-1)*k/2)

    def build_H1(self,B,influence):
        '''
        Project the 1 particle part of H (Hcin + "MFHpot" outside the influence zone)
        :param B: Orbitals
        :param influence: sites of influence
        :return: projected single particle part of H
        '''

        # Build the matrix of interest before rotation, i.e. kinetic part + potential outside influence zone
        H_1 = self.compute_T_tilda(B)+self.compute_U_mf_tilda(B,influence)
        return H_1

    def compute_T_tilda(self,B):
        '''
        Compute the hopping matrix in the b+ basis
        :return: 2*2*L X 2*2*L matrix
        '''
        # compute the Hopping matrix
        T = np.zeros([2 * self.lattice.nb_sites, 2 * self.lattice.nb_sites])
        # Iterate over all sites
        for i in np.arange(self.lattice.nb_sites):
            for j in self.lattice.get_neighbors(i + 1):
                T[j - 1 + self.lattice.nb_sites, i + self.lattice.nb_sites] = -self.t
                T[j - 1, i] = -self.t
        return scipy.matmul(scipy.matmul(np.transpose(B), T),np.conjugate( B))

    def compute_U_mf_tilda(self,B,influence):
        '''
        Outside the influence zone, we work in a mean_field approach
        :param influence:
        :return:
        '''
        size = np.shape(self.potential)
        u = np.zeros(size)
        for i in np.arange(self.nb_sites):
            for j in np.arange(self.nb_sites):
                if i not in influence:
                    if j not in influence:
                        u[i,j]=self.potential[i,j]
                        u[i+self.nb_sites,j]=self.potential[i+self.nb_sites,j]
                        u[i,j+self.nb_sites] = self.potential[i,j+self.nb_sites]
                        u[i+self.nb_sites,j+self.nb_sites] = self.potential[i+self.nb_sites,j+self.nb_sites]
        return scipy.matmul(scipy.matmul(np.transpose(B), u), np.conjugate(B))

    def build_U2_tilda(self,B,impurity,influence):
        '''
        Transform the term -U ai+aj+akal into the impurity basis
        :param B: schmidt orbitlas
        :param influence: site to transform
        :return:
        '''

        # The tricks is that we can only consider the terms ck+cl+cmcn with k<l and m<n by symmetrizing U2 tilda conveniently
        # Therefore, we only need to store half of the (2spin*(n_impurity+N-bath))**2 - the diagonal
        r = int(4*len(impurity))
        s = int(r*(r-1)/2)
        # since we only care about k<l and m<n, we can store the 4th order as a matrix
        U2_tilda =  np.zeros([s,s],dtype=complex)
        # We need to compute each Uklmn :
        influence_down = [x + self.nb_sites for x in influence]

        for k in np.arange(r):
            for l in np.arange(k+1,r):
                # Compute the index x corresponding to kl
                id_x = int(l-k-1+(2*r-(k+1))*k/2)

                for m in np.arange(r):
                    for n in np.arange(m+1,r):
                        # compute teh index y similaraly
                        id_y = int(n-m+(2*r-(m+1))*(m)/2 -1)
                        # Now, there will be 4 different contributions (ck cl -> -cl ck etc)
                        v1_up = (B[influence,k])
                        v1_down = (B[influence_down, k])

                        v2_up = (B[influence, l])
                        v2_down = (B[influence_down, l])

                        v3_up = np.conjugate(B[influence, m])
                        v3_down = np.conjugate(B[influence_down, m])

                        v4_up = np.conjugate(B[influence, n])
                        v4_down = np.conjugate(B[influence_down, n])

                        # compute the term (change the sign each time we do a permutation)
                        U2_tilda[id_x,id_y]=-self.u*np.sum(v1_up*v2_down*v3_up*v4_down-v1_down*v2_up*v3_up*v4_down+v1_down*v2_up*v3_down*v4_up-v1_up*v2_down*v3_down*v4_up)

        return U2_tilda

    def occupation_between(self,state,a,b):
        '''
        Return the number of occupied sites between a and b
        :param state: integer representation of a quantum state
        :param a: "min" site
        :param b: "max" site
        :return: number of occupied states between min and max
        '''
        # cast the state
        state = int(state)
        # make sure that a is the min and b the max
        a,b=np.sort([a,b])

        nb_el=0 # numer of electrons between the two sites
        # loop over all sites between a and b
        for runner in np.arange(a+1,b,dtype="int"):
            runner = int(runner)
            # add an electron if present at site runner
            nb_el+=gmpy2.bit_test(state,runner)

        return nb_el

    def apply_c(self,state,a):
        '''
        Compute c_a|state>
        :param state: integer representation of quantum state
        :param a: spin-site where we will apply c
        :return: sign (+-1) and new state
        '''
        # Make sure that state and the site are casted correctly
        state = int(state)
        a = int(a)
        # check the occupation until a
        sgn =(-1)**self.occupation_between(state,-1,a) # -1 because we count the first state at 0

        # check that there is an electron at a
        if not gmpy2.bit_test(state, a):
            # if there isn't an electron, set the sgn to 0 (and state to 0 as a convention)
            sgn=0
            state=0
        else :
            # otherwise we flip the bit
            state = gmpy2.bit_flip(state,a)

        return sgn,state

    def apply_c_dagger(self,state,a):
        '''
        Compute c_a+|state>
        :param state: integer representation of quantum state
        :param a: spin-site where we will apply c
        :return: sign (+-1) and new state
        '''
        # Make sure that state and the site are casted correctly
        state = int(state)
        a = int(a)
        # check the occupation until a
        sgn = (-1)**self.occupation_between(state,-1,a) # -1 because we count the first state at 0

        # check that there is an electron at a
        if gmpy2.bit_test(state, a):
            # if there is already an electron, set the sgn to 0 (and state to 0 as a convention)
            sgn=0
            state=0
        else :
            # otherwise we flip the bit
            state = gmpy2.bit_flip(state,a)

        return sgn,state
