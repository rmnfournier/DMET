import numpy as np

'''
:author: Romain Fournier
:date: 28.03.2019 
:role: Handle a HoneyComb lattice
:todo: Generalize this class to a general 2D lattice
:contact: romain.fournier@epfl.ch
'''


class HC_Lattice:
    def __init__(self, length, height):
        '''
        :param length: (int) number of squares in the lattice matrix (horizontally)
        :param height: (int) number of squares in the lattice matrix (vertically)
        '''
        # Save the lattice size
        self.length = length
        self.height = height
        # number of sites
        self.nb_sites = 0
        # Initialization of the lattice
        self.lattice = np.zeros([height, length], dtype=int)
        # Fill it
        self.init_lattice()

    def init_lattice(self):
        '''
        Create a HC lattice. This method fills the matrix with 0 where there isn't any site, and with the site number othervise.
        '''
        # iterate over all sites
        for i in np.arange(self.height):
            for j in np.arange(self.length):
                # check if the square is occupied
                if (self.is_occupied(i, j)):
                    # give the next number to the square ij
                    self.nb_sites += 1
                    self.lattice[i, j] = self.nb_sites

    def is_occupied(self, i, j):
        '''
        This method checks if the site i,j is occupied or not
        :param i: (int) row of the matrix
        :param j: (int) column of the matrix
        :return: true if (i,j) is occupied, false othervise
        '''
        # if outside, return false
        if (i < 0 or j < 0 or i >= self.height or j >= self.length):
            return False

        # even line
        if (j % 4 < 2):
            # return true if it is an even line, false othervise
            return True and (i % 2 == 0)
        else:
            # return true if it is an odd line, false othervise
            return True and (i % 2 == 1)

    def get_neighbors(self, site):
        '''
        Get the neighbors of site
        :param site: (int) the id of the site of interest
        :return: list(int) corresponding to the ids of the nearest neighbors sites
        '''
        nn_list = []
        # Get the coordinate of the site
        coord = self.find_site(site)
        # Define the possible n.n.
        nn = [[0, -1], [1, 1], [-1, 1], [0, 1], [1, -1], [-1, -1]]
        for neighbor in nn:
            nn_coord = [sum(x) for x in zip(neighbor, coord)]
            if self.is_occupied(nn_coord[0], nn_coord[1]):
                nn_list.append(self.lattice[nn_coord[0], nn_coord[1]][0])
        return nn_list

    def find_site(self, site):
        '''
        Get the coordinate of the site with id site
        :param site: (int) id of the state of interest
        :return: [x,y] corresponding to the indices of the site in the lattice matrix
        '''
        return np.where(self.lattice == site)
