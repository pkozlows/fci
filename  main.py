import numpy as np
import condon
import itertools
from scipy.sparse import lil_matrix
import cProfile
# load in the intervals
one_elec_ints = np.load("h1e.npy")
two_elec_ints = np.load("h2e.npy")
integrals = (one_elec_ints, two_elec_ints) 
def generation(integrals):  
    """generate the for configuration interaction matrix and find the lowest eigenvalue"""
    
    def determinant_diagonal(determinant):
        """returns the diagonal fci mel of a determinant for later ordering in basis"""
        return condon.condon((determinant, determinant), (one_elec_ints, two_elec_ints))
    
    def possible_determinants():
        """create a list of all possible determinants"""
        basis=list()
        for x in itertools.combinations(range(condon.orbs_in_system*2),condon.elec_in_system):
            basis.append(set(x))
        # sort the determinants based on the basis of the diagonal fci mel
        basis.sort(key = determinant_diagonal)
        # create a sparse matrix with the number of rows and columns equal to the number of determinants
        # print(basis)
        mat = np.empty((len(basis), len(basis)))
        print(basis[0])
        print(determinant_diagonal(basis[0]))
        hd_index = basis.index({0,1,2,3,4,5})
        highest_index = basis.index({6,7,8,9,10,11})
        print(determinant_diagonal(basis[hd_index]))
        print(determinant_diagonal(basis[highest_index]))
        assert({0,1,2,3,4,5} in basis)
        assert({6, 7,8,9,10})
        return basis, mat

    # def populate (mat, basis):
    #     """populate the matrix with the condon elements and then diagonalize it to find the lowest eigenvalue"""
    #     for det_pair in condon.gen_unique_pairs(condon.elec_in_system, condon.orbs_in_system):
    #         # get the determinants from the pair
    #         bra = det_pair[0]
    #         ket = det_pair[1]
    #         # implement the above commented out code in more efficient way using numpy
    #         # find the indices of the determinants in the determinant basis
    #         bra_index = basis.index(bra)
    #         ket_index = basis.index(ket)
    #         # find the condon element
    #         condon_element = condon.condon(det_pair, integrals)
    #         # populate the matrix
    #         mat[bra_index, ket_index] = condon_element
    #     # find just the eigenvalues of mat by diagonalizing it
    #     eigenvalues = np.linalg.eigvalsh(mat)
    #     return eigenvalues[0]
    possible_determinants()
print(generation(integrals))
    # return populate(mat, basis)
# cProfile.run("generation(integrals)")




    
