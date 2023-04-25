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
    
    def create_basis():
        """create ordered basis with all possible determinants"""
        basis=list()
        for x in itertools.combinations(range(condon.orbs_in_system*2),condon.elec_in_system):
            basis.append(set(x))
        # sort the determinants based on the basis of the diagonal fci mel
        basis.sort(key = determinant_diagonal)
        return basis
    
    def populate(basis):
        """populate the matrix with the condon elements and then diagonalize it to find the lowest eigenvalue"""
        # create a sparse matrix with the number of rows and columns equal to the number of determinants
        mat = np.empty((len(basis), len(basis)))
        # iterate over bras and kets
        for i, bra in enumerate(basis):
            for j, ket in enumerate(basis):
                number_of_differences = bra.difference(ket)
                # if more than two differences, the condon element is zero and no need to call condon()
                if len(number_of_differences) >= 2:
                    pass
                else:
                    # find the condon element
                    condon_element = condon.condon((bra, ket), integrals)
                    # populate the matrix
                    mat[i,j] = condon_element
        # find just the eigenvalues of mat by diagonalizing it
        eigenvalues = np.linalg.eigvalsh(mat)
        return eigenvalues[0]
    
    return populate(create_basis())
#print((1/2)*generation(integrals))
    
cProfile.run("generation(integrals)")




    
