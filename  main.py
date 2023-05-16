import numpy as np
from slater import condon
import itertools
import cProfile

# define the system inputs
elec_in_system=6
orbs_in_system=6
# load in the intervals
one_elec_ints = np.load("h1e.npy")
two_elec_ints = np.load("h2e.npy")
integrals = (one_elec_ints, two_elec_ints)
# np.set_printoptions(threshold=np.inf)

def generation(integrals):  
    """generate the for configuration interaction matrix and find the lowest eigenvalue"""
    
    def determinant_diagonal(determinant):
        """returns the diagonal fci mel of a determinant for later ordering in basis"""
        return condon((determinant, determinant), (one_elec_ints, two_elec_ints))
    
    def create_basis():
        """create ordered basis with all possible determinants"""
        # create all possible alpha strings i.e. only with even orbitals
        alpha_strings = itertools.combinations(range(0, orbs_in_system*2, 2), elec_in_system//2)
        # create all possible beta strings i.e. only with odd orbitals
        beta_strings = itertools.combinations(range(1, orbs_in_system*2, 2), elec_in_system//2)
        # create a basis of possible determinants with total spin zero
        basis = []
        for alpha in alpha_strings:
            for beta in beta_strings:
                # create a determinant object for each determinant
                basis.append(set(alpha) | set(beta))
        # sort the determinants based on the basis of the diagonal fci mel
        basis.sort(key = determinant_diagonal)
        return basis
    
    def populate(basis):
        """populate the matrix with the condon elements and then diagonalize it to find the lowest eigenvalue"""
        # create a sparse matrix with the number of rows and columns equal to the number of determinants
        mat = np.zeros((len(basis), len(basis)))
        # iterate over bras and kets
        for i, bra in enumerate(basis):
            for j, ket in enumerate(basis):
                number_of_differences = bra.difference(ket)
                # if more than two differences, the condon element is zero and no need to call condon()
                if len(number_of_differences) >= 2:
                    pass
                else:
                    # find the condon element
                    condon_element = condon((bra, ket), integrals)
                    # populate the matrix
                    mat[i, j] = condon_element
        # find just the eigenvalues of mat by diagonalizing it
        # check that did this a real, symmetric matrix
        assert np.allclose(mat, mat.T)    
        eigenvalues = np.linalg.eigvalsh(mat)
        return eigenvalues[0]
    
    return populate(create_basis())
print(generation(integrals))
    
# cProfile.run("generation(integrals)")




    
