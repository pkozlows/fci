import numpy as np
from full_matrix.slater import condon
import itertools
import cProfile

# define the system inputs
spin_of_system=0
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
        number_of_alpha_electrons = (elec_in_system + spin_of_system) // 2
        number_of_beta_electrons = (elec_in_system - spin_of_system) // 2
        # create all possible alpha strings
        alpha_strings = list(itertools.combinations(range(0, orbs_in_system), number_of_alpha_electrons))
        # create all possible beta strings
        beta_strings = list(itertools.combinations(range(0, orbs_in_system), number_of_beta_electrons))
        number_of_strings = len(alpha_strings) * len(beta_strings)
        # check that the number of alpha and beta strings are equal if the spin of the system is zero
        if spin_of_system == 0:
            assert len(alpha_strings) == len(beta_strings)
        # create a basis of possible determinants
        basis = []
        for alpha in alpha_strings:
            for beta in beta_strings:
                # create a determinant object for each pair of alpha and beta strings
                basis.append((set(alpha), set(beta)))
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
                # find the differences between the bra and ket
                alpha_differences = bra[0].difference(ket[0])
                number_alpha_differences = len(alpha_differences)
                beta_differences = bra[1].difference(ket[1])
                number_beta_differences = len(beta_differences)
                # combine the differences
                number_total_differences = number_alpha_differences + number_beta_differences

                # if more than two differences, the condon element is zero and no need to call condon()
                if number_total_differences > 2:
                    pass
                else:
                    # find the condon element
                    condon_element = condon((bra, ket), integrals)
                    # populate the matrix
                    mat[i, j] = condon_element
        return mat
    
    return populate(create_basis())
assert(np.linalg.eigvalsh(generation(integrals))[0] - -7.8399080148963369 < 1e-10)
    
# cProfile.run("generation(integrals)")




    
