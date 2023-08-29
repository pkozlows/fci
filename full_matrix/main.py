import time
import numpy as np
# from slater import condon
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



def anti_commutator(pair):
  """takes tuple of a pair of determine, which is each represented by two tuples, which contains the offa and beta orbs, respectively. the format for this one is: ((bra_alpha, bra_beta), (ket_alpha, ket_beta)).returns an integer that represents the face factor."""
  # initialize the unsorted string sets
  bra_alpha = pair[0][0]
  bra_beta = pair[0][1]
  ket_alpha = pair[1][0]
  ket_beta = pair[1][1]
  # determine the alpha differences between the two determinants
  alpha_excitation = (sorted(bra_alpha.difference(ket_alpha)), sorted(ket_alpha.difference(bra_alpha)))
  assert(len(alpha_excitation[0]) == len(alpha_excitation[1]))
  # determine the beta differences between the two determinants
  beta_excitation = (sorted(bra_beta.difference(ket_beta)), sorted(ket_beta.difference(bra_beta)))
  assert(len(beta_excitation[0]) == len(beta_excitation[1]))
  # determine the number of differences between the two determinants
  assert len(alpha_excitation[0]) + len(beta_excitation[0]) == len(alpha_excitation[1]) + len(beta_excitation[1])
  number_of_differences = len(alpha_excitation[0]) + len(beta_excitation[0])   
  # treat the number of differences the same, counting the number of swabs anyways for each spin string
  def bubble_sort(string, excitation):
    """takes two unsorted lists, one that has a whole determinant and the other that has the unique orbs of the determinant. returns the number of swaps needed to sort the list."""
    swaps = 0
    for i, orb in enumerate(excitation):
      swaps += sorted(string).index(orb) - i
    return swaps
    # sort the lists and get the number of swaps
  bra_alpha_swaps = bubble_sort(bra_alpha, alpha_excitation[0])
  bra_beta_swaps = bubble_sort(bra_beta, beta_excitation[0])
  ket_alpha_swaps = bubble_sort(ket_alpha, alpha_excitation[1])
  ket_beta_swaps = bubble_sort(ket_beta, beta_excitation[1])

  # return the face factor
  return (-1)**(bra_alpha_swaps + bra_beta_swaps + ket_alpha_swaps + ket_beta_swaps)
def condon(pair: tuple, integrals: tuple) -> int:
    """
    Takes tuple of a pair of determinants that contains a tuple of two spin strings sets, which are alpha and beta, respectively, and also a tuple with one and two electron integrals.
    Returns the matrix element between the pair.
    Args:
        pair (tuple): Tuple of determinants ((alpha, beta), (alpha, beta)).
                      Each determinant is a tuple of two spin strings.

        integrals (tuple): Tuple of one-electron and two-electron integrals.
                           Each integral is a numpy ndarray.
    
    Returns:
        int: Matrix element between the pair.
    """
    one_elec_ints = integrals[0]
    two_elec_ints = integrals[1]
    # determine the alpha differences between the two determinants
    bra_alpha_excitation = sorted(pair[0][0].difference(pair[1][0]))
    ket_alpha_excitation = sorted(pair[1][0].difference(pair[0][0]))
    # determine the beta differences between the two determinants
    bra_beta_excitation = sorted(pair[0][1].difference(pair[1][1]))
    ket_beta_excitation = sorted(pair[1][1].difference(pair[0][1]))
    # determine the number of differences between the two determinants
    assert len(bra_alpha_excitation) + len(bra_beta_excitation) == len(ket_alpha_excitation) + len(ket_beta_excitation)
    number_of_differences = len(bra_alpha_excitation) + len(bra_beta_excitation)    
    # take the intersection of the spin orbitals
    alpha_intersection = sorted(pair[0][0].intersection(pair[1][0]))
    beta_intersection = sorted(pair[0][1].intersection(pair[1][1]))
    total_intersection = alpha_intersection + beta_intersection
    if number_of_differences == 0:
        # if there is no difference between the determinants

        # the one electron part
        one_electron = np.einsum('ii->', one_elec_ints[np.ix_(alpha_intersection, alpha_intersection)]) + np.einsum('ii->', one_elec_ints[np.ix_(beta_intersection, beta_intersection)])

        # the two electron part
        coulumb = np.einsum('iijj->', two_elec_ints[np.ix_(alpha_intersection, alpha_intersection, beta_intersection, beta_intersection)]) + 0.5 * np.einsum('iijj->', two_elec_ints[np.ix_(alpha_intersection, alpha_intersection, alpha_intersection, alpha_intersection)]) + 0.5 * np.einsum('iijj->', two_elec_ints[np.ix_(beta_intersection, beta_intersection, beta_intersection, beta_intersection)])
        
        exchange = 0.5 * np.einsum('ijji->', two_elec_ints[np.ix_(alpha_intersection, alpha_intersection, alpha_intersection, alpha_intersection)]) + 0.5 * np.einsum('ijji->', two_elec_ints[np.ix_(beta_intersection, beta_intersection, beta_intersection, beta_intersection)])
        # add them up
        element = one_electron + coulumb - exchange
        assert(anti_commutator(pair) == 1)
    # if there is one difference, m and p, between the determinants
    if number_of_differences == 1:
      # check if it is an alpha difference
      if len(bra_alpha_excitation) == 1:
        m = bra_alpha_excitation
        p = ket_alpha_excitation
        # the conditional two electron part
        exchange_matrix = two_elec_ints[np.ix_(m, alpha_intersection, alpha_intersection, p)]
      # check if it is a beta difference
      if len(bra_beta_excitation) == 1:
        m = bra_beta_excitation
        p = ket_beta_excitation
        # the conditional two electron part
        exchange_matrix = two_elec_ints[np.ix_(m, beta_intersection, beta_intersection, p)]
      # the one electron part
      one_electron = one_elec_ints[m, p]
      # the unconditional two electron part
      coulomb = np.einsum('ijkk->', two_elec_ints[np.ix_(m, p, total_intersection, total_intersection)])
      exchange = np.einsum('ikkj->', exchange_matrix)
      # add them up
      element = anti_commutator(pair)*(one_electron + coulomb - exchange)
    # if there are two differences between the determinants, where in the first determinant there are orbs m and n and in the ket determinant there are orbs p and q
    if number_of_differences == 2:
      # determine if we are dealing with the same or mixed spin
      mixed_spin = False
      # the first case is when the differences are only composed of electrons with the same spin
      if len(bra_beta_excitation) == 2 or len(bra_alpha_excitation) == 2:
        assert(len(bra_beta_excitation) != len(bra_alpha_excitation))
        beta = False
        if len(bra_alpha_excitation) == 2:
          # set the indices if they are only alpha excitations
          m = bra_alpha_excitation[0]
          m_spin = m * 2
          n = bra_alpha_excitation[1]
          n_spin = n * 2
          p = ket_alpha_excitation[0]
          p_spin = p * 2
          q = ket_alpha_excitation[1]
          q_spin = q * 2
        elif len(bra_beta_excitation) == 2:
          beta = True
          # set the indices
          m = bra_beta_excitation[0]
          m_spin = m * 2 + 1
          n = bra_beta_excitation[1]
          n_spin = n * 2 + 1
          p = ket_beta_excitation[0]
          p_spin = p * 2 + 1
          q = ket_beta_excitation[1]
          q_spin = q * 2 + 1
        # both terms are involved
        element = anti_commutator(pair)*(two_elec_ints[m,p,n,q] - two_elec_ints[m,q,n,p])
      # the second case is when the excitations are composed of electrons with different spins
      if len(bra_beta_excitation) == 1 and len(bra_alpha_excitation) == 1:
        assert(len(ket_alpha_excitation) == 1)
        assert(len(ket_alpha_excitation) == len(ket_beta_excitation))
        mixed_spin = True
        # set the indices
        m = bra_alpha_excitation[0]
        m_spin = m * 2
        n = bra_beta_excitation[0]
        n_spin = n * 2 + 1
        p = ket_alpha_excitation[0]
        p_spin = p * 2
        q = ket_beta_excitation[0]
        q_spin = q * 2 + 1
        # only the first term survives
        element = anti_commutator(pair)*(two_elec_ints[m,p,n,q])
    return element
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


start_generation_time = time.time()
# assert(np.linalg.eigvalsh(generation(integrals))[0] - -7.8399080148963369 < 1e-10)
generation(integrals)
end_generation_time = time.time()
print("generation time:", end_generation_time - start_generation_time)
    
# cProfile.run("generation(integrals)", sort="cumtime")




    
