import math
import numpy as np
from pytest import approx
from face_factor import anti_commutator

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
        exchange = 0.5 * np.einsum('ijji->', two_elec_ints[np.ix_(alpha_intersection, alpha_intersection, beta_intersection, beta_intersection)]) + 0.5 * np.einsum('ijji->', two_elec_ints[np.ix_(beta_intersection, beta_intersection, alpha_intersection, alpha_intersection)])
        element = one_electron + coulumb - exchange
    # if there is one difference, m and p, between the determinants
    if number_of_differences == 1:
      # check if it is an alpha difference
      if len(bra_alpha_excitation) == 1:
        m = bra_alpha_excitation
        p = ket_alpha_excitation
        # the one electron part
        one_elec_mel += anti_commutator(pair)*one_elec_ints[m, p]
        # the two electron part
        coulomb = np.einsum('ijkk->ij', two_elec_ints[np.ix_(range(6), range(6), total_intersection, total_intersection)])
        exchange = np.einsum('ijjk->ik', two_elec_ints[np.ix_(range(6), alpha_intersection, alpha_intersection, range(6))])
        two_elec_mel += anti_commutator(pair)*((coulomb - exchange)[m,p])
      # check if it is a beta difference
      if len(bra_beta_excitation) == 1:
        m = bra_beta_excitation
        p = ket_beta_excitation
        # the one electron part
        one_elec_mel += anti_commutator(pair)*one_elec_ints[m, p]
        # the two electron part
        coulomb = np.einsum('ijkk->ij', two_elec_ints[np.ix_(range(6), range(6), total_intersection, total_intersection)])
        exchange = np.einsum('ijjk->ik', two_elec_ints[np.ix_(range(6), beta_intersection, beta_intersection, range(6))])
        two_elec_mel += anti_commutator(pair)*((coulomb - exchange)[m,p])
    # if there are two differences between the determinants, where in the first determinant there are orbs m and n and in the ket determinant there are orbs p and q
    if number_of_differences == 2:
      # determine if we are dealing with the same or mixed spin
      mixed_spin = False
      # the first case is when the differences are only composed of electrons with the same spin
      if len(bra_beta_excitation) == 2 or len(bra_alpha_excitation) == 2:
        assert(len(bra_beta_excitation) != len(bra_alpha_excitation))
        beta = False
        # initialize the indices to false
        m = False
        n = False
        p = False
        q = False
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
        two_elec_mel += anti_commutator(pair)*(two_elec_ints[m,p,n,q] - two_elec_ints[m,q,n,p])
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
        two_elec_mel += anti_commutator(pair)*(two_elec_ints[m,p,n,q])
    return element
# integrals
one_elec_ints = np.load("h1e.npy")
two_elec_ints = np.load("h2e.npy")
# unit testing
# the hf case
assert math.isclose(condon((({0,1,2},{0,1,2}), ({0,1,2},{0,1,2})), (one_elec_ints, two_elec_ints)), -7.739373948970316, rel_tol=1e-7, abs_tol=1e-7)
# the 1 excitation case
# assert math.isclose(condon((({0,1,2},{0,1,2}), ({0,1,2},{0,1,3})), (one_elec_ints, two_elec_ints)), 0, rel_tol=1e-9, abs_tol=1e-12)
# assert math.isclose(condon((({0,1,2},{0,1,2}), ({0,1,2},{1,2,5})), (one_elec_ints, two_elec_ints)), 0, rel_tol=1e-9, abs_tol=1e-12)
# assert math.isclose(condon((({0,1,2},{0,1,2}), ({1,2,5},{0,1,2})), (one_elec_ints, two_elec_ints)), 0, rel_tol=1e-9, abs_tol=1e-12)
# assert math.isclose(condon((({0,1,2},{0,1,2}), ({1,2,5},{0,1,2})), (one_elec_ints, two_elec_ints)), 0, rel_tol=1e-9, abs_tol=1e-12)
# assert math.isclose(condon((({0,1,2},{0,1,2}), ({0,1,2},{1,2,5})), (one_elec_ints, two_elec_ints)), 0, rel_tol=1e-9, abs_tol=1e-12)
# assert math.isclose(condon((({0,1,2},{0,1,2}), ({0,1,2},{0,1,5})), (one_elec_ints, two_elec_ints)), 0, rel_tol=1e-9, abs_tol=1e-12)
# assert math.isclose(condon((({0,1,2},{0,1,2}), ({0,1,5},{0,1,2})), (one_elec_ints, two_elec_ints)), 0, rel_tol=1e-9, abs_tol=1e-12)
# # the above are passing tests
# assert math.isclose(condon((({0,1,2},{0,2,5}), ({0,1,2},{0,1,2})), (one_elec_ints, two_elec_ints)), 0, rel_tol=1e-7, abs_tol=1e-7)
# assert math.isclose(condon((({0,1,2},{0,1,2}), ({0,2,3},{0,1,2})), (one_elec_ints, two_elec_ints)), 0, rel_tol=1e-7, abs_tol=1e-7)
# # the 2 excitation case
# # only beta difference
# assert math.isclose(condon((({0,1,2},{0,1,2}), ({0,1,2},{0,3,4})), (one_elec_ints, two_elec_ints)), -0.04655311805628327, rel_tol=1e-9, abs_tol=1e-12)
# # only alpha difference
# assert math.isclose(condon((({2,4,5},{0,1,2}), ({0,1,2},{0,1,2})), (one_elec_ints, two_elec_ints)), 0.016177186667624063, rel_tol=1e-9, abs_tol=1e-12)
# # mixed differences
# assert math.isclose(condon((({1,2,4},{0,1,2}), ({1,2,5},{0,1,4})), (one_elec_ints, two_elec_ints)), 3.452099717193846e-16, rel_tol=1e-9, abs_tol=1e-12)


        


    