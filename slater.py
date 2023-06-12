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
    total_intersection = sorted(alpha_intersection + beta_intersection)
    # initialize the matrix element
    one_elec_mel = 0
    two_elec_mel = 0
    if number_of_differences == 0:
        # initialize meshes for converting spin orbs to special ints
        one_elec_special_union_xgrid = np.ix_(total_intersection, total_intersection)
        two_elec_special_union_xgrid=np.ix_(total_intersection, total_intersection, total_intersection, total_intersection)
        one_elec_special_union_ints = one_elec_ints[one_elec_special_union_xgrid]
        two_elec_special_union_ints = two_elec_ints[two_elec_special_union_xgrid]
        # if there is no difference between the determinants
        one_elec_mel += np.einsum('ii->', one_elec_special_union_ints)
        coulumb = np.einsum('iijj->', two_elec_special_union_ints) 
        exchange = np.einsum('ijji->', two_elec_special_union_ints)
        two_elec_mel += (1/2)*(coulumb - (1/2)*exchange)
    # if there is one difference, m and p, between the determinants
    if number_of_differences == 1:
      # check if it is an alpha or beater difference
      if len(bra_alpha_excitation) == 1:
        m_special = bra_alpha_excitation
        p_special = ket_alpha_excitation
      if len(bra_beta_excitation) == 1:
        m_special = bra_beta_excitation
        p_special = ket_beta_excitation
        one_elec_mel += anti_commutator(pair)*one_elec_ints[m_special, p_special]
        # def kronecker(m, n):
        #   """takes 2 spin orbitals. returns 1 if they are equal in spin and 0 if they are not."""
        #   if m % 2 == n % 2:
        #       return 1
        #   else:
        #       return 0
        # # make custom kronecker function that outputs the fraction from \delta_{[m][n]} or \delta_{[n][p]}
        # def kronecker_fraction(m_spin, spin_intersection):
        #   """takes a difference spin orbital and a set of common spin orbitals for pair. returns fraction corresponding to relevant sum of delta functions."""
        #   delta_functions_sum = 0
        #   for orb in spin_intersection:
        #       delta_functions_sum += kronecker(m_spin, orb)
        #   return delta_functions_sum/len(spin_intersection)
        two_elec_xgrid_coloumb = np.ix_(range(6), range(6), total_intersection, total_intersection)
        two_elec_xgrid_exchange = np.ix_(range(6), total_intersection, total_intersection, range(6))
        two_electron_coulomb = two_elec_ints[two_elec_xgrid_coloumb]
        two_electron_exchange = two_elec_ints[two_elec_xgrid_exchange]
        two_elec_mel += anti_commutator(pair)*((np.einsum('ijkk->ij', two_electron_coulomb) - (4/25)*np.einsum('ijjk->ik', two_electron_exchange))[m_special,p_special])
    # if there are two differences between the determinants, where in the first determinant there are orbs m and n and in the ket determinant there are orbs p and q
    if number_of_differences == 2:
      # determine if we are dealing with the same or mixed spin
      mixed_spin = False
      # the first case is when the differences are only composed of electrons with the same spin
      if len(bra_beta_excitation) == 2 or len(bra_alpha_excitation) == 2:
        beta = False
        # initialize the indices to false
        m_special = False
        n_special = False
        p_special = False
        q_special = False
        if len(bra_beta_excitation) == 2:
          beta = True
          # set the indices
          m_special = bra_beta_excitation[0]
          m_spin = m_special * 2 + 1
          n_special = bra_beta_excitation[1]
          n_spin = n_special * 2 + 1
          p_special = ket_beta_excitation[0]
          p_spin = p_special * 2 + 1
          q_special = ket_beta_excitation[1]
          q_spin = q_special * 2 + 1
        elif len(bra_alpha_excitation) == 2:
          # set the indices if they are only alpha excitations
          m_special = bra_alpha_excitation[0]
          m_spin = m_special * 2
          n_special = bra_alpha_excitation[1]
          n_spin = n_special * 2
          p_special = ket_alpha_excitation[0]
          p_spin = p_special * 2
          q_special = ket_alpha_excitation[1]
          q_spin = q_special * 2
        # both terms are involved
        two_elec_mel += anti_commutator(pair)*(two_elec_ints[m_special,p_special,n_special,q_special] - two_elec_ints[m_special,q_special,n_special,p_special])
      # the second case is when the excitations are composed of electrons with different spins
      if len(bra_beta_excitation) == 1 and len(bra_alpha_excitation) == 1:
        mixed_spin = True
        # set the indices
        m_special = bra_alpha_excitation[0]
        m_spin = m_special * 2
        n_special = bra_beta_excitation[0]
        n_spin = n_special * 2 + 1
        p_special = ket_alpha_excitation[0]
        p_spin = p_special * 2
        q_special = ket_beta_excitation[0]
        q_spin = q_special * 2 + 1
        # only the first term survives
        two_elec_mel += anti_commutator(pair)*(two_elec_ints[m_special,p_special,n_special,q_special])
    return one_elec_mel + two_elec_mel
# integrals
one_elec_ints = np.load("h1e.npy")
two_elec_ints = np.load("h2e.npy")
# unit testing
# print(condon(({0,1,2,3,4,7}, {0,1,2,3,4,7}), (one_elec_ints, two_elec_ints)))
# print(condon(({0,1,2,3,4,5}, {0,1,2,3,6,7}), (one_elec_ints, two_elec_ints)))
#print(condon(({0,1,2,3,5,6}, {0,1,2,3,8,9}), (one_elec_ints, two_elec_ints)))
#print(condon(({0,1,2,3,5,6}, {0,1,2,3,8,7}), (one_elec_ints, two_elec_ints)))
# the hf case
assert math.isclose(condon((({0,1,2},{0,1,2}), ({0,1,2},{0,1,2})), (one_elec_ints, two_elec_ints)), -7.739373948970316, rel_tol=1e-9, abs_tol=1e-12)
# the 1 excitation case
assert math.isclose(condon((({0,1,2},{0,1,2}), ({0,1,2},{0,1,3})), (one_elec_ints, two_elec_ints)), 0, rel_tol=1e-9, abs_tol=1e-12)
# the 2 excitation case
# only beta difference
assert math.isclose(condon((({0,1,2},{0,1,2}), ({0,1,2},{0,3,4})), (one_elec_ints, two_elec_ints)), -0.04655311805628327, rel_tol=1e-9, abs_tol=1e-12)
# only alpha difference
assert math.isclose(condon((({2,4,5},{0,1,2}), ({0,1,2},{0,1,2})), (one_elec_ints, two_elec_ints)), 0.016177186667624063, rel_tol=1e-9, abs_tol=1e-12)
# mixed differences
assert math.isclose(condon((({1,2,4},{0,1,2}), ({1,2,5},{0,1,4})), (one_elec_ints, two_elec_ints)), 3.452099717193846e-16, rel_tol=1e-9, abs_tol=1e-12)


        


    