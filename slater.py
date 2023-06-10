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
    # determine the number of alpha differences between the two determinants
    alpha_differences = pair[0][0].difference(pair[1][0])
    # determine the number of beta differences between the two determinants
    beta_differences = pair[0][1].difference(pair[1][1])
    # determine the number of differences between the two determinants
    number_of_differences = len(alpha_differences) + len(beta_differences)    
    # make a global set that contains the combination of the individual spin strings of the determinant
    determinant_one = pair[0][0].union(pair[0][1])
    determinant_two = pair[1][0].union(pair[1][1])
    assert len(determinant_one) == len(determinant_two)
    assert len(determinant_one.difference(determinant_two)) == number_of_differences
    # take the intersection of the spin orbitals
    alpha_intersection = pair[0][0].intersection(pair[1][0])
    beta_intersection = pair[0][1].intersection(pair[1][1])
    # convert the set of spin orbs to special orbs
    alpha_special_intersection = [orb // 2 for orb in alpha_intersection]
    beta_special_intersection = [orb // 2 for orb in beta_intersection]
    # create a list of the total intersections in the pair
    spatial_intersection = alpha_special_intersection + beta_special_intersection
    # initialize the matrix element
    one_elec_mel = 0
    two_elec_mel = 0
    if number_of_differences == 0:
        # initialize meshes for converting spin orbs to special ints
        one_elec_special_union_xgrid = np.ix_(spatial_intersection, spatial_intersection)
        two_elec_special_union_xgrid=np.ix_(spatial_intersection, spatial_intersection, spatial_intersection, spatial_intersection)
        one_elec_special_union_ints = one_elec_ints[one_elec_special_union_xgrid]
        two_elec_special_union_ints = two_elec_ints[two_elec_special_union_xgrid]
        # if there is no difference between the determinants
        one_elec_mel += np.einsum('ii->', one_elec_special_union_ints)
        coulumb = np.einsum('iijj->', two_elec_special_union_ints) 
        exchange = np.einsum('ijji->', two_elec_special_union_ints)
        two_elec_mel += (1/2)*(coulumb - (1/2)*exchange)
    # if there is one difference, m and p, between the determinants
    if number_of_differences == 1:
      # save the spin and special indices of the first difference
        m_spin = differences[0][0]
        m_special = m_spin // 2
        p_spin = differences[1][0]
        p_special = p_spin // 2
        one_elec_mel += anti_commutator(pair)*one_elec_ints[m_special, p_special]
        def kronecker(m, n):
          """takes 2 spin orbitals. returns 1 if they are equal in spin and 0 if they are not."""
          if m % 2 == n % 2:
              return 1
          else:
              return 0
        # make custom kronecker function that outputs the fraction from \delta_{[m][n]} or \delta_{[n][p]}
        def kronecker_fraction(m_spin, spin_intersection):
          """takes a difference spin orbital and a set of common spin orbitals for pair. returns fraction corresponding to relevant sum of delta functions."""
          delta_functions_sum = 0
          for orb in spin_intersection:
              delta_functions_sum += kronecker(m_spin, orb)
          return delta_functions_sum/len(spin_intersection)
        two_elec_xgrid_coloumb = np.ix_(range(6), range(6), spatial_intersection, spatial_intersection)
        two_elec_xgrid_exchange = np.ix_(range(6), spatial_intersection, spatial_intersection, range(6))
        two_electron_coulomb = two_elec_ints[two_elec_xgrid_coloumb]
        two_electron_exchange = two_elec_ints[two_elec_xgrid_exchange]
        two_elec_mel += anti_commutator(pair)*((np.einsum('ijkk->ij', two_electron_coulomb) - (kronecker_fraction(m_spin, spin_intersection)*kronecker_fraction(p_spin, spin_intersection))*np.einsum('ijjk->ik', two_electron_exchange))[m_special,p_special])
    # if there are two differences between the determinants, where in the first determinant there are orbs m and n and in the second determinant there are orbs p and q
    if number_of_differences == 2:
      # save the spin and special indices of the first difference
      m_spin = differences[0][0]
      m_special = m_spin // 2
      p_spin = differences[1][0]
      p_special = p_spin // 2
      n_spin = differences[0][1]
      n_special = n_spin // 2
      q_spin = differences[1][1]
      q_special = q_spin // 2
      # the first case is when the {differences are only composed of electrons with the same spin
      if (m_spin % 2) == (n_spin % 2) and (p_spin % 2) == (q_spin % 2):
        assert(m_spin % 2 == p_spin % 2)
        # both terms are involved
        two_elec_mel += anti_commutator(pair)*(two_elec_ints[m_special,p_special,n_special,q_special] - two_elec_ints[m_special,q_special,n_special,p_special])
      # the second case is when the excitations are composed of electrons with different spins
      if (m_spin % 2) != (n_spin % 2) and (p_spin % 2) != (q_spin % 2):
        assert(n_spin % 2 == q_spin % 2)
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
assert math.isclose(condon((({0,2,4},{1,3,5}), ({0,2,4},{1,3,5})), (one_elec_ints, two_elec_ints)), -7.739373948970316, rel_tol=1e-9, abs_tol=1e-12)
assert(math.isclose(condon(({0,2,4},{1,3,5}, {0,2,4},{1,3,7}), (one_elec_ints, two_elec_ints)), 0, rel_tol=1e-9, abs_tol=1e-12))
# assert(math.isclose(condon(({0,1,2,3,4,5}, {0,1,2,3,5,6}), (one_elec_ints, two_elec_ints)), 0, rel_tol=1e-9, abs_tol=1e-12))


        


    