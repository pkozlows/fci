import numpy as np
import cProfile
from pytest import approx
from anti_commutator import anti_commutator

# integrals
one_elec_ints = np.load("h1e.npy")
two_elec_ints = np.load("h2e.npy")

class braket:
  """This object takes a   determinant pair.
    it contains many methods that allow for the determinant pair to be
      dulled with in second quantization."""
  def __init__(self, pair):
    self.pair = pair
  def diff(self):
    """returns tuple of  two sets that describe the differences that described the difference between pair.""" 
    return (self.pair[0].difference(self.pair[1]), self.pair[1].difference(self.pair[0]))
  # define a ket in second quantizationo
  def ket(self):
    """. Converts the orbs of the second determined
      into a list of appropriate ordered creation operators."""
    ket_sq = list((orbital, 1) for orbital in self.pair[1])
    return ket_sq
  # define a bra in second quantization
  def bra(self):
    """Converts the orbs of the first determined into a list of
     appropriate ordered annihilation operators."""
    bra_sq = list((orbital, 0) for orbital in self.pair[0])
    bra_sq.reverse()
    return bra_sq
  def diff(self):
   """returns tuple of  two sets that describe the differences that described the difference between pair.""" 
   return (self.pair[0].difference(self.pair[1]), self.pair[1].difference(self.pair[0]))
  def combined(self):
    """Returns unsimplified list of the creation and angulation ops Form each determinant."""
    _braket = self.bra() + self.ket()
    return _braket

def condon(pair, integrals):
    """Takes tuple that is two sets which constitutes a pair of determinants and also a tuple with one and two electron integrals. returns the matrix element Between the pair."""
    one_elec_ints = integrals[0]
    two_elec_ints = integrals[1]
    sq = braket(pair)
    # get the difference between the two determinants and its length
    diff = sq.diff()
    number_of_differences = len(diff[0])
    # initializes the mels
    one_elec_mel = 0
    two_elec_mel = 0
    # create a set which contains the union of spin orbs in the pair
    spin_union = sq.pair[0].union(sq.pair[1])
    # create a set which contains the intersection of spin orbs in the pair
    spin_intersection = sq.pair[0].intersection(sq.pair[1])
    # convert the set of spin orbs to special orbs
    spacial_union = [orb // 2 for orb in spin_union]
    spacial_intersection = [orb // 2 for orb in spin_intersection]
    # initialize meshes for converting spin orbs to special ints
    one_elec_special_union_xgrid = np.ix_(spacial_union, spacial_union)
    to_elec_special_union_xgrid=np.ix_(spacial_union, spacial_union, spacial_union, spacial_union)
    one_elec_special_union_ints = one_elec_ints[one_elec_special_union_xgrid]
    two_elec_special_union_ints = two_elec_ints[to_elec_special_union_xgrid]
    # if there is no difference between the determinants
    if number_of_differences == 0:
      one_elec_mel += np.einsum('ii->', one_elec_special_union_ints)
      two_elec_mel += (1/2)*(np.einsum('iijj->', two_elec_special_union_ints) - (1/2)*np.einsum('ijji->',two_elec_special_union_ints))
    # save the spin orb differences between the determinants and then convert them into special indices for later use the access ints
    if number_of_differences >= 1:
        m_spin = list(diff[0])[0]
        m_special = m_spin // 2
        p_spin = list(diff[1])[0]
        p_special = p_spin // 2
        if number_of_differences >= 2:
            n_spin = list(diff[0])[1]
            n_special = n_spin // 2
            q_spin = list(diff[1])[1]
            q_special = q_spin // 2
    def kronecker(m, n):
        """takes 2 spin orbitals. returns 1 if they are equal in spin and 0 if they are not."""
        if m % 2 == n % 2:
            return 1
        else:
            return 0
    # if there is one difference, m and p, between the determinants
    if number_of_differences == 1:
        one_elec_mel += anti_commutator(pair)*kronecker(m_spin, p_spin)*one_elec_special_union_ints[m_special, p_special]
        # make custom kronecker function that outputs the fraction from \delta_{[m][n]} or \delta_{[n][p]}
        def kronecker_fraction(m_spin, spin_intersection):
          """takes a difference spin orbital and a set of common spin orbitals for pair. returns fraction corresponding to 1/sum of delta functions."""
          delta_functions_sum = 0
          for orb in spin_intersection:
              delta_functions_sum += kronecker(m_spin, orb)
          if delta_functions_sum == 0:
              return 0
          else:
              return 1/delta_functions_sum
        two_elec_xgrid_coloumb = np.ix_(spacial_union, spacial_union, spacial_intersection, spacial_intersection)
        two_elec_xgrid_exchange = np.ix_(spacial_union, spacial_intersection, spacial_intersection, spacial_union)
        two_electron_coulomb = two_elec_ints[two_elec_xgrid_coloumb]
        two_electron_exchange = two_elec_ints[two_elec_xgrid_exchange]
        two_elec_mel += anti_commutator(pair)*kronecker(m_spin, p_spin)*(np.einsum('ijkk->ij', two_electron_coulomb) - np.einsum('ijjk->ik', two_electron_exchange))[m_special, p_special]
    # if there are two differences, m p and n q, between the determinants
    if number_of_differences == 2:
      two_elec_mel += anti_commutator(pair)*(kronecker(m_spin, p_spin)*kronecker(n_spin, q_spin)*two_elec_special_union_ints[m_special,p_special,n_special,q_special] - kronecker(m_spin, q_spin)*kronecker(n_spin, q_spin)*two_elec_special_union_ints[m_special,q_special,n_special,p_special])
    return one_elec_mel + two_elec_mel
# unit testing
assert(condon(({0,1,2,3,4,5}, {0,1,2,3,4,5}), (one_elec_ints, two_elec_ints)) == -7.739373948970316)
assert(condon(({0,1,2,3,4,5}, {0,1,2,3,4,6}), (one_elec_ints, two_elec_ints)) == 0)
# print(condon(({0,1,2,3,4,5}, {0,1,2,3,4,7}), (one_elec_ints, two_elec_ints)))
# print(condon(({0,1,2,3,4,5}, {0,1,2,3,5,6}), (one_elec_ints, two_elec_ints)))

        


    