import numpy as np
import itertools
from copy import deepcopy
import cProfile
from pytest import approx
from anti_commutator import anti_commutator

# in puts
elec_in_system=6
orbs_in_system=6
one_elec_ints = np.load("h1e.npy")
two_elec_ints = np.load("h2e.npy")
def gen_unique_pairs(electrons, orbs):
   """takes in the number of electrons and orbitals and the system.
   Returns a list of tuples of each unique pair of determinants."""
   # all possible spin orbitals of this system in a list.
   possible_determinants=list()
   for x in itertools.combinations(range(orbs*2),electrons):
      possible_determinants.append(set(x))
   # create all unique pairs of determinants
   pairs=itertools.combinations_with_replacement(list(possible_determinants),2)
   return pairs

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
    '''takes tuple of two sets with the determinant pair and a
        tuple with the 1e and 2e integrals. returns matrix element'''
    one_elec_ints = integrals[0]
    two_elec_ints = integrals[1]
    sq = braket(pair)
    diff = sq.diff()
    number_of_differences = len(diff[0])
    one_elec_mel = 0
    two_elec_mel = 0
    spin_orbs = set()
    for operator in sq.combined():
      spin_orbs.add(operator[0])
    # initialize meshes for converting spin to special indices
    spacial_indices = [orb // 2 for orb in list(spin_orbs)]
    one_elec_xgrid = np.ix_(spacial_indices, spacial_indices)
    to_electron_grid=np.ix_(spacial_indices, spacial_indices, spacial_indices, spacial_indices)
    # create the relevant matrix for the determined pair
    # if there is no difference between two determinants
    if number_of_differences == 0:
      one_elec_mel += np.einsum('ii->', one_elec_ints[one_elec_xgrid])
      two_elec_mel += (1/2)*(np.einsum('iijj->', two_elec_ints[to_electron_grid]) - (1/2)*np.einsum('ijji->',two_elec_ints[to_electron_grid]))     
    # store first difference between determinants and convert the spin to spatial index, for later use to access integrals
    if number_of_differences >= 1:
        m_spin = list(diff[0])[0]
        m_special = m_spin // 2
        p_spin = list(diff[1])[0]
        p_special = p_spin // 2
        # store the 2nd difference
        if number_of_differences >= 2:
            n_spin = list(diff[0])[1]
            n_special = n_spin // 2
            q_spin = list(diff[1])[1]
            q_special = q_spin // 2
    # one difference
    if number_of_differences == 1:
        # m and p are the orbitals of difference
        # check if the spins are the same with a kronecker like implementation
        same_spin = 0
        if (m_spin % 2 == p_spin % 2):
          same_spin = 1
        one_elec_mel += anti_commutator(sq.combined())*(same_spin*one_elec_ints[m_special,p_special])
        # one einsum is conditional on the spins being the same
        two_elec_mel += anti_commutator(sq.combined())*(same_spin*np.einsum('ijkk->ij', two_elec_ints[to_electron_grid])[m_special,p_special] - (1/4)*np.einsum('ijjk->ik',two_elec_ints[to_electron_grid])[m_special,p_special])
    # 2 differences
    # m,p and n,q are orb differences
    if number_of_differences == 2:
      # check if the spins are the same with a kronecker like implementation
      same_spin_1 = 0
      if (m_spin % 2 == p_spin % 2):
        same_spin_1 = 1
      same_spin_2 = 0
      if (n_spin % 2 == q_spin % 2):
        same_spin_2 = 1
      same_spin_3 = 0
      if (m_spin % 2 == q_spin % 2):
        same_spin_3 = 1
      same_spin_4 = 0
      if (n_spin % 2 == p_spin % 2):
        same_spin_4 = 1      
      # einsums are conditional on the spins being the same
      two_elec_mel += (same_spin_1*same_spin_2)*two_elec_ints[m_special,p_special,n_special,q_special] - (same_spin_3*same_spin_4)*two_elec_ints[m_special,q_special,n_special,p_special]
    return (one_elec_mel + two_elec_mel)

assert(condon(({0,1,2,3,4,5}, {0,1,2,3,4,5}), (one_elec_ints, two_elec_ints)) == -7.739373948970316)
print(condon(({0,1,2,3,4,5}, {0,1,2,3,4,6}), (one_elec_ints, two_elec_ints)))
# assert(condon(({0,1,2,3,4,5}, {0,1,2,3,4,6}), (one_elec_ints, two_elec_ints)) == 0)


