import numpy as np
import itertools
from copy import deepcopy
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
  def no_difference(one_elec_special_union_ints, two_elec_special_union_ints):
    """Returns the matrix element if there are no differences between the two determinants."""
    one_elec = 0
    two_elec = 0
    # get the one electron matrix element
    one_elec += np.einsum('ii->', one_elec_special_union_ints)
    # get the two electron matrix element
    two_elec_mel += (1/2)*(np.einsum('iijj->', two_elec_special_union_ints) - (1/2)*np.einsum('ijji->',two_elec_special_union_ints))
    return one_elec + two_elec
  
    

assert(condon(({0,1,2,3,4,5}, {0,1,2,3,4,5}), (one_elec_ints, two_elec_ints)) == -7.739373948970316)
print(condon(({0,1,2,3,4,5}, {0,1,2,3,4,6}), (one_elec_ints, two_elec_ints)))
# assert(condon(({0,1,2,3,4,5}, {0,1,2,3,4,6}), (one_elec_ints, two_elec_ints)) == 0)


