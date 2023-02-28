# from braket import braket
# from cancellation import anti_commutator
import numpy as np
import itertools
from copy import deepcopy

# in puts
elec_in_system=6
orbs_in_system=6

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
  # define a ket in second quantizationo
  def ket(self):
    """. Converts the orbs of the second determined
      into a list of appropriate ordered creation operators."""
    ket_sq = list()
    determinant = deepcopy(self.pair[1])
    for i in range(len(determinant)):
       #  Taking out the smallest or brutal of the set ensures the correct ordering of the creations operators
       minimum = min(determinant)
       # creation operators are indicated by a 1
       ket_sq.append((minimum, 1))
       determinant.discard(minimum)
    return ket_sq
  # define a bra in second quantization
  def bra(self):
    """Converts the orbs of the first determined into a list of
     appropriate ordered annihilation operators."""
    bra_sq = list()
    determinant = deepcopy(self.pair[0])
    for i in range(len(determinant)):
       # this ensures the correct ordering of the annihilation operators
       maximum = max(determinant)
       # annihilation operators are indicated by a 0
       bra_sq.append((maximum, 0))
       determinant.discard(maximum)
    return bra_sq
  def diff(self):
   """returns tuple of  two sets that describe the differences that described the difference between pair.""" 
   return (self.pair[0].difference(self.pair[1]), self.pair[1].difference(self.pair[0]))
  def combined(self):
    """Returns unsimplified list of the creation and angulation ops Form each determinant."""
    _braket = self.bra() + self.ket()
    return _braket


def anti_commutator(ops): 
    """takes a second quantization op list. simplifies the list, and returns either the face factor, or zero."""    
    # initialize the phase factor to unity
    operator_list = list(ops)
    phase_factor = 1
    # for each op in the inputted list
    for op in ops:
      # the angulation ops should be before the creation partner so just need to iterate over them
      annhltn = op 
      crtn = (op[0], 1)
      reverse = deepcopy(operator_list)
      reverse.reverse()
      # check if # of angulation ops = # of creation ops and if all angulation ops before their creation partner
      if operator_list.count(annhltn) == operator_list.count(crtn) and reverse.index(crtn)<reverse.index(annhltn):
        # continue the lope while there are still are ops to be canceled
        while annhltn and crtn in operator_list:
            for index, op in enumerate(operator_list):
                # if creation and annihilation indices next to each other, remove them and stop the while loop
                if operator_list[index] == annhltn and operator_list[index+1] == crtn:
                    operator_list.remove(annhltn)
                    operator_list.remove(crtn)
                # if the creation and angulation indices not next to each other, swap neighboring ops
                elif operator_list[index] == annhltn:
                    current = deepcopy(operator_list[index])
                    next = deepcopy(operator_list[index+1])
                    operator_list[index] = next
                    operator_list[index+1] = current
                    # add the appropriate face factor
                    phase_factor *= -1
      else:
         return 0
    # if the op list is empty and zero has not been returned yet comma then return the pace factor
    if operator_list == list():
        return phase_factor

def condon(pair, integrals): 
    '''takes tuple of two sets with the determinant pair and a
        tuple with the 1e and 2e integrals. returns matrix element'''
    one_elec_ints = integrals[0]
    two_elec_ints = integrals[1]
    sq = braket(pair)
    one_elec_mel = 0
    two_elec_mel = 0
    for i in range(orbs_in_system*2):
      for j in range(orbs_in_system*2):
        op_list = sq.bra() + [(i,1), (j,0)] + sq.ket()
        one_elec_mel += anti_commutator(op_list) * one_elec_ints[i//2,j//2]
    for i in range(orbs_in_system*2):
      for j in range(orbs_in_system*2):
        for k in range(orbs_in_system*2):
          for l in range(orbs_in_system*2):
            op_list = sq.bra() + [(i,1), (j,1), (k,0), (l,0)] + sq.ket()
            two_elec_mel += (1/2) * anti_commutator(op_list) * two_elec_ints[i//2,k//2,j//2,l//2]
    return (one_elec_mel + two_elec_mel)

