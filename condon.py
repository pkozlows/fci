import numpy as np
import itertools
from copy import deepcopy
import cProfile

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
# test = braket((set([0,1,2,3,4,5]), set([0,1,2,3,4,6])))
# print(test.ket())
# print(test.bra())

def bubbleSort(arr):
    n = len(arr)
    # optimize code, so if the array is already sorted, it doesn't need
    # to go through the entire process
    swapped = False
    # Traverse through all array elements
    for i in range(n-1):
        # range(n) also work but outer loop will
        # repeat one time more than needed.
        # Last i elements are already in place
        for j in range(0, n-i-1):
 
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
         
        if not swapped:
            # if we haven't needed to make a single swap, we
            # can just exit the main loop.
            return

def anti_commutator(ops): 
    """takes a second quantization op list. simplifies the list, and returns either the face factor, or zero."""    
    # initialize the phase factor to unity
    phase_factor = 1
    n = len(ops)
    # create a set to check if the number of annihalation ops is equal to the number of creation ops
    spin_orbs = set()
    for op in ops:
      spin_orbs.add(op[0])
    for orb in spin_orbs:
      if ops.count((orb,0)) != ops.count((orb,1)):
        return 0      
    # Traverse through all array elements
    for i in range(n-1):
      # define the i as a creation or and ideation op
      annhltn = (ops[0][0],0)            
      crtn = (ops[0][0],1)
      # make sure no creation ops come before their partner
      if ops.index(annhltn) > ops.index(crtn):
        return 0
      for j in range(0, n-i-1):
        # traverse the array from 0 to n-i-1
        # if creation and annihilation indices next to each other, stop the while loop
        if ops[j] == annhltn and ops[j+1] == crtn:
          break
        # if the creation and angulation indices not next to each other, swap neighboring ops
        else:
          ops[j], ops[j + 1] = ops[j + 1], ops[j]
          # add the appropriate face factor
          phase_factor *= -1
    return phase_factor

# make unit tests for the anti commutator function
assert(anti_commutator([(0, 0), (0, 1)]) == 1)
assert(anti_commutator([(0, 0), (0,1), (0,0), (0, 1)]) == 1)
assert(anti_commutator([(0, 0), (0,1), (0,1), (0,0), (0,0), (0, 1)]) == 1)
assert(anti_commutator([(0, 0), (0,1), (0,1), (0,0), (0,0), (0, 1), (0,0), (0, 1)]) == 1)



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
      two_elec_mel += (1/2)*(3*np.einsum('ijji->', two_elec_ints[to_electron_grid]) - 2*np.einsum('iijj->',two_elec_ints[to_electron_grid]))     
    # store first difference between determinants and convert the spin to spatial index, for later use to access integrals
    if number_of_differences >= 1:
        m = list(diff[0])[0] // 2
        p = list(diff[1])[0] // 2
        # store the 2nd difference
        if number_of_differences >= 2:
            q = list(diff[1])[1] // 2
            n = list(diff[0])[1] // 2
    # one difference
    if number_of_differences == 1:
        # m and p are the orbitals of difference
        one_elec_mel += one_elec_ints[m,p]
        for i in spin_orbs:          
          two_elec_mel += anti_commutator(sq.combined())*(1/2)*(np.einsum('ijjk->ik', two_elec_ints[to_electron_grid])-np.einsum('ijik->jk', two_elec_ints[to_electron_grid])-np.einsum('ijkj->ik', two_elec_ints[to_electron_grid])+np.einsum('ijki->jk', two_elec_ints[to_electron_grid]))[m, p]
    # 2 differences
    # m,p and n,q are orb differences
        if number_of_differences == 2:
          two_elec_mel += two_elec_ints[m,p,n,q] - two_elec_ints[m,q,n,p]
    return (one_elec_mel + two_elec_mel)
# load in the intervals
# one_elec_ints = np.load("h1e.npy")
# two_elec_ints = np.load("h2e.npy")
# print(1)

# cProfile.run('condon(({0,1,2,3,4,5}, {0,1,2,3,4,5}), (one_elec_ints, two_elec_ints))', sort='cumtime')

