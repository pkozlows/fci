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

def anti_commutator(pair): 
    """Takes a pair of determinants, which are inside of a tuple.  
    returns a simplified version of the list, composed of the second quantization ops
      of the bra and ket. this list is simplified in the sense that all possible cancellations are made."""    
    sq_pair = braket(pair) 
    op_list = sq_pair.combined()
    # initialize the phase factor to unity
    phase_factor = 1
    # change each shared orb to the appropriate second quantization operator for later cancellation
    for orb in sq_pair.pair[0].intersection(sq_pair.pair[1]):
        annihilation = (orb, 0)
        creation = (orb, 1)
        # continue the lope while there are still are ops to be canceled
        while annihilation and creation in op_list:
            for index, op in enumerate(op_list):
                # if the creation and annihilation indices are next to each other, remove them and stop the loop
                if op_list[index] == annihilation and op_list[index+1] == creation:
                    op_list.remove(annihilation)
                    op_list.remove(creation)
                # if the creation and angulation in disease are not next to each other, swap neighboring ops
                elif op_list[index] == annihilation:
                    current = deepcopy(op_list[index])
                    next = deepcopy(op_list[index+1])
                    op_list[index] = next
                    op_list[index+1] = current
                    # add the appropriate face factor
                    phase_factor *= -1
    return phase_factor

# load in the intervals
one_elec_ints = np.load("h1e.npy")
two_elec_ints = np.load("h2e.npy")
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
    # create the relevant matrix for the determined pair
    # if there is no difference between two determinants
    if number_of_differences == 0:
      spin_orbs = set()
      for operator in sq.combined():
        spin_orbs.add(operator[0])
      spacial_indices = [orb // 2 for orb in list(spin_orbs)]
      one_elec_xgrid = np.ix_(spacial_indices, spacial_indices)  
      one_elec_mel += np.einsum('ii->', one_elec_ints[one_elec_xgrid])
      for m in spin_orbs:
         for n in spin_orbs:
            # if the spin orbs have different spins
            if (m % 2 == 0 and n % 2 > 0) or (m % 2 > 0 and n % 2 == 0):
               two_elec_mel -= two_elec_ints[m//2,n//2,n//2,m//2]              
            # if the spin orbs have both spin up
            if m % 2 == 0 and n % 2 == 0:
               two_elec_mel += two_elec_ints[m//2,m//2,n//2,n//2] - two_elec_ints[m//2,n//2,n//2,m//2]
            # if the spin orbs have but spin down
            if m % 2 > 0 and n % 2 > 0:
              two_elec_mel += two_elec_ints[m//2,m//2,n//2,n//2] - two_elec_ints[m//2,n//2,n//2,m//2]
      # multiply the two electron made sucks lament by a half
      two_elec_mel *= 0.5
      # two_elec_xgrid = np.ix_(spacial_indices, spacial_indices, spacial_indices, spacial_indices) 
      # two_elec_mel += (0.5)*(np.einsum('iijj->', two_elec_ints[two_elec_xgrid])-np.einsum('ijji->', two_elec_ints[two_elec_xgrid]))      
      
    # if there is one difference between two determinants
    # if there are two differences bertone to determinants
    # if len(diff[0])  <= 2:
    # #   if len(diff[0]) = 1:
        
    #   if len(diff[0]) == 0:       
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
        one_elec_mel += one_elec_ints[m,p]
        # m and p are the orbitals of difference
        two_elec_mel += np.einsum('ijkk->ij', two_elec_ints)[m,p]-np.einsum('ijjk->ik', two_elec_ints)[m,p]
    # 2 differences
    #
        # m,p and n,q are orb differences
        # two_elec_mel += to_election_integrals[m,p,n,q]-[to_election_integrals](one_elec_mel)
    # print(two_elec_mel)
    return anti_commutator(pair)*(one_elec_mel + two_elec_mel)

