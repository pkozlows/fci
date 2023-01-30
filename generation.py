import itertools
import input
from random import randint
def gen_unique_pairs(electrons, orbs):
   """takes in the number of electrons and orbitals and the system.
   Returns a list of tuples of each unique pair of determinants."""
   # all possible spin orbitals of this system in a list.
   possible_determinants=list()
   for x in itertools.combinations(range(orbs*2),electrons):
      possible_determinants.append(set(x))
   # asserting whether the ground state determinant is in the object
   # assert({*range(input.elec_in_system)} in list(possible_determinants))
   # create all unique pairs of determinants
   pairs=itertools.combinations_with_replacement(list(possible_determinants),2)
   return pairs
e = randint(1, 6)
o = randint(1, 6)
print(list(gen_unique_pairs(e, o)))
print((e, o))