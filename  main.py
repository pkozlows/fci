import itertools
# in puts
elec_in_system=6
orbs_in_system=6
def gen_unique_pairs(electrons, orbs):
   """takes in the number of electrons and orbitals and the system.
   Returns a list of tuples of each unique pair of determinants."""
   # all possible spin orbitals of this system in a list.
   possible_determinants=list()
   for determinant in itertools.combinations(range(orbs*2),electrons):
      possible_determinants.append(set(determinant))
   # asserting whether the ground state determinant is in the object
   assert({*range(elec_in_system)} in list(possible_determinants))
   # create all unique pairs of determinants
   pairs=itertools.combinations_with_replacement(list(possible_determinants),2)
   return pairs
print(compare_determinants(list(gen_unique_pairs(elec_in_system,orbs_in_system))[8]))
print(list(gen_unique_pairs(elec_in_system,orbs_in_system))[8])