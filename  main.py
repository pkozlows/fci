import itertools
elec_in_system=6
orbs_in_system=6
def generate_determinants(electrons, orbs):
   possible_determinants=list() 
   # all possible spin orbitals of this system in a list.
   for x in itertools.combinations(range(orbs*2),electrons):
      possible_determinants.append(set(x))
   return iter(possible_determinants)
# asserting whether the ground state determinant is in the object
assert({*range(elec_in_system)} in generate_determinants(elec_in_system,orbs_in_system))