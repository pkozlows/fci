# inputs
# number of electrons
electrons=6
# number of special orbitals
orbs=6
# all possible spin orbitals of this system in a list.
# and odd reflex a spin down orbital and
# and even reflex a spin up orbital 
spin_orbs=list()
for orb in range(1,(orbs * 2)+1):
     spin_orbs.append(orb)

possible_determinants=list()
determinant=set()
for electron in range(electrons):
    determinant.add(spin_orbs[electron])
# this is the ground state determinants
print(determinant)
     