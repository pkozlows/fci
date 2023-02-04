import numpy as np
import input
from generation import gen_unique_pairs
from braket import braket
from cancellation import anti_commutator
from condon import condon
# load in the intervals
one_elec_ints = np.load("h1e.npy")
two_elec_ints = np.load("h2e.npy")
# generate the unique pairs for the system
for determinant_pair in gen_unique_pairs(input.elec_in_system, input.orbs_in_system):
   mel = condon(determinant_pair, anti_commutator(determinant_pair), (one_elec_ints, two_elec_ints))
   print(mel)

