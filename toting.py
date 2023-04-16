import numpy as np
import condon
# load in the intervals
one_elec_ints = np.load("h1e.npy")
two_elec_ints = np.load("h2e.npy")
integrals = (one_elec_ints, two_elec_ints)
# dets = condon.gen_unique_pairs(condon.elec_in_system, condon.orbs_in_system)
# print(list(dets)[0])
# # party fog grand state
print(condon.anti_commutator([(5, 0), (4, 0), (3, 0), (2, 0), (1, 0), (0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]))
# print(condon.condon(({0,1,2,3,4,5}, {0,1,2,3,4,5}), integrals))
# hf = condon.braket(({0,1,2,3,4,5}, {0,1,2,3,4,5}))

# # print(hf.combined())
# print(condon.anti_commutator(hf.combined()))
# # to permanents with one difference
# condon.condon(({0,1,2,3,4,5}, {0,1,2,3,4,5}), 1, (condon.one_elec_ints, condon.two_elec_ints))
# # determinants with two differences
# condon.condon(({0,1,2,3,4,5}, {0,1,2,3,4,5}), 1, (condon.one_elec_ints, condon.two_elec_ints))    