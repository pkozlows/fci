import numpy as np
import condon
import itertools
# load in the intervals
one_elec_ints = np.load("h1e.npy")
two_elec_ints = np.load("h2e.npy")
integrals = (one_elec_ints, two_elec_ints)
# implementing a determinant bases ordered based of value of diagonal fci mel
def determinant_diagonal(determinant):
   return condon.condon((determinant, determinant), (one_elec_ints, two_elec_ints))
# initialize matrix with 0s for later modification when get the Condon values
poss_dets=list()
for x in itertools.combinations(range(condon.orbs_in_system*2),condon.elec_in_system):
   poss_dets.append(set(x))
poss_dets.sort(key = determinant_diagonal)
mat = np.zeros(shape = (len(poss_dets),len(poss_dets)))
# fill in the matrix with the Condon values
for det_pair in condon.gen_unique_pairs(condon.elec_in_system, condon.orbs_in_system):
   mat[poss_dets.index(det_pair[0])][poss_dets.index(det_pair[1])] = condon.condon(det_pair, integrals)
# Find the eigenvalues and eigenvectors of A
eigenvalues, eigenvectors = np.linalg.eig(mat)
# Create a diagonal matrix D with the eigenvalues
D = np.diag(eigenvalues)
print(D)
