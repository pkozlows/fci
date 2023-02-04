from braket import braket
from cancellation import anti_commutator
import numpy as np
import input
# load in the intervals
one_elec_ints = np.load("h1e.npy")
two_elec_ints = np.load("h2e.npy")
def condon(pair, phase_factor, integrals): 
    '''takes tuple of two sets that describes how determinants vary,the associated
      face factor for pudding that determinant pair into maximum coincidence, and a
        tuple with the 1e and 2e integrals. returns matrix element'''
    sq = braket(pair)
    diff = sq.diff()
    one_elec_mel = 0
    two_elec_mel = 0
    # store first difference between determinants and convert the spin to spatial index, for later use to access integrals
    if len(diff[0]) >= 1:
        m = list(diff[0])[0] // 2
        p = list(diff[1])[0] // 2
        # store the 2nd difference
        if len(diff[0]) >= 2:
            q = list(diff[1])[1] // 2
            n = list(diff[0])[1] // 2
    # no differences
    if len(diff[0]) == 0:
        one_elec_mel += np.einsum('ii->', integrals[0])
        two_elec_mel += (0.5)*(np.einsum('iijj->', integrals[1])-np.einsum('ijji->', integrals[1]))
    # one difference
    if len(diff[0]) == 1:
        one_elec_mel += integrals[0][m,p]
        # m and p are the orbitals of difference
        two_elec_mel += np.einsum('ijkk->ij', integrals[1])[m,p]-np.einsum('ijjk->ik', integrals[1])[m,p]
    # 2 differences
    if len(diff[0]) == 2:
        # m,p and n,q are orb differences
        two_elec_mel += integrals[1][m,p,n,q]-integrals[1][m,q,n,p]
    return phase_factor*(one_elec_mel + two_elec_mel)
# testing
assert(condon(({0,1,2,3,4,5}, {0,1,2,3,4,5}), 1, (one_elec_ints, two_elec_ints)) == -6.277825297355697)
assert(condon(({0,1,3,5,7,9}, {0,1,3,6,7,9}), -1, (one_elec_ints, two_elec_ints)) == 1.3834419720915037e-16)
assert(condon(({0,1,2,5,7,11}, {0,1,3,5,7,9}), -1, (one_elec_ints, two_elec_ints)) == -2.539743589047294e-17)