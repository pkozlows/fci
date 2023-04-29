from condon import braket
import numpy as np
import cProfile
from pytest import approx
from anti_commutator import anti_commutator

# integrals
one_elec_ints = np.load("h1e.npy")
two_elec_ints = np.load("h2e.npy")

def condon(pair, integrals):
    """Takes tuple that is two sets which constitutes a pair of determinants and also a tuple with one and two electron integrals. returns the matrix element Between the pair."""
    one_elec_ints = integrals[0]
    two_elec_ints = integrals[1]
    sq = braket(pair)
    # get the difference between the two determinants and its length
    diff = sq.diff()
    number_of_differences = len(diff[0])
    # initializes the mels
    one_elec_mel = 0
    two_elec_mel = 0
    # create a set which contains the spin orbs in the pair
    orbs_in_pair = sq.pair[0].union(sq.pair[1])
    # initialize meshes for converting spin orbs to special ints
    spacial_indices = [orb // 2 for orb in orbs_in_pair]
    one_elec_xgrid = np.ix_(spacial_indices, spacial_indices)
    to_electron_grid=np.ix_(spacial_indices, spacial_indices, spacial_indices, spacial_indices)
    # if there is no difference between the determinants
    if number_of_differences == 0:
      one_elec_mel += np.einsum('ii->', one_elec_ints[one_elec_xgrid])
      two_elec_mel += (1/2)*(np.einsum('iijj->', two_elec_ints[to_electron_grid]) - (1/2)*np.einsum('ijji->',two_elec_ints[to_electron_grid]))
    # save the spin orb differences between the determinants and then convert them into special indices for later use the access ints
    if number_of_differences >= 1:
        m_spin = list(diff[0])[0]
        m_special = m_spin // 2
        p_spin = list(diff[1])[0]
        p_special = p_spin // 2
        if number_of_differences >= 2:
            n_spin = list(diff[0])[1]
            n_special = n_spin // 2
            q_spin = list(diff[1])[1]
            q_special = q_spin // 2
    def kronecker(m, n):
        """takes 2 spin orbitals. returns 1 if they are equal in spin and 0 if they are not."""
        if m % 2 == n % 2:
            return 1
        else:
            return 0
    # if there is one difference, m and p, between the determinants
    if number_of_differences == 1:
        one_elec_mel += anti_commutator(sq.combined())*kronecker(m_spin, p_spin)*one_elec_ints[one_elec_xgrid][m_special, p_special]
        two_elec_mel += (1/2)*kronecker(m_spin, p_spin)*(np.einsum('ijij->', two_elec_ints[to_electron_grid]) - (1/2)*np.einsum('ijji->',two_elec_ints[to_electron_grid]))

        


    