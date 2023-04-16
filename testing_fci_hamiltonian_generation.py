import unittest
import condon
import numpy as np
# load in the intervals
one_elec_ints = np.load("h1e.npy")
two_elec_ints = np.load("h2e.npy")
integrals = (one_elec_ints, two_elec_ints)
class testing_generation_of_fci_hamiltonian(unittest.TestCase):
    def test_braket(self):
      det = condon.braket(list(condon.gen_unique_pairs(condon.elec_in_system, condon.orbs_in_system)[46])
      self.assertEqual(det.ket(), [(0, 1), (1, 1), (2, 1), (4, 1), (9, 1), (10, 1)])
      self.assertEqual(det.combined(), [(5, 0), (4, 0), (3, 0), (2, 0), (1, 0), (0, 0), (0, 1), (1, 1), (2, 1), (4, 1), (9, 1), (10, 1)])
      self.assertEqual(det.diff(), ({3, 5}, {9, 10}))
    def test_antiCommutator(self):
      self.assertEqual(condon.anti_commutator(({0,2,4,6,8,10}, {0,3,5,6,9,10})), 1)
    def test_condon(self):
      self.assertAlmostEqual(condon.condon(({0,1,2,3,4,5}, {0,1,2,3,4,5}), integrals), -7.739373948970316)
      self.assertAlmostEqual(condon.condon(({0,1,3,5,7,9}, {0,1,3,6,7,9}), integrals), 1.3834419720915037e-16)
      self.assertAlmostEqual(condon.condon(({0,1,2,5,7,11}, {0,1,3,5,7,9}),integrals), -2.539743589047294e-17)
if __name__ == '__main__':
    unittest.main()