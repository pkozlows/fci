def anti_commutator(pair):
  """takes tuple of a pair of determine, which is each represented by two tuples, which contains the offa and beta orbs, respectively. the format for this one is: ((bra_alpha, bra_beta), (ket_alpha, ket_beta)).returns an integer that represents the face factor."""
  # initialize the sorted lists
  bra_total_alpha = sorted(pair[0][0])
  bra_total_beta = sorted(pair[0][1])
  ket_total_alpha = sorted(pair[1][0])
  ket_total_beta = sorted(pair[1][1])
  # determine the alpha differences between the two determinants
  bra_alpha_difference = sorted(pair[0][0].difference(pair[1][0]))
  ket_alpha_difference = sorted(pair[1][0].difference(pair[0][0]))
  # determine the beta differences between the two determinants
  bra_beta_difference = sorted(pair[0][1].difference(pair[1][1]))
  ket_beta_difference = sorted(pair[1][1].difference(pair[0][1]))
  # determine the number of differences between the two determinants
  assert len(bra_alpha_difference) + len(bra_beta_difference) == len(ket_alpha_difference) + len(ket_beta_difference)
  number_of_differences = len(bra_alpha_difference) + len(bra_beta_difference)   
  differences = ((bra_alpha_difference, bra_beta_difference), (ket_alpha_difference, ket_beta_difference)) 
  # divine the spin strings for the alpha and the beta orbitals in the bra and ket
  bra_unique_alpha = differences[0][0]
  bra_unique_beta = differences[0][1]
  ket_unique_alpha = differences[1][0]
  ket_unique_beta = differences[1][1]
  assert(len(bra_unique_alpha) == len(ket_unique_alpha))
  assert(len(bra_unique_beta) == len(ket_unique_beta))
  # treat the number of differences the same, counting the number of swabs and noways for each spin string
  def bubble_sort(string, unique_orbs):
    """takes two unsorted lists, one that has a whole determinant and the other that has the unique orbs of the determinant. returns the number of swaps needed to sort the list."""
    swaps = 0
    for i, orb in enumerate(unique_orbs):
      swaps += string.index(orb) - i
    return swaps
    # sort the lists and get the number of swaps
  bra_alpha_swaps = bubble_sort(bra_total_alpha, bra_unique_alpha)
  bra_beta_swaps = bubble_sort(bra_total_beta, bra_unique_beta)
  total_bra_swaps = bra_alpha_swaps + bra_beta_swaps
  ket_alpha_swaps = bubble_sort(ket_total_alpha, ket_unique_alpha)
  ket_beta_swaps = bubble_sort(ket_total_beta, ket_unique_beta)
  total_ket_swaps = ket_alpha_swaps + ket_beta_swaps
  # return the face factor
  return (-1)**(total_bra_swaps + total_ket_swaps)
# test cases for single difference
assert(anti_commutator((({0,2,8}, {1,7,9}), ({0,2,8}, {1,7,11}))) == 1)
assert(anti_commutator((({0,2,4}, {1,3,5}), ({0,2,6}, {1,3,5}))) == -1)
assert(anti_commutator(({0,1,2,3,4,5}, {0,1,2,3,5,6})) == -1)
assert(anti_commutator(({0,1,2,3,4,5}, {0,1,2,3,5,10})) == -1)
# test cases for two differences
assert(anti_commutator(({0,1,2,3,5,6}, {0,1,2,3,8,9})) == 1)
# assert(anti_commutator(({0,1,2,3,5,6}, {0,1,2,3,8,7})) == -1)
assert(anti_commutator(({0,1,2,3,4,5}, {0,1,2,3,6,7})) == 1)
assert(anti_commutator(({0,1,2,3,4,5}, {0,1,2,4,6,8})) == -1)