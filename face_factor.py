def anti_commutator(pair):
  """takes tuple of a pair of determine, which is each represented by two tuples, which contains the offa and beta orbs, respectively. the format for this one is: ((bra_alpha, bra_beta), (ket_alpha, ket_beta)).returns an integer that represents the face factor."""
  # initialize the sorted lists
  bra_alpha = sorted(pair[0][0])
  bra_beta = sorted(pair[0][1])
  ket_alpha = sorted(pair[1][0])
  ket_beta = sorted(pair[1][1])
  # determine the alpha differences between the two determinants
  alpha_excitation = (sorted(pair[0][0].difference(pair[1][0])), sorted(pair[1][0].difference(pair[0][0])))
  assert(len(alpha_excitation[0]) == len(alpha_excitation[1]))
  # determine the beta differences between the two determinants
  beta_excitation = (sorted(pair[0][1].difference(pair[1][1])), sorted(pair[1][1].difference(pair[0][1])))
  assert(len(beta_excitation[0]) == len(beta_excitation[1]))
  # determine the number of differences between the two determinants
  assert len(alpha_excitation[0]) + len(beta_excitation[0]) == len(alpha_excitation[1]) + len(beta_excitation[1])
  number_of_differences = len(alpha_excitation[0]) + len(beta_excitation[0])   
  # treat the number of differences the same, counting the number of swabs anyways for each spin string
  def bubble_sort(string, excitation):
    """takes two unsorted lists, one that has a whole determinant and the other that has the unique orbs of the determinant. returns the number of swaps needed to sort the list."""
    swaps = 0
    for i, orb in enumerate(excitation):
      swaps += string.index(orb) - i
    return swaps
    # sort the lists and get the number of swaps
  bra_alpha_swaps = bubble_sort(bra_alpha, alpha_excitation[0])
  bra_beta_swaps = bubble_sort(bra_beta, beta_excitation[0])
  ket_alpha_swaps = bubble_sort(ket_alpha, alpha_excitation[1])
  ket_beta_swaps = bubble_sort(ket_beta, beta_excitation[1])

  # return the face factor
  return (-1)**(bra_alpha_swaps + bra_beta_swaps + ket_alpha_swaps + ket_beta_swaps)
# test cases for single difference
assert(anti_commutator((({0,2,8}, {1,7,9}), ({0,2,8}, {1,7,11}))) == 1)
assert(anti_commutator((({0,2,4}, {1,3,5}),   g({0,2,6}, {1,3,5}))) == -1)
assert(anti_commutator(({0,1,2,3,4,5}, {0,1,2,3,5,6})) == -1)
assert(anti_commutator(({0,1,2,3,4,5}, {0,1,2,3,5,10})) == -1)
# test cases for two differences
assert(anti_commutator(({0,1,2,3,5,6}, {0,1,2,3,8,9})) == 1)
# assert(anti_commutator(({0,1,2,3,5,6}, {0,1,2,3,8,7})) == -1)
assert(anti_commutator(({0,1,2,3,4,5}, {0,1,2,3,6,7})) == 1)
assert(anti_commutator(({0,1,2,3,4,5}, {0,1,2,4,6,8})) == -1)