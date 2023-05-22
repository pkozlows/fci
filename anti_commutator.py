def anti_commutator(pair):
  """takes a pair of determinat sets, and returns the face factor. can only deal with cases where there is one or two differences between determinants."""
  # make list for what orbs they differ in
  bra_unique_orbs = list(pair[0].difference(pair[1]))
  ket_unique_orbs = list(pair[1].difference(pair[0]))
  # initialize the lists to be sorted
  bra = list(pair[0])
  ket = list(pair[1])
  # make sure they had the same length
  assert(len(bra) == len(ket))
  def bubble_sort(determinant, unique_orbs):
    """takes two unsorted lists, one that has a whole determinant and the other that has the unique orbs of the determinant. returns the number of swaps needed to sort the list."""
    swaps = 0
    for i, orb in enumerate(unique_orbs):
      swaps += determinant.index(orb) - i
    return swaps
  # sort the lists and get the number of swaps
  bra_swaps = bubble_sort(bra, bra_unique_orbs)
  ket_swaps = bubble_sort(ket, ket_unique_orbs)
  # return the face factor
  return (-1)**(bra_swaps + ket_swaps)
# test cases for single difference
# assert(anti_commutator(({0,1,2,3,4,5}, {0,1,2,3,5,6})) == -1)
# test cases for two differences
# assert(anti_commutator(({0,1,2,3,4,5}, {0,1,2,3,6,7})) == 1)
# assert(anti_commutator(({0,1,2,3,4,5}, {0,1,2,4,6,8})) == -1)