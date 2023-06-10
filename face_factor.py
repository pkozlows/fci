def anti_commutator(pair):
  """
  Takes tuple of a pair of determinants that contains a tuple of two spin strings sets, which are alpha and beta, respectively, and returns the face factor.can only deal with cases where there is one or two differences between determinants.
    Returns the face factor between the pair.
    
    Args:
        pair (tuple): Tuple of determinants ((alpha, beta), (alpha, beta)).
                      Each determinant is a tuple of two spin strings.
    Returns:
        int: face factor between the pair."""
  # divine the spin strings for the alpha and the beta orbitals in the bra and ket
  bra_alpha = sorted(pair[0][0])
  bra_beta = sorted(pair[0][1])
  ket_alpha = sorted(pair[1][0])
  ket_beta = sorted(pair[1][1] )
  # determine the alpha differences between the two determinants
  bra_alpha_difference = sorted(bra_alpha.difference(ket_alpha))
  ket_alpha_difference = sorted(ket_alpha.difference(bra_alpha))
  assert(len(bra_alpha_difference) == len(ket_alpha_difference))
  # determine the beta differences between the two determinants
  bra_beta_difference = sorted(bra_beta.difference(ket_beta))
  ket_beta_difference = sorted(ket_beta.difference(bra_beta))
  assert(len(bra_beta_difference) == len(ket_beta_difference))
  # first treat the case with only one difference of the same spin
  if len(bra_alpha_difference) == 1 or len(bra_beta_difference) == 1:
    # determine whether we are dealing with a difference in alpha or beta
    if len(bra_alpha_difference) == 1:
      bra_unique_orbs = bra_alpha_difference
      ket_unique_orbs = ket_alpha_difference
    elif len(bra_beta_difference) == 1:
      bra_unique_orbs = bra_beta_difference
      ket_unique_orbs = ket_beta_difference
  
  # treat the case with 2 differences of mixed spin
  if len(bra_unique_orbs) == 2 and bra_unique_orbs[0] % 2 != bra_unique_orbs[1] % 2:
    # check whether spins are indeed mixed
    assert(ket_unique_orbs[0] % 2 != ket_unique_orbs[1] % 2)
    # reorder the differences to first go from alpha and then beater
    ket_spin = [bra_unique_orbs[0] % 2, bra_unique_orbs[1] % 2]
    bra_spin = [ket_unique_orbs[0] % 2, ket_unique_orbs[1] % 2]
    # reorder the differences if they are not an canonical order
    if ket_spin == [1, 0] and ket_unique_orbs != sorted(ket_unique_orbs):
      ket_unique_orbs.reverse()
    if bra_spin == [1, 0] and bra_unique_orbs != sorted(bra_unique_orbs):
      bra_unique_orbs.reverse()   
  # initialize the sorted lists
  bra = sorted(pair[0])
  ket = sorted(pair[1])
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
# test cases for two differences
assert(anti_commutator(({0,1,2,3,5,6}, {0,1,2,3,8,9})) == 1)
# assert(anti_commutator(({0,1,2,3,5,6}, {0,1,2,3,8,7})) == -1)
assert(anti_commutator(({0,1,2,3,4,5}, {0,1,2,3,6,7})) == 1)
assert(anti_commutator(({0,1,2,3,4,5}, {0,1,2,4,6,8})) == -1)
# test cases for single difference
assert(anti_commutator(({0,1,2,7,8,9}, {0,1,2,7,8,11})) == 1)
assert(anti_commutator(({0,1,2,3,4,5}, {0,1,2,3,5,6})) == -1)
assert(anti_commutator(({0,1,2,3,4,5}, {0,1,2,3,5,10})) == -1)