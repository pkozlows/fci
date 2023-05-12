# def anti_commutator(operator_list):
#   """takes a second quantization op list. simplifies the list, and returns the face factor. can only deal with cases where there is one or two differences between determinants."""
#   # initialize the phase factor to unity
#   phase_factor = 1
#   n = len(operator_list)
#   # a loop for each sort operation
#   for i in range(n):
#     swapped = False
#     # if the first annihalation operator does not have creation partner in the list, remove it
#     if ((operator_list[0][1] == 0) and (operator_list[0][0], 1) not in operator_list):
#        del operator_list[i]
#     # update the length of the list
#     k = len(operator_list)
#     # traverse the list of operators
#     for j in range(k-i-1):
#       if operator_list[j][0] > operator_list[j+1][0]:
#         # Swap adjacent elements
#         operator_list[j], operator_list[j+1] = operator_list[j+1], operator_list[j]
#         # add the appropriate face factor
#         phase_factor *= -1
#         swapped = True
#     # If no swaps were made during the pass, the list is already sorted
#     if not swapped:
#         break
#   return phase_factor
def anti_commutator(pair):
  """takes a pair of determinat sets, and returns the face factor. can only deal with cases where there is one or two differences between determinants."""
  # make set for what orbs they differ in
  bra_unique_orbs = list(pair[0].difference(pair[1]))
  ket_unique_orbs = list(pair[1].difference(pair[0]))
  # initialize the lists to be sorted
  bra = list(pair[0])
  ket = list(pair[1])
  # make sure they had the same length
  assert(len(bra) == len(ket))
  def bubble_sort(determinant, unique_orbs):
    """takes two sorted list, one that has a whole determinant and the other that has the unique orbs of the determinant. returns the number of swaps needed to sort the list."""
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
assert(anti_commutator(({0,1,2,3,4,5}, {0,1,2,3,4,6})) == 1)
assert(anti_commutator(({0,1,2,3,4,5}, {0,1,2,3,5,6})) == -1)
# test cases for two differences
assert(anti_commutator(({0,1,2,3,4,5}, {0,1,2,3,6,7})) == 1)
assert(anti_commutator(({0,1,2,3,4,5}, {0,1,2,4,6,7})) == -1)