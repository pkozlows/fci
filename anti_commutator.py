def anti_commutator(operator_list):
  """takes a second quantization op list. simplifies the list, and returns the face factor. can only deal with cases where there is one or two differences between determinants."""
  # initialize the phase factor to unity
  phase_factor = 1
  n = len(operator_list)
  # a loop for each sort operation
  for i in range(n):
    swapped = False
    # if the first annihalation operator does not have creation partner in the list, remove it
    if ((operator_list[0][1] == 0) and (operator_list[0][0], 1) not in operator_list):
       del operator_list[i]
    # update the length of the list
    k = len(operator_list)
    # traverse the list of operators
    for j in range(k-i-1):
      if operator_list[j][0] > operator_list[j+1][0]:
        # Swap adjacent elements
        operator_list[j], operator_list[j+1] = operator_list[j+1], operator_list[j]
        # add the appropriate face factor
        phase_factor *= -1
        swapped = True
    # If no swaps were made during the pass, the list is already sorted
    if not swapped:
        break
  return phase_factor
assert(anti_commutator([(5, 0), (4, 0), (3, 0), (2, 0), (1, 0), (0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (6, 1)]) == 1)