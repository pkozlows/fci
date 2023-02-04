import generation
import input
from braket import braket
from copy import deepcopy
def anti_commutator(pair): 
    """Takes a pair of determinants, which are inside of a tuple.  
    returns a simplified version of the list, composed of the second quantization ops
      of the bra and ket. this list is simplified in the sense that all possible cancellations are made."""    
    sq_pair = braket(pair) 
    op_list = sq_pair.combined()
    # initialize the phase factor to unity
    phase_factor = 1
    # change each shared orb to the appropriate second quantization operator for later cancellation
    for orb in sq_pair.pair[0].intersection(sq_pair.pair[1]):
        annihilation = (orb, 0)
        creation = (orb, 1)
        # continue the lope while there are still are ops to be canceled
        while annihilation and creation in op_list:
            for index, op in enumerate(op_list):
                # if the creation and annihilation indices are next to each other, remove them and stop the loop
                if op_list[index] == annihilation and op_list[index+1] == creation:
                    op_list.remove(annihilation)
                    op_list.remove(creation)
                # if the creation and angulation in disease are not next to each other, swap neighboring ops
                elif op_list[index] == annihilation:
                    current = deepcopy(op_list[index])
                    next = deepcopy(op_list[index+1])
                    op_list[index] = next
                    op_list[index+1] = current
                    # add the appropriate face factor
                    phase_factor *= -1
    return phase_factor
# testing
assert(anti_commutator(({0,2,4,6,8,10}, {0,3,5,6,9,10})) == -1)
assert(anti_commutator(({0,1,2},{0,2,3})) == 1)