import generation
import input
from braket import braket
from copy import deepcopy
def anti_commutator(pair): 
    """Takes a pair of determinants, which are inside of a tuple.  
    very turns a simplified version of the list, composed of the tanked quantization providers
      of the bra and ket. this list is simplified in the sense that all possible cancellations are made. """    
    sq_pair = braket(pair) 
    op_list = sq_pair.combined()
    phase_factor = -1
    for orb in sq_pair.pair[0].intersection(sq_pair.pair[1]):
        print(orb)
        annihilation = (orb, 0)
        creation = (orb, 1)
        # print(sq_pair.pair[0].intersection(sq_pair.pair[1]))
        # continue the lope while there are still are ops to be canceled
        while annihilation and creation in op_list:
            # if the creation and annihilation indices are next to each other, remove them and stop the loop
            for index, op in enumerate(op_list):
                print(op_list)
                print(phase_factor)
                # print(phase_factor)
                if op_list[index] == annihilation and op_list[index+1] == creation:
                    print(op_list[index])
                    print(op_list[index+1])
                    op_list.remove(annihilation)
                    op_list.remove(creation)
                    print(op_list)
                elif op_list[index] == annihilation:
                    print('x')
                    current = deepcopy(op_list[index])
                    next = deepcopy(op_list[index+1])
                    op_list[index] = next
                    op_list[index+1] = current
                    phase_factor *= -1
    return op_list
det = braket(list(generation.gen_unique_pairs(input.elec_in_system, input.orbs_in_system))[46])
anti_commutator(det.pair)
            