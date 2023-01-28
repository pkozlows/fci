import generation
import input
class braket:
  def __init__(self, pair):
    self.pair = pair
# define a ket in second quantization
  def ket(self):
    second_quantization = list()
    determinant = self.pair[1]
    for i in range(len(determinant)):
       # this ensures the correct ordering of the creations operators
       minimum = min(determinant)
       # creation operators are indicated by a 1
       second_quantization.append((minimum, 1))
       determinant.discard(minimum)
    return second_quantization
  # define a bra in second quantization
  def bra(self):
    second_quantization = list()
    determinant = self.pair[0]
    for i in range(len(determinant)):
       # this ensures the correct ordering of the annihilation operators
       maximum = max(determinant)
       # annihilation operators are indicated by a 0
       second_quantization.append((maximum, 0))
       determinant.discard(maximum)
    return second_quantization
# need to decide how to test this class
# assert(braket(list(generation.gen_unique_pairs(input.elec_in_system, input.orbs_in_system))[46]).bra() == )