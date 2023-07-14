import time
import numpy as np
from Davidson import Davidson
from main import generation, integrals
from handy import handy_transformer
import mentor_handy

# first make a trial vector 
# trial_vector = np.zeros(400)
# trial_vector[0] = 1
# we want to find out whether the expensive first hamiltonian multiplication of the Davidson algorithm itugging while it was on so it automatically recognized my speech but I compressed like a for loving to toggle it faster without generating the full hamiltonian in the handy implementation
# start_ordinary = time.time()
hamiltonian = generation(integrals)
# get the eigen values and vectors of our hamiltonian
eigenvalues, eigenvectors = np.linalg.eig(hamiltonian)
trial_vector = eigenvectors[:, 0]
ordinary_matmult = np.dot(hamiltonian, trial_vector)
# and_ordinary = time.time()
# start_handy = time.time()
handy = handy_transformer(trial_vector, 6, 6, integrals)

mentor_candy = mentor_handy.knowles_handy_full_ci_transformer(integrals[0], integrals[1], 6, 0)
mentor_candy_vector = mentor_candy(trial_vector)
# and_candy = time.time()
# print(and_ordinary - start_ordinary)
# print(and_candy - start_handy)
print(mentor_candy_vector - handy)
# assert(np.allclose(mentor_candy_vector, ordinary_matmult))
# print(handy)
# # fun the time date Davidson da economization takes
# start_davidson = time.time()
# assert(Davidson(hamiltonian, 1) - -7.8399080148963369 < 1e-10)
# # find the time that Davidson digestion takes
# end_davidson = time.time()
# # find the time that numpy digestion takes
# start_numpy = time.time()
# # eigenvalues, eigenvectors = np.linalg.eig(hamiltonian)# thinner
# # find the time that numpy digestion takes
# end_numpy = time.time()
# assert((end_numpy - start_numpy) > (end_davidson - start_davidson))