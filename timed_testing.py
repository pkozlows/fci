import time
import numpy as np
from Davidson import Davidson
from main import generation, integrals
from handy import handy_transformer

matrix = generation(integrals)
# first make a trial vector
trial_vector = np.zeros(400)
trial_vector[0] = 1
# we want to find out whether the expensive first matrix multiplication of the Davidson algorithm is faster without generating the full antonian matrix in the handy implementation
start_ordinary = time.time()
matrix = generation(integrals)
knew_vector = np.dot(matrix, trial_vector)
and_ordinary = time.time()
start_handy = time.time()
handy_vector = handy_transformer(trial_vector, 6, 6, integrals)
and_candy = time.time()
print(knew_vector.shape)
print(handy_vector.shape)
print(and_ordinary - start_ordinary)
print(and_candy - start_handy)
print(knew_vector - handy_vector)
# assert(np.allclose(knew_vector, handy_vector))
# fun the time date Davidson da economization takes
start_davidson = time.time()
assert(Davidson(matrix, 1) - -7.8399080148963369 < 1e-10)
# find the time that Davidson digestion takes
end_davidson = time.time()
# find the time that numpy digestion takes
start_numpy = time.time()
eigenvalues, eigenvectors = np.linalg.eig(matrix)
# find the time that numpy digestion takes
end_numpy = time.time()
assert((end_numpy - start_numpy) > (end_davidson - start_davidson))