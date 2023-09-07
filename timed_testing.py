import cProfile
import time
import numpy as np
from full_matrix.main import generation, integrals
from full_matrix.Davidson import matrix_davidson
from handy import diaconal, handy_transformer, handy_davidson

# generate the hamiltonian
start_matrix_generation = time.time()
hamiltonian = generation(integrals)
end_matrix_generation = time.time()
# print the timing out
print("matrix generation time:", end_matrix_generation - start_matrix_generation)

start_numpy = time.time()
hamiltonian = generation(integrals)
# get the eigen values and vectors of our hamiltonian
eigenvalues, eigenvectors = np.linalg.eig(hamiltonian)
# sort the eigen system
# sort the eigenvalues and eigenvectors
sorted_indices = eigenvalues.argsort()
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
numpy_result = eigenvalues[0]
and_numpy = time.time()

start_davidson = time.time()
hamiltonian = generation(integrals)
davidson_result = matrix_davidson(hamiltonian, 1)
and_davidson = time.time()


start_handy = time.time()
Diagonal = diaconal(0, 6, 6, integrals)
transformer = handy_transformer(6, 6, integrals)
handy_result = handy_davidson(transformer, Diagonal, 0, 1, 400)
and_handy = time.time()

# print out the results and the times
print("numpy_result:", numpy_result)
print("davidson_result:", davidson_result)
print("handy_result:", handy_result)
print("numpy time:", and_numpy - start_numpy)
print("davidson time:", and_davidson - start_davidson)
print("handy time:", and_handy - start_handy)