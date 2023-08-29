import cProfile
import time
import numpy as np
from full_matrix.main import generation, integrals
import my_functions
import mentors_functions

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
davidson_result = my_functions.matrix_davidson(hamiltonian, 1)
and_davidson = time.time()

start_candy = time.time()
my_diag = my_functions.my_diag(0, 6, 6, integrals)
# my_transformer = my_functions.handy_transformer(6, 6, integrals)
# cProfile.run('my_functions.handy_davidson(my_transformer, my_diag, 0, 2)', sort='cumtime')
# candy_result = my_functions.handy_davidson(my_transformer, my_diag, 0, 2)
and_candy = time.time()

stared_mentor_handy = time.time()
mentor_diag = mentors_functions.mentor_diag(0, 6, integrals)
mentor_transformer = mentors_functions.knowles_handy_full_ci_transformer(integrals[0], integrals[1], 6)
# cProfile.run('my_functions.handy_davidson(mentor_transformer, mentor_diag, 0, 1, 400)', sort='cumtime')
ended_mentor_handy = time.time()

compare_prison_time = time.time()
# start_changing_handy = time.time()
comparison_transformer = my_functions.comparison_transformer(6, 6, integrals)
# cProfile.run('my_functions.handy_davidson(comparison_transformer, my_diag, 0, 1, 400)', sort='cumtime')
comparison = my_functions.handy_davidson(comparison_transformer, my_diag, 0, 1, 400)
compare_prison_time = time.time()

# print out the results and the times
print("numpy:", numpy_result)
print("davidson:", davidson_result)
print("comparison:", comparison)
print("numpy time:", and_numpy - start_numpy)
print("davidson time:", and_davidson - start_davidson)
print("candy time:", and_candy - start_candy)
print("compare prison time:", compare_prison_time - and_candy)
print("mentor_handy time:", ended_mentor_handy - stared_mentor_handy)