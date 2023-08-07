import time
import numpy as np
from full_matrix.Davidson import Davidson
from full_matrix.main import generation, integrals
from handy.handy import handy_transformer
from handy.comparison import changing_handy
import handy.mentor_handy as mentor_handy

# first make a trial vector 
# trial_vector = np.zeros(400)
# trial_vector[0] = 1
# we want to find out whether the expensive first hamiltonian multiplication of the Davidson algorithm itugging while it was on so it automatically recognized my speech but I compressed like a for loving to toggle it faster without generating the full hamiltonian in the handy implementation
# start_ordinary = time.time()
hamiltonian = generation(integrals)
# get the eigen values and vectors of our hamiltonian
eigenvalues, eigenvectors = np.linalg.eig(hamiltonian)
# sort the eigen system
# sort the eigenvalues and eigenvectors
sorted_indices = eigenvalues.argsort()
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
# loop over  pairs of eigenvectors and print the norm of their difference 
# for i in range(len(eigenvalues)):
#     for j in range(i + 1, len(eigenvalues)):
        # print(np.linalg.norm(eigenvectors[:, i]))
trial_vector = eigenvectors[:, 0]
ordinary_matmult = np.dot(hamiltonian, trial_vector) / np.linalg.norm(np.dot(hamiltonian, trial_vector))
# assert(np.allclose(ordinary_matmult, np.einsum('ij,i->j', hamiltonian, trial_vector)))
# and_ordinary = time.time()
# start_handy = time.time(*)
mentor_candy = mentor_handy.knowles_handy_full_ci_transformer(integrals[0], integrals[1], 6)
mentor_candy_vector = mentor_candy(trial_vector) / np.linalg.norm(mentor_candy(trial_vector))
handy = handy_transformer(trial_vector, 6, 6, integrals) / np.linalg.norm(handy_transformer(trial_vector, 6, 6, integrals))
change_handy = changing_handy(trial_vector, 6, 6, integrals) / np.linalg.norm(changing_handy(trial_vector, 6, 6, integrals))


# calculate the the two vectors
stable_difference = handy - ordinary_matmult
mentor_difference = mentor_candy_vector - ordinary_matmult
changing_difference = change_handy - handy
difference_between_candies = mentor_candy_vector - handy
difference_with_comparison = mentor_candy_vector - change_handy

# calculate the Euclidean norm of the difference
stable_norm = np.linalg.norm(stable_difference)
mentor_norm = np.linalg.norm(mentor_difference)
changing_norm = np.linalg.norm(changing_difference)
norm_difference_between_candies = np.linalg.norm(difference_between_candies)
norm_difference_with_comparison = np.linalg.norm(difference_with_comparison)
print("handy and ordinary_matmult:", stable_norm)
print("mentor_candy_vector and ordinary_matmult:", mentor_norm)
print("change_handy and handy:", changing_norm)
print("mentor_candy_vector and handy:", norm_difference_between_candies)
print("mentor_candy_vector and change_handy:", norm_difference_with_comparison)


# and_candy = time.time()
# print(and_ordinary - start_ordinary)
# print(and_candy - start_handy)
# print(mentor_candy_vector - ordinary_matmult)
# assert(np.allclose(mentor_candy_vector, ordinary_matmult))
# print(handy)