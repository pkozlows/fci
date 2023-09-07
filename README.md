# full FCI matrix
I generated my Hamiltonian in main.py and then evaluated the matrix using the Slater-Condon rules in slater.py with the condon function. The face factor is evaluated in face_factor.py with the anti_commutator function. In addition to performing a full diagonalization using numpy, I implemented the Davidson algorithm in , which I use to compute only the desired few eigenvalues.
# handy
Furthermore, I have implemented functions that do not require generating the full FCI matrix. I have implemented the handy transformer and the determinant diagonal, which I use as the preconditioner. Finally, the Davidson algorithm is used to make all of this work.
# goal
The goal of this program is to work for a chain of 6 hydrogen atoms, with 6 special orbitals and 6 electrons.