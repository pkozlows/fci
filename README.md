# Goal
Given a set of one and two electron integrals, this program computes the energy for a chain of 6 hydrogen atoms, with 6 spatial orbitals and 6 electrons.
# Computational Details
## Full Matrix
Within the full matrix folder, I generated my Hamiltonian in main.py and then evaluated the matrix elements using the Slater-Condon rules in slater.py. The phase factor is evaluated in face_factor.py. In addition to performing a full diagonalization using numpy, I implemented the Davidson algorithm in full_matrix/Davidson.py, which I use to compute only the desired few eigenvalues.
## [Handy and Knowles Algorithm (1984)](https://doi.org/10.1016/0009-2614(84)85513-X)
Furthermore, I have implemented functions that do not require generating the full FCI matrix in handy.py. I have implemented the handy transformer (handy_transformer) and the determinant diagonal (diaconal), which I use as the preconditioner. Finally, the Davidson algorithm (handy_davidson) is used to make all of this work.
