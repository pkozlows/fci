# Full Matrix
Within the full matrix folder, I generated my Hamiltonian in main.py and then evaluated the matrix elements using the Slater-Condon rules in slater.py. The phase factor is evaluated in face_factor.py. In addition to performing a full diagonalization using numpy, I implemented the Davidson algorithm in full_matrix/Davidson.py, which I use to compute only the desired few eigenvalues.
# Handy
Furthermore, I have implemented functions that do not require generating the full FCI matrix in handy.py. I have implemented the handy transformer (handy_transformer) and the determinant diagonal (diaconal), which I use as the preconditioner. Finally, the Davidson algorithm (handy_davidson) is used to make all of this work.
# Goal
The goal of this program is to work for a chain of 6 hydrogen atoms, with 6 spatial orbitals and 6 electrons.