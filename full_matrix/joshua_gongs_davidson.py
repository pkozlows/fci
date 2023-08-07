import math
import numpy as np
import time
from main import generation, integrals




def Davidson(matrix, eig, k, tol = 1e-8):
    n = np.shape(matrix)[0] # Dimension of matrix
    mmax = n//2				# Maximum number of iterations
    t = np.eye(n,k)			# set of k unit vectors as guess
    V = np.zeros((n,n))		# array of zeros to hold guess vec
    I = np.eye(n)			# identity matrix same dimen as A
    for m in range(k,mmax,k):
        if m <= k:
            for j in range(0,k):
                V[:,j] = t[:,j]/np.linalg.norm(t[:,j])
            theta_old = 1 
        elif m > k:
            theta_old = theta[:eig]
        V[:,:m],R = np.linalg.qr(V[:,:m])
        T = np.dot(V[:,:m].T,np.dot(matrix,V[:,:m]))
        THETA,S = np.linalg.eig(T)
        idx = THETA.argsort()
        theta = THETA[idx]
        s = S[:,idx]
        for j in range(0,k):
            w = np.dot((matrix - theta[j]*I),np.dot(V[:,:m],s[:,j])) 
            q = w/(theta[j]-matrix[j,j])
            V[:,(m+j)] = q
        norm = np.linalg.norm(theta[:eig] - theta_old)
        if norm < tol:
            break
    return theta[:eig]

matrix = generation(integrals)
eig = 8
# Begin block Davidson routine

start_davidson = time.time()
Davidson = Davidson(matrix, eig, eig * 2)
end_davidson = time.time()

# End of block Davidson. Print results.
print("davidson = ", Davidson,";",
    end_davidson - start_davidson, "seconds")

# Begin Numpy diagonalization of A

start_numpy = time.time()

E,Vec = np.linalg.eig(matrix)
E = np.sort(E)

end_numpy = time.time()

# End of Numpy diagonalization. Print results.

print("numpy = ", E[:eig],";",
     end_numpy - start_numpy, "seconds") 