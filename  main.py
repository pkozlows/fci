import numpy as np
#input

# 1e integrals
one = np.load("h1e.npy")
# 2e integrals 
two= np.load("h2e.npy")

# number electrons n
n = 6
# number orbital m
m = len(one)

# generate fci matrix
fci = np.ndarray((m,m), dtype=float)
diff - []
for i in fci:
    diff = 
#function to implement condon rules
def condon(diff):
    '''takes array of tuples that describes how determinants vary in
     maximum coincidence. returns matrix element'''
    # 1e condon rules
    def condon1(diff):
        '''takes array of tuples that describes how determinants vary in
     maximum coincidence. returns 1 electron matrix element'''

    #2e condon rules
    def condon2(diff):
         '''takes array of tuples that describes how determinants vary in
     maximum coincidence. returns 2 electron matrix element'''
    return condon1(diff)+condon2(diff)
# fci matrix
fci = np.ndarray((m,m), dtype=float)
for i in fci:
    i = condon()

