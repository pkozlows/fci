import numpy as np 
import inp
#function to implement condon rules
def condon(diff):
    '''takes array of tuples that describes how determinants vary in
     maximum coincidence. returns matrix element'''
    # 1e condon rules
    def condon1(diff):
        '''takes array of tuples that describes how determinants vary in
     maximum coincidence. returns 1 electron matrix element'''
        # no diff
        if diff.len()==0:
            return
        #1 diff
        if diff.len()==1:
            return
        #2 diff
        if diff.len()==2:
            return
        #>2 diff
        if diff.len()>2:
            return 0
    #2e condon rules
    def condon2(diff):
         '''takes array of tuples that describes how determinants vary in
     maximum coincidence. returns 2 electron matrix element'''
    return condon1(diff)+condon2(diff)

'''# generate fci matrix
fci = np.ndarray((input.m ,input.m), dtype=float)
forhn fci:
r * 10215
# #set() matrixh6&dfrom  import 
#set() = np.ndarray((input.mfci,input.m), dtype=float)2b
for i in fci:
    i = c20(from condon import contentfor printbrp''self and set(list(def over(FUNK):))
        if a
        if if .PREFER KE.10c4c -  =  /=  +=  !=  !=  >=  and  or  in  not in  :cr8b r def linked(£10):
while :
match : r 
cas'''

# calculate the number of unique determinants
def unique_dets(electrons, orbs):
    """this number will depend on electrons and orbs"""

while len(all_possible) < unique_dets: 
# for each iteration of this loop, generate list of occupied # that will be later incorporated into a det
occupied_orbs=list()
# I don't know what to append below
occupied_orbs.append()5
def add_determinant(occupied_orbs):
    """"function to add a det to all_possible.I believe this should be done bypassing a lest into
      the function itselfthat specify which orbs are to be occupied in the specific det,
        which is the point that I haven't been able to figure out yet."""""
    det=set()
    # iterate through electrons in system, which can take any open orbital spot
    for electron in range(electrons):
        # iterate through spin orbs in system
        
    return all_possible.append(det)

# what condition would upon this wild top to run until.I believe I would
# need to calculate the total number of all possible determine to now when to stop the lope?

    
    
        