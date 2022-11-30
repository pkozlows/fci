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

    #2e condon rules
    def condon2(diff):
         '''takes array of tuples that describes how determinants vary in
     maximum coincidence. returns 2 electron matrix element'''
    return condon1(diff)+condon2(diff)