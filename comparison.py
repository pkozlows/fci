def compare_determinants(determinants):
   """takes tuple with a determinant pair. 
   returns tuple of  two sets that describe the differences that described the difference between pair.""" 
   return (determinants[0].difference(determinants[1]),determinants[1].difference(determinants[0]))