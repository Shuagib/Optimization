import numpy as np 
import matplotlib.pyplot as plt


#test = np.linspace(0,20,20)

#def func_euclidean2(x: np.ndarray):
"""Function Lp Norm 2  collapsing over the coloums and taking the sum of each next point and current.
       This creates A vector of size (n-1,2)"""
    #return np.sum(np.sum(np.pow(x[1:] - x[:-1],2)))


def func_eclidean1(x:np.array):
   """ This function is the Lp norm 2 and takes the difference  squared and sums 
   them for all x_n points in the trajectory"""
   result = 0
   N = len(x)
   for i in range(N-1):
      difference = (x[i+1] - x[i])
      squared = difference**2
      result += np.sum(squared)
   return result 
        


