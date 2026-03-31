import numpy as np 
import matplotlib.pyplot as plt

def func_euclidean(x: np.ndarray):
    """Function Lp Norm 2  collapsing over the coloums and taking the sum of each next point and current.
       This creates A vector of size (n-1,2)  """
    return np.sum(np.sum(np.pow(x[1:] - x[:-1],2),axis=1))



