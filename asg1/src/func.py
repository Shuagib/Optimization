import numpy as np 
import matplotlib.pyplot as plt

def func_euclidean(x: np.ndarray):
    """Function Lp Norm 2  Summing over the coloums and taking the sum of each next point and current.   """
    return np.sum(np.pow(x[1:] - x[:-1]),2,axis=1)

