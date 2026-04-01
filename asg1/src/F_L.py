
import numpy as np 
import matplotlib.pyplot as plt
import torch as tp
import math as ma


def f_L(x:np.array):
   """ This function is the Lp norm 2 and takes the difference  squared and sums 
   them for all x_n points in the trajectory. It calculate the path lenght """
   result = 0
   N = len(x)
   for i in range(N-1):
      difference = (x[i+1] - x[i])
      squared = difference**2
      result += np.sum(squared)
   return result 
   


def gradientf_L(x:np.array):
   """ Take the gradient of Lp 2 norm. Uses Autograd from Torch.
   Takes the function as input and returns its derivate with respect to x"""
   grad = tp.autograd.grad(f_L)
   return grad(x) #Gradient Lp norm 2


