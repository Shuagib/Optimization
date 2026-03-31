import numpy as np 
import matplotlib.pyplot as plt
import torch as tp

test = np.linspace(0,20,20)

#def func_euclidean2(x: np.ndarray):
"""Function Lp Norm 2  collapsing over the coloums and taking the sum of each next point and current.
       This creates A vector of size (n-1,2)"""
    #return np.sum(np.sum(np.pow(x[1:] - x[:-1],2)))


def func_eclideannorm(x:np.array):
   """ This function is the Lp norm 2 and takes the difference  squared and sums 
   them for all x_n points in the trajectory
   Takes a vector as input and Returns a scalar"""
   result = 0
   N = len(x)
   for i in range(N-1):
      difference = (x[i+1] - x[i])
      squared = difference**2
      result += np.sum(squared)
   return result 
   


def gradient_eclideannorm(x:np.array):
   """ Take the gradient of Lp 2 norm. Uses Autograd from Torch.
   Takes the function as input and returns its derivate with respect to x"""
   grad = tp.autograd.grad(func_eclideannorm)
   return grad(x)


#Creating the datastructurer of a circle, we need its position and radius

ob = [(( 19.0 , 19.0), 1), (( 5.0 , 5.0), 1)]
#print(len(ob)) has the size of two


def detector(x:np.array,cir):
   """ Helper function which measure the distance between obstacle
   and tracjectory point"""

   return np.linalg.norm(x - cir[0])
    
      


def Obstacle_func(x:np.array):
   pass 