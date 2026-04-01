
import numpy as np 
import matplotlib.pyplot as plt
import torch as tp
import math as ma





def detector(x:np.array,obj):
   """ Helper function which measure the distance between obstacle
   and tracjectory point. Since our object is a tuple, we are only interested at (x,y) coordinates
                           """
   return np.linalg.norm(x - obj[0]) #Returns the difference between a given point and object




def f_O(x:np.array, obj):
   """ Obstacle Model that add penality large penality if we are close the obstacles and small if we are far away.
         Returns a penality score """
   N = len(x) # The lenght of the matrix
   penalty = 0 #Pentalty
   for y in obj: #Loop through each tuple
      r  = y[1] #Get the radius
      i = 0 # Control variable 
      while i < N: #Continues running as long we are i are smaller then the lenght
         dis = detector(x[i],y)  #Creating penalty givn each trajcectory and object
         if dis > r: #First condtion measure how if we are far enough away
            dis_diff_pow = np.pow(dis-r,2)  #add square it
            penalty += 1/dis_diff_pow #Add small penalty since it since we are far awy
         else:
            penalty += ma.inf #Reaches the obstacle need large penality
         i += 1 
   return penalty
         

def gradientf_O(x:np.array,obj):
   """ Gradient of Obstacle Model"""
   grad = tp.autograd.grad(f_O)
   return grad(x,obj)


def f_O_2(x:np.array,obj,alpha=0.01):
   """ Obstacle model. The second penalty"""
   N = len(x)
   penalty = 0 
   a = alpha
   for y in obj:
      r = y[1]
      j = 0
      while j < N:
         dis = detector(x[j],y)
         penalty += ma.exp(-a *(dis**2 - r**2))
         j+=1 
   return penalty

def gradient_f_O_2(x:np.array,obj,alpha=0.01):
   """ Gradient for obstacle model 2 """
   grad = tp.autograd.grad(f_O_2)
   return grad(x,obj,alpha)

