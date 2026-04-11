
from autograd import grad
import autograd.numpy as np
import math as ma 



def detector(x:np.array, obj):
   """ Measures distance between a point and the center of a obstacle """
   c = obj[0] #Getting the coordinates 
   return np.linalg.norm(x - c)


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
            dis_diff_pow = (dis-r)**2  #add square it
            penalty += 1/dis_diff_pow #Add small penalty since it since we are far awy
         else:
            penalty += np.inf #Reaches the obstacle need large penality and it gives a simple penalty that is easy to correct
         i += 1 
   return penalty

def f_O_2(x: np.array,obj,alpha=0.01):
   """ Obstacle model. The second penalty"""
   N = len(x)
   penalty = 0 
   a = alpha
   for y in obj:
      r = y[1]
      j = 0
      while j < N:
         dis = detector(x[j],y)
         penalty += np.exp(-a *(dis**2 - r**2))
         j+=1 
   return penalty


grad_f_O_1 = grad(f_O)
grad_f_O_2 = grad(f_O_2)

def gradient_f_O_1(x, obj):
    return grad_f_O_1(x, obj)

def gradient_f_O_2(x, obj, alpha=0.01):
    return grad_f_O_2(x, obj, alpha)

number = np.inf

print(number)