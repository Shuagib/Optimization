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


def objective_function(x:np.array,obj,𝜆, alpha: 0.01, 𝜇 ):
   """ Objective function waiting on Anne to be done with the second function"""
   ob_func = f_L(x) + 𝜆 * None + 𝜇 * f_O(x,obj)
   ob_grad = gradientf_L(x) + 𝜆* None + 𝜇  * gradient_f_O_2(x,obj)
   return (ob_func,ob_grad)


""" Testing the different functions to check it returns stable values"""
#Creating a tester for the functions
a = np.linspace(0,20,20)
b = np.linspace(0,20,20)
n2_array = np.column_stack((a,b))

#Creating the datastructurer of a circle. its position (x,y) and radius (r) ((x,y),r)
ob_main = [(( 16.0 , 19.0), 3.14), (( 6.0 , 7.0), 2.19)]
ob_second = [((16.0, 5.0), 2.0), ((4.0, 15.0), 2.0)]


#print(func_eclideannorm(n2_array)) Works
#print(pentality_func_1(n2_array,ob)) Returning inf, since it goe through the points
#print(pentality_func_1(n2_array,ob_second)) returnes 0.6064
#print(pentality_func_2(n2_array,ob_main)) Returns a highere penalty 20.55
#print(pentality_func_2(n2_array,ob_second)) Returns a lowere penalty 13.049
