import numpy as np 
import matplotlib.pyplot as plt
import torch as tp
import math as ma
from F_L import f_L,gradientf_L
from F_O import f_O, f_O_2,gradient_f_O_2,gradientf_O








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


#print(f_L(n2_array)) #Returns 42.105
#print(f_O(n2_array,ob_main)) #Returning inf, since it goe through the points
#print(f_O(n2_array,ob_second)) #returnes 0.6064
#print(f_O_2(n2_array,ob_main)) #Returns a highere penalty 20.55
#print(f_O_2(n2_array,ob_second)) #Returns a lowere penalty 13.049
