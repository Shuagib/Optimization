import numpy as np 
import matplotlib.pyplot as plt
import torch as tp
import math as ma
from F_L import f_L,gradientf_L
from F_O import f_O, f_O_2,gradient_f_O_2,gradientf_O
from smooth import smoothness_value, smoothness_residuals, gradient_smoothness, least_squares_func, build_D, flatten, unflatten
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

start_point_x, start_point_y = (0.0, 0.0)
end_point_x,   end_point_y   = (22.4, 22.4)

x_start = np.array([start_point_x, start_point_y])
x_goal  = np.array([end_point_x,   end_point_y])


# def objective_function(x:np.array,obj,lam, mu ):
#    """ Objective function waiting on Anne to be done with the second function"""
#    ob_func = f_L(x) + lam * 0 + mu * f_O(x,obj)
#    return ob_func

def objective_function(x_flat, n, x_start, x_goal, D, obj, lam, mu):
    x       = unflatten(x_flat, n, x_start, x_goal)
    ob_func = f_L(x) + lam * np.sum(smoothness_residuals(x_flat, n, x_start, x_goal, D)**2) + mu * f_O(x, obj)
    return ob_func

# def gradient_objectivefunc(x:np.array,obj,lam, mu ):
#        ob_grad = gradientf_L(x) + lam * 0 + mu  * gradient_f_O_2(x,obj)
#        return ob_grad

def gradient_objective(x_flat, n, x_start, x_goal, D, obj, lam, mu):
   x = unflatten(x_flat, n, x_start, x_goal)
   ob_grad = gradientf_L(x) + lam * gradient_smoothness(x_flat, n, x_start, x_goal, D) + mu * gradient_f_O_2(x, obj)
   return ob_grad



""" Testing the different functions to check it returns stable values"""
#Creating a tester for the functions
a = np.linspace(0,20,20)
b = np.linspace(0,20,20)
n2_array = np.column_stack((a,b))

#Creating the datastructurer of a circle. its position (x,y) and radius (r) ((x,y),r)
ob_main = [(( 16.0 , 19.0), 3.14), (( 6.0 , 7.0), 2.19)]
ob_second = [((16.0, 5.0), 2.0), ((4.0, 15.0), 2.0)]


print(f_L(n2_array)) #Returns 42.105
print(f_O(n2_array,ob_main)) #Returning inf, since it goes through the points
print(f_O(n2_array,ob_second)) #returnes 0.6064
print(f_O_2(n2_array,ob_main)) #Returns a highere penalty 20.55
print(f_O_2(n2_array,ob_second)) #Returns a lowere penalty 13.049

