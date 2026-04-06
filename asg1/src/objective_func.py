import numpy as np 
import matplotlib.pyplot as plt
import torch as tp
import math as ma
from F_L import f_L,gradientf_L
from F_O import f_O, f_O_2,gradient_f_O_2,gradientf_O
from smooth import smoothness_residuals, gradient_smoothness, least_squares_func, D, flatten, unflatten
from Path import path
from Bt_LineSearch import backtracking_line_search

def objective_function(x_flat, n, x_start, x_goal, D, obj, lam, mu):
    x = unflatten(x_flat, n, x_start, x_goal)
    ob_func = f_L(x) + lam * np.sum(smoothness_residuals(x_flat, n, x_start, x_goal, D)**2) + mu * f_O(x, obj)
    return ob_func


# def gradient_objective(x_flat, n, x_start, x_goal, D, obj, lam, mu):
#    x = unflatten(x_flat, n, x_start, x_goal)
#    ob_grad = gradientf_L(x) + lam * gradient_smoothness(x_flat, n, x_start, x_goal, D) + mu * gradient_f_O_2(x, obj)
#    return ob_grad

def gradient_objective(x_flat, n, x_start, x_goal, D, obj, lam, mu):
    # 1. Reconstruct 2D path for the functions that need it
    x = unflatten(x_flat, n, x_start, x_goal)
    
    # 2. Calculate gradients in 2D (N, 2)
    grad_L_2d = gradientf_L(x)
    grad_O_2d = gradient_f_O_2(x, obj)
    
    # 3. Flatten them to (36,) by removing start/goal rows
    # Use your existing flatten() function here!
    grad_L_flat = flatten(grad_L_2d) 
    grad_O_flat = flatten(grad_O_2d)
    
    # 4. Get the smoothness gradient (which is already flat)
    grad_S_flat = gradient_smoothness(x_flat, n, x_start, x_goal, D)
    
    # 5. Now they all have shape (36,), so you can sum them
    ob_grad = grad_L_flat + (lam * grad_S_flat) + (mu * grad_O_flat)
    
    return ob_grad


N = 20

""" Testing the different functions to check it returns stable values"""

start_point_x, start_point_y = (0.0, 0.0)
end_point_x,   end_point_y   = (22.4, 22.4)

x_start = np.array([start_point_x, start_point_y])
x_goal  = np.array([end_point_x,   end_point_y])


a = np.linspace(0,20,20)
b = np.linspace(0,20,20)
n2_array = np.column_stack((a,b))

#Creating the datastructurer of a circle. its position (x,y) and radius (r) ((x,y),r)
ob_main = [(( 16.0 , 19.0), 3.14), (( 6.0 , 7.0), 2.19)]
ob_second = [((16.0, 5.0), 2.0), ((4.0, 15.0), 2.0)]


#print(f_L(n2_array)) #Returns 42.105
#print(f_O(n2_array,ob_main)) #Returning inf, since it goes through the points
#print(f_O(n2_array,ob_second)) #returnes 0.6064
#print(f_O_2(n2_array,ob_main)) #Returns a highere penalty 20.55
#print(f_O_2(n2_array,ob_second)) #Returns a lowere penalty 13.049

