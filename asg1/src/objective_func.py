
import numpy as np 
import matplotlib.pyplot as plt
import torch as tp
import math as ma
import F_L as fl
import F_O as fo
import smooth as sm

alpha_0 = 0.5

def objective_function(x_flat, n, x_start, x_goal, D, obj, lam, mu):
    x = sm.unflatten(x_flat, n, x_start, x_goal)
    al = 0.5
    path = fl.func_L(x)
    smoothness = np.sum(sm.smoothness_residuals(x_flat, n, x_start, x_goal, D)**2)
    penalty = fo.f_O_2(x,obj, al)
    ob_func = path + lam * smoothness + mu * penalty
    #Computing the Gradient
    grad_L_2d = fl.gradientf_L(x)
    grad_O_2d = fo.gradient_f_O_2(x, obj, al)
    grad_L_flat = sm.flatten(grad_L_2d) 
    grad_O_flat = sm.flatten(grad_O_2d)
    grad_S_flat = sm.gradient_smoothness(x_flat, n, x_start, x_goal, D)
    ob_grad = grad_L_flat + (lam * grad_S_flat) + (mu * grad_O_flat)
    

    return (ob_func,ob_grad)





""" Testing the different functions to check it returns stable values"""



#print(f_L(n2_array)) #Returns 42.105
#print(f_O(n2_array,ob_main)) #Returning inf, since it goes through the points
#print(f_O(n2_array,ob_second)) #returnes 0.6064
#print(f_O_2(n2_array,ob_main)) #Returns a highere penalty 20.55
#print(f_O_2(n2_array,ob_second)) #Returns a lowere penalty 13.049

