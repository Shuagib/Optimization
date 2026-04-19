import numpy as np 
import F_L as fl
import F_O as fo
import F_S as sm

alpha_0 = 0.5

def objective_function(x_flat, n, x_start, x_goal, D, obj, lam, mu):
    """ Computes the total cost and its analytical gradient in regards to the optimizers and the path."""
    # Construction
    x = sm.unflatten(x_flat, n, x_start, x_goal)
    al = 0.5

    # Computing the cost components
    path = fl.func_L(x)
    smoothness = np.sum(sm.smoothness_residuals(x_flat, n, x_start, x_goal, D)**2)
    penalty = fo.f_O_2(x,obj, al) 
    ob_func = path + lam * smoothness + mu * penalty # Total cost

    #Computing the Gradient
    grad_L_2d = fl.gradientf_L(x)
    grad_O_2d = fo.gradient_f_O_2(x, obj, al)

    # Flatten
    grad_L_flat = sm.flatten(grad_L_2d) 
    grad_O_flat = sm.flatten(grad_O_2d)
    grad_S_flat = sm.gradient_smoothness(x_flat, n, x_start, x_goal, D)
    
    # Total Gradient
    ob_grad = grad_L_flat + (lam * grad_S_flat) + (mu * grad_O_flat)
    
    return (ob_func,ob_grad)


