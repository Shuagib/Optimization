import numpy as np 
import matplotlib.pyplot as plt
import torch as tp
import math as ma

def f_L(x):
    # Convert numpy array to torch tensor if needed
    x = tp.tensor(x, dtype=tp.float)
        
    # Now tp.diff will work regardless of the original input type
    diffs = tp.diff(x, axis=0)
    return tp.sum(diffs**2)



def gradientf_L(x_np: np.array):
    # 1. Convert to torch and track history
    x_torch = tp.tensor(x_np, dtype=tp.float64, requires_grad=True)
    
    # 2. Calculate the value (The "Output")
    # This MUST use torch operations inside f_L
    value = f_L(x_torch)
    
    # 3. Use grad(output, input)
    # This returns a tuple, so we take the first element [0]
    grad_tensor = tp.autograd.grad(value, x_torch)[0]
    
    return grad_tensor.detach().numpy()

# def f_L(x:np.array):
#    """ This function is the Lp norm 2 and takes the difference  squared and sums 
#    them for all x_n points in the trajectory. It calculate the path lenght """
#    result = 0
#    N = len(x)
#    for i in range(N-1):
#       difference = (x[i+1] - x[i])
#       squared = difference**2
#       result += np.sum(squared)
#    return result 
   

# def gradientf_L(x:np.array):
#    """ Take the gradient of Lp 2 norm. Uses Autograd from Torch.
#   Takes the function as input and returns its derivate with respect to x"""
#    grad = tp.autograd.grad(f_L,x)
#    return grad #Gradient Lp norm 2