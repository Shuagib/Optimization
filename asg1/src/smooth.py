import numpy as np
from scipy.optimize import least_squares
import torch as tp
from autograd import grad

# MATRIX D given in the assingment

def build_D(n):
    D = np.zeros((n-2, n))
    for i in range(n-2):
        D[i, i]   =  1
        D[i, i+1] = -2
        D[i, i+2] =  1
    return D


#Shape the dimensions
def flatten(x):
    """Removes start and goal points and flattens to 1D."""
    return x[1:-1].flatten()

def unflatten(x_flat, n, x_start, x_goal):
    inside = x_flat.np.reshape(n-2, 2)
    return np.vstack([
        x_start.np.reshape(1, 2),
        inside,  
        x_goal.np.reshape(1, 2)
    ])


def unflatten(x_flat, n, x_start, x_goal):
    """Reconstructs the full (N, 2) path from the flattened 1D array."""
    # Correcting the reshape syntax: np.reshape(array, shape) or array.reshape(shape)
    inside = x_flat.reshape(n - 2, 2)
    
    # Ensure start and goal are (1, 2) for stacking
    return np.vstack([
        x_start.reshape(1, 2),
        inside,
        x_goal.reshape(1, 2)
    ])

#Goes from shape (1, 2) to shape (n-2, 2) to shape (1, 2) to shape (n, 2)


# second_diff = D @ x = (n-2, 2)
#return second_diff.flatten()  --> 1D vector

def smoothness_residuals(x_flat, n, x_start, x_goal, D):
    """ Transform the linear transformation on the """
    x = unflatten(x_flat, n, x_start, x_goal)
    second_diff = D @ x           
    return second_diff.flatten()  

# def smoothness_value(x, D):
#     Dx = D @ x
#     return np.sum(Dx**2)



def least_squares_func(x0, n, x_start, x_goal, D, method='trf', verbose=1):
    result = least_squares(
        lambda x_flat: smoothness_residuals(x_flat, n, x_start, x_goal, D),
        x0,
        method=method,
        verbose=verbose
    )
    return result

def gradient_smoothness(x_flat, n, x_start, x_goal, D):
    x    = unflatten(x_flat, n, x_start, x_goal) 
    grad = 2 * np.transpose(D) @ D @ x                         
    return grad[1:-1].flatten() 

