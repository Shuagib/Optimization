import numpy as np

# MATRIX D given in the assingment
def build_D(n):
    """Constructs the second-order finite difference matrix D, 
    used to approximate the second derivative (curvature)  of the path."""
    D = np.zeros((n-2, n))
    for i in range(n-2):
        D[i, i]   =  1
        D[i, i+1] = -2
        D[i, i+2] =  1
    return D


#Shape the dimensions
def flatten(x):
    """ Removes the fixed start and goal points and flattens the trajectory into a 1D."""
    return x[1:-1].flatten()


def unflatten(x_flat, n, x_start, x_goal):
    """Reconstructs the full (N, 2) path from the flattened 1D array."""
    inside = x_flat.reshape(n - 2, 2) # Reshape the flattened intermediate points back to (n-2, 2)
    # Stack the fixed start, optimized middle, and fixed goal
    return np.vstack([
        x_start.reshape(1, 2), # Shape (1, 2)
        inside, # Shape (n-2, 2)
        x_goal.reshape(1, 2) # Shape (1, 2)
    ])


def smoothness_residuals(x_flat, n, x_start, x_goal, D):
    """ Calculates the second-difference residuals of the trajectory."""
    x = unflatten(x_flat, n, x_start, x_goal)
    second_diff = D @ x # D (n-2, n) @ x (n, 2) to (n-2, 2)        
    return second_diff.flatten()  # Return as 1D vector



def gradient_smoothness(x_flat, n, x_start, x_goal, D):
    """ Gradient of the smoothness objective. Returns  gradient for the points.
    """
    x    = unflatten(x_flat, n, x_start, x_goal) 
    grad = 2 * np.transpose(D) @ D @ x                         
    return grad[1:-1].flatten() 

