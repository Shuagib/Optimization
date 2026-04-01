import numpy as np
from scipy.optimize import least_squares

start_point_x, start_point_y = (0.0, 0.0)
end_point_x,   end_point_y   = (22.4, 22.4)

n = 20

x_start = np.array([start_point_x, start_point_y])
x_goal  = np.array([end_point_x,   end_point_y])

# straight line initial
x_init = np.linspace(x_start, x_goal, n) 

# MATRIX D given in the assingment

def build_D(n):
    D = np.zeros((n-2, n))
    for i in range(n-2):
        D[i, i]   =  1
        D[i, i+1] = -2
        D[i, i+2] =  1
    return D

D = build_D(n)

#Shape the dimensions
def flatten(x):
    return x[1:-1].flatten()

def unflatten(x_flat):
    interior = x_flat.reshape(n-2, 2)
    return np.vstack([
        x_start.reshape(1, 2),
        interior,  
        x_goal.reshape(1, 2)
    ])

#Goes from shape (1, 2) to shape (n-2, 2) to shape (1, 2) to shape (n, 2)

# second_diff = D @ x = (n-2, 2)
#return second_diff.flatten()  --> 1D vector


def smoothness_residuals(x_flat):
    x           = unflatten(x_flat)
    second_diff = D @ x           
    return second_diff.flatten()  

def smoothness_value(x):
    Dx = D @ x
    return np.sum(Dx**2)


x0 = flatten(x_init)

result_ls = least_squares(
    smoothness_residuals,
    x0,
    method='trf',
    verbose=1
)

print(f"Success: {result_ls.success}")
print(f"Final cost: {result_ls.cost:.4f}")