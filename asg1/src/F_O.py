import numpy as np 
import matplotlib.pyplot as plt
import torch as tp
import math as ma


# def detector(x:np.array,obj):
#    """ Helper function which measure the distance between obstacle
#    and tracjectory point. Since our object is a tuple, we are only interested at (x,y) coordinates """
#    return np.linalg.norm(x - obj[0]) #Returns the difference between a given point and object

def detector(point, obstacle_dict):
   """ Measures distance between a point and a dictionary-based obstacle """
   center = tp.tensor(obstacle_dict['center'], dtype=tp.float64)
   return tp.norm(point - center)

def f_O(x, obj):
    """
    Obstacle Model with Infinite Penalty.
    x: Can be a NumPy array (for plotting) or a Torch Tensor (for gradients).
    obj: List of dictionaries [{'center': np.array, 'radius': float}]
    """
    # 1. Ensure input is a Tensor for consistent math
    if not isinstance(x, tp.Tensor):
        x = tp.tensor(x, dtype=tp.float64)
    
    penalty = tp.tensor(0.0, dtype=tp.float64)
    
    for obstacle in obj:
        center = tp.tensor(obstacle['center'], dtype=tp.float64)
        r = obstacle['radius']
        
        # 2. Vectorized distance calculation (The 'Detector' logic)
        # Calculates distance from all points in x to this specific center
        distances = tp.norm(x - center, dim=1)
        
        # 3. The Infinite Wall check
        if tp.any(distances <= r):
            return tp.tensor(float('inf'), dtype=tp.float64)
        
        # 4. The Repulsive Penalty: 1 / (dist - r)^2
        # This creates the "Slope" seen in your Alpha-Value Mapping
        penalty += tp.sum(1.0 / tp.pow(distances - r, 2))
        
    return penalty
         

def gradientf_O(x:np.array,obj):
   """ Gradient of Obstacle Model"""
   grad = tp.autograd.grad(f_O)
   return grad(x,obj)



def f_O_2(x: tp.Tensor, obj, alpha=0.01):
    penalty = tp.tensor(0.0, dtype=tp.float64)
    for y in obj:
        center = tp.tensor(y['center'], dtype=tp.float64) # Access by key
        r = y['radius']                                   # Access by key
        distances = tp.norm(x - center, dim=1) 
        penalty += tp.sum(tp.exp(-alpha * (tp.pow(distances, 2) - r**2)))
    return penalty

def gradient_f_O_2(x_np, obj, alpha=0.01):
    # 1. Prepare the input for Torch
    x_torch = tp.tensor(x_np, dtype=tp.float64, requires_grad=True)
    
    # 2. Run the function to get a result (Scalar)
    # Ensure f_O_2 uses tp.exp, tp.norm, etc.
    penalty = f_O_2(x_torch, obj, alpha)
    
    # 3. Request the gradient of (penalty) with respect to (x_torch)
    # It returns a tuple, so we take the first element [0]
    grad_tuple = tp.autograd.grad(penalty, x_torch)
    grad_np = grad_tuple[0].detach().numpy()
    
    return grad_np


# def f_O(x:np.array, obj):
#    """ Obstacle Model that add penality large penality if we are close the obstacles and small if we are far away.
#          Returns a penality score """
#    N = len(x) # The lenght of the matrix
#    penalty = 0 #Pentalty
#    for y in obj: #Loop through each tuple
#       r  = y[1] #Get the radius
#       i = 0 # Control variable 
#       while i < N: #Continues running as long we are i are smaller then the lenght
#          dis = detector(x[i],y)  #Creating penalty givn each trajcectory and object
#          if dis > r: #First condtion measure how if we are far enough away
#             dis_diff_pow = np.pow(dis-r,2)  #add square it
#             penalty += 1/dis_diff_pow #Add small penalty since it since we are far awy
#          else:
#             penalty += ma.inf #Reaches the obstacle need large penality
#          i += 1 
#    return penalty

# def f_O_2(x:np.array,obj,alpha=0.01):
#    """ Obstacle model. The second penalty"""
#    N = len(x)
#    penalty = 0 
#    a = alpha
#    for y in obj:
#       r = y[1]
#       j = 0
#       while j < N:
#          dis = detector(x[j],y)
#          penalty += ma.exp(-a *(dis**2 - r**2))
#          j+=1 
#    return penalty

# def gradient_f_O_2(x:np.array,obj,alpha=0.01):
#    """ Gradient for obstacle model 2 """
#    grad = tp.autograd.grad(f_O_2)
#    return grad(x,obj,alpha)

# def f_O(x: tp.Tensor, obj):
#     """ 
#     Obstacle Model with Infinite Penalty.
#     x: (N, 2) tensor of coordinates
#     obj: list of (center_array, radius)
#     """
#     penalty = tp.tensor(0.0, dtype=tp.float64)
#     for center_np, r in obj:
#         center = tp.tensor(center_np, dtype=tp.float64)
#         # Calculate Euclidean distance for all points
#         dist = tp.norm(x - center, dim=1)
        
#         # If any point is inside the radius, the whole path is 'Infinite' cost
#         if tp.any(dist <= r):
#             return tp.tensor(float('inf'), dtype=tp.float64)
        
#         # Otherwise, add the 'Far Away' penalty: 1 / (dist - r)^2
#         penalty += tp.sum(1.0 / tp.pow(dist - r, 2))
        
#     return penalty

def gradientf_O(x_np: np.array, obj):
    """ Gradient that handles the Infinite Wall """
    x_torch = tp.tensor(x_np, dtype=tp.float64, requires_grad=True)
    penalty = f_O(x_torch, obj)
    
    # Check if we hit infinity
    if tp.isinf(penalty):
        # We can't take a gradient of inf. 
        # Return a zero gradient or a 'push-back' gradient manually
        return np.zeros_like(x_np) 
    
    # If not infinite, use autograd normally
    grad_out = tp.autograd.grad(penalty, x_torch)[0]
    return grad_out.detach().numpy()
