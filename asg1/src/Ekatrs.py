import numpy as np 
import matplotlib.pyplot as plt
import torch as tp
import math as ma
from F_L import f_L,gradientf_L
from F_O import f_O, f_O_2,gradient_f_O_2,gradientf_O
from smooth import smoothness_residuals, gradient_smoothness, least_squares_func, D, flatten, unflatten
from Path import path
from Bt_LineSearch import backtracking_line_search
from objective_func import *

""" Mapping functions with path"""


# 2. obj: The obstacles usually a list of dictionaries or objects containing (center, radius)
obj = [
    {'center': np.array([16.0, 19.0]), 'radius': 1.5},
    {'center': np.array([6.0, 7.0]), 'radius': 1.5}
]

# 3. lambda Smoothness weight

lam = 0.5 

# 4. mu obstacle avoidance weight

mu = 50.0

# 'f_simple' takes ONLY x_flat as an input. 
# It internally passes all your constants to your objective_function.
f_simple = lambda x_flat: objective_function(x_flat, N, x_start, x_goal, D, obj, lam, mu)

# 'grad_simple' does the same for the gradient.
grad_simple = lambda x_flat: gradient_objective(x_flat, N, x_start, x_goal, D, obj, lam, mu)

x_curr = flatten(path)
d = -grad_simple(x_curr) # Move in the direction of steepest descent


# 1. Define the range for alpha (0 is current path, 1 is full gradient step)
alpha_range = np.linspace(0, 1.2, 100)
x_axis = np.linspace(start_point_x,end_point_y, N)

# 2. Calculate values 
y_values = [f_simple(x_curr + a * d) for a in alpha_range]

# 3. Plot
plt.figure(figsize=(8, 5))
plt.plot(alpha_range, y_values, color='blue', lw=2, label='Total Cost')

# If values are huge (like 10^6), use a log scale to see the curve
if max(y_values) > 1000:
    plt.yscale('log')

plt.xlabel(r'Step Size ($\alpha$)')
plt.ylabel(r'Objective Value $f(x + \alpha d)$')
plt.title('function')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()



# 3. Run your backtracking function to find the "ideal" alpha
optimal_alpha = backtracking_line_search(f_simple, grad_simple, x_curr, d)

# Create a range of alphas to visualize the curve

y_values = [f_simple(x_curr + a * d) for a in alpha_range]

# Calculate the Armijo boundary line (the "sufficient decrease" condition)
# y = f(x) + beta * alpha * (grad' * d)
g = grad_simple(x_curr)
armijo_line = [f_simple(x_curr) + 1e-4 * a * np.dot(g, d) for a in alpha_range]

plt.figure(figsize=(10, 6))

# Plot the actual function value
plt.plot(alpha_range, y_values, label='Objective Function $f(x + \\alpha d)$', color='blue')

# Plot the Armijo condition (the straight descent line)
plt.plot(alpha_range, armijo_line, '--', color='red', label='Armijo Condition (Target)')

# Mark the alpha chosen by your function
plt.scatter(optimal_alpha, f_simple(x_curr + optimal_alpha * d), 
            color='green', s=100, zorder=5, label=f'Chosen Alpha: {optimal_alpha:.4f}')




for a in alpha_range:
    x_test = x_curr + a * d
    cost = f_simple(x_test)
    
    # If cost is infinite, we give it a very high number for the plot
    if cost == float('inf') or np.isinf(cost):
        y_values.append(1e10) # A "representative" infinity
    else:
        y_values.append(cost)

# # Plotting the "Barrier"
# plt.figure(figsize=(8, 5))
# plt.plot(alphas, y_values, label="Path Cost")
# plt.yscale('log') # Log scale helps see the jump to infinity
# plt.xlabel("Step Size (Alpha)")
# plt.ylabel("Objective Value (Log Scale)")
# plt.title("Alpha-Value Mapping with Infinite Penalty")
# plt.grid(True)
# plt.show()

costs = []

# 3. Scan the "Landscape"
for a in alpha_range:
    # Step out by alpha along the gradient direction
    x_test = x_curr + a * d
    
    # Calculate total cost (Length + Smoothness + Obstacles)
    current_cost = f_simple(x_test)
    costs.append(current_cost)

# 4. Create the Map
plt.figure(figsize=(9, 5))
plt.plot(alpha_range, costs, color='navy', label='Path Cost $f(x + \\alpha d)$')

# Run the backtracking search to find the stopping point
opt_alpha = backtracking_line_search(f_simple, grad_simple, x_curr, d)
plt.scatter(opt_alpha, f_simple(x_curr + opt_alpha * d), 
            color='crimson', label=f'Backtracking Choice (Alpha={opt_alpha:.4f})', zorder=5)

# plt.title("Alpha-Value Mapping: Visualizing the Step Size")
# plt.xlabel("Step Size (Alpha)")
# plt.ylabel("Objective Function Value")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.show()

# def plot_path_evolution(x_curr, direction, obj, x_start, x_goal):
#     alphas = [0.0, 0.5, 1.0]  # Check the start, middle, and final step
#     colors = ['red', 'orange', 'green']
    
#     plt.figure(figsize=(8,8))
    
#     # 1. Plot Obstacles
#     for y in obj:
#         circle = plt.Circle(y['center'], y['radius'], color='orange', alpha=0.3)
#         plt.gca().add_patch(circle)
        
#     # 2. Plot the Path "Mapping"
#     for a, col in zip(alphas, colors):
#         # Calculate the test path
#         x_test_flat = x_curr + a * direction
        
#         # Reshape back to (N, 2) and add anchors
#         x_2d = x_test_flat.reshape(-1, 2)
#         full_path = np.vstack([x_start, x_2d, x_goal])
        
#         # --- FIXED: This must be INDENTED to be inside the loop ---
#         plt.plot(full_path[:,0], full_path[:,1], marker='o', color=col, label=f'Alpha={a}')
    
#     plt.title("Alpha-Value Mapping: Visualizing the Step Size")
#     plt.xlabel("Step Size (Alpha)")
#     plt.ylabel("Objective Function Value")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.title("Physical Mapping: How the Path Curves Around Obstacles")
#     plt.axis('equal') 
#     plt.show()

# def plot_path_evolution(x_curr, direction, obj, x_start, x_goal):
#     alphas = [0.0, 0.5, 1.0]  
#     colors = ['red', 'orange', 'green']
    
#     plt.figure(figsize=(10, 8))
#     ax = plt.gca()
    
#     # 1. Plot Obstacles
#     for y in obj:
#         circle = plt.Circle(y['center'], y['radius'], color='orange', alpha=0.3, label='Obstacle')
#         ax.add_patch(circle)
        
#     # 2. Plot the Path "Mapping"
#     for a, col in zip(alphas, colors):
#         # Ensure we are working with a step calculation
#         x_test_flat = x_curr + a * direction
        
#         # FORCE RESHAPE: Turn flat array (40,) into (20, 2)
#         x_2d = x_test_flat.reshape(-1, 2)
        
#         # Ensure anchors are 2D (1, 2) before stacking
#         s = np.array(x_start).reshape(1, 2)
#         g = np.array(x_goal).reshape(1, 2)
        
#         # Combine: Start -> Optimized Points -> Goal
#         full_path = np.vstack([s, x_2d, g])
        
#         # DEBUG: Check if numbers exist
#         print(f"Plotting Alpha {a}: Path shape {full_path.shape}")
        
#         # PLOT: Marker='o' helps see individual points
#         ax.plot(full_path[:, 0], full_path[:, 1], 
#                 marker='o', linestyle='-', color=col, 
#                 linewidth=2, label=f'Alpha={a}')

#     # 3. Final Formatting
#     all_centers = np.array([y['center'] for y in obj])
#     plt.xlim(np.min(all_centers[:,0]) - 10, np.max(all_centers[:,0]) + 10)
#     plt.ylim(np.min(all_centers[:,1]) - 10, np.max(all_centers[:,1]) + 10)
#     plt.axhline(0, color='black', lw=1, alpha=0.2) # Show X-axis
#     plt.axvline(0, color='black', lw=1, alpha=0.2) # Show Y-axis
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.axis('equal')
#     plt.title("Physical Mapping: Path Evolution Around Obstacles")
#     print("Showing plot now...")
#     plt.show()


