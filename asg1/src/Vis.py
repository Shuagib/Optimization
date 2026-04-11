import numpy as np 
import matplotlib.pyplot as plt
import torch as tp
import math as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from F_L import f_L,gradientf_L
from F_O import f_O, f_O_2,gradient_f_O_2,gradientf_O
from smooth import D, smoothness_residuals, gradient_smoothness, least_squares_func, build_D, flatten, unflatten
from objective_func import objective_function, gradient_objective
from Ekatrs import objective_function, gradient_objective, grad_simple
import matplotlib.transforms as mtransforms
from line_search import strong_backtracking
from line_search import backtracking_line_search
from objective_func import objective_function
from Path import *

alpha_range = np.linspace(0, 1.2, 100)

#Visualising the bracketing
# def Bplot_the_search(f_alpha, x, d, alpha_range):
#     alpha_samples = np.arange(0, 4., 0.01)
#     y_values = [f(x + a * d) for a in alpha_range]
#     plt.plot(alpha_samples, f_alpha(alpha_samples,x,d), '-')
    
#     plt.plot(alpha_range, y_values, 'o')

#     #plt.plot(alphas, y_values, 'o')
#     #plt.text(x, y, '%d, %d' % (int(x), int(y)),
#     trans_offset = mtransforms.offset_copy(plt.gca().transData,  fig=plt.gcf(),
#                                         x=0.05, y=0.10, units='inches')
#     for i in range(len(alpha_range)):
#         plt.text(alpha_range[i], f_alpha(alpha_range[i],x,d), '%d' % i,
#         transform=trans_offset,
#         fontsize=10, verticalalignment='bottom', horizontalalignment='right')
#     plt.axis((0, 4, -15, 50))
#     plt.show()

def Bplot_the_search(f_alpha, x, d, alpha_range):
    # 1. Use consistent naming: use f_alpha instead of f
    # 2. Ensure your y_values calculation matches how f_alpha is defined
    y_values = [f_alpha(a, x, d) for a in alpha_range]
    
    # Define samples for a smooth line plot
    alpha_samples = np.linspace(0, 4, 400)
    y_samples = [f_alpha(a, x, d) for a in alpha_samples]
    
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_samples, y_samples, '-', label='Objective Curve', color='blue', alpha=0.5)
    plt.plot(alpha_range, y_values, 'o', label='Bracket Points', color='red')

    # Offset for text labels
    import matplotlib.transforms as mtransforms
    trans_offset = mtransforms.offset_copy(plt.gca().transData, fig=plt.gcf(),
                                        x=0.05, y=0.10, units='inches')
    
    for i, a in enumerate(alpha_range):
        plt.text(a, y_values[i], f'{i}',
                 transform=trans_offset,
                 fontsize=10, va='bottom', ha='right')
                 
    plt.xlim(0, 4)
    plt.ylim(-15, 50)
    plt.xlabel(r'Step Size $\alpha$')
    plt.ylabel('Objective Value')
    plt.title('Line Search Bracketing Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


f = lambda x: x[0]**4-5*x[0]**2+x[1]**2
nabla = lambda x: np.array([4*x[0]**3-10*x[0],2*x[1]])

f_alpha = lambda alpha,x,d: (x[0]+alpha*d[0])**4-5*(x[0]+alpha*d[0])**2+\
                            (x[1]+alpha*d[1])**2
x = np.array([0,1])
d = np.array([1,-1])


best_alpha = backtracking_line_search(f, nabla, x, d)
#all_points = bracket_points + zoom_points + [best_alpha]
Bplot_the_search(f_alpha, x, d, alpha_range)



#Visualising the strong bracketing
def SBplot_the_search(f_alpha, x, d, alpha_range):
    alpha_samples = np.arange(0, 4., 0.01)
    y_values = [f(x + a * d) for a in alpha_range]
    plt.plot(alpha_samples, f_alpha(alpha_samples,x,d), '-')
    
    plt.plot(alpha_range, y_values, 'o')

    #plt.plot(alphas, y_values, 'o')
    #plt.text(x, y, '%d, %d' % (int(x), int(y)),
    trans_offset = mtransforms.offset_copy(plt.gca().transData,  fig=plt.gcf(),
                                        x=0.05, y=0.10, units='inches')
    for i in range(len(alpha_range)):
        plt.text(alpha_range[i], f_alpha(alpha_range[i],x,d), '%d' % i,
        transform=trans_offset,
        fontsize=10, verticalalignment='bottom', horizontalalignment='right')
    plt.axis((0, 4, -15, 50))
    plt.show()


f = lambda x: x[0]**4-5*x[0]**2+x[1]**2
nabla = lambda x: np.array([4*x[0]**3-10*x[0],2*x[1]])

f_alpha = lambda alpha,x,d: (x[0]+alpha*d[0])**4-5*(x[0]+alpha*d[0])**2+\
                            (x[1]+alpha*d[1])**2
x = np.array([0,1])
d = np.array([1,-1])

best_alpha, zoom_points, bracket_points = strong_backtracking(f, nabla, x, d)
all_points = bracket_points + zoom_points + [best_alpha]

SBplot_the_search(f_alpha, x, d, all_points)

#VISUALISING THE Conjugate gradient

#

# start_point_x, start_point_y = (0.0, 0.0)
# end_point_x,   end_point_y   = (22.4, 22.4)

x_start = np.array([start_point_x, start_point_y])
x_goal  = np.array([end_point_x,   end_point_y])

obj = [
    {'center': np.array([16.0, 19.0]), 'radius': 1.5},
    {'center': np.array([6.0, 7.0]), 'radius': 1.5}
]

N=20

# lamada Smoothness weight
lam = 50 

# mu Obstacle avoidance weight
mu = 70.0

 

x_current = flatten(path) 

f_simple = lambda x_flat: objective_function(x_flat, N, x_start, x_goal, D, obj, lam, mu)

alpha_range = np.linspace(0, 19, 20)
x_axis = np.linspace(start_point_x,end_point_y, N)

d = -grad_simple(x_current)

y_values = [f_simple(x_current + a * d) for a in alpha_range]

plt.figure(figsize=(8, 5))
plt.plot(alpha_range, y_values, color='blue', lw=2, label='Total Cost')


if max(y_values) > 1000:
    plt.yscale('log')

plt.xlabel(r'Step Size ($\alpha$)')
plt.ylabel(r'Objective Value $f(x + \alpha d)$')
plt.title('Function22')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

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
alpha_range = np.linspace(0, 19, 100)

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
plt.title('Function')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# 1. Calculate the cost at each of the N points
# We create a list to store the 'Obstacle Penalty' or 'Total Cost' for each point

point_costs = []

for i in range(N):
    point = path[i] # This is [x_i, y_i]
    
    # Calculate the obstacle cost for THIS specific point
    # f_O usually checks distance to all objects in 'obj'
    cost = 0
    for o in obj:
        dist = np.linalg.norm(point - o['center'])
        if dist < o['radius']:
            # If inside the circle, the cost spikes based on mu
            cost += mu * (o['radius'] - dist)**2 
            
    point_costs.append(cost)

# 2. Plotting against the x-coordinates of your path (0 to 19)
plt.figure(figsize=(10, 5))
plt.plot(path[:, 0], point_costs, 'o-', color='firebrick', label='Obstacle Penalty')

# Let's fill the area to make it look like a 'cost landscape'
plt.fill_between(path[:, 0], point_costs, color='red', alpha=0.1)

plt.xlabel('X-coordinate along path (0 to 19)')
plt.ylabel('Cost Value')
plt.title('Objective Function obstacle along the Path')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


