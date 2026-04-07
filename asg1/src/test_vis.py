import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np 
import matplotlib.pyplot as plt
from smooth import *
from F_L import *
from F_O import *
from GD import *
#Choose start and Goal
start_point  = (0.5,0.0)
end_point = (19.0,22.0)


start_point_x, start_point_y = start_point
end_point_x, end_point_y = end_point

#Amount of Points 
N = 20



#Creaing x and y axis
x_axis = np.linspace(start_point_x, end_point_x, N)
y_axis = np.linspace(start_point_y, end_point_y, N)

# Creating path as an (n,2) Array


#print(path)
#Creating Circular Obstacles (More Ecliples then circles)
Ob1 = plt.Circle(( 16.0 , 19.0), 1,color="darkorange",zorder=2)
Ob2 = plt.Circle(( 6.0 , 7.0), 1,color="darkorange", zorder= 2)


#Plotting the Initial path, start and end points and Obstacles
#fig, ax = plt.subplots()
#ax.plot(path,'o',color="orange")
#ax.set_title("Initial Path and Obstacles")
#ax.add_patch(Ob1)
#ax.add_patch(Ob2)
#ax.plot(start_point_x,start_point_y,'s', color='black')
#ax.plot(end_point_x,end_point_y,'s', color='black')
#plt.show()

### Let's visualize it 



l = 1 #Smoothness
m = 0.001  #Penalty
D_matrix = build_D(N)
alpha0 = 1.0

x_start = np.array([start_point_x, start_point_y])
x_goal = np.array([end_point_x, end_point_y])

initial_path = np.column_stack((x_axis, y_axis))
trajectory_path = flatten(initial_path)
ob_main = [((16.0, 19.0), 1), ((6.0, 7.0), 1)]

x_point, travel_x, f_value, alphz = GradientDescent(trajectory_path, alpha0, l, m, ob_main, N, D_matrix, x_start, x_goal).opt(N)





print(len(travel_x))
print(f_value)    
print(len(alphz))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left plot - Path evolution
ax1 = axes[0]


# Left plot - Path evolution

ax1.add_patch(plt.Circle((16.0, 19.0), 1, color='darkorange', alpha=0.5))
ax1.add_patch(plt.Circle((6.0, 7.0), 1, color='darkorange', alpha=0.5))



len = len(travel_x)
index = [0, len//4, len//3, len//2, len -1]

for run in index:
    current_path = unflatten(travel_x[run], N, x_start, x_goal)
    xes = current_path[:,0]
    yis = current_path[:,1]
    ax1.plot(xes, yis, label=f"Steps {run}")

ax1.plot(start_point_x, start_point_y, 's', color='black', markersize=10, label='start')
ax1.plot(end_point_x, end_point_y, '*', color='black', markersize=10, label='goal')
ax1.set_aspect('equal')
ax1.legend()
ax1.set_title('Path Evolution')

# Right plot - Convergence
ax2 = axes[1]
ax2.semilogy(f_value)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('f(x)')
ax2.set_title('Convergence')
ax2.grid()

plt.tight_layout()
plt.show()

