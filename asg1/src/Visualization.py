
import numpy as np 
import matplotlib.pyplot as plt
from smooth import *
from F_L import *
from F_O import *
from GD import *
from CG import * 
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



l = 3 #Smoothness
m = 30 #Penalty
D_matrix = build_D(N)
alpha0 = 0.5

x_start = np.array([start_point_x, start_point_y])
x_goal = np.array([end_point_x, end_point_y])

initial_path = np.column_stack((x_axis, y_axis))
trajectory_path = flatten(initial_path)
ob_main = [((16.0, 19.0), 3), ((6.0, 7.0), 3)]

x_point, travel_x, f_value, alphz, stepz = GradientDescent(trajectory_path, alpha0, l, m, ob_main, N, D_matrix, x_start, x_goal).opt(N)


#print(len(travel_x),"The rigt Amount is:")
#print(f_value, "The right amount is " ) 
#print(len(alphz), "")

fig, axes = plt.subplots(1, 3, figsize=(12, 5))

for run in range(len(travel_x)):
    last_run = unflatten(travel_x[-1], N, x_start, x_goal)
    xes = last_run[:,0]
    yis = last_run[:,1]
    axes[0].plot(xes, yis,color="orange",marker='o')

#Maybe plot each plots for each iteration to show how it converges better
#Plotting the first graph with 
axes[0].add_patch(plt.Circle((16.0, 19.0), 3, color='darkorange', alpha=0.5))
axes[0].add_patch(plt.Circle((6.0, 7.0), 3, color='darkorange', alpha=0.5))
axes[0].plot(start_point_x, start_point_y, 's', color='black', markersize=10, label='start')
axes[0].plot(end_point_x, end_point_y, 's', color='black', markersize=10, label='goal')
axes[0].set_aspect('equal')
axes[0].legend()
axes[0].set_title('Path Evolution Gradient Descent')

# Right plot - Convergence
axes[1].plot(x_axis,f_value)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('f(x)')
axes[1].set_title('Convergence Gradient Descent')
axes[1].grid()

#Plotting the Alpha's to show that they varies, Why does my alpha intialstart do 
axes[2].plot(alphz)
axes[2].set_title('Step sizes (alpha)')
axes[2].set_xlabel('Optimizers iteterations')

plt.tight_layout()
plt.show()


#Intilizaing CG
updat_x, alpha_list,x_points,func_values,alpha_tried, alpha_rejected  = Conjugate_Gradient(trajectory_path, alpha0, l, m, ob_main, N, D_matrix, x_start, x_goal).opt(N)


print(len(x_points), "This is the lenght")
print(len(func_values), "This is the the amount of function evaluations")
print(len(alpha_list), "This is the amount of alphas")
#Plotting Conjugate Gradient 

fig_C, axes_C = plt.subplots(1, 3, figsize=(12, 5))



