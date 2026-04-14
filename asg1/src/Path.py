import numpy as np 
import matplotlib.pyplot as plt

#Starting points and end point 
start_point  = (0.5,0.0)
end_point = (19.0,22.0)

# Initial Amount of Points which is uniformaly distributed 
N_amount = 20



#Creating the 2D-grid
start_point_x, start_point_y = start_point
end_point_x, end_point_y = end_point
x_axis = np.linspace(start_point_x, end_point_x, N_amount)
y_axis = np.linspace(start_point_y, end_point_y, N_amount)

x_start = np.array([start_point_x, start_point_y])
x_goal = np.array([end_point_x, end_point_y])

ob_main = [((16.0, 19.0), 3), (( 6.0 , 7.0), 3)]

#Plotting the Grid
#Creating Circular Obstacles (More Ecliples then circles)
Ob1 = plt.Circle(( 16.0 , 19.0), 2,color="darkorange",zorder=2)
Ob2 = plt.Circle(( 6.0 , 7.0), 2,color="darkorange", zorder= 2)
fig, ax = plt.subplots()
ax.plot(x_axis,y_axis,'o',color="orange")
ax.set_title("Initial Path and Obstacles")
ax.add_patch(Ob1)
ax.add_patch(Ob2)
ax.plot(start_point_x,start_point_y,'s', color='black')
ax.plot(end_point_x,end_point_y,'s', color='black')
plt.show()

