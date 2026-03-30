import numpy as np 
import matplotlib.pyplot as plt

#Choose start and Goal
start_point_x,start_point_y = (0.0,0.0)
end_point_x,end_point_y = (22.7,22.5)

#Amount of Points 
N = 20

#Creaing x and y axis
x_axis = np.linspace(start_point_x,end_point_x,N)
y_axis = np.linspace(start_point_y,end_point_y,N)

#Creating Circular Obstacles (More Ecliples then circles)
Ob1 = plt.Circle(( 19.0 , 19.0), 1.5,color="darkorange",zorder=2)
Ob2 = plt.Circle(( 4.0 , 4.0), 1.5,color="darkorange", zorder= 2)

#Plotting the Initial path, start and end points and Obstacles
fig, ax = plt.subplots()
ax.plot(x_axis,y_axis,'o',color="orange")
ax.set_title("Initial Path and Obstacles")
ax.add_patch(Ob1)
ax.add_patch(Ob2)
ax.plot(start_point_x,start_point_y,'s', color='black')
ax.plot(end_point_x,end_point_y,'s', color='black')
plt.show()

