#Fixing porblem with Noise problems 
initial_path = np.column_stack((x_axis, y_axis))
noise = ra.normal(0.000001, size = initial_path.shape)
noise[0] = [0.0,0.0]
noise[-1] = [0.0,0.0]
initial_path += noise
