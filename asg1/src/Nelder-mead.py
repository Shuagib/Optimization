import numpy as np
import matplotlib.pyplot as plt
import smooth as sm
from Path import *
# from Visualization import x_start, x_goal
from objective_func import objective_function
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# obj = [
#     plt.Circle((16.0, 19.0), 1, color="darkorange", zorder=2),
#     plt.Circle((6.0, 7.0), 1, color="darkorange", zorder=2)
# ]

N_amount = 20

# For the Optimizer (Math)
# Format: [[x, y], radius]

x_axis = np.linspace(start_point_x, end_point_x, N_amount)
y_axis = np.linspace(start_point_y, end_point_y, N_amount)



x_start = np.array([start_point_x, start_point_y])
x_goal = np.array([end_point_x, end_point_y])

obj = [
    [[16.0, 19.0], 1.0],
    [[6.0, 7.0], 1.0]
]

# For the Plotting (Visuals)
obj_visual = {
    "Ob1": plt.Circle((16.0, 19.0), 2, color="darkorange"),
    "Ob2": plt.Circle((6.0, 7.0), 2, color="darkorange")
}

# start_point_x, start_point_y = start_point
# end_point_x, end_point_y = end_point





l = 5 #Smoothness
m = 50 #Penalty

f_simple = lambda x_flat: objective_function(x_flat, N, x_start, x_goal, sm.D, obj, l, m)

def nelder_mead(f, x_initial_flat, tol=1e-5, max_iter=1000):
    """ Direct implementation of neder_mead for path optimization."""
    n_vars = len(x_initial_flat)
    simplex = [x_initial_flat]
    f_history = []
    for i in range(n_vars):
        x_new = np.array(x_initial_flat, copy=True)
        x_new[i] += 0.5  # Step size for the initial simplex
        simplex.append(x_new)

    for iteration in range(max_iter):
        # Sort so simplex[0] is the best point
        simplex.sort(key=f)
        
        # 2. Record the best value found in this iteration
        current_best_f = f(simplex[0])
        f_history.append(current_best_f)

        # Best point (x1), worst point (xn+1), and second worst (xn)
        x1 = simplex[0]
        xn = simplex[-2]
        x_worst = simplex[-1]
        
        #Termination condition, Using the standard deviation of function values across the simplex
        f_vals = [f(v) for v in simplex]
        if np.std(f_vals) < tol:
            break

        x_bar = np.mean(simplex[:-1], axis=0)

        #Reflection point xR = x(-1)
        xr = 2 * x_bar - x_worst
        fr = f(xr)

        if f(x1) <= fr < f(xn):
            # Case: f(x1) <= fR < f(xn)
            simplex[-1] = xr
            continue
            
        elif fr < f(x1):
            # Case 2: fR < f(x1) ie Expansion
            xe = 3 * x_bar - 2 * x_worst
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
            else:
                simplex[-1] = xr
            continue
            
        elif fr >= f(xn):
            # Case 3: fR >= f(xn), Outside Contraction: xC = x(-1/2)
            if f(xn) <= fr < f(x_worst):
                xc = 1.5 * x_bar - 0.5 * x_worst
                if f(xc) < fr:
                    simplex[-1] = xc
                    continue
            else:
                #case 4; Inside Contraction: xC = x(1/2)
                xc = 0.5 * x_bar + 0.5 * x_worst
                if f(xc) < f(x_worst):
                    simplex[-1] = xc
                    continue

        # 5. Shrink: if no other conditions were met
        for i in range(1, len(simplex)):
            simplex[i] = 0.5 * (x1 + simplex[i])

    return simplex[0], f_history

x_initial = sm.flatten(path) 

#Use if f1 <= fr < fn: (after assigning f1 = f_vals[0] and fn = f_vals[-2]).

print("Starting Nelder-mead...")
#optimized_flat = nelder_mead(f_simple, x_initial)


#final_path = sm.unflatten(optimized_flat, N, x_start, x_goal)

# Unpack the two return values into two separate variables
optimized_flat, x_values = nelder_mead(f_simple, x_initial)

# Now optimized_flat is just the array again, and unflatten will work!
final_path = sm.unflatten(optimized_flat, N, x_start, x_goal)



# Check for NaN or Inf values
if np.any(np.isnan(final_path)):
    print("TEST FAILED: Result contains NaNs. Check your penalty division by zero.")
else:
    print("TEST PASSED: Path generated.")

def plot_Optimalpath(path, start, goal, obstacles, title="Path Optimization with Nelder-Mead"):
    """
    Plots the optimal path with Nelder-mead.
    """
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # 1. Plot the obstacles
    for name, circle in obstacles.items():
        # Important: create a copy or re-add the patch if needed
        # Depending on your setup, you might need to recreate the patch here
        ax.add_patch(circle)

    # 2. Plot the optimized path
    plt.plot(path[:, 0], path[:, 1], '-o', color='royalblue', 
             label='Optimized Path', markersize=4, zorder=3)

    # 3. Plot Start and Goal (Corrected to 'ks' for black squares)
    plt.plot(start[0], start[1], 'ks', markersize=10, label='Start', zorder=4)
    plt.plot(goal[0], goal[1], 'rs', markersize=10, label='Goal', zorder=4)

    # 4. Formatting
    plt.title(title)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Crucial: This ensures circles stay circular, not ellipses
    plt.axis('equal') 
    
    plt.tight_layout()
    plt.show()

plot_Optimalpath(final_path, x_start, x_goal, obj_visual)
# plt.figure(figsize=(10, 8))
# ax = plt.gca()

# # Plot the obstacles
# # We use the 'obj_visual' dictionary you created
# for name, circle in obj_visual.items():
#     ax.add_patch(circle)

# # Unpack the path coordinates
# path_x = final_path[:, 0]
# path_y = final_path[:, 1]

# # Plot the optimized path
# plt.plot(path_x, path_y, '-o', color='royalblue', label='Optimized Path', markersize=4)

# # Plot Start and Goal points specifically
# plt.plot(x_start[0], x_start[1], 'sk', markersize=10, label='Start')
# plt.plot(x_goal[0], x_goal[1], 'sk', markersize=10, label='Goal')

# # Formatting the plot
# plt.title("Path Optimization with Nelder-Mead")
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)

# # Ensure the circles aren't squashed into ellipses
# plt.axis('equal') 

# plt.show()

# To shoe the steps of nelder-mead
#Setup the Contour Plot
# plt.figure(figsize=(10, 8))

# # Create a grid for the background "landscape" (your objective function)
# x_range = np.linspace(0, 20, 100)
# y_range = np.linspace(0, 20, 100)
# X, Y = np.meshgrid(x_range, y_range)

# # Note: This part can be slow if your objective function is complex
# Z = np.zeros(X.shape)
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         # We need to reshape the grid point to match what your function expects
#         Z[i,j] = f_simple(np.array([X[i,j], Y[i,j]])) 

# plt.contour(X, Y, Z, levels=50, cmap="viridis", alpha=0.4)

# # 3. Plot the history of the Simplex (The Triangles)
# for i, s in enumerate(simplex_history):
#     # Close the triangle by adding the first point to the end
#     # Assuming 2D, s is shape (3, 2)
#     triangle = np.vstack([s, s[0]]) 
    
#     # Plot with decreasing alpha (transparency) so the latest ones are bold
#     alpha_val = 0.3 if i < len(simplex_history) - 1 else 1.0
#     color = "red" if i < len(simplex_history) - 1 else "blue"
    
#     plt.plot(triangle[:, 0], triangle[:, 1], color=color, alpha=alpha_val, linewidth=1)

# # 4. Final touches
# plt.plot(x_start[0], x_start[1], 'go', label="Start")
# plt.plot(x_goal[0], x_goal[1], 'ro', label="Goal")
# plt.title("Nelder-Mead Path Optimization Progress")
# plt.axis('equal')
# plt.legend()
# plt.show()





# fig, ax = plt.subplots(figsize=(10, 4))

# #ax.plot(f_history, label='Best Function Value', color='#1f77b4', linewidth=1.5)

# # Adding labels and styling
# ax.set_xlabel('Iteration')
# ax.set_ylabel('Function Value $f(x)$')
# ax.set_title('Nelder-Mead Convergence')
# ax.grid(True, linestyle='--', alpha=0.6)

# # Optional: If your values span many orders of magnitude
# # ax.set_yscale('log') 
# plt.plot(x_values) 
# plt.title("Convergence of Nelder-Mead")
# plt.tight_layout()
# plt.show()

def plot_convergence(history, title="Nelder-Mead Convergence", use_log=False):
    """
    Plots the objective function value over iterations.
    
    Parameters:
    history (list): The f_history or simplex_history from the optimizer.
    title (str): Title of the plot.
    use_log (bool): If True, sets the y-axis to a logarithmic scale.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot the data
    ax.plot(history, label='Best Function Value $f(x)$', color='#1f77b4', linewidth=1.5)
    
    # Styling
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Function Value')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    if use_log:
        ax.set_yscale('log')
        ax.set_ylabel('Function Value (Log Scale)')
    
    ax.legend()
    plt.tight_layout()
    plt.show()


# plot_convergence(simplex_history)

optimized_flat, simplex_history = nelder_mead(f_simple, x_initial)

# Plot the convergence
plot_convergence(simplex_history, title="Path Optimization Convergence")

print("Starting Nelder-mead..2.")

#def nelder_mead_with_path_history(f, x_initial_flat, n_points, start, goal, save_interval=100, tol=1e-5, max_iter=2000):
def nelder_mead_with_path_history(f, x_initial_flat, N, x_start, x_goal, tol=1e-5, max_iter=2000, save_interval=100):
    """
    Modified Nelder-Mead that records the full best path at intervals.
    """
    n_vars = len(x_initial_flat)
    simplex = [x_initial_flat]
    f_history = []
    
    # Store tuples of (iteration_number, path_2d_array)
    path_evolution = []
    
    for iteration in range(max_iter):
        simplex.sort(key=f)
        
        # Step 2: Save a snapshot every X iterations
        if iteration % save_interval == 0:
            current_path = sm.unflatten(simplex[0], N, x_start, x_goal)
            path_evolution.append((iteration, current_path))


    # Initialize Simplex
    for i in range(n_vars):
        x_new = np.array(x_initial_flat, copy=True)
        x_new[i] += 0.5 
        simplex.append(x_new)

        # Capture best function value
        f_history.append(f(simplex[0]))

        # --- (Rest of standard Nelder-Mead logic: best, worst, x_bar, xr, xe, xc, shrink) ---
        x1 = simplex[0]
        xn = simplex[-2]
        x_worst = simplex[-1]
        
        f_vals = [f(v) for v in simplex]
        if np.std(f_vals) < tol:
            break

        x_bar = np.mean(simplex[:-1], axis=0)
        xr = 2 * x_bar - x_worst
        fr = f(xr)

        if f(x1) <= fr < f(xn):
            simplex[-1] = xr
            continue
        elif fr < f(x1):
            xe = 3 * x_bar - 2 * x_worst
            fe = f(xe)
            simplex[-1] = xe if fe < fr else xr
            continue
        elif fr >= f(xn):
            if f(xn) <= fr < f(x_worst):
                xc = 1.5 * x_bar - 0.5 * x_worst
                if f(xc) < fr:
                    simplex[-1] = xc
                    continue
            else:
                xc = 0.5 * x_bar + 0.5 * x_worst
                if f(xc) < f(x_worst):
                    simplex[-1] = xc
                    continue

        for i in range(1, len(simplex)):
            simplex[i] = 0.5 * (x1 + simplex[i])

    return simplex[0], f_history, path_evolution
