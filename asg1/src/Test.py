from scipy.optimize import minimize
import numpy as np 
import matplotlib.pyplot as plt
from Smooth import *
from F_L import *
from F_O import *
from GD import *
from CG import * 
from quasi import *
from Visualization import gradient_CG, funcv, trajectory_path,x_start,x_goal,D_matrix,ob_main,l,m,optimal_path_CG,N_amount
from Nelder_mead import *

Ob1 = plt.Circle(( 16.0 , 19.0), 3,color="darkorange",zorder=2)
Ob2 = plt.Circle(( 6.0 , 7.0), 3,color="darkorange", zorder= 2)

x = trajectory_path #Is already flatten to 1D 


fun = lambda x: objective_function(x,N_amount, x_start ,x_goal ,D_matrix,ob_main,l,m)

CD_Scipy_Optimal_path = minimize(fun,x,method= 'CG',jac=True,tol=0.0001,options={'maxiter': 50,'disp': True, })

minimize_line = np.reshape(CD_Scipy_Optimal_path.x,(-1,2))



print(f'After {N_amount} this is what we found from CD : \n',f'The Optimal Path found by Conjugate Descent: {optimal_path_CG} \n ' + 
      f'My_CG Gradient convergencee {np.linalg.norm(gradient_CG)} \n' + 
        f'My_CGThe last function evaluation: {funcv[-1]} \n' + 
        f' For comparison using scipy.optimize Minimize using CG: {CD_Scipy_Optimal_path} ')


get_path = unflatten(optimal_path_CG, N_amount, x_start, x_goal)

fig, ax = plt.subplots(figsize=(8, 8))

# Obstacles
ax.add_patch(plt.Circle((16.0, 19.0), 3, color='darkorange', alpha=0.5))
ax.add_patch(plt.Circle(( 6.0 , 7.0), 3, color='darkorange', alpha=0.5))

# Paths
ax.plot(minimize_line[:,0], minimize_line[:,1], label='Scipy CG', marker='o', markersize=3)
ax.plot(get_path[:,0],      get_path[:,1],      label='My CG',    marker='o', markersize=3)

# Start / goal markers
ax.scatter(*x_start, s=100, c='black', marker='s', zorder=5, label='Start')
ax.scatter(*x_goal,  s=100, c='black', marker='s', zorder=5, label='Goal')

# Formatting
ax.set_title(f'CG Path Comparison (N={N_amount})', fontsize=14)

ax.set_aspect('equal')
ax.autoscale()  
ax.legend(fontsize=11)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()





# Create a wrapper for Scipy (since it only wants the score, not the gradient)
func_no_grad = lambda x_flat: objective_function(
    x_flat, N_amount, x_start, x_goal, D_matrix, ob_main, l, m)[0]


# 2. Run Scipy's version of Nelder-Mead
print("\nRunning Scipy Nelder-Mead for benchmark...")
NM_Scipy_Result = minimize(func_no_grad, x, method='Nelder-Mead', tol=0.01, options={'maxiter': 1000})

# Process Results for plotting
scipy_nm_coords = unflatten(NM_Scipy_Result.x, N_amount, x_start, x_goal)

# 3. Final Benchmark Comparison
print("-" * 30)
print(f"BENCHMARK RESULTS (N={N_amount})")
print("-" * 30)

print(f"Scipy NM Final Cost: {NM_Scipy_Result.fun:.6f}")
print(f"Scipy Success:       {NM_Scipy_Result.success}")
print(f"Scipy Iterations:    {NM_Scipy_Result.nit}")
print("-" * 30)

fig, ax = plt.subplots(figsize=(8, 6))

# Obstacles
ax.add_patch(plt.Circle((16.0, 19.0), 3, color='darkorange', alpha=0.5))
ax.add_patch(plt.Circle(( 6.0 , 7.0), 3, color='darkorange', alpha=0.5))


# Plot start, goal, and path
ax.scatter(*x_start, color='black', s=100, zorder=5, label='Start')
ax.scatter(*x_goal, color='black', s=100, zorder=5, label='Goal')
ax.plot(scipy_nm_coords[:, 0], scipy_nm_coords[:, 1], 'b-o', markersize=4, label='Scipy NM Path')

ax.set_title(f'Scipy Nelder-Mead Path (Cost: {NM_Scipy_Result.fun:.4f})')
ax.legend()
ax.grid(True)
ax.set_aspect('equal')
plt.tight_layout()
plt.show()

