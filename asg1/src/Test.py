from scipy.optimize import minimize
import numpy as np 
import matplotlib.pyplot as plt
from smooth import *
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

fig, ax = plt.subplots()
ax.set_title("Initial Path and Obstacles")
fig, ax = plt.subplots()
ax.add_patch(Ob1)
ax.add_patch(Ob2)
plt.plot(minimize_line[:,0],minimize_line[:,1])
plt.plot(get_path[:,0],get_path[:,1])
plt.show()




# Initialize
# optimizer_nm = NMOptimizer(x_start, x_goal, ob_main, N_amount, l, m)

# # If trajectory_path is ALREADY flat, don't flatten it again inside run
# # Or just pass it in. Let's ensure the variables exist:
# try:
#     # Run the optimizer
#     # my_nm_path will be [20, 2] because run() calls unflatten at the end
#     my_nm_path, my_f_history, my_path_history = optimizer_nm.run(trajectory_path)
    
#     # Now these will work:
#     my_x = my_nm_path[:, 0]
#     my_y = my_nm_path[:, 1]
    
# except Exception as e:
#     print(f"Error during run: {e}")

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

