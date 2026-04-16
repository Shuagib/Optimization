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


x = trajectory_path #Is already flatten to 1D 


fun = lambda x: objective_function(x,N_amount, x_start ,x_goal ,D_matrix,ob_main,l,m)

CD_Scipy_Optimal_path = minimize(fun,x,method= 'CG',jac=True,tol=0.0001,options={'maxiter': 50,'disp': True, })

minimize_line = np.reshape(CD_Scipy_Optimal_path.x,(-1,2))



print(f'After {N_amount} this is what we found from CD : \n',f'The Optimal Path found by Conjugate Descent: {optimal_path_CG} \n ' + 
      f'My_CG Gradient convergencee {np.linalg.norm(gradient_CG)} \n' + 
        f'My_CGThe last function evaluation: {funcv[-1]} \n' + 
        f' For comparison using scipy.optimize Minimize using CG: {CD_Scipy_Optimal_path} ')



get_path = unflatten(optimal_path_CG, N_amount, x_start, x_goal)
plt.plot(minimize_line[:,0],minimize_line[:,1])
plt.plot(get_path[:,0],get_path[:,1])
plt.show()
