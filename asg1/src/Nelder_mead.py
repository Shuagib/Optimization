import numpy as np
import Smooth as sm
from objective_func import objective_function
from Path import *
from F_O import f_O, f_O_2
from F_L import func_L

path = np.column_stack((x_axis,y_axis))
alpha = 1
#l = 3 #Smoothness
#m = 15 #Penalty
D_matrix = sm.build_D(N_amount)
alpha0 = 1
ob_main = [((16.0, 19.0), 3), (( 6.0 , 7.0), 3)]

#f_simple = lambda x_flat: objective_function(x_flat, N, x_start, x_goal, sm.D, obj, l, m)

class NMOptimizer:
    def __init__(self, x_start, x_goal, obj, N, l, m):
        """Initializes the optimizer"""
        self.x_start = x_start
        self.x_goal = x_goal
        self.obj = obj
        self.N = N
        self.l = l
        self.m = m

    def get_objective(self, x_flat):
        """External objective function using class attributes."""
        ob_func, ob_grad = objective_function(
            x_flat, self.N, self.x_start, self.x_goal, D_matrix, self.obj, self.l, self.m)
        return ob_func  # Nelder-Mead only needs the value, not the gradient
    
    def get_objective(self, x_flat):
        """External objective function using class attributes."""
        # We catch both, but ONLY return the scalar score
        score, gradient = objective_function(
            x_flat, self.N, self.x_start, self.x_goal, 
            D_matrix, self.obj, self.l, self.m
        )
        return score

    def nelder_mead(self, x_initial_flat, tol=1e-4, max_iter=1000):
        """Implementation of Nelder-Mead for path optimization."""
        n_vars = len(x_initial_flat)
        simplex = [x_initial_flat]
        f_history = []
        path_history = []
        f_prev_best = np.inf

        for i in range(n_vars):
            x_new = np.array(x_initial_flat, copy=True)
            x_new[i] += 0.5 
            simplex.append(x_new)


        f = self.get_objective
        
        # Sort so simplex[0] is the best point
        for iteration in range(max_iter):
            simplex.sort(key=f)

            # Record the best value and path found in this iteration
            current_best_f = f(simplex[0])
            f_history.append(current_best_f)
            path_history.append(simplex[0].copy())
            print(f"Iteration {iteration}: best f = {current_best_f:.6f}")

            # Best point (x1), worst point (xn+1), and second worst (xn)
            x1 = simplex[0]
            xn = simplex[-2]
            x_worst = simplex[-1]
            
            #Termination condition, Using the standard deviation of function values across the simplex
            # f_vals = [f(v) for v in simplex]
            if abs(f(simplex[0]) - f_prev_best) < tol:
            #if np.std(f_vals) < tol:
                print(f"Converged at iteration {iteration}")
                print(f"Converged at iteration {iteration}, final f = {current_best_f:.6f}")
                break

            # Centroid 
            x_bar = np.mean(simplex[:-1], axis=0)

            # Reflection xr = x_bar + alpha * (x_bar - x_worst)
            xr = (1 + alpha0) * x_bar - alpha0 * x_worst
            fr = f(xr)

            if f(x1) <= fr < f(xn):
                # Case 1: f(x1) <= fR < f(xn)
                simplex[-1] = xr
                continue
                
            elif fr < f(x1):
                # Case 2: fR < f(x1) ie Expansion
                xe = 1.5 * x_bar - 2 * x_worst
                fe = f(xe)
                simplex[-1] = xe if fe < fr else xr
                continue
                
            elif fr >= f(xn):
                # Case 3: fR >= f(xn), Outside Contraction: xC = x(-1/2)
                if f(xn) <= fr < f(x_worst):
                    # Outside Contraction
                    xc = 1.5 * x_bar - 0.5 * x_worst
                    if f(xc) < fr:
                        simplex[-1] = xc
                        continue
                else:
                    # case 4; Inside Contraction: xC = x(1/2)
                    xc = 0.5 * x_bar + 0.5 * x_worst
                    if f(xc) < f(x_worst):
                        simplex[-1] = xc
                        continue

            # 5. Shrink: if no conditions were met
            for i in range(1, len(simplex)):
                simplex[i] = 0.5 * (x1 + simplex[i])
        
        return simplex[0], f_history, path_history

    def run(self, initial_path, **kwargs):
        """Helper to flatten, run NM, and unflatten the result."""
        x_initial_flat = sm.flatten(initial_path)

        
        print("Starting Nelder-mead...")
        optimized_flat, f_history, path_history = self.nelder_mead(x_initial_flat, **kwargs)
        
        final_path = sm.unflatten(optimized_flat, self.N, self.x_start, self.x_goal)
        
        print(f"Optimization complete. Final objective value: {self.get_objective(optimized_flat):.6f}")
        
        return final_path, f_history, path_history
    
