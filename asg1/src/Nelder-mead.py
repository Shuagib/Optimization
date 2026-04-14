import numpy as np
import smooth as sm
from objective_func import objective_function
from Path import *

class PathOptimizer:
    def __init__(self, x_start, x_goal, obj, N, l=5, m=50):
        """Initializes the optimizer"""
        self.x_start = x_start
        self.x_goal = x_goal
        self.obj = obj
        self.N = N
        self.l = l
        self.m = m

    def get_objective(self, x_flat):
        """External objective function using class attributes."""
        return objective_function(
            x_flat, self.N, self.x_start, self.x_goal, 
            sm.D, self.obj, self.l, self.m
        )

    def nelder_mead(self, x_initial_flat, tol=1e-5, max_iter=100, 
                    alpha=1.0, beta=2.0, gamma=0.5):
        """Implementation of Nelder-Mead for path optimization."""
        n_vars = len(x_initial_flat)
        simplex = [x_initial_flat]
        f_history = []

        for i in range(n_vars):
            x_new = np.array(x_initial_flat, copy=True)
            x_new[i] += 0.5 
            simplex.append(x_new)

        f = self.get_objective
        
        # Sort so simplex[0] is the best point
        for iteration in range(max_iter):
            simplex.sort(key=f)
            
            # Record the best value found in this iteration
            current_best_f = f(simplex[0])
            f_history.append(current_best_f)

            # Best point (x1), worst point (xn+1), and second worst (xn)
            x1 = simplex[0]
            xn = simplex[-2]
            x_worst = simplex[-1]
            
            # Termination condition, Using the standard deviation of function values across the simplex
            f_vals = [f(v) for v in simplex]
            if np.std(f_vals) < tol:
                print(f"Converged at iteration {iteration}")
                break

            # Centroid 
            x_bar = np.mean(simplex[:-1], axis=0)

            # Reflection xr = x_bar + alpha * (x_bar - x_worst)
            xr = (1 + alpha) * x_bar - alpha * x_worst
            fr = f(xr)

            if f(x1) <= fr < f(xn):
                # Case 1: f(x1) <= fR < f(xn)
                simplex[-1] = xr
                continue
                
            elif fr < f(x1):
                # Case 2: fR < f(x1) ie Expansion
                xe = (1 + beta) * x_bar - beta * x_worst
                fe = f(xe)
                simplex[-1] = xe if fe < fr else xr
                continue
                
            elif fr >= f(xn):
                # Case 3: fR >= f(xn), Outside Contraction: xC = x(-1/2)
                if f(xn) <= fr < f(x_worst):
                    # Outside Contraction
                    xc = (1 + gamma) * x_bar - gamma * x_worst
                    if f(xc) < fr:
                        simplex[-1] = xc
                        continue
                else:
                    # case 4; Inside Contraction: xC = x(1/2)
                    xc = (1 - gamma) * x_bar + gamma * x_worst
                    if f(xc) < f(x_worst):
                        simplex[-1] = xc
                        continue

            # 5. Shrink: if no conditions were met
            for i in range(1, len(simplex)):
                simplex[i] = 0.5 * (x1 + simplex[i])

        return simplex[0], f_history

    def run(self, initial_path, **kwargs):
        """Helper to flatten, run NM, and unflatten the result."""
        x_initial_flat = sm.flatten(initial_path)
        
        print("Starting Nelder-mead...")
        optimized_flat, f_history = self.nelder_mead(x_initial_flat, **kwargs)
        
        final_path = sm.unflatten(optimized_flat, self.N, self.x_start, self.x_goal)
        return final_path, f_history