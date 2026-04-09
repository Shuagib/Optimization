import numpy as np
import matplotlib.pyplot as plt
from objective_func import objective_function, gradient_objective
from Path import *
from Visualization import x_start, x_goal
import smooth as sm
from Vis import f_simple


# obj = [
#     plt.Circle((16.0, 19.0), 1, color="darkorange", zorder=2),
#     plt.Circle((6.0, 7.0), 1, color="darkorange", zorder=2)
# ]
obj = {
    "Ob1": plt.Circle((16.0, 19.0), 1, color="darkorange", zorder=2),
    "Ob2": plt.Circle((6.0, 7.0), 1, color="darkorange", zorder=2)
}

def nelder_mead(f, x_initial_flat, tol=1e-5, max_iter=2000):
    """ Direct implementation of neder_mead for path optimization."""
    n_vars = len(x_initial_flat)
    simplex = [x_initial_flat]
    for i in range(n_vars):
        x_new = np.array(x_initial_flat, copy=True)
        x_new[i] += 0.5  # Step size for the initial simplex
        simplex.append(x_new)

    for iteration in range(max_iter):
        simplex.sort(key=f)
        
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

    return simplex[0]

x_initial = sm.flatten(path) 
print("Starting optimization...")
optimized_flat = nelder_mead(f_simple, x_initial)


# 4. Verification
final_path = sm.unflatten(optimized_flat, N, x_start, x_goal)

# Check for NaN or Inf values
if np.any(np.isnan(final_path)):
    print("TEST FAILED: Result contains NaNs. Check your penalty division by zero.")
else:
    print("TEST PASSED: Path generated.")

# 5. Visual Verification
plt.plot(final_path[:,0], final_path[:,1], '-o')