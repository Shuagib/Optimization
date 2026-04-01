import numpy as np 

def backtracking_line_search(f, grad, x, d, alpha_0=1, p=0.5, beta=1e-4):
    y, g, alpha = f(x), grad(x), alpha_0    
    while ( f(x + alpha * d) > y + beta * alpha * np.sum(g, d) ) :
        alpha *= p
    return alpha