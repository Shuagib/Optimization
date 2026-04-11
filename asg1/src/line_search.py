import numpy as np

def strong_backtracking(f, nabla, x, d, alpha=1, beta=1e-4, sigma=0.1):
    y0, g0, y_prev, alpha_prev = f(x), nabla(x) @ d, None, 0
    alpha_lo, alpha_hi = None, None
    rejected     = []   # failed, shrink
    tried_alpha = []

    # bracket phase
    while True:
        tried_alpha.append(alpha)
        y = f(x + alpha*d)
        if y > y0 + beta*alpha*g0 or (y_prev is not None and y >= y_prev):
            alpha_lo, alpha_hi = alpha_prev, alpha
            #print(f"low: {alpha_lo:.6f}  high:{alpha_hi:.6f}")
            break
        dir_gradient = nabla(x + alpha*d) @ d
        if abs(dir_gradient) <= -sigma * g0:
            #print("strong Wolfe in " r"$\alpha$" f"{alpha:.6f}")
            return alpha, rejected, tried_alpha
        elif dir_gradient >= 0:
            alpha_lo, alpha_hi = alpha, alpha_prev
            #print(f"bracket done: low={alpha_lo:.6f}  high={alpha_hi:.6f}")
            break
        y_prev, alpha_prev, alpha = y, alpha, 2 * alpha

    # zoom phase
    ylo = f(x + alpha_lo*d)
    while abs(alpha_hi - alpha_lo) > 1e-10:
        alpha = (alpha_lo + alpha_hi)/2
        rejected.append(alpha)
        y = f(x + alpha*d)
        #print(f"alpha={alpha:.6f}  f={y:.6f}  low={alpha_lo:.6f}  high={alpha_hi:.6f}")
        if y > y0 + beta*alpha*g0 or y >= ylo:
            alpha_hi = alpha
        else:
            g = nabla(x + alpha*d) @ d
            if abs(g) <= -sigma*g0:
                #print(f"Zoom done: lo={alpha_lo:.6f}  hi={alpha_hi:.6f}")
                return alpha, rejected, tried_alpha
            elif g*(alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha
    #print(f"Zoom done: lo={alpha_lo:.6f}  hi={alpha_hi:.6f}")
    return alpha_lo, rejected, tried_alpha




def backtracking_line_search(f, grad, x, d, alpha_0=0.5, p=0.5, beta=1e-4):
    y, g, alpha = f(x), grad(x), alpha_0    
    while ( f(x + alpha * d) > y + beta * alpha * np.dot(g, d) ) :
        alpha *= p
    return alpha