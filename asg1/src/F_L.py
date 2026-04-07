from autograd import grad
import autograd.numpy as an


def func_L(x):
    """ This function is the Lp norm 2 and takes the difference  squared and sums 
    them for all x_n points in the trajectory. It calculate the path lenght """
    result = 0
    N = len(x)
    for i in range(N-1):
        difference = (x[i+1] - x[i])
        squared = difference**2
        result += an.sum(squared)
    return result 

grad_func_L = grad(func_L)   

def gradientf_L(x):
    return grad_func_L(x) 