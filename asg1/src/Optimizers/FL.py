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


def gradientf_L(x):
    """ Take the gradient of Lp 2 norm. Uses Autograd from Torch.
    Takes the function as input and returns its derivate with respect to x"""
    grad = grad(func_L)
    return grad(x) #Gradient Lp norm 2

