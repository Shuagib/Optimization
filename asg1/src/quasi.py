from objective_func import gradient_objectivefunc, objective_function
import numpy as np
from strong_brack import strong_backtracking
class DescentMethod:
    alpha : float 


class Quasi_NewtonMethod(DescentMethod):
    """ Using Quasi Newton Method which it approximate the hessian and goes in the steepest descent direction """
    def __init__(self,x:np.array,alpha:float,lam: float,mu:float,obj,grad):
        self.alpha = alpha
        self.lam = lam
        self.mu = mu
        self.x = x
        self.obj = obj
        self.grad = grad
        self.I = np.identity(2,dtype = float)
        if lam  < 0 or mu < 0:
            raise ValueError("Error: Must be strictly bigger than 0")
        
    def step(self):
        H_I = self.I #Defining the identity matrix
        Q, g = H_I, self.grad
        #self.x, alpha = strong_backtracking




           