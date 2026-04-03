from objective_func import gradient_objectivefunc, objective_function
import numpy as np

class DescentMethod:
    alpha : float 


class GradientDescent(DescentMethod):
    """ Gradient descent class. First order descent method.
        Uses a fixed Alpha value """
    def __init__(self,x:np.array,alpha:float,lam: float,mu:float,obj):
        self.alpha = alpha
        self.lam = lam
        self.mu = mu
        self.x = x
        self.obj = obj
        if lam  < 0 or mu < 0:
            raise ValueError("Error: Must be strictly bigger than 0")
           

    def step(self):
        """Compute the next design point according to:"""
        grad = gradient_objectivefunc(self.x,self.obj,self.lam,self.mu)
        self.x = self.x - self.alpha * grad
        return self.x
    

    def opt(self,kmax):
        """ Gradient descent for using Conjugate method to find optimial design point"""
        x_poins = []
        f_values = []
        k = 0
        ep = 0.001

        while k < kmax: 
            f_values.append(objective_function(self.x, self.obj, self.lam, self.mu))
            x_poins.append(self.x.copy())
            x_old = self.x.copy()
            self.x = self.step(self.obj)
            f_new = objective_function(self.x, self.obj, self.lam, self.mu)
            f_old = objective_function(x_old, self.obj, self.lam, self.mu)
            if f_old - f_new < ep * np.linalg.norm(f_old):
                break
            k +=1 
        return self.x, x_poins, f_values






