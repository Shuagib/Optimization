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
        """ Descent method """
        grad = gradient_objectivefunc(self.x,self.obj,self.lam,self.mu)
        self.x = self.x - self.alpha * grad
        return self.x
    

    def opt(self, iteration):
        """ Gradient descent for using Conjugate method to find optimial design point"""
        x_poins = []
        f_values = []

        for k in range(iteration):
            f_values.append(objective_function(self.x, self.obj, self.lam, self.mu))
            x_poins.append(self.x.copy())
            x_old = self.x.copy()
            self.step(self.obj)
            if np.linalg.norm(self.x - x_old) <= 0.001:
                break
        return self.x, x_poins, f_values






