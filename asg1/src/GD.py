from objective_func import gradient_objectivefunc, objective_function
import numpy as np

class DescentMethod:
    alpha : float 


class GradientDescent(DescentMethod):
    def __init__(self,x:np.array,alpha:float,lam: float,mu:float):
        self.alpha = alpha
        self.lam = lam
        self.mu = mu
        self.x = x
        if lam  < 0 or mu < 0:
            raise ValueError("Error: Must be strictly bigger than 0")
           

    def step(self,obj):
        grad = gradient_objectivefunc(self.x,obj,self.lam,self.mu)
        self.x = self.x - self.alpha * grad
        return self.x
    

    def opt(self, obj, max_iter):
        x_poins = []
        f_values = []
        for k in range(max_iter):
            f_values.append(objective_function(self.x, obj, self.lam, self.mu))
            x_poins.append(self.x.copy())
            x_old = self.x.copy()
            self.step(obj)
            if np.linalg.norm(self.x - x_old) <= 0.001:
                break
        return self.x, x_poins, f_values






