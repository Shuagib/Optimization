from objective_func import gradient_objective, objective_function
import numpy as np
from Bt_LineSearch import backtracking_line_search
class DescentMethod:
    alpha : float 


class GradientDescent(DescentMethod):
    """ Gradient descent class. First order descent method.
        Uses a fixed Alpha value """
    def __init__(self,x:np.array,alpha:float,lam: float,mu:float,obj,n,x_start,x_goal,D):
        self.alpha = alpha
        self.lam = lam
        self.mu = mu
        self.x = x
        self.n = n
        self.x_start = x_start
        self.x_goal = x_goal
        self.obj = obj
        self.D = D
        self.f = objective_function(self.x,self.n,self.x_start,self.x_goal,self.D)
        if lam  < 0 or mu < 0:
            raise ValueError("Error: Must be strictly bigger than 0")
           

    def step(self):
        """Compute the next design point according to:"""
        alpha = backtracking_line_search(self.f,grad,self.x,-grad,self.alpha)
        grad = gradient_objective(self.x,self.obj,self.lam,self.mu)
        self.x = self.x - self.alpha * grad
        return self.x, alpha
    

    def opt(self,n):
        """ Gradient descent for using Conjugate method to find optimial design point"""
        x_poins = []
        f_values = []
        k = 0
        ep = 0.001
        alphz = []

        while k < n: 
            f_values.append(objective_function(self.x, self.obj, self.lam, self.mu))
            x_poins.append(self.x.copy())
            x_old = self.x.copy()
            self.x, self.alpha = self.step()
            alphz.append(self.alpha)
            f_new = objective_function(self.x, self.obj, self.lam, self.mu)
            f_old = objective_function(x_old, self.obj, self.lam, self.mu)
            if f_old - f_new < ep * np.linalg.norm(f_old):
                break
            k +=1 
        return self.x, x_poins, f_values,alphz






