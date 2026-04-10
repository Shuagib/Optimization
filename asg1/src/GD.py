from objective_func import gradient_objective, objective_function
import numpy as np
from Bt_LineSearch import backtracking_line_search
import autograd.numpy as an

class DescentMethod:
    alpha : float 


class GradientDescent(DescentMethod):
    """ Gradient descent class. First order descent method.
        Uses a fixed Alpha value """
    def __init__(self,x:an.array,alpha:float,lam: float,mu:float,obj,n,D,start,goal):
        self.alpha = alpha
        self.lam = lam
        self.start = start
        self.goal = goal
        self.mu = mu
        self.x = x
        self.n = n
        self.obj = obj
        self.D = D
        self.gradient = gradient_objective(self.x, self.n, self.start, self.goal, self.D, self.obj, self.lam, self.mu)
        self.f = lambda x: objective_function(x,self.n, self.start , self.goal ,self.D,self.obj,self.lam,self.mu)
        if lam  < 0 or mu < 0:
            raise ValueError("Error: Must be strictly bigger than 0")
           

    def step(self):
        """Compute the next design point according to:"""
        grad_func = lambda y: gradient_objective(y, self.n, self.start, self.goal, self.D, self.obj, self.lam, self.mu)
        grad = gradient_objective(self.x, self.n, self.start, self.goal, self.D, self.obj, self.lam, self.mu)
        d = -grad                   
        alpha = backtracking_line_search(self.f, grad_func, self.x, d, self.alpha)
        self.x = self.x - alpha * grad
        return self.x , alpha, grad
    

    def opt(self,iter=100):
        """ Gradient descent for using Conjugate method to find optimial design point"""
        x_poins = []
        f_values = []
        stepz = []
        k = 0
        ep = 0.0001
        alphz = []
        gradlist = []

        while k < iter: 
            f_values.append(objective_function(self.x,self.n, self.start , self.goal ,self.D,self.obj,self.lam,self.mu))
            x_poins.append(self.x[:]) #Creating a new list
            x_old = self.x[:]
            step_x, alphaz, grad = self.step()
            #print(grad,"GD gradient")
            print(np.linalg.norm(grad),"GD norm gradient")
            scalr_grad = np.linalg.norm(grad)
            gradlist.append(scalr_grad)
            #print(self.x)
            #print(alphaz, "This is a alpha")
            alphz.append(alphaz)
            stepz.append(step_x)
            f_new = objective_function(self.x,self.n, self.start , self.goal ,self.D,self.obj,self.lam,self.mu)
            f_old = objective_function(x_old,self.n, self.start , self.goal ,self.D,self.obj,self.lam,self.mu)
            if f_old - f_new < ep * abs(f_old):
                print(f"f_old: {f_old}, f_new: {f_new}, diff: {f_old - f_new}")
                break
            k +=1 
            #print(alphz)
        return x_poins, self.x, f_values,alphz, stepz,gradlist






