from objective_func import objective_function
import numpy as np
from line_search import backtracking_line_search
import autograd.numpy as an
from F_L import func_L
from F_O import f_O,f_O_2
from F_S import smoothness_residuals
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
        self.gradient = objective_function(x,self.n, self.start , self.goal ,self.D,self.obj,self.lam,self.mu)[1]
        self.f = lambda x: objective_function(x,self.n, self.start , self.goal ,self.D,self.obj,self.lam,self.mu)[0]
        if lam  < 0 or mu < 0:
            raise ValueError("Error: Must be strictly bigger than 0")
           

    def step(self):
        """Compute the next design point according to:"""
        grad_func = lambda y: objective_function(y,self.n, self.start , self.goal ,self.D,self.obj,self.lam,self.mu)[1]
        grad = objective_function(self.x,self.n, self.start , self.goal ,self.D,self.obj,self.lam,self.mu)[1]
        d = -grad                   
        alpha = backtracking_line_search(self.f, grad_func, self.x, d, self.alpha)
        self.x = self.x - alpha * grad
        return self.x , alpha, grad
    

    def opt(self,kmax):
        """ Gradient descent for using Conjugate method to find optimial design point"""
        x_poins = []
        f_values = []
        stepz = []
        k = 0
        ep = 0.0001
        alphz = []
        gradlist = []
        penlist = []
        pathlist = []
        smothlist = []

        print(f"--- Starting Gradient Descent (Max Iterations: {kmax}) ---")
        
        while k < kmax:
            fx, nabla =  objective_function(self.x,self.n, self.start , self.goal ,self.D,self.obj,self.lam,self.mu)
            smooth = smoothness_residuals(self.x,self.n,self.start,self.goal,self.D)
            penalty = f_O_2(self.x,self.obj, self.alpha)
            lenpath = func_L(self.x)
          
            smothlist.append(smooth)
            f_values.append(fx)
            penlist.append(penalty)

            pathlist.append(lenpath)
          
            x_poins.append(self.x[:]) #Creating a new list
            x_old = self.x[:]
            step_x, alphaz, nabla = self.step()
           
            scalr_grad = np.linalg.norm(nabla)
            gradlist.append(scalr_grad)
        
            alphz.append(alphaz)
            stepz.append(step_x)
            f_new = objective_function(self.x,self.n, self.start , self.goal ,self.D,self.obj,self.lam,self.mu)[0]
            f_old = objective_function(x_old,self.n, self.start , self.goal ,self.D,self.obj,self.lam,self.mu)[0]
            if f_old - f_new < ep * abs(f_old):
                print(f"f_old: {f_old}, f_new: {f_new}, diff: {f_old - f_new}")
                break
            k +=1 
           
        return [x_poins, self.x, f_values,alphz, stepz,nabla,pathlist,penlist,smothlist,gradlist]






