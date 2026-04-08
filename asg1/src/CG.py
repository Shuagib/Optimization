from objective_func import gradient_objective, objective_function
import numpy as np
from strong_brack import  strong_backtracking
from smooth import flatten as fl
import autograd.numpy as an

class DescentMethod:
    alpha : float 


class Conjugate_Gradient(DescentMethod):
    """ Conjugate Gradient method uses bracktracking line search for finding the best alpha"""
    def __init__(self,x:an.array,alpha:float,lam: float,mu:float,obj,n,D,start,goal):
        self.alpha = alpha
        self.lam = lam
        self.mu = mu
        self.start = start
        self.goal = goal
        self.x = x
        self.obj = obj
        self.D = D
        self.n = n
        self.f =  lambda x: objective_function(x,self.n, self.start , self.goal ,self.D,self.obj,self.lam,self.mu)
        self.d = -gradient_objective(self.x, self.n, self.start, self.goal, self.D, self.obj, self.lam, self.mu)
        self.gradient = gradient_objective(self.x, self.n, self.start, self.goal, self.D, self.obj, self.lam, self.mu)
        if lam  < 0 or mu < 0:
            raise ValueError("Error: Must be strictly bigger than 0")
        

    def step(self):
        """Compute the next design point according to:"""
        grad_func = lambda y: gradient_objective(y, self.n, self.start, self.goal, self.D, self.obj, self.lam, self.mu)
        func = lambda x: objective_function(x,self.n, self.start , self.goal ,self.D,self.obj,self.lam,self.mu)
        grad = gradient_objective(self.x, self.n, self.start, self.goal, self.D, self.obj, self.lam, self.mu) #NewGradient
        d = -grad #Gradient direction
        alpha_z, alpha_tried, alpha_rejected = strong_backtracking(func,grad_func,self.x,d,self.alpha) #Using Stron_backtracking as line search
        beta_PR =  np.dot(np.transpose(grad), (grad - self.gradient)) / np.dot(np.transpose(self.gradient), self.gradient) 
        beta_FR = max(beta_PR,0)
        self.gradient = grad #Old gradient
        updat_x = self.x + alpha_z * d
        self.x = updat_x 
        self.d = - self.gradient  + beta_FR * self.d
         #Updating rule  for next position x 
         #Polak-Ribière Method for keeping track of the difference between our Gradient directions
         #New descnt direction given our negative gradient, difference  (Beta) and old descent direction
                                     #Using Strong_Bracketing Line search search for finding best alpha
        return updat_x, alpha_z,grad,alpha_tried, alpha_rejected
    
    def opt(self, kmax=100):
        """ Optimizer for using Conjugate method to find optimial design point"""
        x_points = []
        func_values = []
        k = 0
        ep = 0.01
        alpha_list = []
        while k < kmax: #Determination method if the new postion and old postion has very little divergence then step
            func_values.append(objective_function(self.x,self.n, self.start , self.goal ,self.D,self.obj,self.lam,self.mu)) #Function value
            x_points.append(self.x.copy()) #current Positions append in a list 
            updat_x, alpha_z,grad, alpha_tried, alpha_rejected = self.step()
            print(f"The current next step is: {updat_x}")
            alpha_list.append(alpha_z)
            print(f"The current Alphaz list is: {alpha_list}")
            print(f'The gradient is: {grad}')
            if np.linalg.norm(grad) < ep: #Termination Condition
                print(f'The Gradient is: {grad} and is absolute value is { np.linalg.norm(grad)}, Has convedged yet: {np.linalg.norm(grad) < ep} away from converging')
                break 
            k +=1
            print(f"The iteration is: {k}") 
        return updat_x, alpha_list,x_points,func_values,alpha_tried, alpha_rejected #Return current position, alphas list, x position, function values
    