import torch.autograd as ta
import numpy as np
from objective_func import gradient_objective, objective_function,gradient_objective,objective_function
import numpy as np
from strong_brack import strong_backtracking
class DescentMethod:
    alpha : float 


class Quasi_NewtonMethod(DescentMethod):
    """ Using Quasi Newton Method which it approximate the hessian and goes in the steepest descent direction """
    def __init__(self,x_flat, n, x_start, x_goal,alpha ,lam,mu,obj,grad,D,f):
        self.alpha = alpha
        self.lam = lam
        self.mu = mu
        self.x_flat = x_flat
        self.n = n
        self.x_start = x_start
        self.x_goal = x_goal
        self.obj = obj
        self.grad = grad
        self.D = D
        self.f = f
        self.Q =  np.identity(2*(self.n-2),dtype = float)  #Defining the identity matrix so it fits a path 2(n-2)
        self.g = gradient_objective(self.x_flat, self.n, self.x_start, self.x_goal, self.D, self.obj, self.lam, self.mu)
        self.func = lambda x: objective_function(x, self.n, self.x_start, self.x_goal, self.D, self.obj, self.lam, self.mu) #Creating one function so it's easier to pass since strongbracket uses f(x+alpha*d)
        self.nabla = lambda y: gradient_objective(y, self.n, self.x_start, self.x_goal, self.D, self.obj, self.lam, self.mu)

        if lam  < 0 or mu < 0:
            raise ValueError("Error: Must be strictly bigger than 0")
        
    def DFP(self):
        """ Using Davidon-Fletcher-Powell (DFP) method """
        d = -self.Q @ self.nabla(self.x_flat) #Computing the search direction
        alpha_lo, rejected, tried_alpha = strong_backtracking(self.func,self.nabla,self.x_flat,d,self.alpha) #Getting the fist alpha
        new_x = self.x_flat + alpha_lo *d 
        delta  = new_x - self.x_flat
        gamma  = self.nabla(new_x) - self.nabla(self.x_flat)
        Q_new = (self.Q - np.outer(self.Q @ gamma, np.transpose(self.Q) @ gamma) / np.dot(gamma, self.Q @ gamma) + np.outer(delta, delta) / np.dot(delta, gamma))   #approximating our matrix using np.outer since we are creating matrix and np.dot to create scalars
        self.Q = Q_new 
        self.x_flat = new_x 
        return self.x_flat, alpha_lo,rejected,tried_alpha
    

    def BFGS(self):
        d = -self.Q @ self.nabla(self.x_flat) #Computing the search direction
        alpha_lo, rejected, tried_alpha = strong_backtracking(self.func,self.nabla,self.x_flat,d,self.alpha) #Getting the fist alpha
        new_x = self.x_flat + alpha_lo *d 
        delta  = new_x - self.x_flat
        gamma  = self.nabla(new_x) - self.nabla(self.x_flat)
        Q_new = self.Q - (np.dot(np.outer(delta,gamma), self.Q) + np.dot(self.Q, np.outer(gamma,delta))) /np.dot(delta,gamma) + (1 + np.dot(gamma, np.dot(self.Q,gamma))/np.dot(delta,gamma)) * (np.outer(delta,delta))/np.dot(delta,gamma)
        self.Q = Q_new
        self.x_flat = new_x 
        return self.x_flat, alpha_lo,rejected,tried_alpha
    


    
    def opt_DFP(self,kmax=20,ep=0.001):
        x_point = []
        f_value = []
        k = 0
        while np.linalg.norm(self.g) > ep: #Convergences rate
            f_value.append(self.func(self.x_flat)) #Function values
            x_point.append(self.x_flat.copy()) #Keeping track of all the tracjectories
            self.x_flat, alpha = self.DFP()
            self.g = self.nabla(self.x_flat)
            k +=1 
            #We are running 10 iteration as standard, 
            if k >= kmax:
                break 
        return self.x_flat, alpha, x_point,f_value #Returns a lost of given x points that one have traveled, good enough alphas, All the old positiosns and function values 
    

    def opt_BFGS(self,kmax=20,ep=0.001):
        x_point = []
        f_value = []
        k = 0
        while np.linalg.norm(self.g) > ep: #Convergences rate
            f_value.append(self.func(self.x_flat)) #Function values
            x_point.append(self.x_flat.copy()) #Keeping track of all the tracjectories
            self.x_flat, alpha = self.BFGS()
            self.g = self.nabla(self.x_flat)
            k +=1 
            #We are running 10 iteration as standard, 
            if k >= kmax:
                break 
        return self.x_flat, alpha, x_point,f_value #Returns a lost of given x points that one have traveled, good enough alphas, All the old positiosns and function values 







            










           