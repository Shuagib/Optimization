import torch.autograd as ta
import numpy as np
from objective_func import objective_function
import numpy as np
from line_search import strong_backtracking
class DescentMethod:
    alpha : float 


class Quasi_NewtonMethod(DescentMethod):
    """ Using Quasi Newton Method which it approximate the hessian and goes in the steepest descent direction """
    def __init__(self,x, x_start, x_goal,alpha ,lam,mu,obj,D,n):
        self.alpha = alpha
        self.lam = lam
        self.mu = mu
        self.x = x
        self.n = n
        self.x_start = x_start
        self.x_goal = x_goal
        self.obj = obj
        self.D = D
        self.Q =  np.identity(2*(self.n-2),dtype = float)  #Defining the identity matrix so it fits a path 2(n-2)
        self.g = objective_function(self.x, self.n, self.x_start, self.x_goal, self.D, self.obj, self.lam, self.mu)[4]
        self.func = lambda x: objective_function(x, self.n, self.x_start, self.x_goal, self.D, self.obj, self.lam, self.mu)[0] #Creating one function so it's easier to pass since strongbracket uses f(x+alpha*d)
        self.nabla = lambda y: objective_function(y, self.n, self.x_start, self.x_goal, self.D, self.obj, self.lam, self.mu)[4]

        if lam  < 0 or mu < 0:
            raise ValueError("Error: Must be strictly bigger than 0")
        
    def DFP(self):
        """ Using Davidon-Fletcher-Powell (DFP) method """
        d = -self.Q @ self.nabla(self.x) #Computing the search direction
        alpha_lo, rejected, tried_alpha = strong_backtracking(self.func,self.nabla,self.x,d,self.alpha) #Getting the fist alpha
        new_x = self.x + alpha_lo *d 
        delta  = new_x - self.x
        gamma  = self.nabla(new_x) - self.nabla(self.x)
        Q_new = (self.Q - np.outer(self.Q @ gamma, np.transpose(self.Q) @ gamma) / np.dot(gamma, self.Q @ gamma) + np.outer(delta, delta) / np.dot(delta, gamma))   #approximating our matrix using np.outer since we are creating matrix and np.dot to create scalars
        self.Q = Q_new 
        self.x = new_x 
        return self.x, alpha_lo,rejected,tried_alpha
    

    def BFGS(self):
        d = -self.Q @ self.nabla(self.x) #Computing the search direction
        alpha_lo, rejected, tried_alpha = strong_backtracking(self.func,self.nabla,self.x,d,self.alpha) #Getting the fist alpha
        new_x = self.x + alpha_lo *d 
        delta  = new_x - self.x
        gamma  = self.nabla(new_x) - self.nabla(self.x)
        Q_new = self.Q - (np.dot(np.outer(delta,gamma), self.Q) + np.dot(self.Q, np.outer(gamma,delta))) /np.dot(delta,gamma) + (1 + np.dot(gamma, np.dot(self.Q,gamma))/np.dot(delta,gamma)) * (np.outer(delta,delta))/np.dot(delta,gamma)
        self.Q = Q_new
        self.x = new_x 
        return self.x, alpha_lo, rejected,tried_alpha
    


    
    def opt_DFP(self,kmax=20,ep=0.001):
        x_point = []
        f_value = []
        pen_val = []
        path_val = []
        alpha_list = []
        sm = []
        optimal_x = []
        grad_list = []
        k = 0
        while np.linalg.norm(self.g) > ep: #Convergences rate
            fx, pen, path, smooth = objective_function(self.x, self.n, self.x_start, self.x_goal, self.D, self.obj, self.lam, self.mu)
            f_value.append(fx) #Function values
            pen_val.append(pen)
            path_val.append(path)
            sm.append(smooth)
            #print(f_value)
            x_point.append(self.x[:]) #Keeping track of all the tracjectories
            #print(x_point)
            new_x, alpha,rejected,tried_alpha = self.DFP()
            optimal_x.append(new_x)
            alpha_list.append(alpha)
            self.g = self.nabla(new_x)
            grad_list.append(self.g)
            scal_g = np.linalg.norm(grad_list)
            print(scal_g)
            k +=1 
            #We are running 10 iteration as standard, 
            if k >= kmax:
                break 
        return optimal_x, alpha_list, x_point,f_value,rejected,tried_alpha  #Returns a lost of given x points that one have traveled, good enough alphas, All the old positiosns and function values 
    

    def opt_BFGS(self,kmax):
        x_point = []
        f_value = []
        grad = []
        k = 0
        ep=0.001
        while np.linalg.norm(self.g) > ep: #Convergences rate tjeeking the gradient
            fx, pen, path, smooth = objective_function(self.x, self.n, self.x_start, self.x_goal, self.D, self.obj, self.lam, self.mu)
            f_value.append(fx) #Function values
            #print(f_value)
            x_point.append(self.x.copy()) #Keeping track of all the tracjectories
            #print(x_point)
            new_x, alpha, alpha_rejected, alpha_tried = self.BFGS()

            self.g = self.nabla(new_x)
            grad.append(self.g)
            #print(grad)
            sc = np.linalg.norm(grad)
            print(sc)
            k +=1 
            #We are running 10 iteration as standard, 
            if k >= kmax:
                break 
        return self.x, alpha, x_point,f_value #Returns a lost of given x points that one have traveled, good enough alphas, All the old positiosns and function values 







            










           