from objective_func import gradient_objectivefunc, objective_function
import numpy as np
from Bt_LineSearch import  backtracking_line_search
from smooth import flatten as fl

class DescentMethod:
    alpha : float 


class Conjugate_Gradient(DescentMethod):
    """ Conjugate Gradient method uses bracktracking line search for finding the best alpha"""
    def __init__(self,x:np.array,alpha:float,lam: float,mu:float,obj,f, grad):
        self.alpha = alpha
        self.lam = lam
        self.mu = mu
        self.x = x
        self.obj = obj
        self.f = f
        self.grad = grad
        self.r = -gradient_objectivefunc(self.x,obj,self.lam,self.mu)
        self.d = self.r.copy()
        if lam  < 0 or mu < 0:
            raise ValueError("Error: Must be strictly bigger than 0")
        

    def step(self):
        """ Descent direction method using Conjugate Gradient method"""
        alpha = backtracking_line_search(self.f,self.grad,self.x,self.d,self.alpha) #Using bracktracking search for finding best alpha
        self.x = self.x + alpha * self.d #Updating rule  for next position x 
        r_prev = self.r.copy() #old Gradient
        self.r = -gradient_objectivefunc(self.x, self.obj, self.lam, self.mu) #Updated Gradient
        beta =  np.dot(self.r.flatten(), (self.r - r_prev).flatten()) / np.linalg.norm(r_prev)**2 #Polak-Ribière Method for keeping track of the difference between our Gradient directions
        self.d = self.r + beta * self.d  #New descnt direction given our negative gradient, difference  (Beta) and old descent direction
        return self.x,alpha
    
    def opt(self, iteration):
        x_points = []
        func_values = []
        a_array = []
        j = 0
        while j < iteration:
            func_values.append(objective_function(self.x, self.obj, self.lam, self.mu)) #Function value
            x_points.append(self.x.copy()) #current Positions append in a list 
            x_old = self.x.copy() #old posistion
            self.x, self.alpha = self.step() #compute the current position and alpha step
            a_array.append(self.alpha ) #Keep the alphas 
            if np.linalg.norm(self.x - x_old) <= 0.001: #Determination method if the new postion and old postion has very little divergence then step
                break
            j +=1 
        return self.x, a_array ,x_points,func_values #Return current position, alphas list, x position, function values


            




           