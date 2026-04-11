from objective_func import objective_function
import numpy as np
from line_search import  strong_backtracking
from smooth import flatten as fl,gradient_smoothness,smoothness_residuals
import autograd.numpy as an
from F_O import f_O,f_O_2
from F_L import func_L
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
        self.f =  lambda x: objective_function(x,self.n, self.start , self.goal ,self.D,self.obj,self.lam,self.mu)[0]
        self.d = -objective_function(self.x, self.n, self.start, self.goal, self.D, self.obj, self.lam, self.mu)[1] #Starting descent direction
        self.gradient =objective_function(self.x, self.n, self.start, self.goal, self.D, self.obj, self.lam, self.mu)[1]
        if lam  < 0 or mu < 0:
            raise ValueError("Error: Must be strictly bigger than 0")
        

    def step(self):
        """Compute the next design point according to:"""
        grad_func = lambda y: objective_function(y, self.n, self.start, self.goal, self.D, self.obj, self.lam, self.mu)[1]
        func = lambda x: objective_function(x,self.n, self.start , self.goal ,self.D,self.obj,self.lam,self.mu)[0]
        grad = objective_function(self.x, self.n, self.start, self.goal, self.D, self.obj, self.lam, self.mu)[1]#NewGradient
        beta_PR =  np.dot(np.transpose(grad), (grad - self.gradient)) / np.dot(np.transpose(self.gradient), self.gradient) 
        beta_FR = max(beta_PR,0)
        old_d = self.d
        self.d = - grad  + beta_FR * old_d #Descent direction
        alpha_z,  rejected, tried_alpha = strong_backtracking(func,grad_func,self.x, self.d,self.alpha) #Using Stron_backtracking as line searc
        updat_x = self.x + alpha_z * self.d
        self.gradient = grad #Old gradient
        self.x = updat_x
         #Updating rule  for next position x 
         #Polak-Ribière Method for keeping track of the difference between our Gradient directions
         #New descnt direction given our negative gradient, difference  (Beta) and old descent direction
                                     #Using Strong_Bracketing Line search search for finding best alpha
        return self.x, alpha_z,grad, rejected, tried_alpha
    
    def opt(self, kmax=100):
        """ Optimizer for using Conjugate method to find optimial design point"""
        x_points = []
        func_values = []
        stepz = []
        k = 0
        ep = 0.0001
        alpha_list = []
        penalty_list = []
        path_list = []
        smooth_list = []
        converlist = []
        #path,pena,smooth
        while k < kmax: #Determination method if the new postion and old postion has very little divergence then step
            fx, nabla =  objective_function(self.x,self.n, self.start , self.goal ,self.D,self.obj,self.lam,self.mu)
            smooth = smoothness_residuals(self.x,self.n,self.start,self.goal,self.D)
            pena = f_O(self.x,self.obj)
            path = func_L(self.x)
            #print(f"Smoothness is {smooth}")
            smooth_list.append(smooth)
            func_values.append(fx) #Function value
            #print(func_values, "f(x)")
            penalty_list.append(pena)
            print(penalty_list)
            path_list.append(path)
            x_points.append(self.x[:]) #current Positions append in a list 
            updat_x, alpha_z,nabla, rejected, tried_alpha = self.step()
            #print(f"The current next step is: {updat_x}")
            alpha_list.append(alpha_z)
            stepz.append(updat_x)
            norm = np.linalg.norm(nabla)
            converlist.append(norm)
            #print(f"The current Alphaz list is: {alpha_list}")c
            #print(f'The gradient is: {grad}')
            #print(np.linalg.norm(nabla),"CG norm gradient")
            #print(grad_con, "This is the Gradient norm")
            #print( np.linalg.norm(gradient_list), "This is the gradietn list norm")
            #print(f'This is the Penalty For our CG: {penalty_list} ')
            #print(f"alpha that has been searched:  {alpha_tried}")
            #print(f"Alphat that been rejected: {alpha_rejected}")
            #print(f'The points we have benn: {x_points}')
            k +=1
            if np.linalg.norm(nabla) < ep: #Termination Condition
                print(f'The Gradient is: {nabla} and is absolute value is {np.linalg.norm(nabla)}, and stopped at: {self.x[k]} away from converging')
                break 
            
            #print(f"The iteration is: {k}") 
        return [x_points,updat_x,func_values, nabla,stepz,alpha_list,rejected, tried_alpha,penalty_list,path_list,smooth_list,converlist] #Return current position, alphas list, x position, function values
    