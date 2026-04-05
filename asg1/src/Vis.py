import numpy as np 
import matplotlib.pyplot as plt
import torch as tp
import math as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from F_L import f_L,gradientf_L
#from F_O import f_O, f_O_2,gradient_f_O_2,gradientf_O
#from smooth import smoothness_value, smoothness_residuals, gradient_smoothness, least_squares_func, build_D, flatten, unflatten
#from objective_func import *
import matplotlib.transforms as mtransforms
from strong_brack import strong_backtracking
from Bt_LineSearch import backtracking_line_search

#Visualising the bracketing
def Bplot_the_search(f_alpha, x, d, alphas):
    alpha_samples = np.arange(0, 4., 0.01)
    y_values = [f(x + a * d) for a in alphas]
    plt.plot(alpha_samples, f_alpha(alpha_samples,x,d), '-')
    
    plt.plot(alphas, y_values, 'o')

    #plt.plot(alphas, y_values, 'o')
    #plt.text(x, y, '%d, %d' % (int(x), int(y)),
    trans_offset = mtransforms.offset_copy(plt.gca().transData,  fig=plt.gcf(),
                                        x=0.05, y=0.10, units='inches')
    for i in range(len(alphas)):
        plt.text(alphas[i], f_alpha(alphas[i],x,d), '%d' % i,
        transform=trans_offset,
        fontsize=10, verticalalignment='bottom', horizontalalignment='right')
    plt.axis((0, 4, -15, 50))
    plt.show()


f = lambda x: x[0]**4-5*x[0]**2+x[1]**2
nabla = lambda x: np.array([4*x[0]**3-10*x[0],2*x[1]])

f_alpha = lambda alpha,x,d: (x[0]+alpha*d[0])**4-5*(x[0]+alpha*d[0])**2+\
                            (x[1]+alpha*d[1])**2
x = np.array([0,1])
d = np.array([1,-1])


best_alpha = backtracking_line_search(f, nabla, x, d)
#all_points = bracket_points + zoom_points + [best_alpha]
Bplot_the_search(f_alpha, x, d)



#Visualising the strong bracketing
def SBplot_the_search(f_alpha, x, d, alphas):
    alpha_samples = np.arange(0, 4., 0.01)
    y_values = [f(x + a * d) for a in alphas]
    plt.plot(alpha_samples, f_alpha(alpha_samples,x,d), '-')
    
    plt.plot(alphas, y_values, 'o')

    #plt.plot(alphas, y_values, 'o')
    #plt.text(x, y, '%d, %d' % (int(x), int(y)),
    trans_offset = mtransforms.offset_copy(plt.gca().transData,  fig=plt.gcf(),
                                        x=0.05, y=0.10, units='inches')
    for i in range(len(alphas)):
        plt.text(alphas[i], f_alpha(alphas[i],x,d), '%d' % i,
        transform=trans_offset,
        fontsize=10, verticalalignment='bottom', horizontalalignment='right')
    plt.axis((0, 4, -15, 50))
    plt.show()


f = lambda x: x[0]**4-5*x[0]**2+x[1]**2
nabla = lambda x: np.array([4*x[0]**3-10*x[0],2*x[1]])

f_alpha = lambda alpha,x,d: (x[0]+alpha*d[0])**4-5*(x[0]+alpha*d[0])**2+\
                            (x[1]+alpha*d[1])**2
x = np.array([0,1])
d = np.array([1,-1])

best_alpha, zoom_points, bracket_points = strong_backtracking(f, nabla, x, d)
all_points = bracket_points + zoom_points + [best_alpha]
SBplot_the_search(f_alpha, x, d, all_points)

#VISUALISING THE Conjugate gradient