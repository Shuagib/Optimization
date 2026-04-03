import numpy as np 
import matplotlib.pyplot as plt
import torch as tp
import math as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from F_L import f_L,gradientf_L
from F_O import f_O, f_O_2,gradient_f_O_2,gradientf_O
from smooth import smoothness_value, smoothness_residuals, gradient_smoothness, least_squares_func, build_D, flatten, unflatten
from objective_func import *
