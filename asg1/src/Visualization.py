import numpy as np 
import matplotlib.pyplot as plt
from smooth import *
from F_L import *
from F_O import *
from GD import *
from CG import * 
from quasi import *
#Choose start and Goal
start_point  = (0.5,0.0)
end_point = (19.0,22.0)


start_point_x, start_point_y = start_point
end_point_x, end_point_y = end_point

#Amount of Points 
N_amount = 20



#Creaing x and y axis
x_axis = np.linspace(start_point_x, end_point_x, N_amount)
y_axis = np.linspace(start_point_y, end_point_y, N_amount)

### Let's visualize it 
l = 0.1 #Smoothness
m = 0.5 #Penalty
D_matrix = build_D(N_amount)
alpha0 = 0.5

x_start = np.array([start_point_x, start_point_y])
x_goal = np.array([end_point_x, end_point_y])

initial_path = np.column_stack((x_axis, y_axis))
trajectory_path = flatten(initial_path)
ob_main = [((16.0, 19.0), 3), ((6.0, 7.0), 3)]


#print(len(travel_x),"The rigt Amount is:")
#print(f_value, "The right amount is " ) 
#print(len(alphz), "")


def plotting_Gradient_Descent(path_evolution,optimal_path,gradientz,func_v,alphz):
    #Plotting the Evolutions pr. iteration. Iteration = N_Amount
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
  
    for i in range(len(path_evolution)):
        last_run = unflatten(path_evolution[0], N_amount, x_start, x_goal)
        xes = last_run[:,0]
        yis = last_run[:,1]
        axes[0].plot(xes, yis,color="orange",marker='o')

    axes[0].add_patch(plt.Circle((16.0, 19.0), 3, color='darkorange', alpha=0.5))
    axes[0].add_patch(plt.Circle((6.0, 7.0), 3, color='darkorange', alpha=0.5))
    axes[0].plot(start_point_x, start_point_y, 's', color='black', markersize=10, label='start')
    axes[0].plot(end_point_x, end_point_y, 's', color='black', markersize=10, label='goal')
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].legend()
    axes[0].set_title('Path Evolution Gradient Descent')
  



 

    get_path = unflatten(optimal_path, N_amount, x_start, x_goal)
    xes = get_path[:,0]
    yis = get_path[:,1]
    axes[1].plot(xes, yis,color="orange",marker='o')
    axes[1].add_patch(plt.Circle((16.0, 19.0), 3, color='darkorange', alpha=0.5))
    axes[1].add_patch(plt.Circle((6.0, 7.0), 3, color='darkorange', alpha=0.5))
    axes[1].plot(start_point_x, start_point_y, 's', color='black', markersize=10, label='start')
    axes[1].plot(end_point_x, end_point_y, 's', color='black', markersize=10, label='goal')
    axes[1].set_aspect('equal')
    axes[1].legend()
    axes[1].set_title(' Optimal path Gradient Descent')



    #Convergence of gradient magnitude
    axes[2].plot(gradientz,color='blue')
    axes[2].set_xlabel('Points at each gradient',fontsize='10')
    axes[2].set_ylabel('Iterations',fontsize='10')
    axes[2].set_title('Magnitude of Change',fontsize='10')
 



    # Convergence of function evalutions 
    axes[3].plot(func_v)
    axes[3].set_title('function values Gradient Descent', fontsize = '10')
    axes[3].set_xlabel('Iteration',fontsize='10')
    axes[3].set_ylabel('Evaluations',fontsize='10')

    axes[3].grid()

    #Plotting the Alpha's to show that they varies, Why does my alpha intialstart do 
    axes[4].plot(alphz)
    axes[4].set_title('Brackting Line Search Alpha',fontsize='10')
    axes[4].set_xlabel('Alpha search',fontsize='10')
    axes[4].set_ylabel('Alphas',fontsize='10')

    # axes[5].plot(path)
    # axes[5].set_title('Path Lenght Evolutions',fontsize='10')
    # axes[5].set_xlabel('Iterations',fontsize='10')
    # axes[5].set_ylabel('Lenght of Path ',fontsize='10')

    # axes[6].plot(penalty)
    # axes[6].set_title('Penalty Function',fontsize='10')
    # axes[6].set_xlabel('Iterations',fontsize='10')
    # axes[6].set_ylabel('Cost ',fontsize='10')


    plt.tight_layout()
    plt.show()



#print(len(x_points), "This is the lenght")
#print(len(func_values), "This is the the amount of function evaluations")
#print(len(alpha_list), "This is the amount of alphas")


####Plotting Conjugate Gradient 



def plotting_CG_Path(optimal_path,path_ev,func_v,gradlist,alpha_z,alpha_rejected, alpha_tried_alpha):
    _, axes = plt.subplots(1, 5, figsize=(15, 5))
    for run in range(len(path_ev)):
        last_run = unflatten(path_ev[0], N_amount, x_start, x_goal)
        xes = last_run[:,0]
        yis = last_run[:,1]
        axes[0].plot(xes, yis,color="orange",marker='o')
    axes[0].add_patch(plt.Circle((16.0, 19.0), 3, color='darkorange', alpha=0.5))
    axes[0].add_patch(plt.Circle((6.0, 7.0), 3, color='darkorange', alpha=0.5))
    axes[0].plot(start_point_x, start_point_y, 's', color='black', markersize=5, label='start')
    axes[0].plot(end_point_x, end_point_y, 's', color='black', markersize=5, label='goal')
    axes[0].set_title('Path Evolution Conjugate Gradient',fontsize = '10')
    axes[0].set_aspect('equal')
    axes[0].legend()

    

 

    get_path = unflatten(optimal_path, N_amount, x_start, x_goal)
    xes = get_path[:,0]
    yis = get_path[:,1]
    axes[1].plot(xes, yis,color="orange",marker='o')
    axes[1].add_patch(plt.Circle((16.0, 19.0), 3, color='darkorange', alpha=0.5))
    axes[1].add_patch(plt.Circle((6.0, 7.0), 3, color='darkorange', alpha=0.5))
    axes[1].plot(start_point_x, start_point_y, 's', color='black', markersize=10, label='start')
    axes[1].plot(end_point_x, end_point_y, 's', color='black', markersize=10, label='goal')
    axes[1].set_aspect('equal')
    axes[1].legend()
    axes[1].set_title('Optimal Path Conjugate Gradient',fontsize = '10')



    #Convergence of gradient magnitude
    axes[2].plot(gradlist,color='blue')
    axes[2].set_xlabel('Iterations',fontsize='10')
    axes[2].set_ylabel('Gradient Evaluation',fontsize='10')
    axes[2].set_title('Magnitude of Change',fontsize='10')

 



    # Convergence of function evalutions 
    axes[3].plot(func_v)
    axes[3].set_xlabel('Iteration',fontsize='10')
    axes[3].set_ylabel('Evaluations',fontsize='10')
    axes[3].set_title('Function values Conjugate Gradient',fontsize='10')

 
    axes[3].grid()

    #Plotting the Alpha's to show that they varies, Why does my alpha intialstart do 
    axes[4].plot(alpha_z)
    axes[4].plot(alpha_rejected)
    axes[4].plot(alpha_tried_alpha)
    axes[4].set_title('BLS with Strong wolf conditions ',fontsize='10')
    axes[4].set_xlabel("Alpha ",fontsize='10')
    axes[4].set_ylabel("Strong Bracketing Line Search Alpha",fontsize='10')
    plt.tight_layout()
    plt.show()

    



# Creating a plot for the penalty function and path functions

def plot_convergence(penal_GD,Path_GD,Penal_CG,path_GC,N,SmGD,SmCg):
    fig, ax = plt.subplots(1,3,figsize=(15, 5))
    x_axis = [i for i in range(0,N)]
    y_CG_path = path_GC
    y_CG_cost = Penal_CG
    y_GD_path = Path_GD
    y_GD_cost= penal_GD

 
    #Plooting path function
    ax[0].plot(x_axis,y_CG_path, color = 'crimson',label='CD',linestyle = ':')
    ax[0].plot(x_axis,y_GD_path, color = 'royalblue',label='GD',linestyle = ':')
    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel("Path lenght")
    ax[0].set_title("Path Function: GD vs CG")
  
    ax[0].grid()

    #Plotting Penalty cost 
    ax[1].plot(x_axis,y_CG_cost,color = 'olivedrab',label='GD',linestyle = '--')
    ax[1].plot(x_axis,y_GD_cost, color = 'orange',label='GD', linestyle = '--')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Penalty cost')
    ax[1].set_title('Penalty function (1): GD vs CG')

    ax[2].plot(x_axis,SmGD,color = 'darkorange',label='GD',marker='.',linestyle = 'dashdot')
    ax[2].plot(x_axis,SmCg, color = 'green',label='GD',marker ='.',linestyle = 'dashdot')
    ax[2].set_xlabel('Iterations')
    ax[2].set_ylabel('Smootness of trajectory')
    ax[2].set_title('Smootness: GD vs CG')
   
    
   
    
    plt.tight_layout()
    plt.show()





# #Starting all the plotting

# #Intializing Conjugate Gradient

# trav_path,optimal_path,funcv,grad_list,stepx,alpha_list,rejected_list,alpha_tried_list,pena_list_CG,path_list_CG, SMC = Conjugate_Gradient(trajectory_path, alpha0, l, m, ob_main, N_amount, D_matrix, x_start, x_goal).opt(N_amount)

Quasi_NewtonMethod(trajectory_path,x_start,x_goal,alpha0,l,m,ob_main,D_matrix,N_amount).opt_BFGS(N_amount)

# #plotting_CG_Path(optimal_path,trav_path,funcv,grad_list,alpha_list,rejected_list,alpha_tried_list)


# trav_x, optimal_x, f_values, alphz, stepz, gradlist, pathlist_GD, penlist_GD, SMG = GradientDescent(trajectory_path, alpha0, l, m, ob_main, N_amount, D_matrix, x_start, x_goal).opt(N_amount)

# plotting_Gradient_Descent(trav_x,optimal_x,gradlist,f_values,alphz)

# path_GradientDescent = pathlist_GD
# penalty_GradientDescent = penlist_GD
# path_ConjugateGradient = path_list_CG
# Penalty_ConjugateGradient = pena_list_CG
# smoothnessG = SMG
# SmootnnessC = SMC   

# plot_convergence(penalty_GradientDescent,path_GradientDescent,Penalty_ConjugateGradient,path_ConjugateGradient,N_amount,SMG,SMC)







# #Plot penalty cost different penalty functions , and Convergence by the gradient use lin.norm using the 

















# Result of showing Good Convergence between gradients
""" 
4906.5685809566 GD norm gradient
362.7865027335331 GD norm gradient
981.4588931584686 GD norm gradient
717.9531696982509 GD norm gradient
357.71168318304376 GD norm gradient
216.46276378361497 GD norm gradient
355.2729154175717 GD norm gradient
214.52028637590968 GD norm gradient
136.7283371589387 GD norm gradient
94.24925781426217 GD norm gradient
166.06006635236471 GD norm gradient
107.71307992641228 GD norm gradient
75.40302832288292 GD norm gradient
134.60841229077556 GD norm gradient
88.39043849842446 GD norm gradient
62.787679246022826 GD norm gradient
111.02010970853257 GD norm gradient
73.73849515062993 GD norm gradient
53.20386815469168 GD norm gradient
92.49198411361415 GD norm gradient
62.192249573998495 GD norm gradient
45.65375411264919 GD norm gradient
77.65729317756673 GD norm gradient
52.9447142721099 GD norm gradient
39.60967049825252 GD norm gradient
65.65495821586282 GD norm gradient
45.46852763980034 GD norm gradient
34.72147697950438 GD norm gradient
55.87649606073746 GD norm gradient
39.38432995658376 GD norm gradient
72.090069205749 GD norm gradient
47.86910728366489 GD norm gradient
34.407031357702294 GD norm gradient
61.30176887107891 GD norm gradient
41.28620691757128 GD norm gradient
30.316819329374443 GD norm gradient
52.38667663364744 GD norm gradient
35.85729896551197 GD norm gradient
26.941172161772887 GD norm gradient
44.99896129837777 GD norm gradient
31.368323427376897 GD norm gradient
24.143007240082056 GD norm gradient
38.863668280853055 GD norm gradient
27.647790933289404 GD norm gradient
50.40056611023058 GD norm gradient
33.75991150487991 GD norm gradient
24.556916878319402 GD norm gradient
43.348963583463814 GD norm gradient
29.508607738514012 GD norm gradient
21.982483071227787 GD norm gradient
4906.5685809566 CG norm gradient
The iteration is: 1
362.7865027335331 CG norm gradient
The iteration is: 2
341.4594931330908 CG norm gradient
The iteration is: 3
314.9653385287765 CG norm gradient
The iteration is: 4
237.35599776611772 CG norm gradient
The iteration is: 5
214.6781433377702 CG norm gradient
The iteration is: 6
131.59904912437756 CG norm gradient
The iteration is: 7
127.3668117365286 CG norm gradient
The iteration is: 8
107.89809522283493 CG norm gradient
The iteration is: 9
75.52157432305368 CG norm gradient
The iteration is: 10
77.13920591321468 CG norm gradient
The iteration is: 11
71.25095926505401 CG norm gradient
The iteration is: 12
61.79054417640815 CG norm gradient
The iteration is: 13
54.7429077595277 CG norm gradient
The iteration is: 14
60.078205767067004 CG norm gradient
The iteration is: 15
50.6673442580725 CG norm gradient
The iteration is: 16
44.920517596260986 CG norm gradient
The iteration is: 17
41.9319413370228 CG norm gradient
The iteration is: 18
31.33283137904136 CG norm gradient
The iteration is: 19
31.211724277746672 CG norm gradient
The iteration is: 20
35.511803922854206 CG norm gradient
The iteration is: 21
25.147722318654722 CG norm gradient
The iteration is: 22
22.48933123538054 CG norm gradient
The iteration is: 23
20.73780900393269 CG norm gradient
The iteration is: 24
21.658801901288708 CG norm gradient
The iteration is: 25
18.51772360078978 CG norm gradient
The iteration is: 26
15.37678468191944 CG norm gradient
The iteration is: 27
14.281234406925776 CG norm gradient
The iteration is: 28
13.30509117134498 CG norm gradient
The iteration is: 29
11.683336155715065 CG norm gradient
The iteration is: 30
13.159679667123692 CG norm gradient
The iteration is: 31
13.470101281554372 CG norm gradient
The iteration is: 32
12.689487241406727 CG norm gradient
The iteration is: 33
14.195758544159311 CG norm gradient
The iteration is: 34
14.611984611435652 CG norm gradient
The iteration is: 35
14.408411079671025 CG norm gradient
The iteration is: 36
12.629670568167194 CG norm gradient
The iteration is: 37
13.541657570700444 CG norm gradient
The iteration is: 38
7.716488625096728 CG norm gradient
The iteration is: 39
7.249861542423023 CG norm gradient
The iteration is: 40
8.307541857305385 CG norm gradient
The iteration is: 41
10.14146861028916 CG norm gradient
The iteration is: 42
11.557091057709053 CG norm gradient
The iteration is: 43
11.557462646706266 CG norm gradient
The iteration is: 44
7.838070154741479 CG norm gradient
The iteration is: 45
5.095503852746266 CG norm gradient
The iteration is: 46
4.678211940451872 CG norm gradient
The iteration is: 47
5.202340313105237 CG norm gradient
The iteration is: 48
5.66285421698645 CG norm gradient
The iteration is: 49
3.8640155693639375 CG norm gradient
The iteration is: 50
50 This is the lenght
50 This is the the amount of function evaluations
50 This is the amount of alphas """


""" Conjugate Gradient need a low penalty ( can come close without hitting obstacles), but need high penalty for smoothness. Why is that"""