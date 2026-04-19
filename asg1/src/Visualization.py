import numpy as np 
import matplotlib.pyplot as plt
from F_S import *
from F_L import *
from F_O import *
from GD import *
from Nelder_mead import *
from CG import * 

#Choose start and Goal
start_point  = (0.5,0.0)
end_point = (19.0,22.0)
ob_main = [((16.0, 19.0), 3), (( 6.0 , 7.0), 3)]

start_point_x, start_point_y = start_point
end_point_x, end_point_y = end_point

#Amount of Points 
N_amount = 20



#Creaing x and y axis
x_axis = np.linspace(start_point_x, end_point_x, N_amount)
y_axis = np.linspace(start_point_y, end_point_y, N_amount)

l = 2 #Smoothness
m = 15 #Penalty
D_matrix = build_D(N_amount)
alpha0 = 0.5

x_start = np.array([start_point_x, start_point_y])
x_goal = np.array([end_point_x, end_point_y])


initial_path = np.column_stack((x_axis, y_axis))
trajectory_path = flatten(initial_path)
# noise = ra.normal(0.01, size = initial_path.shape)
# noise[0] = [0.0,0.0]
# noise[-1] = [0.0,0.0]
# initial_path += noise

#Fixing porblem with Noise problems 



def plotting_pathevolution(path_evolution):
    """Visualizes the step-by-step progress of the path optimization.."""
    #Plotting the Evolutions pr. iteration. Iteration = N_Amount
    iteration_pr_plot = [0, 1, 2,3,4,5,6,7,8] # The first 9 plots
    _, axes = plt.subplots(3, 3, figsize=(15, 5)) #Creating 3 rows and 3 coloums 3x3 = 9 pictures 


    for id, i in enumerate(iteration_pr_plot):
        row = id//3 #Using 3 coloums  each row []. id  = 0, 0//3 = 0 row[0], id = 1, 1//3 = 1
        col = id % 3  #Using 3 Coloums each col [] 
        path_evalutions = unflatten(path_evolution[i], N_amount, x_start, x_goal)
        #Ends with a plot: axes[0,0]  axes[0,1]  axes[0,2]  axes[0,3] axes[1,0]  axes[1,1]  axes[1,2]  axes[1,3] ...

        xes = path_evalutions[:,0] #Getting the xis of the vector
        yis = path_evalutions[:,1] #Getting the yis of the vector
     
        axes[row,col].plot(xes, yis,color="orange",marker='o')
        axes[row,col].set_xlim(0, 25)
        axes[row,col].set_ylim(0, 25)
        axes[row,col].add_patch(plt.Circle((16.0, 19.0), 3, color='darkorange', alpha=0.5))
        axes[row,col].add_patch(plt.Circle(( 6.0 , 7.0), 3, color='darkorange', alpha=0.5))
        axes[row,col].plot(start_point_x, start_point_y, 's', color='black')
        axes[row,col].plot(end_point_x, end_point_y, 's', color='black')
        axes[row,col].set_aspect('equal')
        axes[row,col].legend()
     
    plt.tight_layout()
    plt.show()


  



def plotting_Gradient_Descent(optimal_path,gradientz,func_v,alphz):
    """ Plots the final results and convergence diagnostics for Gradient Descent."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    get_path = unflatten(optimal_path, N_amount, x_start, x_goal)
    xes = get_path[:,0]
    yis = get_path[:,1]
    axes[0].plot(xes, yis,color="orange",marker='o')
    axes[0].add_patch(plt.Circle((16.0, 19.0), 3, color='darkorange', alpha=0.5))
    axes[0].add_patch(plt.Circle(( 6.0 , 7.0), 3, color='darkorange', alpha=0.5))
    axes[0].plot(start_point_x, start_point_y, 's', color='black', markersize=10, label='start')
    axes[0].plot(end_point_x, end_point_y, 's', color='black', markersize=10, label='goal')
    axes[0].set_aspect('equal')
    axes[0].legend()
    axes[0].set_title(' Optimal path Gradient Descent')

    #Convergence of gradient magnitude
    axes[1].plot(gradientz,color='blue')
    axes[1].set_xlabel('Points at each gradient',fontsize='10')
    axes[1].set_ylabel('Iterations',fontsize='10')
    axes[1].set_title('Magnitude of Change',fontsize='10')

    # Convergence of function evalutions 
    axes[2].plot(func_v)
    axes[2].set_xlabel('Iteration',fontsize='10')
    axes[2].set_ylabel('Evaluations',fontsize='10')
    axes[2].set_title('function values Gradient Descent', fontsize = '10')
    axes[2].grid()

    #Plotting the Alpha's to show that they varies, Why does my alpha intialstart do 
    axes[3].plot(alphz)
    axes[3].set_title('Brackting Line Search Alpha',fontsize='10')
    axes[3].set_xlabel('Alpha search',fontsize='10')
    axes[3].set_ylabel('Alphas',fontsize='10')

    plt.tight_layout()
    plt.show()



# Plotting Conjugate Gradient 


def plotting_CG_Path(optimal_path,func_v,gradlist,alpha_z,alpha_rejected, alpha_tried_alpha):
    """Plots the final results and diagnostics for Conjugate Gradient optimization."""

    _, axes = plt.subplots(1, 4, figsize=(15, 5))
   
    get_path = unflatten(optimal_path, N_amount, x_start, x_goal)
    xes = get_path[:,0]
    yis = get_path[:,1]
    axes[0].plot(xes, yis,color="orange",marker='o')
    axes[0].add_patch(plt.Circle((16.0, 19.0), 3, color='darkorange', alpha=0.5))
    axes[0].add_patch(plt.Circle(( 6.0 , 7.0), 3, color='darkorange', alpha=0.5))
    axes[0].plot(start_point_x, start_point_y, 's', color='black', markersize=10, label='start')
    axes[0].plot(end_point_x, end_point_y, 's', color='black', markersize=10, label='goal')
    axes[0].set_aspect('equal')
    axes[0].legend()
    axes[0].set_title('Optimal Path Conjugate Gradient',fontsize = '10')

    # Convergence of gradient magnitude
    axes[1].plot(gradlist,color='blue')
    axes[1].set_xlabel('Iterations',fontsize='10')
    axes[1].set_ylabel('Gradient Evaluation',fontsize='10')
    axes[1].set_title('Magnitude of Change',fontsize='10')

    # Convergence of function evalutions 
    axes[2].plot(func_v)
    axes[2].set_xlabel('Iteration',fontsize='10')
    axes[2].set_ylabel('Evaluations',fontsize='10')
    axes[2].set_title('Function values Conjugate Gradient',fontsize='10')
    axes[2].grid()

    #Plotting the Alpha's to show that they varies, Why does my alpha intialstart do 
    axes[3].plot(alpha_z)
    axes[3].plot(alpha_rejected)
    axes[3].plot(alpha_tried_alpha)
    axes[3].set_title('BLS with Strong wolf conditions ',fontsize='10')
    axes[3].set_xlabel("Alpha ",fontsize='10')
    axes[3].set_ylabel("Strong Bracketing Line Search Alpha",fontsize='10')
    plt.tight_layout()
    plt.show()

    

# Creating a plot for the penalty function and path functions

def plot_convergence(penal_GD,Path_GD,Penal_CG,path_GC,N,SmGD,SmCg,gradientCG,gradientGD):
    """Provides comparison between GD and CG solvers."""
    fig, ax = plt.subplots(1,4,figsize=(15, 5))
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
    
    ax[2].plot(x_axis,SmGD,color = 'darkorange',label='GD')
    ax[2].plot(x_axis,SmCg, color = 'green',label='GD')
    ax[2].set_xlabel('Iterations')
    ax[2].set_ylabel('Smootness of trajectory')
    ax[2].set_title('Smootness Visulization: GD vs CG')


    ax[3].plot(x_axis,gradientCG,color = 'darkorange',label='GD',marker='.',linestyle = 'dashdot')
    ax[3].plot(x_axis,gradientGD, color = 'green',label='GD',marker ='.',linestyle = 'dashdot')
    ax[3].set_xlabel('Iterations')
    ax[3].set_ylabel('Convergence of Gradient')
    ax[3].set_title('Gradient Convergence: GD vs CG')
    
    plt.tight_layout()
    plt.show()


def plotting_nelder_evolution(path_history, N_amount, x_start, x_goal):
    """Visualizes the evolution of the Nelder-Mead simplex search."""
    max_idx = len(path_history) - 1
    iterations_to_plot = [1, 50, 100, 200, 300, 500, 700, 850, max_idx] 
    
    fig, axes = plt.subplots(3, 3, figsize=(8, 8)) 
    for id, i in enumerate(iterations_to_plot):

        idx = min(i, max_idx)
        row = id // 3
        col = id % 3
        ax = axes[row, col]

        path_coords = unflatten(path_history[idx], N_amount, x_start, x_goal)

        xes = path_coords[:, 0]
        yes = path_coords[:, 1]
     
        ax.plot(xes, yes, color="orange", marker='o', markersize=3, label=f"Iter {idx}")
      
        # Add Obstacles
        ax.add_patch(plt.Circle((16.0, 19.0), 3.0, color='darkorange'))
        ax.add_patch(plt.Circle((6.0, 7.0), 3.0, color='darkorange'))
        
        # Start & end points
        ax.plot(x_start[0], x_start[1], 's', color='black', label="Start")
        ax.plot(x_goal[0], x_goal[1], 's', color='black', label="Goal")
        
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 25)
        ax.set_aspect('equal')
        ax.set_title(f"Iteration {idx}")
        ax.legend(prop={'size': 6})
     
    plt.tight_layout()
    plt.show()


optimizer = NMOptimizer(x_start, x_goal, ob_main, N_amount, l, m)
final_path, f_vals, path_evolution = optimizer.run(path, max_iter=1000)


def plotting_NM(optimal_path, f_history, path_history):
    """ Plots diagnostics for the Nelder-Mead (NM) optimization process."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
   
    # Plot the optimal path with obstacles etc.
    path_coords = optimal_path
    axes[0].plot(path_coords[:,0], path_coords[:,1], color="orange", marker='o', label='NM Path')
    axes[0].add_patch(plt.Circle((16.0, 19.0), 3.0, color='darkorange'))
    axes[0].add_patch(plt.Circle((6.0, 7.0), 3.0, color='darkorange'))
    axes[0].plot(x_start[0], x_start[1], 's', color='black', markersize=10)
    axes[0].plot(x_goal[0], x_goal[1], 's', color='black', markersize=10)
    axes[0].set_aspect('equal')
    axes[0].set_title('Optimal Path (Nelder-Mead)')
    axes[0].legend()

    # Convergence of Simplex 
    # Measuring how much the best path changed from the previous iteration
    path_changes = [np.linalg.norm(path_history[i] - path_history[i-1]) for i in range(1, len(path_history))]
    axes[1].plot(path_changes, color='blue')
    axes[1].set_yscale('log') 
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('Path Displacement')
    axes[1].set_title('Step Magnitude (Simplex Movement)')

    # Convergence of Function Values
    axes[2].plot(f_history, color='green')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Cost (Objective Value)')
    axes[2].set_title('Function Value Convergence')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


nm_penalties = []
nm_smoothness = []
nm_path_lens = []

for p_flat in path_evolution:
    p_coords = unflatten(p_flat, N_amount, x_start, x_goal)
    nm_penalties.append(f_O_2(p_coords, ob_main, alpha=0.5))
    nm_smoothness.append(np.sum(smoothness_residuals(p_flat, N_amount, x_start, x_goal, D_matrix)**2))
    nm_path_lens.append(func_L(p_coords))

def plot_nm_diagnostics(path_lens, penalties, smoothness, total_f):
    """ Deconstructs the Nelder-Mead objective function into its sub-components."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    iterations = np.arange(len(path_lens))

    # Path Length
    axes[0].plot(iterations, path_lens, color='royalblue', lw=2)
    axes[0].set_title("Path Length ($L$)")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Length")

    # Penalty Function
    axes[1].plot(iterations, penalties, color='crimson', lw=2)
    axes[1].set_title("Penalty Cost ($f_O$)")
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Penalty")

    # Smoothness
    axes[2].plot(iterations, smoothness, color='darkorange', lw=2)
    axes[2].set_title("Smoothness ($S$)")
    axes[2].set_xlabel("Iterations")

    # Total Convergence (Objective Function)
    axes[3].plot(iterations, total_f, color='black', lw=2, linestyle='--')
    axes[3].set_yscale('log') # Log scale helps see small improvements
    axes[3].set_title("Total Objective (Log Scale)")
    axes[3].set_xlabel("Iterations")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# #Starting all the plotting

#Conjugate Gradient
trav_path,optimal_path_CG,funcv,gradient_CG,stepx,alpha_list,rejected_list,alpha_tried_list,pena_list_CG,path_list_CG, smc, converlist = Conjugate_Gradient(trajectory_path, alpha0, l, m, ob_main, N_amount, D_matrix, x_start, x_goal).opt(N_amount)

#Gradient Descent
trav_x, optimal_x, f_values, alphz, stepz, gradlist, pathlist_GD, penlist_GD, SMG,converlistG = GradientDescent(trajectory_path, alpha0, l, m, ob_main, N_amount, D_matrix, x_start, x_goal).opt(N_amount)




plotting_pathevolution(trav_x)


plotting_pathevolution(trav_path)



plotting_CG_Path(optimal_path_CG,funcv,gradient_CG,alpha_list,rejected_list,alpha_tried_list)



plotting_Gradient_Descent(optimal_x,gradlist,f_values,alphz)



path_GradientDescent = pathlist_GD
path_ConjugateGradient = path_list_CG
penalty_GradientDescent = penlist_GD
Penalty_ConjugateGradient = pena_list_CG
smoothnessG = SMG
SmootnnessC = smc   

plot_convergence(penalty_GradientDescent,path_GradientDescent,Penalty_ConjugateGradient,path_ConjugateGradient,N_amount,smoothnessG,SmootnnessC,converlist,converlistG)

# Nelder mead plot
plotting_nelder_evolution(path_evolution, N_amount, x_start, x_goal)

plotting_NM(final_path, f_vals, path_evolution)

plot_nm_diagnostics(nm_path_lens, nm_penalties, nm_smoothness, f_vals)
