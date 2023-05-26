import numpy as np
import random
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy.optimize import minimize
import time

def griewank(x):
    return griewank_(x[0],x[1])
    
def griewank_(x1,x2):
    A = x1**2/4000 + x2**2/4000
    B = np.cos(x1/np.sqrt(1))*np.cos(x2/np.sqrt(2))
    return A-B+1


# Function for finding the global optimum
def global_opt(par_3, warmup_iters, do_print=False, do_plot=False, time_speed = False, count_it = False):
    """This function finds the global optimum of the Griewank function using the BFGS algorithm and multi-starts.

    Args:
        par_3 (SimpleNamespace): Namespace containing the parameters for the algorithm.
        warmup_iters (int): Number of iterations before the algorithm starts to update x_k0.

    Returns:
        if print_mean (bool)
            prints the optimal x1 and x2 values, the number of iterations and the time it took to find the global optimum.
        
        if do_plot (bool)
            Plots the effective initial guesses against the iteration counter.
        
        if time_speed (bool)
            Returns the time it took to find the global optimum, the number of iterations and the distans to x_true.
        
        if count_it (bool)
            Returns the number of iterations and the distans to x_true. x_star (list): List of optimal x1 and x2 values.

    """
    t0 = time.time()
    x_star = None
    x_k0_values = []
    iter_counter = []
    # set seed
    np.random.seed(par_3.seed_)

    # Refined global optimizer
    for k in range(par_3.max_iters):
        
        # A. Draw random x^k 
        x_k = np.random.uniform(par_3.bounds[0][0], par_3.bounds[0][1], 2)

        # Saving the best warmup itteration/ inital guess
        if k == (warmup_iters+1):
            x_first_gues = x_star

        # B. Seting random draw as x_k0 for first warmup_iters iterations
        if k < warmup_iters:
            x_k0 = x_k

        else:
        # C. Calculating chi_k
            chi_k = 0.5 * (2 / (1 + np.exp((k - warmup_iters) / 100)))

        #D. Calculating x_k0
            x_k0 = chi_k * x_k + (1 - chi_k) * x_star

        # Save the inital guesss to be used in the optimizer 
        iter_counter.append(k -1)
        x_k0_values.append(x_k0)

        # E. Optimize using BFGS
        result = minimize(griewank, x_k0, method='BFGS', tol=par_3.tau)

        # F. Update x_star
        if x_star is None or griewank(result.x) < griewank(x_star):
            x_star = result.x

        # G. Break if tau is reached
        if griewank(x_star) < par_3.tau:
            break
        t1 = time.time()

    # Print results
    if do_print == True:
        print(f'Global optimum using {warmup_iters} warmup iterations:')
        print(f"x1 = :{x_star[0]:.3f}, x2 = {x_star[1]:.3f}")
        print("Iteration: ", k)
        print(f"After: {t1-t0:.2f} seconds\n")

    # Plot results
    if do_plot == True:
        plot_q3(iter_counter, x_k0_values, warmup_iters)
        
    # Return time and iteration counter
    if time_speed == True and count_it == True:
        #Distans from x_first_gues sum of abs value
        distans_to_xtrue = np.sum(np.abs(x_first_gues)) 

        return t1-t0, k, distans_to_xtrue
  



# Funciton for plotting the results used in global_opt as an optional argument
def plot_q3(iter_counter, x_k0_values, warmup_iters):

    """This function plots the results from the global_opt function.

    Args:
        iter_counter (list): List of iterations.
        x_k0_values (list): List of x_k0 values.
        warmup_iters (int): Number of iterations before the algorithm starts to update x_k0.

    Returns:
        None

    """
    #Plot x_k0 values against iteration counter
    plt.figure(1, figsize=(10,6))
    plt.plot(iter_counter, x_k0_values, '.')
    plt.xlabel('Iteration counter k')
    plt.ylim([-600, 600])
    # plt.xlim([0, 300])
    plt.ylabel('Effective Initial Guesses x_k0')
    plt.title(f'Variation of effective initial guesses with iteration counter (warmup_iters = {warmup_iters})')
    plt.grid(True)



# Function for finding speed of algorithsm 

def speed_func(par_3, loops, print_mean=False,  plot= False, plot_distans_to_xtrue=False):

    """This function finds the speed of the algorithm 

    Args:
        par_3 (SimpleNamespace): Namespace containing the parameters for the algorithm.
        loops (int): Number of loops to run the algorithm.
    
    
    Returns:
        if print_mean (bool)
            prints the mean speed and iterations for n = 10 and n = 100.


        if plot (bool)
            Plots the speed and iterations for n = 10 and n = 100.

        if plot_distans_to_xtrue (bool)
            Plots the distans to x_true for n = 10 and n = 100.


    """
    # A empty lists to store the speed and iterations
    speed_10 = []
    speed_100 = []
    iterations_10 = []
    iterations_100 = []
    x_first_10 = []
    x_first_100 = []

    # Looping over the algorithm
    for i in range(1, loops):
        par_3.seed_ = i

        speed, iterations, x_first = global_opt(par_3, 10, time_speed= True, count_it = True)
        speed_10.append(speed)
        iterations_10.append(iterations)
        x_first_10.append(x_first)

        speed, iterations, x_first = global_opt(par_3, 100, time_speed= True, count_it = True)
        speed_100.append(speed)
        iterations_100.append(iterations)
        x_first_100.append(x_first)

    mean_speed_10 = np.mean(speed_10)
    mean_speed_100 = np.mean(speed_100)
    mean_iterations_10 = np.mean(iterations_10)
    mean_iterations_100 = np.mean(iterations_100)

    
    if print_mean == True:
        print(f"Mean speed for n = 10: {mean_speed_10:.2f} seconds")
        print(f"Mean speed for n = 100: {mean_speed_100:.2f} seconds")
        print(f"Mean iterations for n = 10: {mean_iterations_10:.2f}")
        print(f"Mean iterations for n = 100: {mean_iterations_100:.2f}")
    

    if plot == True:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].hist(speed_10, bins=20, alpha=0.5, label='Warmup iterations = 10', color='r')
        axs[0].hist(speed_100, bins=20, alpha=0.5, label='Warmup iterations = 100', color='b')
        axs[0].axvline(x=mean_speed_10, color='r', linestyle='dashed', linewidth=1, label='Mean speed (n=10)')
        axs[0].axvline(x=mean_speed_100, color='b', linestyle='dashed', linewidth=1, label='Mean speed (n=100)')
        #axs[0].legend(loc='upper right')
        axs[0].set_xlabel('Seconds')
        axs[0].set_ylabel('Density')
        axs[0].set_title('Speed')

        axs[1].hist(iterations_10, bins=20, alpha=0.5, label='Warmup iterations = 10', color='r')
        axs[1].hist(iterations_100, bins=20, alpha=0.5, label='Warmup iterations = 100', color='b')
        axs[1].axvline(x=mean_iterations_10, color='r', linestyle='dashed', linewidth=1, label='Mean iterations (n=10)')
        axs[1].axvline(x=mean_iterations_100, color='b', linestyle='dashed', linewidth=1, label='Mean iterations (n=100)')
        axs[1].legend(loc='upper right')
        axs[1].set_xlabel('Iterations')
        axs[1].set_ylabel('Density')
        axs[1].set_title('Number of Iterations')

        plt.tight_layout()
        plt.show()

    if plot_distans_to_xtrue == True:

        # Scatter plot of the sum of the absalut values of x_first_10[0] and x_first_10[0] against the iteration counter
        plt.figure(figsize=(10, 6))
        plt.scatter(iterations_10, x_first_10, label='n = 10', color='r')
        plt.scatter(iterations_100, x_first_100, label='n = 100', color='b')
        plt.xlabel('Iterations')
        plt.ylabel('Sum of absalut values of x')
        plt.title('Effect of warmup iterations on the iteration counter ')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()
