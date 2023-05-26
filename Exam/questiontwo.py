import numpy as np
import random
import matplotlib.pyplot as plt
from types import SimpleNamespace

# Question 2

# Setting up a simple namespace for the parameters in Question 2
par = SimpleNamespace()
par.eta = 0.5 # fixed demand shock
par.w = 1 # fixed wage rate
par.rho = 0.9 # persistence of shocks in AR(1) process
par.iota = 0.01 # fixed adjustment cost of huring or firing 
par.sigma = 0.1 # shock parameter
par.R = (1 + 0.01) ** (1 / 12) # montly discount factor
par.K = 49 # number of simulations for approximation
par.T = 120 # 10 years, 120 months
par.Delta = 0.00 # change threshold

# Q2.1, functions for discrete kappa
def profit(par, kappa, l):
    """
    Calculate the profit given parameters kappa and labor input l.

    Parameters:
    par (namespace): Namespace object containing the model parameters.
    kappa (float): Value of kappa parameter.
    l (float): Labor input.

    Returns:
    float: Profit value.
    """
    return kappa * l ** (1 - par.eta) - par.w * l


def optimal_l(par, kappa):
    """
    Calculate the optimal labor input given parameter kappa.

    Parameters:
    par (namespace): Namespace object containing the model parameters.
    kappa (float): Value of kappa parameter.

    Returns:
    float: Optimal labor input.
    """
    return ((1 - par.eta) * kappa / par.w) ** (1 / par.eta)



# Q2.2-Q2.5, functions for continuous kappa
def generate_kappa(par, seed):
    """
    Generate the series of kappa_t values based on a given seed.

    Parameters:
    - par: Namespace object containing the model parameters
    - seed: Seed value for random number generation

    Returns:
    - kappa: Numpy array of kappa_t values
    """
    np.random.seed(seed)
    eps = np.random.normal(-0.5 * par.sigma**2, par.sigma, par.T)
    kappa = np.empty(par.T)
    kappa[0] = 1
    for t in range(1, par.T):
        kappa[t] = np.exp(par.rho * np.log(kappa[t-1]) + eps[t])
    return kappa


def calculate_lt(par, kappa):
    """
    Calculate the value of l_t based on the kappa_t values.

    Parameters:
    - par: Namespace object containing the model parameters
    - kappa: Numpy array of kappa_t values

    Returns:
    - lt_star: Numpy array of l_t values
    """
    l = np.empty(par.T)
    l[0] = 0
    l_star = np.empty(par.T)
    for t in range(1, par.T):
        l_star[t] = ((1 - par.eta) * kappa[t] / par.w)**(1 / par.eta)
        if np.abs(l[t-1] - l_star[t]) > par.Delta:
            l[t] = l_star[t]
        else:
            l[t] = l[t-1]    
    return l


def calculate_h(par, kappa, l):
    """
    Calculate the ex post value of h based on the kappa_t and l_t values.

    Parameters:
    - par: Namespace object containing the model parameters
    - kappa: Numpy array of kappa_t values
    - l: Numpy array of l_t values

    Returns:
    - h: Ex post value of h
    """
    h = 0
    for t in range(par.T):
        h += par.R**(-t) * (kappa[t] * l[t]**(1 - par.eta) - par.w * l[t] - int(l[t] != l[t - 1]) * par.iota)
    return h


def calculate_h_extention(par, kappa, l):
    """
    Calculate the ex post value of h based on the kappa_t and l_t values.

    Parameters:
    - par: Namespace object containing the model parameters
    - kappa: Numpy array of kappa_t values
    - l: Numpy array of l_t values

    Returns:
    - h: Ex post value of h using the new policy with part time workers
    """
    h = 0
    for t in range(par.T):
        h += par.R**(-t) * (kappa[t] * l[t]**(1 - par.eta) - par.w * l[t] - (l[t] - l[t - 1]) * par.iota)
    return h


def function_call(par, seed, extention=False):
    """
    Calculate the value of h given the parameters and a seed through kappa and lt.

    Args:
        par (types.SimpleNamespace): Namespace containing the model parameters.
        seed (int): Seed value for generating kappa.

    Returns:
        float: Value of h.

    """
    kappa = generate_kappa(par, seed)
    l = calculate_lt(par, kappa)
    
    if extention==True:
        h = calculate_h_extention(par, kappa, l)
    if extention==False:
        h = calculate_h(par, kappa, l)
    return h


