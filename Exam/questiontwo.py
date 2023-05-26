import numpy as np
import random
import matplotlib.pyplot as plt
from types import SimpleNamespace

# Question 2

# Setting up a simple namespace for the parameters in Question 2
par = SimpleNamespace()
par.eta = 0.5
par.w = 1
par.rho = 0.9
par.iota = 0.01
par.sigma = 0.1
par.R=(1+0.01)**(1/12)
par.kappa_discrete = [1.0, 2.0]


# Q2.1
def profit(par,l):
    return kappa_discrete * l ** (1 - par.eta) - par.w * l

def optimal_l(par):
    return ((1 - par.eta) * par.kappa_discrete / par.w) ** (1 / par.eta)


