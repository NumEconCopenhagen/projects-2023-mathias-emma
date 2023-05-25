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
par.R=(1+0.01)^(1/12)


# Q2.1
def profit(kappa, l):
    return kappa * l ** (1 - eta) - w * l

def optimal_l(kappa):
    return ((1 - eta) * kappa / w) ** (1 / eta)


