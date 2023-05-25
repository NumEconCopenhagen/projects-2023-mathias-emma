import numpy as np
import random
import matplotlib.pyplot as plt
from types import SimpleNamespace

# Question 2

# Setting up a simple namespace for the parameters in Question 2
par = SimpleNamespace(**{'s_h':0.13, 's_k':0.25, 'g':0.016, 'n':0.014, 'alpha':1/3, 'phi':1/3, 'delta':0.02, 'tau':0.1, 'eta':0.05})

# Q2.1
def profit(kappa, l):
    return kappa * l ** (1 - eta) - w * l

def optimal_l(kappa):
    return ((1 - eta) * kappa / w) ** (1 / eta)


