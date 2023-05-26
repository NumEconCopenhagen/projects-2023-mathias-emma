import numpy as np
import sympy as sm
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy.optimize import minimize,root


# Define the symbols for Q1.1
L = sm.symbols('L^*')  # Labor
w = sm.symbols('w')  # Real wage
tau = sm.symbols('tau')  # Labor-income tax rate
C = sm.symbols('C')  # Private consumption
kappa = sm.symbols('kappa')  # Free private consumption component
alpha = sm.symbols('alpha')  # Weight of private consumption
G = sm.symbols('G')  # Government consumption
nu = sm.symbols('nu')  # Disutility of labor scaling factor
tilde_w = sm.conjugate(sm.symbols('w'))

# Calculate optimal labor supply in sympy for Q1.1
def calculate_optimal_labor_supply():

    # Define the utility function
    U = sm.ln(C**alpha * G**(1 - alpha)) - nu * L**2 / 2
    
    # Constraint 
    C_sub = kappa + tilde_w * L
    
    # Substitute the constraint into the utility function
    U_subs_C = U.subs(C, C_sub)
    
    # Take derivative with respect to L
    U_prime = sm.diff(U_subs_C, L)
    
    # Set derivative equal to zero and solve for L
    L_star = sm.solve(U_prime, L)
    
    # Create the equation
    eq = sm.Eq(L, L_star[1])

    return eq


# Setting up a Simple Name Space for numerical analysis
par = SimpleNamespace()
par.alpha_val = 0.5
par.kappa_val = 1
par.nu_val = (1/(2*16**2))
par.tau_val = 0.3
par.G_val = 1
par.w_val = 1

# Lamdify function from Q1.1 for §1.2
def calculate_optimal_labor_supply_func_2(eq):
    # Turning into a python function
    L_star_func = sm.lambdify((kappa, alpha, G, nu, tilde_w), eq.rhs)
    return L_star_func

# Plotting function in Q1.2
def plot_labor_supply(L_star_func, par):
    
    # Define the function to be plotted
    func = lambda x: L_star_func(par.kappa_val, par.alpha_val, par.G_val, par.nu_val, x)

    # List of wages 
    w_list = np.linspace(0.1, 10, 100)

    # Tilde for given wages 
    tilde_w_list = (1 - par.tau_val) * w_list

    # L* for given tilde wages
    L_star_val = func(tilde_w_list)

    # Plotting the relation between tilde_w and L
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(w_list, L_star_val)
    ax.set_xlabel(r'$w$')
    ax.set_ylabel(r'$L^*(\tilde{w})$')
    ax.set_title('Labor supply as a function of the actual wage')
    ax.grid(True)
    plt.show()

# Additional parameters
par.sigma_values = [1.001, 1.5]
par.rho_values = [1.001, 1.5]
par.epsilon_values = [1.0, 1.0]
par.tau = 0.5  # the optimal tax rate found in question 4

# Define the new utility function
def V_v2(L, G, tau=par.tau, w=par.w_val, kappa=par.kappa_val, alpha=par.alpha_val, nu=par.nu_val, rho=par.rho_values[0], sigma=par.sigma_values[0], epsilon=par.epsilon_values[0]):
    tilde_w = (1 - tau) * w
    C = kappa + tilde_w * L
    U = (((alpha * C**((sigma-1)/sigma) + (1-alpha) * G**((sigma-1)/sigma))**(sigma/(sigma - 1)))**(1 - rho) - 1)/(1-rho)
    disutility = nu * L**(1 + epsilon) / (1 + epsilon)
    return U - disutility

# Define a function that solves the worker's problem
def L_star_v2(G, tau=par.tau, rho=par.rho_values[0], sigma=par.sigma_values[0], epsilon=par.epsilon_values[0]):
    objective_func = lambda L: -V_v2(L, tau, G, rho=rho, sigma=sigma, epsilon=epsilon)
    result = minimize(objective_func, 0.5, bounds=[(0, 24)])
    return result.x[0]

# Define a function that solves the government's problem
def G_func_v2(tau=par.tau, rho=par.rho_values[0], sigma=par.sigma_values[0], epsilon=par.epsilon_values[0]):
    func = lambda G: G - tau * par.w_val * L_star_v2(tau, G, rho, sigma, epsilon)
    result = root(func, 1.0)  # Let's start with initial guess of 1.0 for G
    return result.x[0]

def G_func_v2_utility(tau=par.tau, rho=par.rho_values[0], sigma=par.sigma_values[0], epsilon=par.epsilon_values[0]):
    func = lambda G: G - tau * par.w_val * L_star_v2(tau, G, rho, sigma, epsilon)
    result = root(func, 1.0)  # Let's start with initial guess of 1.0 for G
    utility_given_G = V_v2(L_star_v2(tau, result.x[0], rho, sigma, epsilon), result.x[0], tau, rho, sigma, epsilon)
    return result.x[0], utility_given_G
