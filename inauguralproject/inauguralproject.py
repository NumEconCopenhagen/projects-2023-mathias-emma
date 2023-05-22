
from types import SimpleNamespace

import numpy as np
from scipy import optimize
from matplotlib import cm # for colormaps

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # additional parameter for disutility of 
        par.theta = 0
        par.z = 1

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        H = np.nan

        power = (par.sigma - 1)/par.sigma

        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0:
            H = np.fmin(HM, HF)
        else: 
            HM = np.fmax(HM,1e-8)
            HF = np.fmax(HF,1e-8)
            inside = ( (1-par.alpha)  * (HM) **(power) + par.alpha * (HF)**(power)  )
            inside = np.fmax(inside,1e-8)
            H = inside**(1/power)

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_ + LF*par.theta + HM*par.z)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt


    def solve(self,do_print=False):
        """ solve model continously """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
    
    # a. objective function 
        def obj(x):
            LM, HM, LF, HF = x
            return - self.calc_utility(LM, HM, LF, HF)
    
    #b. Constraints and Bounds (to minimize) 
        def constraints(x):
            LM, HM, LF, HF = x
            return [24 - LM-HM, 24 -LF-HF]
    

        constraints = ({'type': 'ineq', 'fun':constraints}) 
        bounds = ((0,24), (0,24), (0,24), (0,24))

        initial_guess = [6,6,6,6]

    #c. Solver 
        solution = optimize.minimize(obj, initial_guess, method="SLSQP", bounds=bounds, constraints=constraints, tol = 0.000000001)

        opt.LM = solution.x[0]
        opt.HM = solution.x[1]
        opt.LF = solution.x[2]
        opt.HF = solution.x[3]
        
        return opt


    def solve_wage_work(self, discrete=False):
        """ plot results
         arg
           """
        
        # a. Setting up parameters
        par = self.par
        sol = self.sol
        wF_vec = par.wF_vec

        # b. Lists for relative wage and hours
        rel_wage = []
        rel_hours = []

        # c. Loop over wF_vec for optimal solution
        for wF in wF_vec:
            par.wF = wF
            if discrete:
                opt = self.solve_discrete()
            else:
                opt = self.solve()
            #opt = self.solve_discrete()
            rel_wage.append(wF/par.wM)
            rel_hours.append(opt.HF/opt.HM)
        
        # d. Log transformation
        log_rel_wage = np.log(rel_wage)
        log_rel_hours = np.log(rel_hours)

        return log_rel_wage, log_rel_hours



    def q1_gen_table(self, list_alpha,list_sigma):
        """ generate table """
    
       #Setting up parameters
        par = self.par
        sol = self.sol

        # empty list to store the relative wage values
        table_data = []

        # loop alpha values
        for alpha in list_alpha:
            # row list for the current alpha value
            row_data = []
            
            # loop  sigma values
            for sigma in list_sigma:
                # parameter values
                par.alpha = alpha
                par.sigma = sigma
                
                # solve for optimal solution
                opt = self.solve_discrete()
                
                # calculate relative wage and append to row list
                relative_wage = opt.HF / opt.HM
                row_data.append(relative_wage)
            
            # append the row list to the table data list
            table_data.append(row_data)

        # create a pandas DataFrame from the table data
        table = pd.DataFrame(table_data, index=list_alpha, columns=list_sigma)
        return table
    

    def q1_plot_table(self,table):
            """ plot table """
                #Illistration 
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # create x, y and z values
            x_data, y_data = np.meshgrid(table.columns, table.index)
            z_data = table.values

            # plot the surface
            ax.plot_surface(x_data, y_data, z_data, cmap=cm.jet)

            # set the axis labels
            ax.set_xlabel('Sigma')
            ax.set_ylabel('ALpha')
            ax.set_zlabel('Relative hours H_F /H_M', labelpad=0)
            # c. invert xaxis to bring Origin in center front
            ax.invert_xaxis()
            fig.tight_layout()
            plt.show()


    def q2_plot(self, log_rel_wage, log_rel_hours):
        """ plot results """
        
        # a. Setting up parameters
        par = self.par
        sol = self.sol

        # b. Plotting
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(log_rel_wage, log_rel_hours, color='black')
        ax.scatter(log_rel_wage, log_rel_hours, marker='o', color='red')
        ax.set_xlabel('log(w_F/w_M)')
        ax.set_ylabel('log(H_F/H_M)')
        ax.set_title('Log wage - log hours')
        plt.show()
        return fig, ax


    def run_regression(self, log_rel_wage, log_rel_hours):
        """ run regression """
        
        #Setting up parameters
        par = self.par
        sol = self.sol

        #Running regression
        x = log_rel_wage
        y = log_rel_hours

        A = np.vstack([np.ones(x.size),x]).T
        beta0, beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

        return beta0, beta1 

