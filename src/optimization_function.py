# Topic idea for South East Asian (maybe African, Indian, Pakistani) data
# does LASSO beat the naive optimization? 
# more advanced: does the non-convex optimizers also beat LASSO?


#%% Loading libraries
from pickletools import optimize
import pandas as pd
import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize

#%% Function that reads in the data and computes daily returns in %

def prep_data(path_to_file:str, time_period:int, drop=True):

    """
    Specify path to xlsx file that contains the data and
    specify time considered for returns. If drop = True,
    all NaN cells are dropped. 
    """

    df = pd.read_excel(path_to_file).set_index('Name')

    df.index.names = ["Date"]

    return df.pct_change(time_period).dropna()



#%% Read in 

d_returns = prep_data('../input/toy_data.xlsx', 1)


# %% Create optimizer

def portfolio_selector(daily_returns: pd.DataFrame, lambda_=0):
    """
    Inputs:
    daily_returns: Dataframe that contains daily returns
     for every asset in one column

    lambda_: regularization_strength [0,1]
     default: 0 -> unregularized

    Output: 
    optimal weight vector w for the MVP
    """

    number_of_assets = daily_returns.shape[1]

    w = np.empty((number_of_assets,1)) # weight vector to find

    Sigma = daily_returns.cov() # sample covariance matrix


    def target_fn(w):

        return w.T @ Sigma @ w + lambda_ * sum(abs(w))
        # 1x6 x 6x6 x 6x1 + 1x1 = 1x1 


    fconst = lambda w: 1 - sum((abs(w)))
    cons   = ({'type':'eq','fun':fconst})


    optimized = minimize(target_fn, x0 = np.repeat(1/number_of_assets,
    number_of_assets), constraints=cons, method="trust-constr")

    return optimized


#%% Create function for 10 fold cross-validation