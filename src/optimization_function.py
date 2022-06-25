# Topic idea for South East Asian (maybe African, Indian, Pakistani) data
# does LASSO beat the naive optimization? 
# more advanced: does the non-convex optimizers also beat LASSO?


#%% Loading libraries
from operator import concat
from pickletools import optimize
import os
import glob
import pandas as pd
import numpy as np
from functools import reduce
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize

#%% Function that reads in the data and computes daily returns in %

def prep_data_toy(path_to_file:str, time_period:int, drop=True):

    """
    Specify path to xlsx file that contains the data and
    specify time considered for returns. If drop = True,
    all NaN cells are dropped. 
    """

    df = pd.read_excel(path_to_file).set_index('Name')

    df.index.names = ["Date"]

    return df.pct_change(time_period).dropna()



def prep_data(path_to_file:str, time_period:int, drop=True):

    """
    Specify path to xlsx file that contains the data and
    specify time considered for returns. If drop = True,
    all NaN cells are dropped. 
    """

    df = pd.read_excel(path_to_file)
    
    # Define date column (first column) to be the index 

    df = df.rename(columns={df.columns[0]: "Date"}).set_index('Date')


    # Kick out error columns

    df = df[df.columns.drop(list(df.filter(regex="ERROR")))]

    # Replace error strings with NaN 

    df = df.replace(to_replace=r"[^A-Za-z0-9]+", value=np.NaN, regex=True)


    return df.pct_change(time_period)





#%% Read in actual data
path = './input/actual_data'
datafiles_pattern = os.path.join(path, '*.xls') 
file_list = glob.glob(datafiles_pattern)

# Read in alls dfs

returns_all_assets = [prep_data(file, time_period=1) for file in file_list]

#desired dimensions (sanity checks)

print(max([asset.shape[0] for asset in returns_all_assets]))
print(sum([asset.shape[1] for asset in returns_all_assets]))

# Join the df list to one df 

d_returns = pd.DataFrame()
for region in range(len(returns_all_assets) - 1):
    df = pd.concat([returns_all_assets[region], returns_all_assets[region+1]], axis=1)
    d_returns = pd.concat([d_returns, df], axis=1)

# Dropping duplicated cols 

d_returns = d_returns.loc[:, ~d_returns.columns.duplicated()].copy()

#%% Read in toy data

#d_returns = prep_data('../input/toy_data.xlsx', 1)


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


#%% Create function to drop assets that containt a certain fraction of missing values

def drop_high_na_assets(daily_returns:pd.DataFrame, fraction:float):
    """
    fraction: if a column has relatively more missings that this figure, it gets dropped
    """

    cleaned_df = daily_returns.loc[:, daily_returns.isin([' ','NULL', np.NaN]).mean() < fraction]
    return cleaned_df.dropna()



#%% Apply drop function to end up with ~ 1400 samples

cleaned_returns = drop_high_na_assets(d_returns, 0.8464)

#%% Create summary stats for data presentation table and text 

# Mean return in %
cleaned_returns.describe().loc["mean"].mean() * 100
# Mean std. dev in %
cleaned_returns.describe().loc["std"].mean() * 100
# Skewness in %
cleaned_returns.skew().mean() 
# Kurtosis in %
cleaned_returns.kurtosis().mean() 


#%% Create a function to compute the performance measures of the portfolio

def eval_selector(daily_returns:pd.DataFrame):

    # Create training subset

    # fuers erste sind 250 weg. dann immer 21

    n_windows = int((daily_returns.shape[0] - 250) / 21)

    for window in n_windows:
        training_data = daily_returns.iloc[21*window:21*window+250]
        
        test_data = daily_returns.iloc[21*window+250+1:21*window+250+21+1]

        weights = portfolio_selector(training_data)
        
        # calculate VaR and Sharpe ratio for all optimal porfolios (55), yielding 55 estimates

    return avg_VaR, avg_sharpe




#%% Create function for cross-validation

def cv_sample_size_lambda(daily_returns:pd.DataFrame, n_folds:int):
    """
    Function that uses cross-validation to find the optimal regularization strength.
    """

