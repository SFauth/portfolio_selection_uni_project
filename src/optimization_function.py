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
import matplotlib.pyplot as plt

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

# Get weight vector.
# buy assets according to weights 
# 10 % Samsung, 90 % SK hynix
# compute return distribution of this portfolio: (cont. returns?)
# we take discrete returns as they have the advantage that we can
# compute the portfolio"s total return as a weighted sum 
# of the individual returns (we dont care about longer periods,
# as we care about daily returns) (CITE FINANZMARKSTATISTIK)
# 0.1 * samsung return + 0.9 * SK hynix return day 1 
# 0.1 * samsung return + 0.9 * SK hynix return day 2 
# ...
# 21 daily returns of this portfolio form the distribution 
# extract VaR, Sharpe ratio




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
cleaned_returns.describe().loc["mean"].mean() 
# Mean std. dev in %
cleaned_returns.describe().loc["std"].mean()
# Skewness in %
cleaned_returns.skew().mean() 
# Kurtosis in %
cleaned_returns.kurtosis().mean() 


#%% Plot function to be optimized 
# Vector function: a vector goes in and a scalar comes out 
# can not plot it 



#%% Create a function to compute the performance measures of the portfolio

def eval_selector(daily_returns:pd.DataFrame, VaR_per=0.1, reg_strength=0):
    """
    Give in daily returns and percentage to calculate VaR.
    Default 10%
    Specify Lasso regularization strength (Default: 0)
    """
    # Create training subset

    # fuers erste sind 250 weg. dann immer 21

    n_windows = int((daily_returns.shape[0] - 250) / 21)

    SharpeRatios = []
    VaRs = []
    shorting_frac = []
    active_frac = [] 

    for window in range(n_windows):
        training_data = daily_returns.iloc[21*window:21*window+250]
        
        test_data = daily_returns.iloc[21*window+250+1:21*window+250+21+1]

        # Try out different lambdas 

        

        weights_obj = portfolio_selector(training_data, lambda_=  reg_strength)

        weights = weights_obj.x 

        returns = sorted(test_data @ weights)

        test_size = test_data.shape[0]

        shorting_frac.append((weights < 0 ).sum() / weights.shape[0])

        active_frac.append(((weights != 0 ).sum()) / weights.shape[0])

        VaRs.append(returns[int(test_size * VaR_per)])
        
        SharpeRatios.append(pd.Series(returns).mean() / pd.Series(returns).std())

        # calculate VaR and Sharpe ratio for all optimal porfolios (55), yielding 55 estimates

    return VaRs, SharpeRatios, shorting_frac, active_frac




#%% Create function for cross-validation
# Should we shuffle the data ? 

def cv_sample_size_lambda(daily_returns:pd.DataFrame, n_folds:int):
    """
    Function that uses cross-validation to find the optimal regularization strength.
    """
    fold_size = int(daily_returns.shape[0] / n_folds)

    for fold in range(len(n_folds)):

        test_data = daily_returns[fold * fold_size: (fold+1) * fold_size]
        
        # Get indices and exclude them from daily returns to get training data

        test_indices = test_data.index

        training_data = daily_returns.drop([test_indices], axis=0)

    
