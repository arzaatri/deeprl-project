import torch
import pandas as pd
import numpy as np

stocks = pd.read_csv('data/all_stocks_5yr.csv')
ab = pd.read_csv('data/alpha_and_beta.csv')
rolling_b = pd.read_csv('data/rolling_betas.csv').set_index(['Name','day'])
rolling_stdevs = pd.read_csv('data/rolling_stdevs.csv').set_index(['Name','day'])
rolling_stdevs_stockvals = pd.read_csv('data/rolling_stdevs.csv').set_index(['Name','day'])

def portfolio_beta(weights, for_loss=True):
    if not for_loss:
        with torch.no_grad():
            return (torch.tensor(ab['beta'].values) * weights[1:]).sum()
    else:
        return (torch.tensor(ab['beta'].values) * weights[1:]).sum()
    
def portfolio_beta_rolling(weights, idx, returns=None, for_loss=True):
    if not for_loss:
        with torch.no_grad():
            return (torch.tensor(rolling_b['beta'].xs(idx, level=1).values) * weights[1:]).sum()
    else:
        return (torch.tensor(rolling_b['beta'].xs(idx, level=1).values) * weights[1:]).sum()
    
def portfolio_alpha(weights, for_loss=True):
    if not for_loss:
        with torch.no_grad():
            return 1
        
def portfolio_stdev(weights, for_loss=True):
    if not for_loss:
        with torch.no_grad():
            return (torch.tensor(stdevs.values) * weights[1:]).sum()
    else:
        return (torch.tensor(stdevs.values) * weights[1:]).sum()
    
def portfolio_stdev_rolling(weights, idx, returns=None, for_loss=True):
    if not for_loss:
        with torch.no_grad():
            return (torch.tensor(rolling_stdevs.xs(idx, level=1)['stdev'].values) * 
                    weights[1:]).sum()
    else:
        return (torch.tensor(rolling_stdevs.xs(idx, level=1)['stdev'].values) * weights[1:]).sum()
    
def naive_sharpe_rolling(weights, idx, returns, for_loss=True):
    if not for_loss:
        with torch.no_grad():
            weighted_stdevs = (
                (torch.tensor(rolling_stdevs.xs(idx, level=1)['stdev'].values) * weights[1:]).sum()
            )
            return returns / weighted_stdevs
    else:
        weighted_stdevs = (
            (torch.tensor(rolling_stdevs.xs(idx, level=1)['stdev'].values) * weights[1:]).sum()
        )
        return returns / weighted_stdevs 
    
def get_all_risks(weights, idx, returns, for_loss=False):
    beta = portfolio_beta_rolling(weights, idx, for_loss=for_loss)
    stdev = portfolio_stdev_rolling(weights, idx, for_loss=for_loss)
    sharpe = naive_sharpe_rolling(weights, idx, returns, for_loss=for_loss)
    return np.array([beta.item(), stdev.item(), sharpe.item()])
    
def add(a, for_loss=True):
    if for_loss:
        return a+torch.ones(len(a))
    else:
        with torch.no_grad():
            return a+torch.ones(len(a))