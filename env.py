import gym
import numpy as np
import torch
import math

class PortfolioEnv():
    '''
    An environment for portfolio management
    
    Parameters:
        window_size: How many previous time steps are used by the model
        portfolio_value: Initial monetary value of the portfolio
        weights: Initial weights
        trade_cost: Percent commission on trades
        interest_rate: Yearly interest, expressed per trading day
        train_size: Size of the train split
    '''
    # Trade cost is based on the portfolio framework paper
    def __init__(self, data_path = 'data.npy', window_size=50, portfolio_value=10000,
                 weights = None, trading_cost = 0.0025, interest_rate = 0.02/250,
                 train_size = 0.70):
        
        self.data = torch.from_numpy(np.load(data_path))
        self.num_stocks = self.data.shape[1]
        
        if weights:
            self.weights = weights
        else: # Random initialization
            w = np.random.rand(self.num_stocks)
            self.weights = w/w.sum()
            
        self.portfolio_value = portfolio_value
        self.window_size = window_size
        self.trading_cost = trading_cost
        self.interest_rate = interest_rate
        
        self.end_time = int((self.data.shape[2]-self.window_size)*train_size)
        
        self.state = None
        self.idx = None
        self.done = False
        
    def reset(self, weights=None, portfolio_value=None, t=0):
        '''
        Resets the environment, allowing for specified initialization
        Args:
            weights: Initial weight vector
            portfolio_value: Initial value of the portfolio
            t: Start time
        '''
        if weights:
            assert math.isclose(np.sum(weights), 1)
            self.weights = weights
        else:
            w = np.random.rand(self.num_stocks)
            self.weights = w/w.sum()
            
        self.portfolio_value = portfolio_value if portfolio_value else 10000
        
        self.idx = self.window_size+t
        self.state = (self.data[:,:,self.idx-self.window_size:self.idx],
                      self.weights, self.portfolio_value)
        self.done = False
        
        return self.state
    
    def step(self, action):
        '''
        Updates weights to be the values given by action, and 
        Returns:
            state at time t+1
            reward, defined as the percent change in portfolio value
            done: boolean indicating whether we have reached end_time
            info: None, included for parity with gym envs
            
        Args:
            action: Output of the actor, redistributed weights
        '''
            
        X_next = self.data[:,:,self.idx:self.idx+self.window_size]
        w = self.state[1]
        p_val = self.state[2]
        
        # Now to calculate the new value of the portfolio
        # First, we need the return of everything defined as Open(t)/Open(t+1), along with interest
        returns = np.array([1+self.interest_rate]+self.data[0,:,self.idx].tolist())
        
        fees = p_val * np.sum(action-w) * self.trading_cost
        
        value_vec = p_val * action
        value_vec_new = (value_vec - np.array([fees]+[0]*self.num_stocks))*returns
        p_val_new = value_vec_new.sum()
        
        reward = (p_val_new - p_val) / p_val
        
        self.weights = value_vec_new / p_val_new
        self.portfolio_value = p_val_new
        
        self.idx = self.idx+1

        self.state = (self.data[:,:,self.idx-self.window_size:self.idx],
                      self.weights, self.portfolio_value)
        if idx > self.end_time:
            self.done = True
            
        return state, reward, done, None