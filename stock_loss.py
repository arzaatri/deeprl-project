import torch
from torch import nn
class ReturnsLoss(nn.Module):
    
    def __init__(self, interest_rate=0.02/250, trading_cost=0.0025, alpha=0.1,
                 secondary=None, gamma=0.2):
        '''
        Parameters:
            alpha: Regularization parameter for max portfolio weight
            secondary: Secondary objective. Should be a function that takes weights as
                input and returns some measure of portfolio risk, e.g. beta, standard deviation,
                sharpe ratio, etc
            gamma: Parameter for secondary loss
        '''
        super().__init__()
        self.interest_rate = torch.tensor([interest_rate], requires_grad=True)
        self.trading_cost = torch.tensor([trading_cost], requires_grad=True)
        self.alpha = alpha
        self.secondary = secondary
        self.gamma = gamma
        
    def forward(self, action, w_prev, p_val_prev, daily_ret, p_val_prev_eq, idx):
        '''
        Loss given as profits from the network less profits from equiweighted portfolio,
        less argmax of the portfolio to prevent algorithm from placing all funds in a 
        single stock
        
        Note that this is currently returns for one round, not cumulative returns. Cumulative
        returns should just have additive gradients anyway so this should be fine.
        
        Args:
            
        '''
        
        num_stocks = len(action)-1
        # Now to calculate the new value of the portfolio
        # update_vec is the price relative vector in the paper, aka yt
        update_vec = torch.tensor([1+self.interest_rate]+daily_ret.tolist(), requires_grad=True)
        
        # Transaction fees
        fee = p_val_prev * torch.sum(torch.abs(action-w_prev)) * self.trading_cost
        
        # Weights * portfolio value = value vector, aka the money put into each asset
        value_vec_prev = p_val_prev * action
        
        # Pay transaction cost on portfolio value, cash
        p_val_trans = p_val_prev - fee 
        value_vec_trans = (value_vec_prev -
                           torch.tensor([fee]+[0]*num_stocks, requires_grad=True))
        
        # Evolve value according to relative price changes from previous period
        value_vec_new = value_vec_trans * update_vec
        # Get portfolio value after changes due to transaction costs, price fluctuation
        p_val_new = torch.sum(value_vec_new)
        
        # Instantaneous reward is returns, i.e. change in portfolio values
        returns = (p_val_new - p_val_prev) / p_val_prev
        
        '''
        # How weights adjust due to transaction costs
        w_prime = yt*action / yt.dot(action)
        # This approximation to the transaction remainder factor is for when
        # sell and purchase costs are the same
        mu = transaction_cost * torch.sum(torch.abs(w_prime - action))
        returns = (mu*yt).dot(w)
        '''
        
        # Equiweighted portfolio returns
        w_eq = torch.ones(action.shape, requires_grad=True) / action.shape[0]
        value_vec_prev_eq = w_eq * p_val_prev_eq
        
        value_vec_new_eq = value_vec_prev_eq * update_vec
        p_val_new_eq = torch.sum(torch.abs(value_vec_new_eq))
        
        returns_eq = (p_val_new_eq - p_val_prev_eq) / p_val_prev_eq
        if self.secondary:
            secondary_loss = self.gamma*self.secondary(action, idx, returns)
        else:
            secondary_loss = 0
            
        # Subtract returns_eq as a baseline, then subtract the value of the max
        # weight to keep the model from putting all its eggs in one basket
        # Negative everything so that we minimize, rather than maximize
        return -(returns - returns_eq.detach() - self.alpha*torch.max(action) - secondary_loss)