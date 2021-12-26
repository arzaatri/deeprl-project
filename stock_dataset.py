from torch.utils.data import Dataset, DataLoader, Sampler
from portfolio_env_pvm import PortfolioEnv
import torch

class StockDataset(Dataset):
    def __init__(self, pvm, env, env_eq, split='train', device='cpu'):
        super().__init__()
        self.pvm = pvm
        self.env = env
        self.env_eq = env_eq
        self.eq_weights = torch.ones(self.env.data.shape[1]+1)
        self.eq_weights = self.eq_weights/len(self.eq_weights)
        self.device = device
        
    def reset_env(self):
        state = self.env.reset()
        state_eq = self.env_eq.reset()
        
    def step(self, action):
        state, reward, done = self.env.step(action)
        X, w, p_val = state
        daily_ret = X[-1,:,-1]
        X = X[:-1] # Just the features
        state_eq, _, _ = self.env_eq.step()
        _, _, p_val_eq = state_eq
        #daily_ret = self.env.data[-1, :, self.env.idx].to(device)
        return X, w, p_val, daily_ret, p_val_eq, reward, done
        
    def reset_pvm(self):
        self.pvm.reset_weights()
        
    def get_index(self):
        return self.env.idx
    
    def __len__(self):
        return len(self.pvm.memory)
    
    def __getitem__(self, idx):
        '''
        Returns:
            X: State of the env at time t=idx
            w_prev: Weight vector from PVM at index idx
            p_val_prev: Value of the portfolio at time t=idx
            daily_ret: Open[t] / Open[t-1] from environment, where t=idx
            p_val_prev_eq: Value of equiweighted portfolio at time t=idx
        '''
        # pvm_env should be returning these as .to(device)
        # We call reset here because this function is only used at the start of a batch
        X, w_prev, p_val_prev = self.env.reset(t=idx+1, weights=self.pvm.get_weight(idx))
        daily_ret = X[-1,:,-1]
        X = X[:-1] # Just the features
        _, _, p_val_prev_eq = self.env_eq.reset(t=idx+1)
        
        return X, w_prev, p_val_prev, daily_ret, p_val_prev_eq
    '''
    After this, model should use X, w_prev to get action
    It then computes loss. Is stepping environment necessary?
    Might be - action isn't precisely the same as next weight. Maybe make the 
        env work entirely with detached tensors?
    Then again, can just do those computations in the loss function, then step after
    '''
    
class GeometricSampler(Sampler):
    def __init__(self, beta, num_indices):
        self.beta = beta
        self.num_indices = num_indices
        
    def __len__(self):
        return self.num_indices
    
    def __iter__(self):
        return
    
def get_dataloader(pvm, env, env_eq, split='train'):
    data = StockDataset(pvm, env, env_eq, split=split)
    sampler = GeometricSampler(self.pvm.beta, len(pvm.memory))
    return DataLoader(data, batch_size=4, shuffle=False)