import torch
import numpy as np

class PVM():
    def __init__(self, num_steps, beta=0.5, num_stocks=16, device='cpu'):
        # Paper says it starts with uniform weights in all
        # Do num_steps-1 because t=0 has no previous weight
        self.num_steps = num_steps
        self.num_stocks = num_stocks
        self.memory = torch.ones(self.num_steps-1, self.num_stocks+1)
        self.memory = self.memory / (num_stocks+1)
        self.beta = beta
        self.device = device
        
    def get_weight(self, t):
        return self.memory[t].to(self.device)
    
    def update_weight(self, t, w):
        self.memory[t] = w
    
    def reset_weights(self):
        self.memory = torch.ones(self.num_steps, self.num_stocks+1)
        self.memory = self.memory / (num_stocks+1)
        
    def draw(self, batch_size):
        while True:
            z = np.random.geometric(p=self.beta)
            tb = self.num_steps - batch_size + 1 - z
            # Note I added the z > 50+bs
            if tb >= 0 and z > (50+batch_size): 
                return tb