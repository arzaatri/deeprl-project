import torch
from torch import nn
import torch.nn.functional as F
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNNAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, (1,3))
        self.conv2 = nn.Conv2d(2, 20, (1, 48))
        self.conv3 = nn.Conv2d(21, 1, 1)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, inp, w):
        '''
        Args:
            inp: The data for the previous time and the previous window_size time steps
            w: The weight vector at the start of this round of learning
        Returns:
            The action, which is the weights for the next round
        '''
        x = torch.tanh(self.conv1(inp))
        x = torch.tanh(self.conv2(x))
        x = torch.cat([x, w[1:].view(1,1,-1,1)], dim=1)
        x = torch.tanh(self.conv3(x)) 
        x = torch.cat([torch.ones([1]*len(x.shape), requires_grad=True), x], dim=2)
        return self.softmax(x.view(-1))
    
class CNNAgentBaselines(BaseFeaturesExtractor):
    def __init__(self, obs_space, features_dim=17):
        super().__init__(obs_space, features_dim)
        self.conv1 = nn.Conv2d(3, 2, (1,3))
        self.conv2 = nn.Conv2d(2, 20, (1, 48))
        self.conv3 = nn.Conv2d(21, 1, 1)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, obs):
        '''
        Args:
            inp: The data for the previous time and the previous window_size time steps
            w: The weight vector at the start of this round of learning
        Returns:
            The action, which is the weights for the next round
        '''
        inp = obs['observation']
        w = obs['prev_w']
        
        x = torch.tanh(self.conv1(inp))
        x = torch.tanh(self.conv2(x))
        x = torch.cat([x, w[1:].view(1,1,-1,1)], dim=1)
        x = torch.tanh(self.conv3(x)) 
        x = torch.cat([torch.ones([1]*len(x.shape), requires_grad=True), x], dim=2)
        return self.softmax(x.view(-1))
    
class RNNAgent2(nn.Module):
    def __init__(self):
        super().__init__()
        # This will result in 16 outputs of dim 20
        for i in range(16):
            setattr(self, f'rnn{i}', nn.LSTM(input_size=3, hidden_size=20, batch_first=True))
        self.conv = nn.Conv1d(21, 1, 1)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, inp, w):
        _, (x, a) = self.rnn0(inp[:,:,0,:])
        x = torch.cat([ x, self.rnn1(inp[:,:,1,:])[1][0] ], dim=1)
        for i in range(2,16):
            x = torch.cat([ x, getattr(self, f'rnn{i}')(inp[:,:,i,:])[1][0] ], dim=1)
        x = x.transpose(1,2)
        #x = torch.transpose(x)
        x = torch.cat([x, w[1:].view(1,1,-1)], dim=1)
        x = F.tanh(self.conv(x))
        x = torch.cat([torch.ones([1]*len(x.shape), requires_grad=True), x], dim=2)
        return self.softmax(x.view(-1))
    
class RNNAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = MV_LSTM(3, 50)
        self.rnn.init_hidden(1)
        self.conv1 = nn.Conv2d(3, 20, (1,20))
        self.conv2 = nn.Conv2d(21, 1, 1)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, inp, w):
        inp = inp.view(3,16,50)
        x = self.rnn(inp)
        x = torch.transpose(torch.stack((x1,x2,x3)), 0, 1)
        x = F.relu(self.conv1(x))
        x = torch.cat([x, w[1:].view(1,1,-1,1)], dim=1)
        x = F.relu(self.conv2(x)) 
        x = torch.cat([x, torch.ones([1]*len(x.shape), requires_grad=True)], dim=2)
        return self.softmax(x.view(-1))
    
# Multivariate LSTM implementation from Tomas Trdla on stackoverflow
class MV_LSTM(torch.nn.Module):
    def __init__(self, n_features, seq_length, n_hidden=20, n_layers=1):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 20 # number of hidden states
        self.n_layers = 1 # number of LSTM layers (stacked)
    
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, 1)
        
    
    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        self.hidden = (hidden_state, cell_state)
    
    
    def forward(self, x):        
        batch_size, seq_len, _ = x.size()
        
        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size,-1)
        print(self.l_linear(x).shape)
        return self.l_linear(x)
    
class RNNAgent_old(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.LSTM(input_size=50, hidden_size=20, dropout=0.3)
        self.rnn2 = nn.LSTM(input_size=50, hidden_size=20, dropout=0.3)
        self.rnn3 = nn.LSTM(input_size=50, hidden_size=20, dropout=0.3)
        self.conv1 = nn.Conv1d(60, 20, 1)
        self.conv2 = nn.Conv1d(21, 1, 1)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, inp, w):
        x1 = self.rnn1(inp[:,0])[0]
        x2 = self.rnn1(inp[:,1])[0]
        x3 = self.rnn1(inp[:,2])[0]
        x = torch.transpose(torch.cat((x1,x2,x3),dim=2), 1, 2).clone()
        x = F.relu(self.conv1(x))
        x = torch.cat([x, w[1:].view(1,1,-1)], dim=1)
        x = F.relu(self.conv2(x)) 
        x = torch.cat([x, torch.ones([1]*len(x.shape), requires_grad=True)], dim=2)
        return self.softmax(x.view(-1))
    