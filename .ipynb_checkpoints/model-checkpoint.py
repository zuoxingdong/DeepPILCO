import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RandomPolicy(nn.Module):
    """Linear policy for the controller"""
    def __init__(self, env):
        super().__init__()
        
        self.env = env
        
    def forward(self, x):
        return Variable(torch.FloatTensor(self.env.action_space.sample()))

class LinearPolicy(nn.Module):
    """Linear policy for the controller"""
    def __init__(self, env):
        super().__init__()
        
        self.env = env
        
        # Fully connected layers
        self.out = nn.Linear(in_features=env.observation_space.shape[0],
                             out_features=1, 
                             bias=False)
    def forward(self, x):
        x = self.out(x)

        return torch.clamp(x[:, 0], self.env.action_space.low[0], self.env.action_space.high[0])

class MLPPolicy(nn.Module):
    """MLP Policy for the controller"""
    def __init__(self, env, hidden_size=50):
        super().__init__()
        
        self.env = env
        
        self.hidden_size = hidden_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=env.observation_space.shape[0]+1,  # [x, dx, polex, poley, dtheta]
                             out_features=self.hidden_size, 
                             bias=True)
        self.out = nn.Linear(in_features=self.hidden_size,
                             out_features=1, # 1D continuous action space [mu, log-space of sigma] 
                             bias=True)
    def forward(self, x):
        polex = torch.sin(x[:, 2])*0.6
        poley = torch.cos(x[:, 2])*0.6
        
        x = torch.stack([x[:, 0], x[:, 1], x[:, 0] + polex, poley, x[:, 3]], 1)
        
        x = F.relu(self.fc1(x))
        x = self.out(x)
        
        #mu = x[:, 0]
        #std = F.sigmoid(x[:, 1])  # Softplus vs sigmoid
        #z = Variable(torch.randn(mu.size()))
        
        #action = mu + std*z ###### For now, try deterministic actions
        
        x = 9/8*torch.sin(x) + 1/8*torch.sin(3*x)
        #x = torch.tanh(x)
        x = x*self.env.action_space.high[0]
        #x[:, 0] = torch.clamp(x[:, 0], self.env.action_space.low[0], self.env.action_space.high[0])

        return x[:, 0]

class DynamicsModel(nn.Module):
    """Learning dynamics model via regression"""
    def __init__(self, env, hidden_size=200, drop_prob=0.0):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        
        # Flag for sampling parameters
        self.sampling = False
        # Fix the random mask for dropout, each batch contains K particles
        self.mask1 = None
        self.mask2 = None

        # Fully connected layer
        self.fc1 = nn.Linear(in_features=env.observation_space.shape[0]+env.action_space.shape[0],  # State + Action
                             out_features=self.hidden_size,
                             bias=True)
        self.fc2 = nn.Linear(in_features=self.hidden_size,
                             out_features=self.hidden_size,
                             bias=True)
        self.out = nn.Linear(in_features=self.hidden_size,
                             out_features=env.observation_space.shape[0],  # Next state
                             bias=True)

    def forward(self, x, delta_target=False, training=True):
        # Check if drop mask with correct dimension
        if self.sampling:
            if self.mask1.size()[0] != x.size()[0] or self.mask2.size()[0] != x.size()[0]:
                raise ValueError('Dimension of fixed masks must match the batch size.')
        
        state = x.clone()[:, :-1]  # CartPoleSwingUp, without action
        x = F.selu(self.fc1(x))  # try sigmoid as DeepPILCO paper
        
        if self.sampling:
            x = x*self.mask1
        else:
            x = F.dropout(x, p=self.drop_prob, training=training)
            
        x = F.selu(self.fc2(x))  # try sigmoid as DeepPILCO paper
        
        if self.sampling:
            x = x*self.mask2
        else:
            x = F.dropout(x, p=self.drop_prob, training=training)
            
        x = self.out(x)
        
        if delta_target:  # return difference in states, for training
            x =  x
        else:  # return next states as s + delta_s
            x = state + x
            
        return x
    
    def set_sampling(self, sampling=None, batch_size=None):
        if sampling is None:
            raise ValueError('Sampling cannot be None.')
            
        self.sampling = sampling
        
        if self.sampling:
            # Sample dropout random masks
            self.mask1 = Variable(torch.bernoulli(torch.zeros(batch_size, self.hidden_size).fill_(1 - self.drop_prob))).cuda()
            self.mask2 = Variable(torch.bernoulli(torch.zeros(batch_size, self.hidden_size).fill_(1 - self.drop_prob))).cuda()
            # Rescale by 1/p to maintain output magnitude
            self.mask1 /= (1 - self.drop_prob)
            self.mask2 /= (1 - self.drop_prob)
            
class TrueDynamics(nn.Module):
    def __init__(self, env, hidden_size=200, drop_prob=0.0):
        super().__init__()
        
        self.env = env
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        
        self.mask1 = None
        
    def forward(self, x):
        th = x[:, 0]
        thdot = x[:, 1]
        u = torch.clamp(x[:, 2], -3, 3)
        
        g = 9.82
        m = 1.
        l = 1.
        dt = 0.08
        
        newthdot = thdot + (-3*g/(2*l)*torch.sin(th + np.pi) + 3./(m*l**2)*u)*dt
        newth = th + newthdot*dt
        
        newthdot = torch.clamp(newthdot, -8, 8)
        
        return torch.stack([newth, newthdot], 1)
        
    def set_sampling(self, sampling=None, batch_size=None):
        if sampling is None:
            raise ValueError('Sampling cannot be None.')
            
        self.sampling = sampling
        
        if self.sampling:
            # Sample dropout random masks
            self.mask1 = Variable(torch.bernoulli(torch.zeros(batch_size, self.hidden_size).fill_(1 - self.drop_prob))).cuda()
            self.mask2 = Variable(torch.bernoulli(torch.zeros(batch_size, self.hidden_size).fill_(1 - self.drop_prob))).cuda()
            # Rescale by 1/p to maintain output magnitude
            self.mask1 /= (1 - self.drop_prob)
            self.mask2 /= (1 - self.drop_prob)
