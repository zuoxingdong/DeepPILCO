import torch
import torch.utils.data as data
from torch.autograd import Variable

import numpy as np


def rollout(env, policy, dynamics=None, T=1, mode=None, init_particle=None):
    """Generate one trajectory with NN or System dynamics, return transitions"""
    
    # Intial state
    if init_particle is not None:
        s = init_particle
        # Set Gym environment consistently
        env.reset()
        env.unwrapped.state = init_particle
    else:
        s = env.reset()
    
    transitions = []
    for _ in range(T):
        # Convert to FloatTensor, Variable and send to GPU
        s = Variable(torch.FloatTensor(s).unsqueeze(0)).cuda()
        # Select an action by policy
        a = policy(s)
        # Take action via NN/System dynamcis
        if mode == 'System':
            s_next, _, _, _ = env.step(a.data.cpu().numpy())
        elif mode == 'NN':
            state_action = torch.cat([s, a.unsqueeze(0)], 1)
            s_next = dynamics(state_action).data.cpu().numpy()[0]
        else:
            raise ValueError('The value of mode must be either NN or System. ')
        
        # Record data
        transitions.append(np.concatenate([s.data.cpu().numpy()[0], a.data.cpu().numpy(), s_next]))
        
        # Update s as s_next for recording next transition
        s = s_next
        
    return np.array(transitions)


class DataBuffer(data.Dataset):
    def __init__(self, env):
        self.data = None
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        self.max_trajectory = 10  # Same as DeepPILCO
        self.buffer = []
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        """Output FloatTensor"""
        data =  self.data[index]
        # Convert to FloatTensor
        data = torch.FloatTensor(data)
        
        state = data[:self.observation_dim]
        target = data[-self.observation_dim:]
        
        # return target data as difference between current and predicted state
        return data[:self.observation_dim+self.action_dim], target - state
    
    def push(self, D):
        self.buffer.append(D)
        if len(self.buffer) > self.max_trajectory:
            del self.buffer[0]  # Delete oldest trajectory
            
        self.data = np.concatenate(self.buffer, axis=0)
        np.random.shuffle(self.data)

