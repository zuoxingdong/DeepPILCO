import time

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from model import LinearPolicy, MLPPolicy

def train_dynamics_model(dynamics, dynamics_optimizer, trainset, epochs=1, batch_size=1):
    # Loss
    criterion = nn.MSELoss()  # MSE/SmoothL1

    # Create Dataloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    list_train_loss = []
    for epoch in range(epochs): # Loop over dataset multiple times
        running_train_losses = []

        start_time = time.time()
        
        for i, data in enumerate(trainloader): # Loop over batches of data
            # Get input batch
            X, Y = data
            # Wrap data tensors as Variable and send to GPU
            X = Variable(X).cuda()
            Y = Variable(Y).cuda()
                
            # Zero out the parameter gradients
            dynamics_optimizer.zero_grad()
            
            # Forward pass
            outputs = dynamics(X, delta_target=True)  # delta_target, return state difference for training
            
            # Loss
            loss = criterion(outputs, Y)

            # Backward pass
            loss.backward()

            # Update params
            dynamics_optimizer.step()

            # Accumulate running losses
            running_train_losses.append(loss.data[0])  # Take out value from 1D Tensor
        
        # Record the mean of training and validation losses in the batch
        batch_train_loss = np.mean(running_train_losses)
        list_train_loss.append(batch_train_loss)
        
        time_duration = time.time() - start_time
        # Logging: Only first, middle and last
        if epoch == 0 or epoch == epochs//2 or epoch == epochs - 1:
            print('\t[Epoch # {:3d} ({:.1f} s)] Train loss: {:.8f}'.format(epoch+1, time_duration, batch_train_loss))

    print('\nFinished training dynamics model. \n')
    
    return np.array(list_train_loss)
    
    
def cost(states, sigma=0.25):
    """Pendulum-v0: Same as OpenAI-Gym"""
    l = 0.6
    
    goal = Variable(torch.FloatTensor([0.0, l])).cuda()

    # Cart position
    cart_x = states[:, 0]
    # Pole angle
    thetas = states[:, 2]
    # Pole position
    x = torch.sin(thetas)*l
    y = torch.cos(thetas)*l
    positions = torch.stack([cart_x + x, y], 1)
    
    squared_distance = torch.sum((goal - positions)**2, 1)
    squared_sigma = sigma**2
    cost = 1 - torch.exp(-0.5*squared_distance/squared_sigma)
    
    return cost


def learn_policy(env, dynamics, policy, policy_optimizer, K=1 ,T=1, gamma=0.99, init_particles=None, moment_matching=True, c_sigma=0.25):
    # Predict system trajectories
    
    #policies = [policy]
    #optimizers = [policy_optimizer]

    #for _ in range(T-1):
    #    cloned_policy = _clone_net(policy, requires_grad=True).cuda()
    #    policies.append(cloned_policy)
    #    optimizers.append(optim.Adam(cloned_policy.parameters(), lr=1e-3))
    
    # Particles for initial state
    if init_particles is not None:
        particles = Variable(torch.FloatTensor(init_particles)).cuda()
    else:
        particles = Variable(torch.FloatTensor([env.reset() for _ in range(K)])).cuda()
        
    # Sample BNN dynamics: fixed dropout masks
    dynamics.set_sampling(sampling=True, batch_size=K)
    
    # List of costs
    list_costs = []
    # list of mu and sigma
    list_moments = []
    for t in range(T):  # time steps
        # K actions from policy given K particles
        actions = policy(particles)
        # Concatenate particles and actions as inputs to Dynamics model
        state_actions = torch.cat([particles, actions.unsqueeze(1)], 1)
        # Get next states from Bayesian Dynamics Model
        next_states = dynamics(state_actions)
        
        # Moment matching
        # Compute mean and standard deviation
        if moment_matching:
            mu = torch.mean(next_states, 0)
            sigma = torch.std(next_states, 0)
            # Standard normal noise for K particles
            z = Variable(torch.randn(K, sigma.size(0))).cuda()
            # Sample K new particles from a Gaussian by location-scale transformation/reparameterization
            particles = mu + sigma*z
            
            # Record mu and sigma
            list_moments.append([mu, sigma])
        else:
            particles = next_states
            
        # Compute the mean cost for the particles in the current time step
        #costs = torch.mean(cost(particles, sigma=c_sigma))
        costs = cost(torch.mean(particles, 0).unsqueeze(0), sigma=c_sigma)
        # Append the list of discounted costs
        list_costs.append((gamma**(t + 1))*costs)
        
    # Optimize policy
    policy_optimizer.zero_grad()
    #[optimizer.zero_grad() for optimizer in optimizers]
    J = torch.sum(torch.cat(list_costs))
    J.backward(retain_graph=True)
    
    #for policy in policies[1:]:
    #    for policy_param, cloned_param in zip(policies[0].parameters(), policy.parameters()):
    #        policy_param.grad.data += cloned_param.grad.data.clone()
    
    # Original policy
    policy_optimizer.step()
    
    return policy, list_costs, list_moments

#""" Use cost
def optimize_policy(list_costs, dynamics, policy, policy_optimizer):
    policy_optimizer.zero_grad()
    
    #for c in list_costs[:-1]:
    #    c.backward(retain_graph=True)
    #list_costs[-1].backward()
    
    J = torch.sum(torch.cat(list_costs))
    J.backward(retain_graph=True)
    
    policy_optimizer.step()
#"""

def _grad_norm(net):
    norm = 0.0
    for param in net.parameters():
        if param.grad is not None:
            norm += np.linalg.norm(param.grad.data.cpu().numpy())
            
    return norm

def _clone_net(net, requires_grad=False):
    if type(net) is LinearPolicy:
        cloned = type(net)(net.env)
    else:
        cloned = type(net)(net.env, net.hidden_size)
        
    for cloned_param, net_param in zip(cloned.parameters(), net.parameters()):
        cloned_param.data = net_param.data.clone()
        cloned_param.requires_grad = requires_grad
        
    return cloned
    