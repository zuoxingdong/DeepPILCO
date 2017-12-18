import argparse

import os
import shutil
import itertools

import numpy as np
import gym

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from data import rollout, DataBuffer
from model import RandomPolicy, LinearPolicy, MLPPolicy, DynamicsModel
from model_utils import optimize_policy, learn_policy, train_dynamics_model, cost, _grad_norm
from utils import plot_databuffer, plot_trajectories, plot_dynamics_uncertainty, plot_actions, test_episodic_reward

# python script_pendulum.py --seed 99 --num_iter_policy  500 --policy_type LinearPolicy --hidden_size 50 --T 25 --num_iter_algo 10

parser = argparse.ArgumentParser(description='DeepPILCO')
parser.add_argument('--seed', type=int)
parser.add_argument('--num_iter_policy', type=int, default=1000)
parser.add_argument('--policy_type', type=str)
parser.add_argument('--hidden_size', type=int, default=50)
parser.add_argument('--T', type=int, default=25)
parser.add_argument('--num_iter_algo', type=int, default=30)
parser.add_argument('--lr_dynamics', type=float)
parser.add_argument('--lr_policy', type=float)
parser.add_argument('--drop_p', type=float)
parser.add_argument('--c_sigma', type=float)

args = parser.parse_args()

# Set up environment
env = gym.make('CartPoleSwingUp-v0')

T = args.T
K = 10
log_interval_policy = 100
ylim = [-7, 7]
plot=False
fig_path=None

# Create log folders, '[lr_policy]_[lr_dynamics]_[p]'
log_path = os.path.join('./log', 'r_' + str(args.num_iter_policy) + '_' + str(args.lr_policy) + '_' + str(args.lr_dynamics) + '_' + str(args.drop_p) + '_' + str(args.c_sigma), str(args.seed))
if os.path.exists(log_path):
    shutil.rmtree(log_path)  # clean directory each time
os.makedirs(log_path)

# Logging settings
log_str_set = '# Settings: [T={:d}, Num_iter_algo={:d}, Num_iter_policy={:d}, lr_dynamics={:.4f}, lr_policy={:.4f}, c_sigma={:.2f}]\n'
print(log_str_set.format(args.T, args.num_iter_algo, args.num_iter_policy, args.lr_dynamics, args.lr_policy, args.c_sigma))

# Create policy and its optimizer
if args.policy_type == 'LinearPolicy':
    policy = LinearPolicy(env).cuda()
elif args.policy_type == 'MLPPolicy':
    policy = MLPPolicy(env, hidden_size=args.hidden_size).cuda()
else:
    raise TypeError('Policy type must be either LinearPolicy or MLPPolicy')

# Initialize policy parameters to ensure small values
for param in policy.parameters():
    nn.init.normal(param, mean=0, std=1e-5)
    
policy_optimizer = optim.Adam(policy.parameters(), lr=args.lr_policy)  # 1e-2, RMSprop

# Create dynamics and its optimizer
dynamics = DynamicsModel(env, hidden_size=200, drop_prob=args.drop_p).cuda()
dynamics_optimizer = optim.Adam(dynamics.parameters(), lr=args.lr_dynamics, weight_decay=1e-4)

for param in dynamics.parameters():
    nn.init.normal(param, mean=0, std=1e-2)

# Create Data buffer
data = DataBuffer(env)
# Initialize with 4 trajectories to make sure batch size 100
[data.push(rollout(env, RandomPolicy(env), T=T, mode='System')) for _ in range(1)]

# Record some useful loggings for each DeepPILCO iteration
list_ep_costs = []
list_test_rewards = []
list_policy_param = []
list_policy_grad = []
for i in range(args.num_iter_algo):
    print('\nDeepPILCO Iteration #', i + 1)

    # Execute system and record data
    data.push(rollout(env, policy, T=T, mode='System'))

    # Make and save plotting of data buffer
    if plot:
        plot_databuffer(data, ylim=ylim).savefig(fig_path + 'databuffer_' + str(i + 1) + '.png')

    # Create dynamics and its optimizer
    dynamics.set_sampling(sampling=False)
    for param in dynamics.parameters():
        nn.init.normal(param, mean=0, std=1e-2)

    # Train dynamics
    train_dynamics_model(dynamics, dynamics_optimizer, data, epochs=1000, batch_size=data.data.shape[0]//10)

    # Make and save plotting of predictive trajectories between system and NN dynamics
    if plot:
        fig_tau = plot_trajectories(env, policy, dynamics, N=K, T=T, list_moments=None, ylim=ylim)
        fig_tau.savefig(fig_path + 'trajectory_' + str(i + 1) + '.png')

    # Update policy
    print('Policy optimization:')
    for j in range(args.num_iter_policy):
        list_costs, list_moments = learn_policy(env, dynamics, policy, policy_optimizer, K=K, T=T, gamma=1.0, moment_matching=True, c_sigma=args.c_sigma)

        # Loggings
        if (j + 1)%log_interval_policy == 1 or (j + 1) == args.num_iter_policy:
            log_str = '\t[Total cost # {:3d}]: {:.6f}, \tGrad norm: {:.6f}'
            print(log_str.format(j+1, torch.cat(list_costs).mean().data.cpu().numpy()[0], _grad_norm(policy)))
            print('\t\t Param of policy: ', next(policy.parameters()).data.cpu().numpy()[0])
            print('\t\t Grad of policy: ', next(policy.parameters()).grad.data.cpu().numpy()[0])
            print('\t\t Test rewards: ', test_episodic_reward(env, policy, N=50, T=T, render=False))
            print('\n')

    print('Done for Iteration #', i + 1)
    # Record data
    list_ep_costs.append(torch.cat(list_costs).mean().data.cpu().numpy()[0])
    np.savetxt(log_path + '/ep_costs', list_ep_costs)
    list_test_rewards.append(test_episodic_reward(env, policy, N=50, T=T, render=False))
    np.savetxt(log_path + '/test_rewards', list_test_rewards)
    list_policy_param.append(next(policy.parameters()).data.cpu().numpy()[0])
    np.savetxt(log_path + '/policy_param', list_policy_param)
    list_policy_grad.append(next(policy.parameters()).grad.data.cpu().numpy()[0])
    np.savetxt(log_path + '/policy_grad', list_policy_grad)







ep_r = test_episodic_reward(env, policy, N=50, T=T, render=False)

# Logging hyperparameters and its final performance on simulation
param_logstr = './log/log_best_hyperparam.txt'
with open(param_logstr, 'a') as f:
    f.write(log_str_set.format(args.T, args.num_iter_algo, args.num_iter_policy, args.lr_dynamics, args.lr_policy, args.c_sigma))
    f.write('\t Test episodic rewards: {:.2f}\n'.format(ep_r))


                

        
        
        
    