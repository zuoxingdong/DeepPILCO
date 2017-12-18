import numpy as np

import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

from data import rollout
from model_utils import learn_policy

def plot_databuffer(data, ylim=None):
    fig, ax = plt.subplots(1, 1)

    for i in range(len(data.buffer)):
        tau = data.buffer[i]
        
        s = np.concatenate([tau[:, 2], [tau[-1, -2]]])

        if i == len(data.buffer) - 1:  # latest trajectory in red
            ax.plot(np.arange(0, s.shape[0]), s, 'red')
        else:
            ax.plot(np.arange(0, s.shape[0]), s, 'blue')

            
    # Figure settings
    ax.grid()
    ax.set_title('Trajectories in data buffer')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Angle (in radians)')
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.close()  # Close figure to save memory
    
    return fig

def plot_trajectories(env, policy, policy_optimizer, dynamics, N=50, T=25, predict_uncertainty=False, ylim=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=[6, 4*2])
    
    init_particles = [env.reset() for _ in range(N)]
    
    # Gaussian-fitted predictive states
    if predict_uncertainty:  # Use same particles, plot moment matched mean and std
        policy, list_costs, list_moments = learn_policy(env, 
                                                        dynamics, 
                                                        policy, 
                                                        policy_optimizer,
                                                        K=N, 
                                                        T=T, 
                                                        init_particles=init_particles,
                                                        moment_matching=True,
                                                        c_sigma=0.25)
        list_mean = []  # fill in maybe not
        list_one_upper = []
        list_one_lower = []
        for mu, sigma in list_moments:
            mu = mu.data.cpu().numpy()
            sigma = sigma.data.cpu().numpy()

            one_upper = mu + 1*sigma
            one_lower = mu - 1*sigma

            list_mean.append(mu[2])
            list_one_upper.append(one_upper[2])
            list_one_lower.append(one_lower[2])

        ax1.plot(np.arange(1, T + 1), list_mean, 'red')
        ax1.fill_between(np.arange(1, T + 1), 
                         list_one_upper, 
                         list_one_lower, 
                         facecolor='red', 
                         alpha=0.5)
    
        # Clone masks for each particles below
        mask1 = dynamics.mask1.clone()
        mask2 = dynamics.mask2.clone()
    
    for n in range(N):
        # Common initial particles
        init_particle = init_particles[n]
        
        # Rollout via system dynamics
        tau_sys = rollout(env, policy, T=T, mode='System', init_particle=init_particle)
        
        # Rollout via NN dynamics
        if predict_uncertainty:  # Use same single mask as with predictive distribution above
            dynamics.mask1 = mask1[n, :].unsqueeze(0)
            dynamics.mask2 = mask2[n, :].unsqueeze(0)
        else:  # No predictive dristribution made, then re-sample new dropout mask
            dynamics.set_sampling(sampling=True, batch_size=1)
        tau_NN = rollout(env, policy, dynamics, T=T, mode='NN', init_particle=init_particle)
        
        # States
        s_sys = np.concatenate([tau_sys[:, 2], [tau_sys[-1, -2]]])
        s_NN = np.concatenate([tau_NN[:, 2], [tau_NN[-1, -2]]])
        
        # Plot states
        if n == 0:  # label
            ax1.plot(np.arange(0, s_sys.shape[0]), s_sys, 'green', label='System')
            ax1.plot(np.arange(0, s_NN.shape[0]), s_NN, 'blue', label='NN')
        else:
            ax1.plot(np.arange(0, s_sys.shape[0]), s_sys, 'green')
            ax1.plot(np.arange(0, s_NN.shape[0]), s_NN, 'blue')
            
        # Plot actions
        if n == 0:  # label
            ax2.plot(np.arange(0, tau_sys.shape[0]), tau_sys[:, 4], 'green', label='System')
            ax2.plot(np.arange(0, tau_NN.shape[0]), tau_NN[:, 4], 'blue', label='NN')
        else:
            ax2.plot(np.arange(0, tau_sys.shape[0]), tau_sys[:, 4], 'green')
            ax2.plot(np.arange(0, tau_NN.shape[0]), tau_NN[:, 4], 'blue')
            
    ax1.set_ylabel('Pole angle (radians)')
    ax2.set_xlabel('Time steps')
    ax2.set_ylabel('Action values')
    ax1.legend()
    ax2.legend()
    ax1.grid()
    ax2.grid()
    if ylim is not None:
        ax1.set_ylim(ylim)
    ax2.set_ylim([env.action_space.low[0] - .3, env.action_space.high[0] + .3])
        
    
    plt.close()  # Close figure to save memory
    return fig

def plot_dynamics_uncertainty(env, policy, dynamics, T=1, N_uncertainty=1, N_plots=1, ylim=None, real_tau=False):
    if ylim is None:
        fig, axes = plt.subplots(N_plots, N_plots, sharex=True, figsize=[6*N_plots, 4*N_plots])
    else:
        fig, axes = plt.subplots(N_plots, N_plots, sharex=True, sharey=True, figsize=[6*N_plots, 4*N_plots])
    if N_plots == 1:
        axes = np.array([axes])
    
    # Record for actions
    actions_sys = []  # [N_plots**2, T, s_a_next_s]
    actions_NN = []  # [N_plots**2, N_uncertainty, T, s_a_next_s]
    
    for ax in axes.reshape(-1):  # each subplot for different initial particle
        # Random initial particle
        init_particle = env.reset()

        # Trajectory via system dynamics
        tau_sys = rollout(env, policy, T=T, mode='System', init_particle=init_particle)
        # Obtain states from set of transitions
        theta_sys = np.concatenate([[init_particle[0]], tau_sys[:, -2]])
        
        # Record actions in this rollout by system dynamics
        actions_sys.append(tau_sys[:, 2])

        # List of trajectories via NN dynamics, each with different dropout mask
        list_tau_NN = []
        for _ in range(N_uncertainty):
            # New dropout mask
            dynamics.set_sampling(sampling=True, batch_size=1)
            # Take one trajectory via NN dynamics with fixed initial particle
            list_tau_NN.append(rollout(env, policy, dynamics, T=T, mode='NN', init_particle=init_particle))

        list_tau_NN = np.array(list_tau_NN)
        
        # Record actions in list of rollouts by NN dynamics
        actions_NN.append(list_tau_NN[:, :, 2])

        # Plot trajectory via system dynamics
        ax.plot(np.arange(0, theta_sys.shape[0]), theta_sys, 'green')
        # Plot uncertainty via NN dynamics, mean and +/- 2 std
        if not real_tau:  # plot shaded uncertainty
            # Calculate list of mu and sigma for set of trajectories via NN dynamics
            mu = [init_particle[0]]  # theta of initial particle
            sigma = [0.0]  # no uncertainty of initial particle
            for t in range(T):
                theta = list_tau_NN[:, t, -2]  # theta at time t for each trajectory 
                mu.append(theta.mean())
                sigma.append(theta.std())

            mu = np.array(mu)
            sigma = np.array(sigma)
            
            # Plotting
            ax.plot(np.arange(0, T + 1), mu, 'blue')
            ax.fill_between(np.arange(0, T + 1), mu - 1*sigma, mu + 1*sigma, facecolor='blue', alpha=0.3)
            ax.fill_between(np.arange(0, T + 1), mu - 2*sigma, mu + 2*sigma, facecolor='blue', alpha=0.2)
        else:  # plot real trajectories
            for tau in list_tau_NN:
                theta_NN = np.concatenate([[init_particle[0]], tau[:, -2]])
                ax.plot(np.arange(0, theta_NN.shape[0]), theta_NN, 'blue')

        ax.set_xlabel('Time steps')
        ax.set_ylabel('Angle (radian)')
        ax.grid()
        if ylim is not None:
            ax.set_ylim(ylim)
    
    plt.close()  # Close figure to save memory
    
    # Adjust subplots to avoid label overlapping
    fig.tight_layout()
    return fig, plot_actions(np.array(actions_sys), np.array(actions_NN), T=T, N_plots=N_plots, ylim=[-2.3, 2.3])

def plot_actions(actions_sys, actions_NN, T=1, N_plots=1, ylim=None):
    if ylim is None:
        fig, axes = plt.subplots(N_plots, N_plots, sharex=True, figsize=[6*N_plots, 4*N_plots])
    else:
        fig, axes = plt.subplots(N_plots, N_plots, sharex=True, sharey=True, figsize=[6*N_plots, 4*N_plots])
    if N_plots == 1:
        axes = np.array([axes])
        
    for ax, action_sys, action_NN in zip(axes.reshape(-1), actions_sys, actions_NN):
        ax.plot(np.arange(1, action_sys.shape[0] + 1), action_sys, 'green')
        for a in action_NN:
            ax.plot(np.arange(1, a.shape[0] + 1), a, 'blue')
            
        ax.set_xlabel('Time steps')
        ax.set_ylabel('Action values')
        ax.grid()
        if ylim is not None:
            ax.set_ylim(ylim)
        
    
    plt.close()  # Close figure to save memory
    # Adjust subplots to avoid label overlapping
    fig.tight_layout()  
    return fig


def test_episodic_reward(env, policy, N=1, T=1, render=False):
    ep_reward = []
    
    for _ in range(N):  # N episodes
        # Initial state
        s = env.reset()
        s = Variable(torch.FloatTensor(s).unsqueeze(0)).cuda()
        
        # Accumulated reward for current episode
        reward = []
        
        for t in range(T):  # T time steps
            # Select action via policy
            a = policy(s).data.cpu().numpy()
            # Take action in the environment
            s_next, r, done, info = env.step(a)
            
            # Record reward
            reward.append(r)
            
            # Update new state
            s = Variable(torch.FloatTensor(s_next).unsqueeze(0)).cuda()
            
            if render:
                env.render()
                
        ep_reward.append(np.sum(reward))
        
    return np.mean(ep_reward)
