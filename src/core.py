import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

"""
This module defines the NNs used to parameterize the value and policy function when we have
a single continuous action.
"""

#Constants used by the NNs to prevent numerical instabilities
LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    return a sequential net of fully connected layers

    Args:
        sizes(tuple(int)): sizes of all the layers
        activation: activation function for all layers except for the output layer
        output_activation: activation to use for output layer only

    Returns:
        stacked fully connected layers
    """
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    """counts the variables of a module """
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActorCritic(nn.Module):
    """
    Module that contains two value functions, self.q1 and self.q2, and the policy self.pi.
    Both are parameterizes using an arbitrary number of fully connected layers.
    WARNING: it only works for 1 continuous action. Should be easy to generalize to more.

    Args:
        observation_space: observation space of the environment
        action_space: action space of sac_envs environments
        hidden_sizes (tuple(int)): size of each of the hidden layers that will be addded to 
            both the value and policy functions
        activation: activation function for all layers except for the output layer

    """
    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        #load size of observation and action space
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_lower_limit = action_space.low[0] #changed to introduce upper and lower limits
        act_upper_limit = action_space.high[0] #changed to introduce upper and lower limits
        
        #raise error if the size isn't supported        
        if act_dim > 1:
            raise NameError("There are various things that are not properly implemented for multiple continuous actions.")
        
        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_lower_limit, act_upper_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        """ return the action, chosen according to deterministic, given a single unbatched observation obs """
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a

class MLPQFunction(nn.Module):
    """
    Class representing a q-value function, implemented with fully connected layers
    that stack the state-action as input, and output the value of such state-action

    Args:
        obs_dim(int): number of continuous state variables
        act_dim(int): number of continuous actions
        hidden_sizes(tuple(int)): list of sizes of hidden layers
        activation: activation function for all layers except for output
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        """
        Args:
            obs(torch.Tensor): batch of observations
            act(torch.Tensor): batch of continuous actions

        Returns:
            (torch.Tensor): 1D tensor with value of each state-action in the batch
        """
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class SquashedGaussianMLPActor(nn.Module):
    """
    Class representing the policy, implemented with fully connected layers
    that take the state as input, and outputs the average and sigma of the
    probability density of chosing a continuous action. We use a squashed gaussian policy.

    Args:
        obs_dim(int): number of continuous state variables
        act_dim(int): number of continuous actions
        hidden_sizes(tuple(int)): list of sizes of hidden layers
        activation: activation function for all layers except for output
        act_lower_limit (float): minimum value of the single continuous action
        act_upper_limit (float): maximum value of the single continuous action
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_lower_limit, act_upper_limit):
        super().__init__()
        #save the bounds for the continuous action
        self.act_lower_limit = act_lower_limit
        self.act_upper_limit = act_upper_limit

        #main network taking the state as input, and passing it through all hidden layers
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        #output layer producing the average of the probability densities of the action
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        #output layer producing log(sigma) of the probability densities of the action
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs, deterministic=False, with_logprob=True):
        """
        Args:
            obs(torch.Tensor): batch of observations
            deterministic(bool): if the actions should be chosen deterministally or not
            with_logprob(bool): if the log of the probability should be computed and returned

        Returns:
            pi_action(torch.Tensor): the chosen continuous action
            logp_pi(torch.Tensor): the log probability of such continuous action
        """
        #run the state through the network
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash (tanh) distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
        
        #if necessary, compute the log of the probabilities
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            #change of distribution when going from gaussian to Tanh
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
            #change of distribution when going from Tanh to the interval [act_lower_limit,act_upper_limit]
            logp_pi -= np.log(0.5*(self.act_upper_limit-self.act_lower_limit)) 
        else:
            logp_pi = None
       
        #Apply Tanh to the sampled gaussian
        pi_action = torch.tanh(pi_action)
        #Apply the shift of the action to the correct interval
        pi_action = self.act_lower_limit + 0.5*(pi_action + 1.)*(self.act_upper_limit-self.act_lower_limit)

        return pi_action, logp_pi



