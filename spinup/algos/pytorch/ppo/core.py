import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

class HierarchicalActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(64,64), activation=nn.Tanh, latent_z_dimension=64):

        super().__init__()
        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

        # Build latent z policy and latent b policy... 
        self.latent_b_policy = MLPCategoricalActor(obs_dim, 1, hidden_sizes, activation)
        self.latent_z_policy = MLPGaussianActor(obs_dim, latent_z_dimension, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)

            latent_b_policy_distribution = self.latent_b_policy._distribution(obs)
            latent_z_policy_distribution = self.latent_z_policy._distribution(obs)

            latent_b = latent_b_policy_distribution.sample()
            latent_z = latent_z_policy_distribution.sample()

            latent_b_logprobabilitity = self.latent_b_policy._log_prob_from_distribution(latent_b_policy_distribution, latent_b)
            latent_z_logprobabilitity = self.latent_z_policy._log_prob_from_distribution(latent_z_policy_distribution, latent_z)

        # return a.numpy(), v.numpy(), logp_a.numpy()
        action_tuple = (a.numpy(), latent_b.numpy(), latent_z.numpy())
        logprobability_tuple = (logp_a.numpy(), latent_b_logprobabilitity.numpy(), latent_z_logprobabilitity.numpy())

        total_logprobabilities = logp_a.numpy() + latent_b_logprobabilitity.sum().numpy() + latent_z_logprobabilitity.sum().numpy()

        # return action_tuple, v.numpy(), logprobability_tuple
        return action_tuple, v.numpy(), total_logprobabilities

    def act(self, obs):
        return self.step(obs)[0]

    def evaluate_batch_logprob(self, obs, action_tuple):

        # Get the distributions from the observation.
        pi = self.pi._distribution(obs)
        latent_b_policy_distribution = self.latent_b_policy._distribution(obs)
        latent_z_policy_distribution = self.latent_z_policy._distribution(obs)

        # Get logprobabilities. 
        logp_a = self.pi._log_prob_from_distribution(pi, action_tuple[0])
        latent_b_logprobabilitity = self.latent_b_policy._log_prob_from_distribution(latent_b_policy_distribution, action_tuple[1])
        latent_z_logprobabilitity = self.latent_z_policy._log_prob_from_distribution(latent_z_policy_distribution, action_tuple[2])

        total_logprobabilities = logp_a.numpy() + latent_b_logprobabilitity.sum().numpy() + latent_z_logprobabilitity.sum().numpy()

        return pi, latent_b_policy_distribution, latent_z_policy_distribution, total_logprobabilities




