import abc

import torch
from gym import spaces
from torch import nn as nn

from habitat.config import Config
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.utils.common import CategoricalNet, GaussianNet


class Policy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, net, dim_actions, policy_config=None):
        super().__init__()
        self.net = net

        self.dim_actions = dim_actions

        self.critic = CriticHead(self.net.output_size, self.dim_actions)

    def forward(self, *x):
        raise NotImplementedError

    def act(self, observations):
        features = self.net(observations)

        action_values = self.critic(features)

        action = action_values.max(1)[0]

        return action_values, action

    def get_value(self, observations):
        features = self.net(observations)
        return self.critic(features)

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space):
        pass


class CriticHead(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


