"""
    Neural Network Dynamics with the following options implented:
        1. Stochastic inference using [mean, std_dev] split
        2. Truely parallel ensemble
        3. API for encoding historical state-action pairs
"""
from ast import JoinedStr
import torch
import torch.nn as nn
import numpy as np
import torch.distributions as D
from torch.distributions.normal import Normal

from auto_learn.dynamics.ensemble.ensemble_linear import EnsembleLinear
from auto_learn.dynamics.divergence.utility import JensenRenyiDivergence


class EnsembleStochasticLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_features, trajectory_size, ensemble_size=1, activation='relu', explore_var='legacy', residual=True, dt=0.1):
        super(EnsembleStochasticLinear, self).__init__()
        self.trajectory_size = trajectory_size
        self.ensemble_size = ensemble_size
        self.residual = residual
        self.dt = dt
        self.explore_var = explore_var
        self.n_states = out_features

        self.lin1 = EnsembleLinear(in_features=in_features,
                                   out_features=hidden_features, ensemble_size=self.ensemble_size, bias=True)
        self.lin2 = EnsembleLinear(in_features=hidden_features,
                                   out_features=hidden_features * 2, ensemble_size=self.ensemble_size, bias=True)
        self.lin3 = EnsembleLinear(in_features=hidden_features * 2,
                                   out_features=hidden_features * 3, ensemble_size=self.ensemble_size, bias=True)
        self.lin4 = EnsembleLinear(in_features=hidden_features * 3,
                                   out_features=hidden_features, ensemble_size=self.ensemble_size, bias=True)
        self.lin5 = EnsembleLinear(in_features=hidden_features,
                                   out_features=out_features*2, ensemble_size=self.ensemble_size, bias=True)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU()
        elif activation == 'softplus':
            self.act = nn.Softplus()
        self.log_std_min = -20
        self.log_std_max = 1  # 2

        self.mix = D.Categorical(torch.ones(self.ensemble_size,
                                            self.trajectory_size, self.n_states))

    def forward(self, x):
        prev_x = x.clone().detach()  # save previous state (history)
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
        x = self.act(self.lin3(x))
        x = self.act(self.lin4(x))
        x = self.lin5(x)

        mu = x[:, :, :self.n_states]
        if self.residual:
            mu = prev_x[:, [15, 16, 17]] + mu  # * self.dt
            # mu = prev_x[:, [21, 22, 23, 24, 25]] + mu  # * self.dt

        current_state = prev_x[:, [15, 16, 17]]

        log_std = x[:, :, self.n_states:]
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # use gaussian mixture model to sample !!!!
        # shape[1] indicates batch size

        if self.explore_var == 'jrd':
            std = torch.exp(log_std)
            jrd_div = JensenRenyiDivergence(
                states=current_state, next_states_mean=mu, next_states_var=std.square()).compute_measure()
            dis = jrd_div.abs().unsqueeze(1)

        rand_idx = np.random.randint(self.ensemble_size, size=x.shape[1])
        batch_idx = np.arange(x.shape[1])
        yhat = mu[rand_idx, batch_idx], log_std[rand_idx, batch_idx]

        return yhat, dis

    def single_forward(self, x, index):
        prev_x = x.clone().detach()  # save previous state
        x = self.act(self.lin1.single_forward(x, index))
        x = self.act(self.lin2.single_forward(x, index))
        x = self.act(self.lin3.single_forward(x, index))
        x = self.act(self.lin4.single_forward(x, index))
        x = self.lin5.single_forward(x, index)

        mu = x[:, :, :self.n_states]
        if self.residual:
            mu = prev_x[:, [15, 16, 17]] + mu  # * self.dt
            # mu = prev_x[:, [21, 22, 23, 24, 25]] + mu  # * self.dt

        log_std = x[:, :, self.n_states:]
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        yhat = mu.squeeze(dim=0), log_std.squeeze(dim=0)  # indexing 0

        return yhat
