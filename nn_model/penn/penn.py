"""
    Neural Network Dynamics with the following options implented:
        1. Stochastic inference using [mean, std_dev] split
        2. Truely parallel ensemble
        3. API for encoding historical state-action pairs
"""
import torch
import torch.nn as nn
import numpy as np
import torch.distributions as D
from torch.distributions.normal import Normal

try:
    from penn.ensemble.ensemble_linear import EnsembleLinear
    from penn.divergence.utility import JensenRenyiDivergence
except:
    from nn_model.penn.ensemble.ensemble_linear import EnsembleLinear
    from nn_model.penn.divergence.utility import JensenRenyiDivergence

class EnsembleStochasticLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_features, ensemble_size=3, activation='relu', explore_var='legacy', residual=True, dt=0.1):
        super(EnsembleStochasticLinear, self).__init__()
        self.ensemble_size = ensemble_size
        self.residual = residual
        self.dt = dt
        self.explore_var = explore_var
        self.n_output = out_features

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

    def forward(self, x):
        prev_x = x.clone().detach()  # save previous state (history)
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
        x = self.act(self.lin3(x))
        x = self.act(self.lin4(x))
        x = self.lin5(x)

        mu = x[:, :, :self.n_output]
        log_std = x[:, :, self.n_output:]
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # List to store mean and log_std of each ensemble
        ensemble_outputs = []
        
        # Loop through each ensemble to collect mu and log_std
        for i in range(self.ensemble_size):
            yhat = (mu[i], log_std[i])
            ensemble_outputs.append(yhat)

        if self.explore_var == 'jrd':
            std = torch.exp(log_std)
            jrd_div = JensenRenyiDivergence(
                states_mean=mu, states_var=std.square()).compute_measure()
            dis = jrd_div.abs().unsqueeze(1)

        return (*ensemble_outputs, dis)

    def single_forward(self, x, index):
        prev_x = x.clone().detach()  # save previous state
        x = self.act(self.lin1.single_forward(x, index))
        x = self.act(self.lin2.single_forward(x, index))
        x = self.act(self.lin3.single_forward(x, index))
        x = self.act(self.lin4.single_forward(x, index))
        x = self.lin5.single_forward(x, index)

        mu = x[:, :, :self.n_output]
        log_std = x[:, :, self.n_output:]
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        yhat = mu.squeeze(dim=0), log_std.squeeze(dim=0)  # indexing 0

        return yhat
