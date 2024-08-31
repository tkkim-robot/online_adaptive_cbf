import numpy as np
import os
import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

PATH = '/home/add/Desktop/auto_ws/src/auto_learn'


class VehicleModel(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden, n_ensemble, n_history, n_trajectory, device, lr=0.001, model_type='vanilla', incremental=False, activation='relu'):
        super(VehicleModel, self).__init__()
        self.device = device
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.n_ensemble = n_ensemble
        self.n_history = n_history
        self.n_trajectory = n_trajectory
        if model_type == 'vanilla':
            from auto_learn.dynamics.nn_sto_ens import EnsembleStochasticLinear
            self.model = EnsembleStochasticLinear(in_features=((self.n_states + self.n_actions)*self.n_history),
                                                  out_features=self.n_states,
                                                  hidden_features=self.n_hidden, trajectory_size=self.n_trajectory, ensemble_size=self.n_ensemble, activation=activation, explore_var='jrd', residual=True)

        self.model = self.model.to(device)  # kaiming init

        if device == 'cuda':
            model = nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True
        print('Am I using CPU or GPU : {}'.format(device))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.GaussianNLLLoss()
        self.mse_loss = nn.MSELoss()
        self.incremental = incremental
        self.dataset = None

        self.best_test_err = 10000.0
        self.history_data = torch.zeros(
            self.n_trajectory, (self.n_states + self.n_actions), self.n_history).to(self.device)

    def forward(self, state, action):
        xu_history = self.make_history_data(state, action)
        with torch.no_grad():  # only forwarding here
            yhat, ens_var = self.model(xu_history)  # Ensemble*K*((X+U)*H)
        return yhat, ens_var

    # JW
    def make_history_data(self, state, action):
        # TODO: measure state and action's tensor dimension
        #       it should be K*S and K*X
        # remove oldest history
        self.history_data = self.history_data[:, :, 1:]
        current_data = torch.cat([state, action], dim=1).unsqueeze(
            2)  # shoule be K*(S+X)*1
        self.history_data = torch.cat(
            [self.history_data, current_data], dim=2)  # K*(X+S)*H

        # 2D tensor with batches: K*((X+S)*H)
        # return self.history_data.view((self.n_trajectory, -1))

        temp_history_data = self.history_data.permute(0, 2, 1)
        return temp_history_data.reshape(self.n_trajectory, -1)

    def initialize_history(self, states):
        # TODO: Expand the states to N Trajectories
        self.history_data = states.to(self.device)

    def inference(self, state, action):
        (res_mu, log_std), ens_var = self.forward(state, action)  # states

        std = torch.exp(log_std)
        dist = Normal(res_mu, std)
        res_sample = dist.rsample()
        # next_state = state.clone().detach() + res_sample

        # return next_state, ens_var
        return res_mu, ens_var

        # u = perturbed_action
        # # u = torch.clamp(perturbed_action, ACTION_LOW, ACTION_HIGH)

        # self.forward(state, u)
        # with torch.no_grad():
        #     (res_mu, log_std), ensemble_var = self.forward(state, u)

        # std = torch.exp(log_std)
        # dist = Normal(res_mu, std)
        # res_sample = dist.rsample()
        # # output dtheta directly so can just add
        # next_state = state.clone().detach() + res_sample

        # return next_state, ensemble_var

    def train(self, train_loader, epoch):
        # train a single epoch
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        err_list = []
        # print('\nEpoch: %d' % epoch)
        train_loss = torch.FloatTensor([0])
        for batch_idx, samples in enumerate(train_loader):
            x_train, y_train = samples
            x_train, y_train = x_train.to(self.device), y_train.to(self.device)
            for model_index in range(self.n_ensemble):
                self.optimizer.zero_grad(set_to_none=True)
                (mu, log_std) = self.model.single_forward(
                    x_train, model_index)

                yhat_mu = mu
                var = torch.square(torch.exp(log_std))

                loss = self.criterion(yhat_mu, y_train, var)
                loss.mean().backward()
                self.optimizer.step()
                train_loss += loss

        err = float(train_loss.item() / float(len(train_loader)))
        print('Training ==> Epoch {:2d}  Cost: {:.6f}'.format(epoch, err))
        print('Data Size:', len(train_loader))
        err_list.append(err)
        return err

    def test(self, test_loader, epoch, verbose=False):
        self.model.eval()
        test_loss = torch.FloatTensor([0])
        test_mse = torch.FloatTensor([0])

        if verbose:
            vx_loss = torch.FloatTensor([0])  # tensor  with 1 dim, 0.0 element
            vy_loss = torch.FloatTensor([0])
            yawrate_loss = torch.FloatTensor([0])

            vx_mse = torch.FloatTensor([0])  # tensor  with 1 dim, 0.0 element
            vy_mse = torch.FloatTensor([0])
            yawrate_mse = torch.FloatTensor([0])

        with torch.no_grad():
            for batch_idx, samples in enumerate(test_loader):
                x_test, y_test = samples
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                yhat, ens_var = self.model(x_test)
                (mu, log_std) = yhat

                yhat_mu = mu
                var = torch.square(torch.exp(log_std))

                test_loss += self.criterion(yhat_mu, y_test, var)

                yhat_std = torch.exp(log_std)
                dist = Normal(yhat_mu, yhat_std)
                dist_sample = dist.rsample()  # Batch*X

                test_mse += self.mse_loss(dist_sample, y_test)

                if verbose:
                    vx_loss += self.criterion(
                        yhat_mu[:, 0], y_test[:, 0], var[:, 0])
                    vy_loss += self.criterion(
                        yhat_mu[:, 1], y_test[:, 1], var[:, 1])
                    yawrate_loss += self.criterion(
                        yhat_mu[:, 2], y_test[:, 2], var[:, 2])
                    vx_mse += self.mse_loss(dist_sample[:, 0], y_test[:, 0])
                    vy_mse += self.mse_loss(dist_sample[:, 1], y_test[:, 1])
                    yawrate_mse += self.mse_loss(
                        dist_sample[:, 2], y_test[:, 2])

        err = float(test_loss.item() / float(len(test_loader)))
        print(
            'Testing ==> Epoch {:2d} Cost: {:.6f}'.format(epoch, err))
        test_rmse = math.sqrt(test_mse.item()/len(test_loader))
        print('test RMSE : {}'.format(test_rmse))

        if verbose:
            vx_loss = float(vx_loss.item() / float(len(test_loader)))
            vy_loss = float(vy_loss.item() / float(len(test_loader)))
            yawrate_loss = float(yawrate_loss.item() / float(len(test_loader)))
            # print('vx Gussian NLL Loss : {}'.format(vx_loss))
            # print('vy Gussian NLL Loss : {}'.format(vy_loss))
            # print('r  Gussian NLL Loss : {}'.format(yawrate_loss))
            vx_rmse = math.sqrt(vx_mse.item()/len(test_loader))
            vy_rmse = math.sqrt(vy_mse.item()/len(test_loader))
            yawrate_rmse = math.sqrt(yawrate_mse.item()/len(test_loader))
            # print('vx RMSE   : {}'.format(vx_rmse))
            # print('vy RMSE   : {}'.format(vy_rmse))
            # print('r  RMSE   : {}'.format(yawrate_rmse))

        if epoch == 0:
            self.best_test_err = 10000.0
        bool_best = False
        if test_rmse < self.best_test_err:
            if not os.path.isdir(PATH + 'auto_learn/checkpoint'):
                os.makedirs(
                    PATH + '/auto_learn/checkpoint/', exist_ok=True)
            print("Best Model Saving...")
            torch.save(
                self.model.state_dict(), PATH + '/auto_learn/checkpoint/temp.pth')
            self.best_test_err = test_rmse
            bool_best = True

        return err, bool_best, test_rmse

    def validation(self, val_loader):
        self.model.eval()
        test_loss = torch.FloatTensor([0])

        vx_loss = torch.FloatTensor([0])  # tensor  with 1 dim, 0.0 element
        vy_loss = torch.FloatTensor([0])  # tensor  with 1 dim, 0.0 element
        yawrate_loss = torch.FloatTensor([0])

        val_mse = torch.FloatTensor([0])
        vx_mse = torch.FloatTensor([0])  # tensor  with 1 dim, 0.0 element
        vy_mse = torch.FloatTensor([0])
        yawrate_mse = torch.FloatTensor([0])

        with torch.no_grad():
            for batch_idx, samples in enumerate(val_loader):
                x_test, y_test = samples
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                yhat, ens_var = self.model(x_test)
                (mu, log_std) = yhat

                yhat_mu = mu
                var = torch.square(torch.exp(log_std))

                test_loss += self.criterion(yhat_mu, y_test, var)
                vx_loss += self.criterion(
                    yhat_mu[:, 0], y_test[:, 0], var[:, 0])
                vy_loss += self.criterion(
                    yhat_mu[:, 1], y_test[:, 1], var[:, 1])
                yawrate_loss += self.criterion(
                    yhat_mu[:, 2], y_test[:, 2], var[:, 2])

                yhat_std = torch.exp(log_std)
                dist = Normal(yhat_mu, yhat_std)
                dist_sample = dist.rsample()  # Batch*X # X t+1

                val_mse += self.mse_loss(dist_sample, y_test)
                vx_mse += self.mse_loss(dist_sample[:, 0], y_test[:, 0])
                vy_mse += self.mse_loss(dist_sample[:, 1], y_test[:, 1])
                yawrate_mse += self.mse_loss(dist_sample[:, 2], y_test[:, 2])

        print('Validation')
        err = float(test_loss.item() / float(len(val_loader)))
        print(
            'Valdation ==> Cost: {:.6f}'.format(err))

        vx_loss = float(vx_loss.item() / float(len(val_loader)))
        vy_loss = float(vy_loss.item() / float(len(val_loader)))
        yawrate_loss = float(yawrate_loss.item() / float(len(val_loader)))
        print('vx Gussian NLL Loss : {}'.format(vx_loss))
        print('vy Gussian NLL Loss : {}'.format(vy_loss))
        print('yaw rate Gussian NLL Loss : {}'.format(yawrate_loss))

        val_rmse = math.sqrt(val_mse.item()/len(val_loader))
        vx_rmse = math.sqrt(vx_mse.item()/len(val_loader))
        vy_rmse = math.sqrt(vy_mse.item()/len(val_loader))
        yawrate_rmse = math.sqrt(yawrate_mse.item()/len(val_loader))
        print('val RMSE : {}'.format(val_rmse))
        print('vx RMSE   : {}'.format(vx_rmse))
        print('vy RMSE   : {}'.format(vy_rmse))
        print('r  RMSE   : {}'.format(yawrate_rmse))
        return vx_loss, vy_loss, yawrate_loss, val_rmse, vx_rmse, vy_rmse, yawrate_rmse


def model_save(model):
    os.makedirs('model', exist_ok=True)
    torch.save(model.state_dict(), 'model/nn_pendulum.pth')


def model_load(model):
    checkpoint = torch.load('model/nn_pendulum.pth')
    model.load_state_dict(checkpoint)
    return model
