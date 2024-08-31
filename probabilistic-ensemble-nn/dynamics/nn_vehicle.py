import numpy as np
import pandas as pd
import os
import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from sklearn.preprocessing import StandardScaler
import joblib

PATH = '/home/add/Desktop/auto_ws/src/auto_learn'


class VehicleModel(nn.Module):
    def __init__(self, n_states, n_output, n_hidden, n_ensemble, device, lr=0.001, model_type='vanilla', incremental=False, activation='relu'):
        super(VehicleModel, self).__init__()
        self.device = device
        self.n_states = n_states
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_ensemble = n_ensemble
        self.scaler = StandardScaler()
        self.model = None  # Define the model here based on your requirements

        # Model type configuration
        if model_type == 'vanilla':
            from dynamics.nn_sto_ens import EnsembleStochasticLinear
            self.model = EnsembleStochasticLinear(in_features=self.n_states,
                                                  out_features=self.n_output,
                                                  hidden_features=self.n_hidden,
                                                  ensemble_size=self.n_ensemble, 
                                                  activation=activation, 
                                                  explore_var='jrd', 
                                                  residual=True)

        self.model = self.model.to(device)

        if device == 'cuda':
            self.model = nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = self.gaussian_nll_loss  # Custom Gaussian NLL Loss
        self.mse_loss = nn.MSELoss()
        self.best_test_err = 10000.0

    def gaussian_nll_loss(self, mu, target, var):
        """
        Custom Gaussian Negative Log Likelihood Loss
        :param mu: Predicted mean
        :param target: True target value
        :param var: Predicted variance
        :return: Loss value
        """
        loss = 0.5 * (torch.log(var) + (target - mu) ** 2 / var)
        return torch.mean(loss)

    def load_and_preprocess_data(self, data_file, scaler_path=None):
        # Load data
        dataset = pd.read_csv(data_file)

        # Define input features and outputs
        # X = dataset[['Distance_obs1', 'Distance_obs2', 'Distance_obs3', 'Velocity', 'Theta', 'Gamma1', 'Gamma2']].values
        X = dataset[['Distance', 'Velocity', 'Theta', 'Gamma1', 'Gamma2']].values
        y = dataset[['Safety Loss', 'Deadlock Time']].values 

        # Transform Theta into sine and cosine components
        Theta = X[:, 2]
        X_transformed = np.column_stack((X[:, :2], np.sin(Theta), np.cos(Theta), X[:, 3:]))

        # Normalize the inputs
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)  # Load existing scaler
        else:
            self.scaler.fit(X_transformed)  # Fit new scaler

        X_scaled = self.scaler.transform(X_transformed)

        # Save the scaler for later use
        if scaler_path:
            joblib.dump(self.scaler, scaler_path)

        # Splitting data into training and testing sets
        train_size = int(0.7 * len(X_scaled))
        train_dataX, test_dataX = X_scaled[:train_size], X_scaled[train_size:]
        train_dataY, test_dataY = y[:train_size], y[train_size:]

        return train_dataX, train_dataY, test_dataX, test_dataY

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
        # print('Data Size:', len(train_loader))
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

    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_path):
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint)
        else:
            print("Model path does not exist. Check the provided path.")
            
            

def model_save(model):
    os.makedirs('model', exist_ok=True)
    torch.save(model.state_dict(), 'model/nn_pendulum.pth')


def model_load(model):
    checkpoint = torch.load('model/nn_pendulum.pth')
    model.load_state_dict(checkpoint)
    return model
