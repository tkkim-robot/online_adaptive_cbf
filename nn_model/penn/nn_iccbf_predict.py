import numpy as np
import pandas as pd
import os
import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from sklearn.mixture import GaussianMixture
import joblib


class ProbabilisticEnsembleNN(nn.Module):
    def __init__(self, n_states=6, n_output=2, n_hidden=40, n_ensemble=3, device='cpu', lr=0.001, activation='relu', theta_index=None):
        super(ProbabilisticEnsembleNN, self).__init__()
        self.device = device
        self.n_states = n_states
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_ensemble = n_ensemble
        self.scaler = None
        # theta_index: index of theta in the input array before transformation
        # Default to 2 for compatibility (DynamicUnicycle2D, KinematicBicycle2D_DPCBF)
        # Quad3D and Quad2D have theta at index 3
        self.theta_index = theta_index if theta_index is not None else 2

        try:
            from penn.penn import EnsembleStochasticLinear
        except:
            from nn_model.penn.penn import EnsembleStochasticLinear
        self.model = EnsembleStochasticLinear(in_features=self.n_states,
                                                out_features=self.n_output,
                                                hidden_features=self.n_hidden,
                                                ensemble_size=self.n_ensemble, 
                                                activation=activation, 
                                                explore_var='jrd', 
                                                residual=True)

        self.model = self.model.to(device)
        if device == 'cuda' and torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs for DataParallel")
            self.model = nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = self.gaussian_nll_loss  # Custom Gaussian NLL Loss
        self.mse_loss = nn.MSELoss()
        self.best_test_err = 10000.0

    def predict(self, input_array):
        input_array = np.atleast_2d(input_array)  # Ensures 2D input
        theta = input_array[:, self.theta_index]
        # Split array: everything before theta, theta, everything after theta
        input_transformed = np.column_stack((input_array[:, :self.theta_index], 
                                           np.sin(theta), np.cos(theta), 
                                           input_array[:, self.theta_index+1:]))
        input_scaled = self.scaler.transform(input_transformed)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            ensemble_outputs = self.model(input_tensor)  # list of (mu, log_std) per ensemble + div

        # Ensemble outputs: list of (mu[B,2], log_std[B,2])
        mu_list, logstd_list = zip(*ensemble_outputs[:-1])
        mu_stack     = torch.stack(mu_list,     dim=0)   # (E,B,2)
        logstd_stack = torch.stack(logstd_list, dim=0)   # (E,B,2)
        sigma_sq     = torch.exp(logstd_stack).pow(2)

        # Transpose to shape (B, E, 2)
        mu_np = mu_stack.permute(1, 0, 2).cpu().numpy()       # (B, E, 2)
        sigma_np = sigma_sq.permute(1, 0, 2).cpu().numpy()    # (B, E, 2)

        # Split predictions
        y_pred_safety_loss = [[list(mv) for mv in zip(m[:, 0], s[:, 0])] for m, s in zip(mu_np, sigma_np)]
        y_pred_deadlock_time = [[list(mv) for mv in zip(m[:, 1], s[:, 1])] for m, s in zip(mu_np, sigma_np)]

        div = ensemble_outputs[-1][:, 0].cpu().numpy().tolist()

        return y_pred_safety_loss, y_pred_deadlock_time, div
    
    def create_gmm(self, predictions, num_components=3):
        num_components = self.n_ensemble
        mu_list, var_list = [], []

        for i in range(num_components):
            mu, sigma_sq = predictions[i] if i < len(predictions) else predictions[0][i]
            mu_list.append(float(mu))
            var_list.append(float(sigma_sq))

        means_arr = np.array(mu_list, dtype=np.float64).reshape(-1, 1)           # (E,1)
        covs_arr  = np.array(var_list, dtype=np.float64).reshape(-1, 1, 1)       # (E,1,1)

        class _SimpleGMM:
            pass
        gmm = _SimpleGMM()               # lightweight placeholder
        gmm.means_        = means_arr
        gmm.covariances_  = covs_arr     # diag cov
        return gmm    
    
    def train(self, train_loader, epoch):
        # train a single epoch
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        err_list = []
        # print('\nEpoch: %d' % epoch)
        train_loss = torch.FloatTensor([0]).to(self.device)
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

    def test(self, test_loader, epoch):
        self.model.eval()
        test_loss = torch.FloatTensor([0]).to(self.device)
        test_mse = torch.FloatTensor([0]).to(self.device)

        with torch.no_grad():
            for batch_idx, samples in enumerate(test_loader):
                x_test, y_test = samples
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                ensemble_outputs = self.model(x_test)

                # Initialize variables for ensemble loss and MSE
                ensemble_loss = torch.FloatTensor([0]).to(self.device)
                ensemble_mse = torch.FloatTensor([0]).to(self.device)

                # Loop through each ensemble
                for ensemble_idx in range(self.n_ensemble):
                    mu, log_std = ensemble_outputs[ensemble_idx]
                    var = torch.square(torch.exp(log_std))

                    # Compute loss for the current ensemble
                    loss = self.criterion(mu, y_test, var)
                    ensemble_loss += loss

                    # Compute MSE for the current ensemble
                    dist = Normal(mu, torch.exp(log_std))
                    dist_sample = dist.rsample()
                    mse = self.mse_loss(dist_sample, y_test)
                    ensemble_mse += mse

                # Average the loss and MSE across all ensembles
                ensemble_loss /= self.n_ensemble
                ensemble_mse /= self.n_ensemble

                test_loss += ensemble_loss
                test_mse += ensemble_mse

        err = float(test_loss.item() / float(len(test_loader)))
        print('Testing ==> Epoch {:2d} Cost: {:.6f}'.format(epoch, err))
        test_rmse = math.sqrt(test_mse.item() / len(test_loader))
        print('test RMSE : {}'.format(test_rmse))

        if epoch == 0:
            self.best_test_err = 10000.0
        bool_best = False
        if test_rmse < self.best_test_err:
            if not os.path.isdir('checkpoint'):
                os.makedirs('checkpoint/', exist_ok=True)
            print("Best Model Saving...")
            torch.save(self.model.state_dict(), 'checkpoint/temp.pth')
            self.best_test_err = test_rmse
            bool_best = True

        return err, bool_best, test_rmse

    def gaussian_nll_loss(self, mu, target, var):
        # Custom Gaussian Negative Log Likelihood Loss
        loss = 0.5 * (torch.log(var) + (target - mu) ** 2 / var)
        return torch.mean(loss)

    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_path):
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Adjust the state_dict keys if they have 'model.' prefix
            if "model." in list(checkpoint.keys())[0]:
                new_state_dict = {}
                for k, v in checkpoint.items():
                    name = k.replace("model.", "")  # remove 'model.' prefix
                    new_state_dict[name] = v
                checkpoint = new_state_dict
            self.model.load_state_dict(checkpoint)       
            self.model.to(self.device)  

        else:
            print("Model path does not exist. Check the provided path.")
            
    def load_scaler(self, scaler_path):
        self.scaler = joblib.load(scaler_path)

