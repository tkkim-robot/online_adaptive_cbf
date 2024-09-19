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
    def __init__(self, n_states=6, n_output=2, n_hidden=40, n_ensemble=3, device='cpu', lr=0.001, activation='relu'):
        super(ProbabilisticEnsembleNN, self).__init__()
        self.device = device
        self.n_states = n_states
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_ensemble = n_ensemble
        self.scaler = None

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

        if device == 'cuda':
            self.model = nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = self.gaussian_nll_loss  # Custom Gaussian NLL Loss
        self.mse_loss = nn.MSELoss()
        self.best_test_err = 10000.0

    def predict(self, input_array):
        # Ensure input is a NumPy array
        input_array = np.array(input_array)

        # If input is 1D, reshape it to 2D
        if input_array.ndim == 1:
            input_array = input_array[np.newaxis, :]

        # Transform Theta into sine and cosine components
        theta = input_array[:, 2]
        input_transformed = np.column_stack((input_array[:, :2], np.sin(theta), np.cos(theta), input_array[:, 3:]))

        # Normalize the inputs using the loaded scaler
        input_scaled = self.scaler.transform(input_transformed)

        # Convert the input to a PyTorch tensor and move it to the device
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(self.device)

        # Predict the output using the trained model
        self.model.eval()
        with torch.no_grad():
            ensemble_outputs = self.model(input_tensor)

        y_pred_safety_loss = []
        y_pred_deadlock_time = []
        div_list = []

        # Extract mean, variance, and divergence for each input in the batch
        for i in range(input_tensor.shape[0]):
            safety_loss_ensembles = []
            deadlock_time_ensembles = []

            # Loop through each ensemble prediction
            for ensemble_idx in range(self.n_ensemble):
                mu, log_std = ensemble_outputs[ensemble_idx]
                yhat_mu = mu[i].cpu().numpy()
                yhat_sig = torch.square(torch.exp(log_std[i]))

                # Collect safety loss and deadlock time predictions
                safety_loss_ensembles.append([yhat_mu[0], yhat_sig[0]])
                deadlock_time_ensembles.append([yhat_mu[1], yhat_sig[1]])

            y_pred_safety_loss.append(safety_loss_ensembles)
            y_pred_deadlock_time.append(deadlock_time_ensembles)

            # Extract divergence value for each input
            div = ensemble_outputs[-1][i].cpu().numpy()[0] 
            div_list.append(div)

        return y_pred_safety_loss, y_pred_deadlock_time, div_list

    def create_gmm(self, predictions, num_components=3):
        num_components = self.n_ensemble
        means = []
        variances = []
        for i in range(num_components):
            try:
                mu, sigma_sq = predictions[i]  
            except:
                mu, sigma_sq = predictions[0][i]  
                
            means.append(mu)  
            variances.append(sigma_sq) 

        means = np.array(means).reshape(-1, 1)  
        variances = np.array(variances).reshape(-1, 1, 1)

        gmm = GaussianMixture(n_components=num_components)
        gmm.means_ = means
        gmm.covariances_ = variances
        gmm.weights_ = np.ones(num_components) / num_components  

        try:
            gmm.precisions_cholesky_ = np.array([np.linalg.cholesky(np.linalg.inv(cov)) for cov in gmm.covariances_])
        except np.linalg.LinAlgError:
            pass
    
        return gmm
    
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

    def test(self, test_loader, epoch):
        self.model.eval()
        test_loss = torch.FloatTensor([0])
        test_mse = torch.FloatTensor([0])

        with torch.no_grad():
            for batch_idx, samples in enumerate(test_loader):
                x_test, y_test = samples
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                ensemble_outputs = self.model(x_test)

                # Initialize variables for ensemble loss and MSE
                ensemble_loss = 0
                ensemble_mse = 0

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
            checkpoint = torch.load(model_path)
            
            # Adjust the state_dict keys if they have 'model.' prefix
            if "model." in list(checkpoint.keys())[0]:
                new_state_dict = {}
                for k, v in checkpoint.items():
                    name = k.replace("model.", "")  # remove 'model.' prefix
                    new_state_dict[name] = v
                checkpoint = new_state_dict
            self.model.load_state_dict(checkpoint)       
            
            # self.model.load_state_dict(checkpoint)
        else:
            print("Model path does not exist. Check the provided path.")
            
    def load_scaler(self, scaler_path):
        self.scaler = joblib.load(scaler_path)

