import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import numpy as np
from torch.distributions.normal import Normal
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.nn.functional import mse_loss
from sklearn.mixture import GaussianMixture
import joblib


class ProbabilisticEnsembleGAT(nn.Module):
    def __init__(self, gat_model, n_output=2, n_hidden=40, n_ensemble=3, device='cpu', lr=0.001, activation='relu'):
        super(ProbabilisticEnsembleGAT, self).__init__()
        self.device = device
        self.n_ensemble = n_ensemble
        self.n_hidden = n_hidden
        self.n_output = n_output
        
        self.gat_model = gat_model  
        self.gat_model.eval()       
        self.gat_model = self.gat_model.to(self.device)

        try:
            from penn.penn import EnsembleStochasticLinear
        except:
            from nn_model.penn.penn import EnsembleStochasticLinear
        self.model = EnsembleStochasticLinear(in_features=18,  # 16D embedding + 2D gamma
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

    def predict(self, data_list):
        loader = GeoDataLoader(data_list, batch_size=1, shuffle=False)
        self.model.eval()
        self.gat_model.eval()

        y_pred_safety_list = []
        y_pred_deadlock_list = []
        div_list = []

        with torch.no_grad():
            for batch_data in loader:
                x, edge_index, edge_attr, y, batch_idx = (
                    batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.y, batch_data.batch
                )
                # GAT embeddings
                robot_emb = self.gat_model.gat.extract_robot_embedding(x, edge_index, edge_attr, batch_idx)
                gamma = getattr(batch_data, 'gamma', None)
                if gamma is None:
                    gamma = torch.zeros((robot_emb.shape[0], 2), dtype=torch.float, device=self.device)
                else:
                    gamma = gamma.view(-1, 2).to(self.device)

                X_input = torch.cat([robot_emb, gamma], dim=1).to(self.device)
                
                # Each ensemble_out[en_idx] => (mu, log_std)
                #   ensemble_out => list of n_ensemble + 1 outputs
                #   each ensemble[i] => (mu, log_std)
                #   ensemble_out[-1] => divergence
                ensemble_out = self.model(X_input)
                
                y_safety_ensembles = []
                y_deadlock_ensembles = []
                for (mu, log_std) in ensemble_out[:-1]:
                    sigma_sq = torch.square(torch.exp(log_std))
                    y_safety_ensembles.append([mu[0,0].item(), sigma_sq[0,0].item()])
                    y_deadlock_ensembles.append([mu[0,1].item(), sigma_sq[0,1].item()])
                divergence = ensemble_out[-1][0,0].item()
                div_list.append(divergence)

                y_pred_safety_list.append(y_safety_ensembles)
                y_pred_deadlock_list.append(y_deadlock_ensembles)

        return y_pred_safety_list, y_pred_deadlock_list, div_list

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
        self.model.train()
        self.gat_model.eval()  

        total_loss = 0.0

        for batch_data in train_loader:
            x, edge_index, edge_attr, y, batch_idx = (
                batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.y, batch_data.batch
            )
            y = y.squeeze(1) if y.dim() == 3 else y  # Ensure shape [batch_size, 2]

            # 1) Extract 16D embedding from GAT
            with torch.no_grad():
                robot_emb = self.gat_model.extract_robot_embedding(x, edge_index, edge_attr, batch_idx)

            # Debugging: Check shape
            # print(f"robot_emb.shape: {robot_emb.shape}")  # Expected: [batch_size, 16]

            # 2) Obtain gamma 2D
            gamma = getattr(batch_data, 'gamma', None)
            gamma = gamma.view(-1, 2).to(self.device)

            # 3) Concatenate => shape [batch_size, 18]
            X_input = torch.cat([robot_emb, gamma], dim=1).to(self.device)
            y_target = y.to(self.device)

            # Debugging: Check final input shape
            # print(f"Final X_input shape: {X_input.shape}")  # Expected: torch.Size([batchsize, 18])

            # 4) Train each ensemble member
            for model_idx in range(self.n_ensemble):
                self.optimizer.zero_grad(set_to_none=True)
                mu, log_std = self.model.single_forward(X_input, model_idx)
                var = torch.square(torch.exp(log_std))
                loss = self.criterion(mu, y_target, var).mean()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'[TRAIN] Epoch {epoch} | Loss: {avg_loss:.6f}')
        return avg_loss

    def test(self, test_loader, epoch):
        self.model.eval()
        self.gat_model.eval()

        total_loss = 0.0
        total_mse = 0.0

        with torch.no_grad():
            for batch_data in test_loader:
                x, edge_index, edge_attr, y, batch_idx = (
                    batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.y, batch_data.batch
                )
                y = y.squeeze(1) if y.dim() == 3 else y

                # Extract GAT embeddings
                robot_emb = self.gat_model.extract_robot_embedding(x, edge_index, edge_attr, batch_idx)
                gamma = getattr(batch_data, 'gamma', None)
                gamma = gamma.view(-1, 2).to(self.device)

                X_input = torch.cat([robot_emb, gamma], dim=1).to(self.device)
                y_target = y.to(self.device)

                # Ensemble predictions
                ensemble_out = self.model(X_input)
                en_loss = 0
                en_mse = 0

                for en_idx in range(self.n_ensemble):
                    mu, log_std = ensemble_out[en_idx]
                    var = torch.square(torch.exp(log_std))
                    loss_ = self.criterion(mu, y_target, var).mean()
                    en_loss += loss_

                    dist = Normal(mu, torch.exp(log_std))
                    dist_samp = dist.rsample()  # One sample from each Gaussian
                    en_mse += self.mse_loss(dist_samp, y_target)

                en_loss /= self.n_ensemble
                en_mse  /= self.n_ensemble

                total_loss += en_loss.item()
                total_mse  += en_mse.item()

        avg_loss = total_loss / len(test_loader)
        avg_mse  = total_mse  / len(test_loader)
        rmse = math.sqrt(avg_mse)
        print(f'[TEST]  Epoch {epoch} | Loss: {avg_loss:.6f} | RMSE: {rmse:.6f}')

        if rmse < self.best_test_err:
            if not os.path.isdir('checkpoint'):
                os.makedirs('checkpoint/', exist_ok=True)
            print("Best Model Saving...")
            torch.save(self.state_dict(), 'checkpoint/best_gat_penn.pth')
            self.best_test_err = rmse

        return avg_loss, rmse

    def train_loop(self, train_data, test_data, epochs=50, batch_size=32):
        """
        Higher-level train loop. Creates the dataloader, calls train_one_epoch & test_one_epoch.
        """
        train_loader = GeoDataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader  = GeoDataLoader(test_data, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            self.train(train_loader, epoch)
            self.test(test_loader, epoch)

    def gaussian_nll_loss(self, mu, target, var):
        # Custom Gaussian Negative Log Likelihood Loss        
        loss = 0.5 * (torch.log(var) + (target - mu)**2 / var)
        return torch.mean(loss)

    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.state_dict(), model_path)

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
            self.model.load_state_dict(checkpoint, strict=False)       
            
        else:
            print("Model path does not exist. Check the provided path.")
