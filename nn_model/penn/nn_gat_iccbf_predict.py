import torch
import torch.nn as nn
import os
import math
import numpy as np
from torch.distributions.normal import Normal
from torch_geometric.loader import DataLoader as GeoDataLoader
from sklearn.mixture import GaussianMixture


class ProbabilisticEnsembleGAT(nn.Module):
    def __init__(self, gat_model, n_output=2, n_hidden=40, n_ensemble=3, gamma_dim=2, device='cpu', lr=0.001, activation='relu'):
        super(ProbabilisticEnsembleGAT, self).__init__()
        self.device = device
        self.n_ensemble = n_ensemble
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.gamma_dim = gamma_dim
        
        self.gat_model = gat_model  
        self.gat_model = self.gat_model.to(self.device)
        
        # Handle both GATModule and GATGraphNetwork cases
        # If gat_model is a GATModule, extract the GATGraphNetwork
        if hasattr(self.gat_model, 'gat'):
            self.gat_network = self.gat_model.gat
        else:
            self.gat_network = self.gat_model

        try:
            from penn.penn import EnsembleStochasticLinear
        except:
            from nn_model.penn.penn import EnsembleStochasticLinear
        self.model = EnsembleStochasticLinear(in_features=16 + self.gamma_dim,  # 16D embedding + gamma dimension
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

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.gat_network.parameters()), lr=lr)
        self.criterion = self.gaussian_nll_loss  # Custom Gaussian NLL Loss
        self.mse_loss = nn.MSELoss()
        self.best_test_err = 10000.0

    def predict(self, data_list):
        loader = GeoDataLoader(data_list, batch_size=len(data_list), shuffle=False)
        self.model.eval()
        self.gat_network.eval()

        with torch.no_grad():
            batch_data = next(iter(loader))  
            x = batch_data.x.to(self.device)
            edge_index = batch_data.edge_index.to(self.device)
            edge_attr = batch_data.edge_attr.to(self.device)
            batch_idx = batch_data.batch.to(self.device)
            gamma = getattr(batch_data, 'gamma', None).view(-1, self.gamma_dim).to(self.device)
            robot_emb = self.gat_network.extract_robot_embedding(x, edge_index, edge_attr, batch_idx)
            
            # robot_emb: (1, emb_dim) → concat with gamma
            if robot_emb.shape[0] == 1 and gamma.shape[0] > 1:
                robot_emb = robot_emb.repeat(gamma.shape[0], 1)

            X_input = torch.cat([robot_emb, gamma], dim=1).to(self.device)  # (B, 16+2)

            # Ensemble prediction
            ensemble_out = self.model(X_input)

            # Output shape: ensemble_out[:-1] = list of (mu, log_std), each of shape (B, 2)
            y_pred_safety_list = []
            y_pred_deadlock_list = []
            for i in range(gamma.shape[0]):
                safety_ensemble, deadlock_ensemble = [], []
                for (mu, log_std) in ensemble_out[:-1]:
                    sigma_sq = torch.square(torch.exp(log_std))
                    deadlock_ensemble.append([mu[i,0].item(), sigma_sq[i,0].item()])
                    safety_ensemble.append([mu[i,1].item(), sigma_sq[i,1].item()])
                y_pred_deadlock_list.append(deadlock_ensemble)
                y_pred_safety_list.append(safety_ensemble)

            # Divergence
            div = ensemble_out[-1][:, 0]  # shape: (B,)
            div_list = div.tolist()

        return y_pred_safety_list, y_pred_deadlock_list, div_list

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
        self.model.train()
        self.gat_network.train()  

        total_loss = 0.0

        for batch_data in train_loader:
            x          = batch_data.x.to(self.device)
            edge_index = batch_data.edge_index.to(self.device)
            edge_attr  = batch_data.edge_attr.to(self.device)
            batch_idx  = batch_data.batch.to(self.device)       
            y          = batch_data.y.to(self.device)                
            y = y.squeeze(1) if y.dim() == 3 else y  # Ensure shape [batch_size, 2]

            # Obtain gamma
            gamma = getattr(batch_data, 'gamma', None).view(-1, self.gamma_dim).to(self.device)

            # Train each ensemble member
            for model_idx in range(self.n_ensemble):
                robot_emb = self.gat_network.extract_robot_embedding(x, edge_index, edge_attr, batch_idx)
                X_input = torch.cat([robot_emb, gamma], dim=1).to(self.device)
                y_target = y.to(self.device)
                
                self.optimizer.zero_grad(set_to_none=True)
                mu, log_std = self.model.single_forward(X_input, model_idx)
                var = torch.square(torch.exp(log_std))
                loss = self.criterion(mu, y_target, var).mean()
                loss.backward()
                # loss.backward(retain_graph=True)
                self.optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'[TRAIN] Epoch {epoch} | Loss: {avg_loss:.6f}')
        return avg_loss

    def test(self, test_loader, epoch):
        self.model.eval()
        self.gat_network.eval()

        total_loss = 0.0
        total_mse = 0.0

        with torch.no_grad():
            for batch_data in test_loader:
                x          = batch_data.x.to(self.device)
                edge_index = batch_data.edge_index.to(self.device)
                edge_attr  = batch_data.edge_attr.to(self.device)
                batch_idx  = batch_data.batch.to(self.device)   
                y          = batch_data.y.to(self.device)    
                y = y.squeeze(1) if y.dim() == 3 else y

                robot_emb = self.gat_network.extract_robot_embedding(x, edge_index, edge_attr, batch_idx)
                gamma = getattr(batch_data, 'gamma', None).view(-1, self.gamma_dim).to(self.device)

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
