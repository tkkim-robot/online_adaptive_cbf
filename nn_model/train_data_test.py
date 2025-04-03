import time
import os
import numpy as np
import pandas as pd
import torch
from module import module
import random
import math
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader
from penn.nn_iccbf_predict import ProbabilisticEnsembleNN

import pickle
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import Batch
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from torch_geometric.data import Data
from gnn_gcbf import GCBFModule  

ACTIVATION = 'relu'

# Name or model and saving path
DATANAME = 'gnn_datagen_50000_DynamicUnicycle2D_mpc_cbf'
MODELNAME_SAVE = 'penn_model_0314_best_gnn'
data_file = 'data/' + DATANAME + '.csv'
pickle_file = 'data/' + DATANAME + '.pkl'
scaler_path = 'checkpoint/scaler_0314.save'
model_path = 'checkpoint/' + MODELNAME_SAVE + '.pth'

robot_model_list = ['DynamicUnicycle2D', 'KinematicBicycle2D', 'Quad2D']
robot_model = robot_model_list[0]

# Neural Network Parameters
if robot_model == 'Quad2D':
    n_states = 7  
else:
    n_states = 6  

n_output = 2
n_hidden = 40
n_ensemble = 3
device = 'cpu'

LR = 0.0001
BATCHSIZE = 32
EPOCH = 2000

USE_GNN_EMBED = True


def load_and_preprocess_data(data_file, scaler_path=None, noise_percentage=0.0, robot_model=None):
    # Load data
    dataset = pd.read_csv(data_file)

    # Define input features and outputs
    if robot_model == 'Quad2D':
        X = dataset[['Distance', 'VelocityX', 'VelocityZ', 'Theta', 'gamma0', 'gamma1']].values
        extra_states = 1
    else:
        X = dataset[['Distance', 'Velocity', 'Theta', 'gamma0', 'gamma1']].values
        extra_states = 0
    y = dataset[['Safety Loss', 'Deadlock Time']].values 

    # Apply noise to Distance, Velocity, and Theta
    noise = np.random.randn(*X[:, :3+extra_states].shape) * noise_percentage / 100
    X[:, :3+extra_states] += X[:, :3+extra_states] * noise

    # Transform Theta into sine and cosine components
    Theta = X[:, 2+extra_states]
    X_transformed = np.column_stack((X[:, :2+extra_states], np.sin(Theta), np.cos(Theta), X[:, 3+extra_states:]))

    # Initialize the scaler
    scaler = StandardScaler()
    
    # Normalize the inputs
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)  # Load existing scaler
    else:
        scaler.fit(X_transformed)  # Fit new scaler

    X_scaled = scaler.transform(X_transformed)

    # Save the scaler for later use
    if scaler_path:
        joblib.dump(scaler, scaler_path)

    # Splitting data into training and testing sets
    train_size = int(0.7 * len(X_scaled))
    train_dataX, test_dataX = X_scaled[:train_size], X_scaled[train_size:]
    train_dataY, test_dataY = y[:train_size], y[train_size:]

    return train_dataX, train_dataY, test_dataX, test_dataY, scaler

def plot_gmm(gmm):
    x = np.linspace(gmm.means_.min() - 3, gmm.means_.max() +
                    3, 1000).reshape(-1, 1)
    logprob = gmm.score_samples(x)
    responsibilities = gmm.predict_proba(x)
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf, '-k', label='GMM')

    for i in range(pdf_individual.shape[1]):
        plt.plot(x, pdf_individual[:, i], '--', label=f'GMM Component {i+1}')

    plt.xlabel('Safety Loss Prediction')
    plt.ylabel('Density')
    plt.title('Gaussian Mixture Model for Safety Loss Predictions')
    plt.legend()
    plt.show()


def load_graph_dataset(pickle_file):
    """
    Load dataset from a pickle file and convert stored dictionaries back into PyG Data objects.
    """
    with open(pickle_file, 'rb') as f:
        results = pickle.load(f)

    data_list = []
    for item in results:
        graph_dict = item["graph_data"]  # Extract stored dictionary

        # Convert back to PyTorch tensors
        graph_data = Data(
            x=torch.tensor(graph_dict["x"], dtype=torch.float),
            edge_index=torch.tensor(graph_dict["edge_index"], dtype=torch.long),
            edge_attr=torch.tensor(graph_dict["edge_attr"], dtype=torch.float)
        )

        if "gamma" in graph_dict:
            graph_data.gamma = torch.tensor(graph_dict["gamma"], dtype=torch.float)

        # Ensure y (labels) is included
        if "y" in graph_dict and graph_dict["y"] is not None:
            graph_data.y = torch.tensor(graph_dict["y"], dtype=torch.float)
        else:
            print("Missing y values in dataset. Assigning default zeros.")
            graph_data.y = torch.zeros((graph_data.x.shape[0], 2), dtype=torch.float)

        data_list.append(graph_data)

    return data_list

def train_gnn_embeddings_penn(gnn, penn, train_data, test_data, epochs=50, batch_size=32):
    """
    Train GNN to extract a 16D robot embedding, concatenate gamma values (2D),
    and pass the 18D input to PENN.
    """
    train_loader = GeoDataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = GeoDataLoader(test_data, batch_size=batch_size, shuffle=False)

    best_rmse = float('inf')

    for epoch in range(epochs):
        # Training phase
        gnn.train()
        penn.model.train()
        total_loss = 0.0

        for batch in train_loader:
            x, edge_index, edge_attr, y, batch_vec = (
                batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch
            )

            # Ensure y is shape [B, 2]
            if y.dim() == 3:  # Handle case where y is [B,1,2]
                y = y.squeeze(1)

            # Extract GNN embeddings
            with torch.no_grad():
                robot_emb = gnn.extract_robot_embedding(x, edge_index, edge_attr, batch_vec)

            # Debugging: Check shape
            # print(f"robot_emb.shape: {robot_emb.shape}")  # Expected: [batch_size, 16]

            # Retrieve gamma and ensure it's a tensor
            gamma = getattr(batch, 'gamma', None)
            if gamma is not None:
                if isinstance(gamma, list) or isinstance(gamma, np.ndarray):  # Convert if needed
                    gamma = torch.tensor(gamma, dtype=torch.float, device=robot_emb.device)

                # Ensure shape is [batch_size, 2]
                gamma = gamma.view(-1, 2)
            else:
                gamma = torch.zeros((robot_emb.shape[0], 2), dtype=torch.float, device=robot_emb.device)

            # Debugging: Check gamma shape
            # print(f"gamma.shape: {gamma.shape}")  # Expected: torch.Size([32, 2])

            # Concatenate embeddings with gamma => Expected shape: [batch_size, 18]
            X_input = torch.cat([robot_emb, gamma], dim=1).to(penn.device)

            # Debugging: Check final input shape
            # print(f"Final X_input shape: {X_input.shape}")  # Expected: torch.Size([32, 18])

            y_train = y.to(penn.device)

            # Train PENN (Ensemble Learning)
            for model_idx in range(penn.n_ensemble):
                penn.optimizer.zero_grad(set_to_none=True)
                mu, log_std = penn.model.single_forward(X_input, model_idx)
                var = torch.square(torch.exp(log_std))
                loss = penn.criterion(mu, y_train, var).mean()
                loss.backward()
                penn.optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Testing phase
        gnn.eval()
        penn.model.eval()
        test_loss = 0.0
        test_mse = 0.0

        with torch.no_grad():
            for batch in test_loader:
                x, edge_index, edge_attr, y, batch_vec = (
                    batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch
                )

                if y.dim() == 3:
                    y = y.squeeze(1)

                # Extract GNN embeddings
                robot_emb = gnn.extract_robot_embedding(x, edge_index, edge_attr, batch_vec)

                # Retrieve gamma and ensure it's a tensor
                gamma = getattr(batch, 'gamma', None)
                if gamma is not None:
                    if isinstance(gamma, list):
                        gamma = torch.tensor(gamma, dtype=torch.float, device=robot_emb.device)
                    gamma = gamma.view(-1, 2)
                else:
                    gamma = torch.zeros((robot_emb.shape[0], 2), dtype=torch.float, device=robot_emb.device)

                # Concatenate embeddings with gamma => shape [B, 18]
                X_input = torch.cat([robot_emb, gamma], dim=1).to(penn.device)
                y_test = y.to(penn.device)

                # PENN ensemble prediction
                ensemble_out = penn.model(X_input)
                en_loss = 0
                en_mse = 0

                for en_idx in range(penn.n_ensemble):
                    mu, log_std = ensemble_out[en_idx]
                    var = torch.square(torch.exp(log_std))
                    loss_ = penn.criterion(mu, y_test, var).mean()
                    en_loss += loss_
                    dist = Normal(mu, torch.exp(log_std))
                    dist_samp = dist.rsample()
                    en_mse += mse_loss(dist_samp, y_test)

                en_loss /= penn.n_ensemble
                en_mse /= penn.n_ensemble
                test_loss += en_loss.item()
                test_mse += en_mse.item()

        avg_val_loss = test_loss / len(test_loader)
        avg_val_mse = test_mse / len(test_loader)
        val_rmse = np.sqrt(avg_val_mse)

        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val RMSE: {val_rmse:.4f}")

        # Save best model based on validation RMSE
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            print("Saving best model so far...")
            os.makedirs('checkpoint/', exist_ok=True)
            torch.save(penn.state_dict(), 'checkpoint/'+MODELNAME_SAVE+'_best_gnn.pth')

def test_gnn_embeddings_penn(gnn_module, penn, graph_list, checkpoint_path, sample_idx=0, device='cpu'):
    """
    Test a trained GNN + PENN model on a single PyG graph sample.
    """    
    penn.load_state_dict(torch.load(checkpoint_path, map_location=device))

    gnn_network = gnn_module.gnn
    gnn_network.eval()
    penn.model.eval()

    sample_data = graph_list[sample_idx]
    sample_batch = Batch.from_data_list([sample_data])

    print("sample_batch")
    print(sample_batch)

    x = sample_batch.x.to(device)
    edge_index = sample_batch.edge_index.to(device)
    edge_attr = sample_batch.edge_attr.to(device)
    batch_idx = sample_batch.batch.to(device)

    with torch.no_grad():
        robot_emb = gnn_network.extract_robot_embedding(x, edge_index, edge_attr, batch_idx)
    gamma = getattr(sample_batch, 'gamma', None)
    gamma = gamma.view(-1, 2).float().to(device)
    
    print("robot_emb")
    print(robot_emb)
    
    y_pred_safety_loss, y_pred_deadlock_time, div_list = penn.predict_gnn(robot_emb, gamma)

    print("Predicted Safety Loss:", y_pred_safety_loss)
    print("Predicted Deadlock Time:", y_pred_deadlock_time)
    print("Divergence List:", div_list)

    gmm_safety = penn.create_gmm(y_pred_safety_loss[0])
    plot_gmm(gmm_safety)



if __name__ == '__main__':
    Test = True

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if Test:
        if not USE_GNN_EMBED:
            penn = ProbabilisticEnsembleNN(n_states, n_output, n_hidden, n_ensemble, device, lr=LR)
            penn.load_scaler(scaler_path)
            penn.load_model(model_path)

            # Example input array [distance, velocity, theta, gamma1, gamma2
            input_data = [2.55, 0.01, 0.001, 0.005, 0.005]
            y_pred_safety_loss, y_pred_deadlock_time, div = penn.predict(input_data)
            print("Predicted Safety Loss:", y_pred_safety_loss)
            print("Predicted Deadlock Time:", y_pred_deadlock_time)

            # Create GMM for safety loss predictions
            gmm_safety = penn.create_gmm(y_pred_safety_loss)
            plot_gmm(gmm_safety)
            
        else: # GNN + PENN approach
            n_states = 18
            penn = ProbabilisticEnsembleNN(n_states, n_output, n_hidden, n_ensemble, device, lr=LR)
            gnn_module = GCBFModule()
            graph_list = load_graph_dataset(pickle_file)

            test_gnn_embeddings_penn(gnn_module=gnn_module, penn=penn, graph_list=graph_list, checkpoint_path=model_path,)
        
    else:
        if not USE_GNN_EMBED:
            penn = ProbabilisticEnsembleNN(n_states, n_output, n_hidden, n_ensemble, device, lr=LR)

            # Load and preprocess data
            train_dataX, train_dataY, test_dataX, test_dataY, scaler = load_and_preprocess_data(data_file, scaler_path, noise_percentage=3.0, robot_model=robot_model)

            # Assign the scaler to the model
            penn.scaler = scaler
            
            # Create datasets and dataloaders
            train_dataset = module.CustomDataset(train_dataX, train_dataY)
            test_dataset = module.CustomDataset(test_dataX, test_dataY)
            train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=1, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False)

            start_epoch = 0
            best_test_rmse = 1000000
            start_time = time.time()
            for epoch in range(start_epoch, start_epoch + EPOCH):
                train_loss = penn.train(train_loader, epoch)
                test_loss, bool_best, test_rmse = penn.test(test_loader, epoch)
                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    print('Saving... \n')
                    state = {
                        'model': penn.state_dict(),
                        'test_rmse': test_rmse,
                        'epoch': epoch,
                        'input_state': n_states,
                    }
                    os.makedirs('checkpoint/', exist_ok=True)
                    torch.save(penn.state_dict(
                    ), 'checkpoint/' + MODELNAME_SAVE + '.pth')

            end_time = time.time()
            print('Learnig Time: {:.1f} min'.format((end_time-start_time)/60))

        else: # GNN + PENN approach
            # 0) 16D embedding and 2D gamma => 18D input to PENN
            n_states = 18  
            penn = ProbabilisticEnsembleNN(n_states, n_output, n_hidden, n_ensemble, device, lr=LR)

            # 1) Load PyG graph dataset from pickle
            graph_list = load_graph_dataset(pickle_file)

            # 2) Create GNN from gnn_gcbf
            gnn_module = GCBFModule() 
            gnn_network = gnn_module.gnn  

            # 3) Split train/test
            n_tot = len(graph_list)
            n_trn = int(0.8 * n_tot)
            train_g, test_g = torch.utils.data.random_split(graph_list, [n_trn, n_tot-n_trn])

            # 4) Train GNN Embedding + PENN
            train_gnn_embeddings_penn(gnn=gnn_network, penn=penn, train_data=train_g, test_data=test_g, epochs=EPOCH, batch_size=BATCHSIZE)
            
            print("Done GNN+PENN training.")
