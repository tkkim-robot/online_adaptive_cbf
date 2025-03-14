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
from gnn_gcbf import GCBFModule  

ACTIVATION = 'relu'

# Name or model and saving path
DATANAME = 'data_generation_DynamicUnicycle2D_cbf_qp_6datapoint'
MODELNAME_SAVE = 'penn_model_0128'
data_file = 'data/' + DATANAME + '.csv'
scaler_path = 'checkpoint/scaler_0128.save'
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

USE_GNN_EMBED = False


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
    Used when USE_GNN_EMBED=True
    Return a list of PyG Data objects that have:
      data.y => [deadlock, risk]
      data.gamma => [gamma0, gamma1]
    """
    with open(pickle_file, 'rb') as f:
        results = pickle.load(f)

    data_list = []
    for item in results:
        graph_data = item["graph_data"]
        data_list.append(graph_data)
    return data_list


def train_gnn_embeddings_penn(gnn, penn, train_data, test_data, epochs=50, batch_size=32):
    """
    Use 'gnn' to extract a 16D robot embedding, plus gamma => final [18D].
    Then feed that embedding to PENN.
    We'll replicate the ensemble training logic from 'penn.train(...)',
    but adapted for GNN batches.
    """
    train_loader = GeoDataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = GeoDataLoader(test_data, batch_size=batch_size, shuffle=False)

    best_rmse = 999999.0
    for epoch in range(epochs):
        # Training
        gnn.train()
        penn.model.train()

        total_loss = 0.0
        for batch in train_loader:
            x, edge_index, edge_attr, y, batch_vec = (batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch)

            # Adjust shapes => y => [B,2]
            if y.dim() == 3:  # e.g. [B,1,2]
                y = y.squeeze(1)

            # Extract GNN embedding
            with torch.no_grad():  # if you want to freeze GNN or else remove no_grad
                robot_emb = gnn.extract_robot_embedding(x, edge_index, edge_attr, batch_vec)

            # Handle gamma => shape [B,1,2] => [B,2]
            gamma = getattr(batch, 'gamma', None)
            if gamma is not None and gamma.dim() == 3:
                gamma = gamma.squeeze(1)

            if gamma is not None:
                X_input = torch.cat([robot_emb, gamma], dim=1)
            else:
                X_input = robot_emb

            # => shape [B, (16+2)], y => [B,2]
            X_input = X_input.to(penn.device)
            y_train = y.to(penn.device)

            # mimic penn.train(...) logic => ensemble
            for model_idx in range(penn.n_ensemble):
                penn.optimizer.zero_grad(set_to_none=True)
                mu, log_std = penn.model.single_forward(X_input, model_idx)
                var = torch.square(torch.exp(log_std))
                loss = penn.criterion(mu, y_train, var).mean()
                loss.backward()
                penn.optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Testing
        gnn.eval()
        penn.model.eval()
        test_loss = 0.0
        test_mse = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x, edge_index, edge_attr, y, batch_vec = (batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch)
                if y.dim() == 3:
                    y = y.squeeze(1)
                robot_emb = gnn.extract_robot_embedding(x, edge_index, edge_attr, batch_vec)
                gamma = getattr(batch, 'gamma', None)
                if gamma is not None and gamma.dim() == 3:
                    gamma = gamma.squeeze(1)

                if gamma is not None:
                    X_input = torch.cat([robot_emb, gamma], dim=1)
                else:
                    X_input = robot_emb

                X_input = X_input.to(penn.device)
                y_test  = y.to(penn.device)

                ensemble_out = penn.model(X_input)
                # average loss
                from torch.distributions.normal import Normal
                from torch.nn.functional import mse_loss
                en_loss = 0
                en_mse  = 0
                for en_idx in range(penn.n_ensemble):
                    mu, log_std = ensemble_out[en_idx]
                    var = torch.square(torch.exp(log_std))
                    loss_ = penn.criterion(mu, y_test, var).mean()
                    en_loss += loss_
                    dist = Normal(mu, torch.exp(log_std))
                    dist_samp = dist.rsample()
                    en_mse  += mse_loss(dist_samp, y_test)

                en_loss /= penn.n_ensemble
                en_mse  /= penn.n_ensemble
                test_loss += en_loss.item()
                test_mse  += en_mse.item()

        avg_val_loss = test_loss / len(test_loader)
        avg_val_mse  = test_mse  / len(test_loader)
        val_rmse     = np.sqrt(avg_val_mse)

        print(f"[Epoch {epoch+1}/{epochs}] TrainLoss: {avg_loss:.4f}, ValLoss: {avg_val_loss:.4f}, ValRMSE: {val_rmse:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            print("Saving best model so far...")
            os.makedirs('checkpoint/', exist_ok=True)
            torch.save(penn.state_dict(), 'checkpoint/'+MODELNAME_SAVE+'_best_gnn.pth')


if __name__ == '__main__':
    Test = False

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize the model (PENN)
    penn = ProbabilisticEnsembleNN(n_states, n_output, n_hidden, n_ensemble, device, lr=LR)

    if Test:
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
        
    else:
        if not USE_GNN_EMBED:
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

        else:
            # GNN + PENN approach

            # 1) Load PyG graph dataset from pickle
            pkl_file = "gnn_datagen_1000_DynamicUnicycle2D_cbf_qp.pkl"
            graph_list = load_graph_dataset(pkl_file)

            # 2) Create GNN from gnn_gcbf
            gnn_module = GCBFModule()  # This includes .gnn => GCBFGraphNetwork
            gnn_network = gnn_module.gnn  # we will call .extract_robot_embedding(...) inside

            # 3) Split train/test
            n_tot = len(graph_list)
            n_trn = int(0.8 * n_tot)
            train_g, test_g = torch.utils.data.random_split(graph_list, [n_trn, n_tot-n_trn])

            # 4) Train GNN Embedding + PENN
            # We need n_states=18 in penn => 16 from embedding + 2 from gamma
            # But for simplicity, we won't re-init penn => or do it:
            #   new_penn = ProbabilisticEnsembleNN(18, n_output, n_hidden, n_ensemble, device, lr=LR)
            #   but let's keep it as is for demonstration
            # Just ensure in single_forward => in_features= n_states=?
            # In practice, set n_states=18 if you do 16+2

            train_gnn_embeddings_penn(gnn=gnn_network, penn=penn, train_data=train_g, test_data=test_g, epochs=50, batch_size=BATCHSIZE)
            print("Done GNN+PENN training.")
