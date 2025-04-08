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

WANDB_FLAG = False
if WANDB_FLAG:
    import wandb

import pickle
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data
from penn.gat import GATModule
from penn.nn_gat_iccbf_predict import ProbabilisticEnsembleGAT  


# Name or model and saving path
DATANAME = 'gat_datagen_10000_DynamicUnicycle2D_mpc_cbf'
MODELNAME_SAVE = 'penn_model_0314_best_gat'
data_file = 'data/' + DATANAME + '.csv'
pickle_file = 'data/' + DATANAME + '.pkl'
scaler_path = 'checkpoint/scaler_0314.save'
model_path = 'checkpoint/' + MODELNAME_SAVE + '.pth'

robot_model_list = ['DynamicUnicycle2D', 'KinematicBicycle2D', 'Quad2D', 'VTOL2D']
robot_model = robot_model_list[0]

# PENN Parameters
if robot_model == 'Quad2D':
    n_states = 7  
else:
    n_states = 6  
n_output = 2
n_hidden = 40
n_ensemble = 3
device = 'cpu'  

ACTIVATION = 'relu'
LR = 0.00005
BATCHSIZE = 32
EPOCH = 2000

TEST_ONLY = False      # If True, just do inference; if False, train then test
USE_GAT_EMBED = True   # False => MLP-only PENN, True => GAT+PENN

TEST_ONLY = False      # If True, just do inference; if False, train then test
USE_GAT_EMBED = True   # False => MLP-only PENN, True => GAT+PENN

if WANDB_FLAG:
    wandb.init(project="your name", config={
        "learning_rate": LR,
        "epochs": EPOCH,
        "batch_size": BATCHSIZE
    })

def load_and_preprocess_data(data_file, scaler_path=None, noise_percentage=0.0, robot_model=None):
    dataset = pd.read_csv(data_file)

    # Prepare X, y depending on robot model
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


def load_graph_dataset(pickle_file):
    with open(pickle_file, 'rb') as f:
        results = pickle.load(f)

    data_list = []
    for item in results:
        graph_dict = item["graph_data"]
        
        graph_data = Data(
            x=torch.tensor(graph_dict["x"], dtype=torch.float),
            edge_index=torch.tensor(graph_dict["edge_index"], dtype=torch.long),
            edge_attr=torch.tensor(graph_dict["edge_attr"], dtype=torch.float)
        )
        
        if "gamma" in graph_dict:
            graph_data.gamma = torch.tensor(graph_dict["gamma"], dtype=torch.float)

        if "y" in graph_dict and graph_dict["y"] is not None:
            graph_data.y = torch.tensor(graph_dict["y"], dtype=torch.float)
        else:
            print("Missing y values in dataset. Assigning default zeros.")
            graph_data.y = torch.zeros((graph_data.x.shape[0], 2), dtype=torch.float)

        data_list.append(graph_data)
        
    return data_list


def plot_gmm(gmm):
    x = np.linspace(gmm.means_.min() - 3, gmm.means_.max() + 3, 1000).reshape(-1, 1)
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






if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not USE_GAT_EMBED:
        # ============= MLP-based (no GAT) approach =============
        penn = ProbabilisticEnsembleNN(n_states, n_output, n_hidden, n_ensemble, device, lr=LR)

        if TEST_ONLY:
            # Load scaler and model, then do predictions
            penn.load_scaler(scaler_path)
            penn.load_model(model_path)

            # Example input array [distance, velocity, theta, gamma1, gamma2]
            input_data = [2.55, 0.01, 0.001, 0.005, 0.005]
            y_pred_safety_loss, y_pred_deadlock_time, div = penn.predict(input_data)
            print("Predicted Safety Loss:", y_pred_safety_loss)
            print("Predicted Deadlock Time:", y_pred_deadlock_time)

            # Plot GMM for safety predictions
            gmm_safety = penn.create_gmm(y_pred_safety_loss)
            plot_gmm(gmm_safety)

        else:
            # Load and preprocess data
            train_dataX, train_dataY, test_dataX, test_dataY, scaler = load_and_preprocess_data(
                data_file, scaler_path, noise_percentage=3.0, robot_model=robot_model
            )
            penn.scaler = scaler

            if WANDB_FLAG:
                wandb.watch(penn, log="all", log_freq=100)

            # Create datasets and dataloaders
            train_dataset = module.CustomDataset(train_dataX, train_dataY)
            test_dataset = module.CustomDataset(test_dataX, test_dataY)
            train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=1, pin_memory=True)
            test_loader  = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False)

            start_epoch = 0
            best_test_rmse = 1000000
            start_time = time.time()

            for epoch in range(start_epoch, start_epoch + EPOCH):
                train_loss = penn.train(train_loader, epoch)
                test_loss, bool_best, test_rmse = penn.test(test_loader, epoch)
                
                if WANDB_FLAG:
                    wandb.log({"train_loss": train_loss, "test_loss": test_loss, "test_rmse": test_rmse, "epoch": epoch})
                
                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    print('Saving...\n')
                    os.makedirs('checkpoint/', exist_ok=True)
                    torch.save(penn.state_dict(), 'checkpoint/' + MODELNAME_SAVE + '.pth')

            end_time = time.time()
            print('Learning Time: {:.1f} min'.format((end_time - start_time) / 60))

    else:
        # ============= GAT + PENN approach =============
        graph_list = load_graph_dataset(pickle_file)
        gat_module = GATModule()  
        gat_network = gat_module.gat
        penn_gat = ProbabilisticEnsembleGAT(gat_network, n_output, n_hidden, n_ensemble, 
                                            device, LR, ACTIVATION).to(device)

        n_tot = len(graph_list)
        n_trn = int(0.8 * n_tot)
        train_g, test_g = torch.utils.data.random_split(graph_list, [n_trn, n_tot - n_trn])

        if TEST_ONLY:
            penn_gat.load_model(model_path)

            sample_data = [test_g[0]]  # must wrap in a list for .predict
            y_safety, y_deadlock, divs = penn_gat.predict(sample_data)

            print("Predicted Safety Loss:", y_safety)
            print("Predicted Deadlock Time:", y_deadlock)
            print("Divergences:", divs)

            # Build GMM for safety loss
            gmm_safety = penn_gat.create_gmm(y_safety[0])
            plot_gmm(gmm_safety)

        else:
            if WANDB_FLAG:
                wandb.watch(penn_gat, log="all", log_freq=100)
            
            train_loader = GeoDataLoader(train_g, batch_size=BATCHSIZE, shuffle=True)
            test_loader = GeoDataLoader(test_g, batch_size=BATCHSIZE, shuffle=False)

            best_test_rmse = 1000000
            start_time = time.time()
            
            for epoch in range(EPOCH):
                train_loss = penn_gat.train(train_loader, epoch)
                test_loss, test_rmse = penn_gat.test(test_loader, epoch)
                
                if WANDB_FLAG:
                    wandb.log({"train_loss": train_loss, "test_loss": test_loss, "test_rmse": test_rmse, "epoch": epoch})
                
                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    print('Saving...\n')
                    os.makedirs('checkpoint/', exist_ok=True)
                    torch.save(penn_gat.state_dict(), 'checkpoint/' + MODELNAME_SAVE + '.pth')

            end_time = time.time()
            print('Training Time: {:.1f} min'.format((end_time - start_time) / 60))