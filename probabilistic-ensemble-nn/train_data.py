# Infinite loop Truckmaker
from distutils.command.build_scripts import first_line_re
import time
import os
import numpy as np
import pandas as pd
import torch
from module import module
import random
import math

from torch.utils.data import DataLoader
from dynamics.nn_vehicle import VehicleModel

# ACTIVATION = 'softplus'
ACTIVATION = 'relu'

# Name or model and saving path
MODELNAME = 'data_generation_results_3datapoint'
MODELNAME_SAVE = 'temp_out_lr_0.0001_all'
data_file = 'data/' + MODELNAME + '.csv'
scaler_path = 'scaler.save'
model_path = 'checkpoint/' + MODELNAME_SAVE + '.pth'

# Neural Network Paramters
device = 'cpu'
n_states = 6
n_output = 2
n_hidden = 40
n_ensemble = 3

LR = 0.0001
BATCHSIZE = 32
EPOCH = 30

if __name__ == '__main__':
    # Random Seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize the vehicle model
    vehicle_model = VehicleModel(n_states, n_output, n_hidden, n_ensemble, device, lr=LR)

    # Load and preprocess data
    train_dataX, train_dataY, test_dataX, test_dataY = vehicle_model.load_and_preprocess_data(data_file, scaler_path)

    # Create datasets and dataloaders
    train_dataset = module.CustomDataset(train_dataX, train_dataY)
    test_dataset = module.CustomDataset(test_dataX, test_dataY)
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False)

    start_epoch = 0
    best_test_rmse = 1000000
    start_time = time.time()
    for epoch in range(start_epoch, start_epoch + EPOCH):
        train_loss = vehicle_model.train(train_loader, epoch)
        test_loss, bool_best, test_rmse = vehicle_model.test(
            test_loader, epoch, verbose=False)
        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            print('Saving... \n')
            state = {
                'model': vehicle_model.state_dict(),
                'test_rmse': test_rmse,
                'epoch': epoch,
                'input_state': n_states,
            }
            os.makedirs('checkpoint/', exist_ok=True)
            torch.save(vehicle_model.state_dict(
            ), 'checkpoint/' + MODELNAME_SAVE + '.pth')

    end_time = time.time()
    print('Learnig Time: {:.1f} min'.format((end_time-start_time)/60))
