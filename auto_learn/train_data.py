# Infinite loop Truckmaker
from distutils.command.build_scripts import first_line_re
import time
import os
import rclpy
import numpy as np
import pandas as pd
import torch
from auto_learn.module import module
import random
import math

from torch.utils.data import DataLoader


from auto_learn.dynamics.nn_vehicle import VehicleModel

# ACTIVATION = 'softplus'
ACTIVATION = 'relu'

# Random Seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# Name or model and saving path
MODELNAME = 'auto_220218_03_all'
MODELNAME_SAVE = 'temp_out_lr_0.0001_all'
PATH = '/home/add/Desktop/auto_ws/src/auto_learn'

LR = 0.0001
BATCHSIZE = 32
EPOCH = 3000
# MAX_DRIVE_INDEX = 180

# Neural Network Paramters
device = 'cpu'
n_states = 5
n_actions = 2
n_history = 4
n_hidden = 40
n_ensemble = 5
n_trajectory = 10000

# Vehicle Model
nn_vehicle_model = VehicleModel(
    n_states, n_actions, n_hidden, n_ensemble, n_history, n_trajectory, device, lr=LR)

train_dataX = []
train_dataY = []

dataset = pd.read_csv(PATH + '/auto_learn/data/' +
                      MODELNAME + '.csv', delimiter=',')
dataset_drive = dataset.loc[:, ['drive_index']]
drive_index = int(dataset_drive[-1:].values[0][0] + 1)
drive_index_first = drive_index


for i in range(drive_index_first):
    temp_history_data, temp_answer_data = module.make_history_data(
        dataset, i, n_states, n_actions, n_history, normal_flag=False)
    train_dataX.extend(temp_history_data)
    train_dataY.extend(temp_answer_data)
print('Length of Previous History Data:', len(train_dataX))

temp = list(zip(train_dataX, train_dataY))
random.shuffle(temp)
train_use_dataX, train_use_dataY = zip(*temp)

len_data = len(train_use_dataX)

test_dataX = train_use_dataX[int(len_data*0.7):]
test_dataY = train_use_dataY[int(len_data*0.7):]

train_dataX = train_use_dataX[0:int(len_data*0.7)]
train_dataY = train_use_dataY[0:int(len_data*0.7)]

train_dataset = module.CustomDataset(train_dataX, train_dataY)
train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True,
                          num_workers=1, pin_memory=True)  # shuffle every epoch
test_dataset = module.CustomDataset(test_dataX, test_dataY)
test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False)

start_epoch = 0

best_test_rmse = 1000000
start_time = time.time()
for epoch in range(start_epoch, start_epoch + EPOCH):
    train_loss = nn_vehicle_model.train(train_loader, epoch)
    test_loss, bool_best, test_rmse = nn_vehicle_model.test(
        test_loader, epoch, verbose=False)
    if test_rmse < best_test_rmse:
        best_test_rmse = test_rmse
        print('Saving... \n')
        state = {
            'model': nn_vehicle_model.state_dict(),
            'test_rmse': test_rmse,
            'epoch': epoch,
            'input_state': n_states,
        }
        os.makedirs(PATH + '/auto_learn/checkpoint/', exist_ok=True)
        torch.save(nn_vehicle_model.state_dict(
        ), PATH + '/auto_learn/checkpoint/' + MODELNAME_SAVE + '.pth')


end_time = time.time()
print('Learnig Time: {:.1f} min'.format((end_time-start_time)/60))
