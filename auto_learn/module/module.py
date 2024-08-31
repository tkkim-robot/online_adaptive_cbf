import pandas as pd
import numpy as np
import csv
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
# import pyautogui


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, item):
        x_ = torch.FloatTensor(self.x_data[item])
        y_ = torch.FloatTensor(self.y_data[item])
        return x_, y_


# def linearize_deq_data(deque, n_history):
#     temp = []
#     for i in range(0, n_history):
#         pop_data = deque.popleft()
#         temp = temp + pop_data
#     return temp


def make_history_data(dataset, drive_index, n_states, n_actions, n_history, normal_flag=True):

    dataset = dataset.loc[:, ['drive_index',
                              'iter', 'car_vx', 'car_vy', 'car_yaw_vel', 'car_roll_vel', 'sideslip_angle', 'mppi_steer', 'mppi_speed']]
    dataset = dataset.loc[dataset['drive_index'] == drive_index]

    # Select specific or arbitrary Column
    # selection_data = ['car_vx', 'car_vy',
    #                   'car_yaw_vel', 'car_roll_vel', 'sideslip_angle', 'mppi_steer', 'mppi_speed']
    selection_data = ['car_vx', 'car_vy',
                      'car_yaw_vel', 'mppi_steer', 'mppi_speed']

    if normal_flag == True:
        # For Normalize
        data_vx = dataset['car_vx'] / 10.0
        data_vy = dataset['car_vy'] / 5.0
        data_yaw_vel = dataset['car_yaw_vel'] / 1.0
        # data_roll_vel = dataset['car_roll_vel'] / 0.5
        # data_sideslip_angle = dataset['sideslip_angle'] / 1.0
        data_mppi_steer = dataset['mppi_steer'] / 1.0
        data_mppi_speed = dataset['mppi_speed'] / 1.0

        # dataset1 = pd.concat([data_vx, data_vy, data_yaw_vel,
        #                       data_roll_vel, data_sideslip_angle, data_mppi_steer, data_mppi_speed], axis=1)

        dataset1 = pd.concat([data_vx, data_vy, data_yaw_vel,
                              data_mppi_steer, data_mppi_speed], axis=1)
    else:
        dataset1 = dataset.loc[:, selection_data]

    np_dataset = dataset1.to_numpy(dtype=np.float64)
    total_temp_data = np.zeros((n_states+n_actions)*(n_history))
    # total_temp_data = np.expand_dims(total_temp_data, axis=0)
    total_history_data = []

    # Answer list
    # selection_answer = ['car_vx', 'car_vy',
    #                     'car_yaw_vel', 'car_roll_vel', 'sideslip_angle']
    selection_answer = ['car_vx', 'car_vy',
                        'car_yaw_vel']
    answer = dataset.loc[:, selection_answer]
    if normal_flag == True:
        answer['car_vx'] = answer['car_vx'] / 10.0
        answer['car_vy'] = answer['car_vy'] / 5.0
        answer['car_yaw_vel'] = answer['car_yaw_vel'] / 1.0
        # answer['car_roll_vel'] = answer['car_roll_vel'] / 0.5
        # answer['sideslip_angle'] = answer['sideslip_angle'] / 1.0
    # print()
    np_answer = answer.to_numpy(dtype=np.float64)
    total_answer_data = np_answer.tolist()

    for i, row in enumerate(np_dataset):
        # row = np.expand_dims(row, axis=0)
        if i >= n_history:
            total_history_data.append((list(total_temp_data)))
        total_temp_data = np.hstack([total_temp_data, row])
        # answer_temp_data = np.hstack(row[1, ])
        # 3 means n_actions+n_states
        total_temp_data = total_temp_data[(n_states+n_actions):]
        # answer_temp_data = answer_temp_data[3:]

    return total_history_data, total_answer_data[n_history:]


def normalize(state):
    state[0] = state[0] / 10.0
    state[1] = state[1] / 5.0
    state[2] = state[2] / 1.0
    # state[3] = state[3] / 0.5
    # state[4] = state[4] / 1.0

    return state


def denormalize(state):
    state[:, 0] = state[:, 0] * 10.0
    state[:, 1] = state[:, 1] * 5.0
    state[:, 2] = state[:, 2] * 1.0
    # state[:, 3] = state[:, 3] * 0.5
    # state[:, 4] = state[:, 4] * 1.0

    return state


def inject_sensor_noise(state):
    # inject sensor noise up to 10 %
    # ex) vx range from 0 ~ 10, 10% of the range is 1.0
    #     3 * sigma = 1.0
    # state[0] += np.random.normal(0.0, 0.3)
    # state[1] += np.random.normal(0.0, 0.15)
    # state[2] += np.random.normal(0.0, 0.03)
    state[0] += np.random.normal(0.0, 0.6)
    state[1] += np.random.normal(0.0, 0.3)
    state[2] += np.random.normal(0.0, 0.06)
    return state


def replace_in_file(file_path, find_name, old, new):
    fr = open(file_path, 'r')
    lines = fr.readlines()
    fr.close()

    fw = open(file_path, 'w')
    for line in lines:
        fw.write(line.replace(find_name + str(old),
                 find_name + str(new)))


# def click_mouse(x_mouse, y_mouse):
#     pyautogui.click(x_mouse, y_mouse, button='left')


# def get_point():
#     print(pyautogui.position())
