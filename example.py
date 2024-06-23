import numpy as np
import matplotlib.pyplot as plt

from tracking import LocalTrackingController

from utils import plotting
from utils import env

"""
Created on June 22nd, 2024
@author: Taekyung Kim

@description: 

@functions to be implemented: 
    An Exploration class
        1. can handle multiple agents by setting up multiple controllers
        2. manage the global map
        3. extract the global frontier

@required-scripts: tracking.py
"""

class ExplorationManager:
    def __init__(self, X0s, type='DynamicUnicycle2D', num_robot=1, dt=0.05,
                  show_animation=False, save_animation=False,
                  plotting=None, env=None):
        self.type = type
        self.num_robot = num_robot
        self.dt = dt

        plot_handler = plotting.Plotting()
        ax, fig = plot_handler.plot_grid("Local Tracking Controller")
        env_handler = env.Env()

        

        self.show_animation = show_animation
        self.save_animation = save_animation

        self.controller_list = []
        for i in range(num_robot):
            X0 = X0s[i]
            tracking_controller = LocalTrackingController(X0, type=type, 
                                         robot_id=i,
                                         dt=dt,
                                         show_animation=show_animation,
                                         save_animation=save_animation,
                                         ax=ax, fig=fig,
                                         env=env_handler)
            self.controller_list.append(tracking_controller)
        
    def set_waypoints(self,):
        
        

def single_agent_main():
    dt = 0.05

    # temporal
    waypoints = [
        [2, 2, math.pi/2],
        [2, 12, 0],
        [10, 12, 0],
        [10, 2, 0]
    ]
    waypoints = np.array(waypoints, dtype=np.float64)

    x_init = waypoints[0]
    x_goal = waypoints[-1]

    plot_handler = plotting.Plotting(x_init, x_goal)
    plot_handler = plotting.Plotting(x_init, x_goal)
    ax, fig = plot_handler.plot_grid("Local Tracking Controller")
    env_handler = env.Env()

    #type = 'Unicycle2D'
    type = 'DynamicUnicycle2D'
    tracking_controller = LocalTrackingController(x_init, type=type, dt=dt,
                                         show_animation=True,
                                         save_animation=False,
                                         ax=ax, fig=fig,
                                         env=env_handler)

    # unknown_obs = np.array([[9.0, 8.8, 0.3]]) 
    # tracking_controller.set_unknown_obs(unknown_obs)
    tracking_controller.set_waypoints(waypoints)
    unexpected_beh = tracking_controller.run_all_steps(tf=30)

if __name__ == "__main__":
    from utils import plotting
    from utils import env
    import math

    