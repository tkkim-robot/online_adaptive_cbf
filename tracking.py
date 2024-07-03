import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cvxpy as cp
import os
import glob
import subprocess
import do_mpc
import casadi
from casadi import *

"""
Created on June 20th, 2024
@author: Taekyung Kim

@description: 
This code implements a local tracking controller for 2D robot navigation using Control Barrier Functions (CBF) and Model Predictive Control (MPC).
It supports both kinematic (Unicycle2D) and dynamic (DynamicUnicycle2D) unicycle models, with functionality for obstacle avoidance and waypoint following.
The controller includes real-time visualization capabilities and can handle both known and unknown obstacles.
The main functions demonstrate single and multi-agent scenarios, showcasing the controller's ability to navigate complex environments.

@required-scripts: robots/robot.py
"""

def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

def compute_alpha_k(h_k, alpha_obs, gamma):
    return alpha_obs * np.exp(-gamma * h_k)

def compute_beta_k(delta_theta, beta_obs, lambda_):
    return beta_obs * np.exp(-lambda_ * (cos(delta_theta)+1))

def compute_safety_loss_function(obs, alpha_obs, beta_obs, gamma, lambda_, z):
    z_k, h_k, delta_theta = obs['z'], obs['h'], obs['d']
    alpha_k = compute_alpha_k(h_k, alpha_obs, gamma)
    beta_k = compute_beta_k(delta_theta, beta_obs, lambda_)
    phi = alpha_k / (beta_k * np.linalg.norm(z - z_k)**2 + 1)
    return phi


class CollisionError(Exception):
    '''
    Exception raised for errors when  
    the robot collides with the obstacle
    '''
    def __init__(self, message="ERROR in Collision"):
        self.message = message
        super().__init__(self.message)


class LocalTrackingController:
    def __init__(self, X0, type='DynamicUnicycle2D', robot_id=0, dt=0.05,
                  show_animation=False, save_animation=False, ax=None, fig=None, env=None, waypoints=None, data_generation=True):
        self.type = type
        self.robot_id = robot_id # robot id = 1 has the plot handler
        self.dt = dt
        self.data_generation = data_generation
        self.waypoints = waypoints

        self.current_goal_index = 0  # Index of the current goal in the path
        self.reached_threshold = 0.5

        if self.type == 'DynamicUnicycle2D':
            self.gamma1 = 0.05
            self.gamma2 = 0.05
            # v_max is set to 1.0 inside the robot class
            self.v_max = 1.0
            self.a_max = 0.5
            self.w_max = 0.5
            X0 = np.array([X0[0], X0[1], X0[2], 0.0]).reshape(-1, 1)

        self.show_animation = show_animation
        self.save_animation = save_animation
        if self.save_animation:
            self.current_directory_path = os.getcwd() 
            if not os.path.exists(self.current_directory_path + "/output/animations"):
                os.makedirs(self.current_directory_path + "/output/animations")
            self.save_per_frame = 2
            self.ani_idx = 0

        self.ax = ax
        self.fig = fig
        self.obs = np.array(env.obs_circle)
        self.unknown_obs = None

        if show_animation:
            # Initialize plotting
            if self.ax is None:
                self.ax = plt.axes()
            if self.fig is None:
                self.fig = plt.figure()
            plt.ion()
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_aspect(1)
        else:
            self.ax = plt.axes() # dummy placeholder

        if data_generation:
            # Parameters for the safety loss function
            self.alpha_obs = 1.0
            self.beta_obs = 1.0
            self.gamma_loss = 0.1
            self.lambda_loss = 1.0

        # Setup DT-MPC problem  
        self.goal = np.array(self.waypoints[self.current_goal_index]).reshape(-1, 1)
        self.near_obs = np.array([10.0,10.0,0.1]).reshape(-1, 1) # Set initial obs far away
        self.horizon = 10
        self.Q = np.diag([50, 50, 0.01, 30]) # State cost matrix
        self.R = np.array([0.5, 0.5]) # Input cost matrix
        self.setup_robot(X0)
        self.model = self.create_model()
        self.mpc = self.create_mpc(self.model)
        self.simulator = self.define_simulator()
        self.estimator = do_mpc.estimator.StateFeedback(self.model)
        self.set_init_state()
        

    def setup_robot(self, X0):
        from robots.robot import BaseRobot
        self.robot = BaseRobot(X0.reshape(-1, 1), self.dt, self.ax, self.type, self.robot_id, self.data_generation, goal=self.goal, obs=self.unknown_obs)

    def create_model(self):
        """Creates a model for the MPC controller"""
        model_type = 'discrete'
        model = do_mpc.model.Model(model_type)

        # States (_x[0] = x pos, _x[1] = y pos, _x[2] = theta _x[3] = velocity)
        n_states = 4
        _x = model.set_variable(var_type='_x', var_name='x', shape=(n_states, 1)) 
        
        # Inputs (u1 = linear acceleration, u2 = angular velocity)
        n_controls = 2
        _u = model.set_variable(var_type='_u', var_name='u', shape=(n_controls, 1))

        # CBF parameters
        _p1 = model.set_variable(var_type='_tvp', var_name='gamma1')
        _p2 = model.set_variable(var_type='_tvp', var_name='gamma2')

        # Waypoints
        _goal = model.set_variable(var_type='_tvp', var_name='goal', shape=(4, 1))

        # Obstacle
        _obs = model.set_variable(var_type='_tvp', var_name='obs', shape=(3, 1)) 

        # State Space equations in one line
        f_x = self.robot.f_casadi(_x) 
        g_x = self.robot.g_casadi(_x)
        X_next = _x + (f_x + casadi.mtimes(g_x, _u)) * self.dt
        
        # Update model RHS
        model.set_rhs('x', X_next)

        # stage and terminal cost of the control problem
        model, cost_expr = self.get_cost_expression(model, _goal)
        model.set_expression(expr_name='cost', expr=cost_expr)
        
        model.setup()
        return model

    def get_cost_expression(self, model, goal):
        """Defines the objective function wrt the state cost"""
        X = SX.zeros(4, 1)
        X = model.x['x'] - goal
        cost_expression = casadi.mtimes([X.T, self.Q, X])
        return model, cost_expression

    def create_mpc(self, model):
        """Creates the MPC problem"""
        mpc = do_mpc.controller.MPC(model)
        mpc.settings.supress_ipopt_output()

        setup_mpc = {'n_robust': 0,  # Robust horizon
                     'n_horizon': self.horizon,
                     't_step': self.dt,
                     'state_discretization': 'discrete',
                     'store_full_solution': True,
                     }
        mpc.set_param(**setup_mpc)

        # Configure objective function
        mterm = self.model.aux['cost'] # Terminal cost
        lterm = self.model.aux['cost'] # Stage cost
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(u=self.R) # Input penalty (R diagonal matrix in objective fun)

        # State and input bounds
        max_x = np.array([10000, 10000, 10000, self.v_max])
        mpc.bounds['lower','_x', 'x'] = -max_x
        mpc.bounds['upper','_x', 'x'] = max_x
        max_u = np.array([self.a_max, self.w_max])
        mpc.bounds['lower', '_u', 'u'] = -max_u
        mpc.bounds['upper', '_u', 'u'] = max_u

        # Set TVP
        mpc = self.set_tvp_for_mpc(mpc)
        
        # Add CBF constraints
        if self.near_obs is not None:
            mpc = self.get_cbf_constraints(mpc) 

        mpc.setup()
        mpc.settings.supress_ipopt_output()
        return mpc

    def get_cbf_constraints(self, mpc):
        """Compute DT-HOCBF constraints for the MPC controller"""
        x_k = self.model.x['x']  # Current state [0] xpos, [1] ypos, [2] orien, [3] velocity
        u_k = self.model.u['u']  # Current control input [0] acc, [1] omega

        gamma1 = self.model.tvp['gamma1']
        gamma2 = self.model.tvp['gamma2']
        
        obs = self.model.tvp['obs']

        cbf_constraints = []
        
        if obs != None:
            hocbf_2nd_order = self.robot.agent_barrier(x_k, u_k, gamma1, gamma2, self.dt, self.robot.robot_radius, obs)
            cbf_constraints.append(-hocbf_2nd_order)
        else:
            pass
        
        i = 0
        for cbc in cbf_constraints:
            mpc.set_nl_cons('cbf_constraint'+str(i), cbc, ub=0)
            i += 1
        
        return mpc        
    
    def set_tvp_for_mpc(self, mpc):
        """Set Time Varying Parameters for MPC controller"""
        tvp_struct_mpc = mpc.get_tvp_template()
        def tvp_fun_mpc(t_now):
            tvp_struct_mpc['_tvp', :, "gamma1"] = self.gamma1
            tvp_struct_mpc['_tvp', :, "gamma2"] = self.gamma2
            tvp_struct_mpc['_tvp', :, "goal"] = self.goal.flatten()
            tvp_struct_mpc['_tvp', :, "obs"] = self.near_obs.flatten()
            return tvp_struct_mpc
        mpc.set_tvp_fun(tvp_fun_mpc)
        return mpc
      
    def define_simulator(self):
        """Set MPC simulator"""
        simulator = do_mpc.simulator.Simulator(self.model)
        simulator.set_param(t_step=self.dt)
        tvp_template = simulator.get_tvp_template()

        def tvp_fun(t_now):
            return tvp_template
        simulator.set_tvp_fun(tvp_fun)

        simulator.setup()

        return simulator

    def set_init_state(self):
        """Sets the initial state in MPC for all components."""
        self.mpc.x0 = self.robot.X.flatten()
        self.simulator.x0 = self.robot.X.flatten()
        self.estimator.x0 = self.robot.X.flatten()
        self.mpc.set_initial_guess()
      


    def set_waypoints(self, waypoints):
        self.waypoints = waypoints
        self.current_goal_index = 0
        if self.show_animation:
            self.ax.scatter(waypoints[:, 0], waypoints[:, 1], c='g', s=10)

    def goal_reached(self, current_position, goal_position):
        return np.linalg.norm(current_position[:2] - goal_position[:2]) < self.reached_threshold

    def set_unknown_obs(self, unknown_obs):
        # set initially
        self.unknown_obs = unknown_obs
        for (ox, oy, r) in self.unknown_obs :
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='orange',
                    fill=True,
                    alpha=0.4
                )
            )
        self.robot.test_type = 'mpc-cbf'

    def get_nearest_obs(self, detected_obs):
        # If there are new obstacles detected, update the obs
        if len(detected_obs) != 0:
            try:
                all_obs = np.vstack((self.obs, detected_obs))
            except:
                all_obs = detected_obs
        else:
            all_obs = self.obs

        if len(all_obs) == 0:
            return None

        radius = all_obs[:, 2]
        distances = np.linalg.norm(all_obs[:, :2] - self.robot.X[:2].T, axis=1)
        min_distance_index = np.argmin(distances - radius)
        nearest_obstacle = all_obs[min_distance_index]
        return nearest_obstacle.reshape(-1, 1)

    def is_collide_unknown(self):
        if self.unknown_obs is None:
            return False
        robot_radius = self.robot.robot_radius
        for obs in self.unknown_obs:
            # Check if the robot collides with the obstacle
            distance = np.linalg.norm(self.robot.X[:2].flatten() - obs[:2])
            if distance < obs[2] + robot_radius:
                return True
        return False

    def update_goal(self):
        '''
        Update the goal from waypoints
        '''
        # Check if all waypoints are reached;
        if self.current_goal_index >= len(self.waypoints):
            return None
        
        if self.goal_reached(self.robot.X, np.array(self.waypoints[self.current_goal_index]).reshape(-1, 1)):
            self.current_goal_index += 1
            if self.current_goal_index >= len(self.waypoints):
                print("All waypoints reached.")
                return None
            self.goal = np.array(self.waypoints[self.current_goal_index]).reshape(-1, 1)
            print("Waypoint changed.")
        goal = np.array(self.waypoints[self.current_goal_index]) # set goal to next waypoint's (x,y,theta,velocity)
        return goal


    def control_step(self):
        '''
        Simulate one step of tracking control with CBF-MPC with the given waypoints.
        Output: 
            - -1: all waypoints reached
            - 0: normal
            - 1: visibility violation
            - raise CollisionError: if the robot collides with the obstacle
        '''

        goal = self.update_goal()

        # 1. Update the detected obstacles
        if self.data_generation == False:
            detected_obs = self.robot.detect_unknown_obs(self.unknown_obs)
            nearest_obs = self.get_nearest_obs(detected_obs)
        else:
            # while generating data, we assume every unknown obstacles can be timely detected
            nearest_obs = self.get_nearest_obs(self.unknown_obs)

        # 2. Update obstacle to nearest_obs
        if nearest_obs is not None:
            self.near_obs = nearest_obs.reshape(-1, 1)
        else:
            self.near_obs = np.array([[10.0, 10.0, 0.1]]).reshape(-1, 1)  # A default far away obstacle

         # 3. Compute control input
        u0 = self.mpc.make_step(self.robot.X.flatten())

        # 4. Raise an error if the robot collides with the obstacle
        collide = self.is_collide_unknown()
        if collide:
            if self.show_animation:
                self.robot.render_plot()
                current_position = self.robot.X[:2].flatten()
                self.ax.text(current_position[0]+0.5, current_position[1]+0.5, '!', color='red', weight='bold', fontsize=22)
                if self.robot_id == 0:
                    self.fig.canvas.draw()
                    plt.pause(5)

                    if self.save_animation:
                        plt.savefig(self.current_directory_path +
                                    "/output/animations/" + "t_step_" + str(self.ani_idx) + ".png")
            raise CollisionError

        # 5. Step the robot
        y_next = self.simulator.make_step(u0)
        x0 = self.estimator.make_step(y_next)
        self.robot.X = x0.reshape(-1, 1)
        if self.show_animation:
            self.robot.render_plot()

        # *. Compute the spatial density function Φ(z) (Only while data generation)
        if self.data_generation:
            z = self.robot.X[:2].flatten()  # Current position of the robot
            relative_angle = np.arctan2(self.near_obs[1] - self.robot.X[1], self.near_obs[0] - self.robot.X[0]) - self.robot.X[2]
            obs = {
                'z': self.near_obs[:2].flatten(),
                'h': self.robot.agent_barrier(self.robot.X, u0, self.gamma1, self.gamma2, self.dt, self.robot.robot_radius, self.near_obs),
                'd': angle_normalize(relative_angle)
            }
            phi = compute_safety_loss_function(obs, self.alpha_obs, self.beta_obs, self.gamma_loss, self.lambda_loss, z)
            print(f"Safety Loss Function Value: {phi}, Normalized Relative angle: {angle_normalize(relative_angle)}, Relative angle: {relative_angle}")

        # 6. Update sensing information (Skipped while data generation)
        if self.data_generation == False:
            self.robot.update_sensing_footprints()
            self.robot.update_safety_area()

            beyond_flag = self.robot.is_beyond_sensing_footprints()
            if beyond_flag and self.show_animation:
                print("Visibility Violation")
        else: 
            beyond_flag = False

        if self.show_animation:
            if self.robot_id == 0:
                self.fig.canvas.draw()
                plt.pause(0.01)
                if self.save_animation and self.ani_idx % self.save_per_frame == 0:
                    plt.savefig(self.current_directory_path +
                                "/output/animations/" + "t_step_" + str(self.ani_idx) + ".png")
                    self.ani_idx += 1

        if goal is None:
            return -1 # all waypoints reached
        return beyond_flag
    
    
    def run_all_steps(self, tf=30):
        print("===================================")
        print("============ Tracking =============")
        print("Start following the generated path.")
        unexpected_beh = 0

        for _ in range(int(tf / self.dt)):
            ret = self.control_step()
            unexpected_beh += ret
            if ret == -1: # all waypoints reached
                break

        # convert the image sequence to a video
        if self.show_animation and self.save_animation:
            subprocess.call(['ffmpeg',
                            '-i', self.current_directory_path+"/output/animations/" + "/t_step_%01d.png",
                            '-r', '60',  # Changes the output FPS to 30
                            '-pix_fmt', 'yuv420p',
                            self.current_directory_path+"/output/animations/tracking.mp4"])

            for file_name in glob.glob(self.current_directory_path +
                            "/output/animations/*.png"):
                os.remove(file_name)

        print("=====   Tracking finished    =====")
        print("===================================\n")
        if self.show_animation:
            plt.ioff()
            plt.close()

        return unexpected_beh

    def plot_safety_loss_function(self):
        if not self.data_generation:
            return

        # Create a grid of points to evaluate the safety loss function
        x_range = np.linspace(self.robot.X[0] - 10, self.robot.X[0] + 5, 100)
        y_range = np.linspace(self.robot.X[1] - 5, self.robot.X[1] + 5, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)

        # Compute the safety loss function for each point in the grid
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                z = np.array([X[i, j], Y[i, j]])
                obs = {
                    'z': self.near_obs[:2].flatten(),
                    'h': self.robot.agent_barrier(self.robot.X, np.array([0, 0]), self.gamma1, self.gamma2, self.dt, self.robot.robot_radius, self.near_obs),
                    'd': angle_normalize(self.robot.X[2])
                }
                Z[i, j] = compute_safety_loss_function(obs, self.alpha_obs, self.beta_obs, self.gamma_loss, self.lambda_loss, z)

        # Plot the safety loss function
        plt.figure()
        cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(cp)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Safety Loss Function')
        plt.scatter(self.near_obs[0], self.near_obs[1], color='red', label='Obstacle')
        plt.legend()
        plt.show()

    def plot_safety_loss_function_grid(self):
        if not self.data_generation:
            return

        alpha_obs_values = [1.5, 1.0, 0.5]
        delta_theta_values = [-0.1, -0.7, -1.4, -2.1]
        beta_obs = 0.5
        fig, axs = plt.subplots(3, 4, figsize=(15, 15), subplot_kw={'projection': '3d'})
        plt.subplots_adjust(hspace=0.4, wspace=0.4)

        x_range = np.linspace(self.robot.X[0] - 10, self.robot.X[0] + 5, 30)
        y_range = np.linspace(self.robot.X[1] - 5, self.robot.X[1] + 5, 30)
        X, Y = np.meshgrid(x_range, y_range)

        for j, delta_theta in enumerate(delta_theta_values):
            for i, alpha_obs in enumerate(alpha_obs_values):
                Z = np.zeros_like(X)
                for m in range(X.shape[0]):
                    for n in range(X.shape[1]):
                        z = np.array([X[m, n], Y[m, n]])
                        obs = {
                            'z': self.near_obs[:2].flatten(),
                            'h': self.robot.agent_barrier(self.robot.X, np.array([0, 0]), self.gamma1, self.gamma2, self.dt, self.robot.robot_radius, self.near_obs),
                            'd': delta_theta
                        }
                        Z[m, n] = compute_safety_loss_function(obs, alpha_obs, beta_obs, self.gamma_loss, self.lambda_loss, z)

                ax = axs[i, j]
                ax.plot_surface(X, Y, Z, cmap='viridis')
                ax.set_title(f'alpha_obs = {alpha_obs}, delta_theta = {delta_theta}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Safety Loss Function')

        plt.show()

    def plot_safety_loss_function_grid_contour(self):
        if not self.data_generation:
            return

        alpha_obs_values = [1.5, 1.0, 0.5]
        delta_theta_values = [-0.1, -0.7, -1.4, -2.1]
        beta_obs = 0.5
        fig, axs = plt.subplots(3, 4, figsize=(15, 15))
        plt.subplots_adjust(hspace=0.4, wspace=0.4)

        x_range = np.linspace(self.robot.X[0] - 10, self.robot.X[0] + 5, 30)
        y_range = np.linspace(self.robot.X[1] - 5, self.robot.X[1] + 5, 30)
        X, Y = np.meshgrid(x_range, y_range)

        for j, delta_theta in enumerate(delta_theta_values):
            for i, alpha_obs in enumerate(alpha_obs_values):
                Z = np.zeros_like(X)
                for m in range(X.shape[0]):
                    for n in range(X.shape[1]):
                        z = np.array([X[m, n], Y[m, n]])
                        obs = {
                            'z': self.near_obs[:2].flatten(),
                            'h': self.robot.agent_barrier(self.robot.X, np.array([0, 0]), self.gamma1, self.gamma2, self.dt, self.robot.robot_radius, self.near_obs),
                            'd': delta_theta
                        }
                        Z[m, n] = compute_safety_loss_function(obs, alpha_obs, beta_obs, self.gamma_loss, self.lambda_loss, z)

                ax = axs[i, j]
                cp = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
                fig.colorbar(cp, ax=ax)
                ax.set_title(f'alpha_obs = {alpha_obs}, delta_theta = {delta_theta}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')

        plt.show()




def single_agent_main():
    dt = 0.05

    # temporal
    waypoints = [
        [2, 2, math.pi/2, 0],
        [2, 4, 0, 0],
        [4, 4, 0, 0],
        [4, 2, 0, 0]
    ]
    waypoints = np.array(waypoints, dtype=np.float64)

    x_init = waypoints[0]
    x_goal = waypoints[-1]

    env_handler = env.Env(width=6.0, height=6.0)
    plot_handler = plotting.Plotting(env_handler)
    ax, fig = plot_handler.plot_grid("Local Tracking Controller")

    type = 'DynamicUnicycle2D'
    tracking_controller = LocalTrackingController(x_init, type=type, dt=dt,
                                         show_animation=True,
                                         save_animation=False,
                                         ax=ax, fig=fig,
                                         env=env_handler,
                                         waypoints=waypoints,
                                         data_generation=True)

    unknown_obs = np.array([[2.0, 3.0, 0.1],
                            [4.0, 3.0, 0.1],
                            ]) 
    tracking_controller.set_unknown_obs(unknown_obs)
    tracking_controller.set_waypoints(waypoints)
    tracking_controller.set_init_state()
    unexpected_beh = tracking_controller.run_all_steps(tf=30)


def multi_agent_main():
    # not implemented yet
    pass
            

if __name__ == "__main__":
    from utils import plotting
    from utils import env
    import math
    
    single_agent_main()