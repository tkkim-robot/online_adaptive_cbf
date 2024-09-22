import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'safe_control'))

import numpy as np
import matplotlib.pyplot as plt
from safe_control.utils import plotting, env
from safe_control.tracking import LocalTrackingController, InfeasibleError
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class SafetyLossFunction:
    def __init__(self, lambda_1=0.4, lambda_2=0.1, beta_1=100.0, beta_2=2.5):
        '''Initialize parameters for the safety loss function'''
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def compute_lambda_j(self, psi_j):
        '''Compute the alpha component based on the control barrier function constraint value (psi_j)'''
        return self.lambda_1 * np.exp(-self.lambda_2 * psi_j)

    def compute_beta_j(self, delta_theta):
        '''Compute the beta component based on the change in angle (delta_theta)'''
        return self.beta_1 * np.exp(-self.beta_2 * (np.cos(delta_theta) + 1))

    def compute_safety_loss_function(self, robot_pos, obs_pos, robot_rad, obs_rad, cbf_constraint_value, delta_theta):
        '''
        Compute the safety loss function value based on robot position, obstacle position,
        control barrier function constraint value, and change in angle
        '''
        lambda_j = self.compute_lambda_j(cbf_constraint_value)
        beta_j = self.compute_beta_j(delta_theta)
        Phi = lambda_j / (beta_j * (np.linalg.norm(robot_pos - obs_pos) - robot_rad - obs_rad)** 2 + 1)
        return Phi

def plot_safety_loss_function_grid(tracking_controller, safety_metric):
    '''
    Plot the safety loss function grid for different lambda_1 and delta_theta values
    We assume a zero control input in this plot
    '''
    lambda_1_values = [0.6, 0.4, 0.2]
    delta_theta_values = [-0.1, -1.5, -2.9]
    
    # Create subplots for each combination of lambda_1 and delta_theta
    fig = make_subplots(rows=3, cols=3, specs=[[{'type': 'surface'}]*3]*3, 
                        subplot_titles=[f'lambda_1 = {lambda_1}, delta_theta = {delta_theta}' 
                                        for lambda_1 in lambda_1_values for delta_theta in delta_theta_values],
                        horizontal_spacing=0.0,   # Reduce this value to decrease horizontal gap
                        vertical_spacing=0.05     # Reduce this value to decrease vertical gap
                        )

    x_range = np.linspace(0, 6, 50)
    y_range = np.linspace(0, 6, 50)
    X, Y = np.meshgrid(x_range, y_range)

    # Update the detected obstacle
    nearest_obs = tracking_controller.obs.flatten()
    obs_x, obs_y, obs_r = nearest_obs

    # Set CBF parameters
    cbf_gamma0 = 0.15
    cbf_gamma1 = 0.15

    # Calculate and plot safety loss function for each grid point
    for i, lambda_1 in enumerate(lambda_1_values):
        for j, delta_theta in enumerate(delta_theta_values):
            Z = np.zeros_like(X)
            for m in range(X.shape[0]):
                for n in range(X.shape[1]):
                    robot_pos = np.array([X[m, n], Y[m, n]])
                    robot_state = np.zeros_like(tracking_controller.robot.X)
                    robot_state[0, 0] = X[m, n]
                    robot_state[1, 0] = Y[m, n]
                    robot_state[2, 0] = delta_theta
                    robot_state[3, 0] = 1
                    robot_rad = tracking_controller.robot.robot_radius
                    obs_pos = nearest_obs[:2].flatten()
                    obs_rad = nearest_obs[2]
                    relative_theta = np.arctan2(obs_pos[1] - robot_state[1, 0], obs_pos[0] - robot_state[0, 0]) - delta_theta
                    h_k, d_h, dd_h = tracking_controller.robot.agent_barrier_dt(
                        robot_state, np.array([0, 0]), nearest_obs.flatten()
                    )
                    cbf_constraint_value = dd_h + (cbf_gamma0 + cbf_gamma1) * d_h + cbf_gamma0 * cbf_gamma1 * h_k
                    safety_metric.lambda_1 = lambda_1
                    Z[m, n] = safety_metric.compute_safety_loss_function(robot_pos, obs_pos, robot_rad, obs_rad, cbf_constraint_value, relative_theta)

            Z_obs = np.where((X - obs_x) ** 2 + (Y - obs_y) ** 2 <= obs_r ** 2, Z, np.nan)

            fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', showscale=False, opacity=0.8), row=i+1, col=j+1)
            fig.add_trace(go.Surface(z=Z_obs, x=X, y=Y, colorscale='Reds', showscale=False), row=i+1, col=j+1)

    zoom = 1.7
    # Update layout to zoom out the plots
    camera = dict(
        eye=dict(x=zoom, y=zoom, z=zoom)
    )
    
    for i in range(1, 4):
        for j in range(1, 4):
            fig.update_scenes(camera=camera, row=i, col=j)

    fig.update_layout(height=960, width=1920, title_text="Safety Loss Function Visualization")
    fig.show()

def safety_loss_function_example():
    '''
    Example function to visualize the safety loss function grid of lambda_1 and delta_theta
    '''
    dt = 0.05

    # Define waypoints for the robot to follow
    waypoints = np.array([
        [1, 3, 0.05],
        [9, 3, 0]
    ], dtype=np.float64)
    waypoints = np.array(waypoints, dtype=np.float64)
    x_init = np.append(waypoints[0], 0)

    known_obs = np.array([[4, 3, 0.2]])

    # Initialize environment and plotting handler
    plot_handler = plotting.Plotting(width=10, height=6, known_obs=known_obs)
    ax, fig = plot_handler.plot_grid("Safety Loss Function Example")
    env_handler = env.Env()

    # Initialize tracking controller with DynamicUnicycle2D model
    robot_spec = {
        'model': 'DynamicUnicycle2D',
        'w_max': 0.5,
        'a_max': 0.5,
        'fov_angle': 70.0,
        'cam_range': 3.0,
        'radius': 0.3
    }
    control_type = 'mpc_cbf'
    tracking_controller = LocalTrackingController(x_init, robot_spec,
                                                control_type=control_type,
                                                dt=dt,
                                                show_animation=True,
                                                save_animation=False,
                                                ax=ax, fig=fig,
                                                env=env_handler)

    # Set gamma values
    tracking_controller.pos_controller.cbf_param['alpha1'] = 0.1
    tracking_controller.pos_controller.cbf_param['alpha2'] = 0.1

    # Define obstacle
    tracking_controller.obs = known_obs
    tracking_controller.set_waypoints(waypoints)
    
    # Setup safety loss function
    lambda_1 = 0.4
    lambda_2 = 0.1 
    beta_1 = 100.0
    beta_2 = 2.5 
    safety_metric = SafetyLossFunction(lambda_1, lambda_2, beta_1, beta_2)

    for _ in range(int(20 / dt)):
        ret = tracking_controller.control_step()
        tracking_controller.draw_plot()
        
        robot_state = tracking_controller.robot.X
        robot_pos = tracking_controller.robot.X[:2].flatten()
        robot_rad = tracking_controller.robot.robot_radius
        obs_state = tracking_controller.obs[:2].flatten()
        obs_pos = tracking_controller.obs[:2].flatten()[:2]
        obs_rad = tracking_controller.obs[0][2]
        delta_theta = np.arctan2(obs_pos[1] - robot_state[1, 0], obs_pos[0] - robot_state[0, 0]) - robot_state[2, 0]
        h_k, d_h, dd_h = tracking_controller.robot.agent_barrier_dt(robot_state, np.array([0, 0]), obs_state.flatten())
        cbf_constraint_value = dd_h + (0.05 + 0.05) * d_h + 0.05 * 0.05 * h_k        
        safety_loss = safety_metric.compute_safety_loss_function(robot_pos, obs_pos, robot_rad, obs_rad, cbf_constraint_value, delta_theta)
        print(f"Safetyloss: {safety_loss}")
        
        if ret == -1:
            break
        
    # Plot safety loss function grid
    plot_safety_loss_function_grid(tracking_controller, safety_metric)
    
def dead_lock_example(deadlock_threshold=0.2, max_sim_time=15):
    '''
    Example function to simulate a scenario and check for deadlocks
    '''
    try:
        dt = 0.05

        # Define waypoints and unknown obstacles based on sampled parameters
        waypoints = [
                [0.25, 1.5, 0.01],
                [5.75, 1.5, 0]
        ]
        waypoints = np.array(waypoints, dtype=np.float64)
        x_init = waypoints[0]

        known_obs = np.array([[0.75, 1.5, 0.1]])

        # Initialize environment and plotting handler
        plot_handler = plotting.Plotting(width=6, height=3, known_obs=known_obs)
        ax, fig = plot_handler.plot_grid("Dead Lock Example")
        env_handler = env.Env()

        # Initialize tracking controller with DynamicUnicycle2D model
        robot_spec = {
            'model': 'DynamicUnicycle2D',
            'w_max': 0.5,
            'a_max': 0.5,
            'fov_angle': 70.0,
            'cam_range': 3.0,
            'radius': 0.3
        }
        control_type = 'mpc_cbf'
        tracking_controller = LocalTrackingController(x_init, robot_spec,
                                                    control_type=control_type,
                                                    dt=dt,
                                                    show_animation=True,
                                                    save_animation=False,
                                                    ax=ax, fig=fig,
                                                    env=env_handler)

        # Set gamma values for control barrier function
        tracking_controller.pos_controller.cbf_param['alpha1'] = 0.1
        tracking_controller.pos_controller.cbf_param['alpha2'] = 0.1
        
        # Define obstacles
        tracking_controller.obs = known_obs
        tracking_controller.set_waypoints(waypoints)

        # Run simulation to check for deadlocks
        unexpected_beh = 0
        deadlock_time = 0.0
        sim_time = 0.0

        for _ in range(int(max_sim_time / dt)):
            try:
                ret = tracking_controller.control_step()
                tracking_controller.draw_plot()

                unexpected_beh += ret
                sim_time += dt

                # Check for deadlock
                if np.abs(tracking_controller.robot.X[3]) < deadlock_threshold:
                    deadlock_time += dt

                print(f"Current Velocity: {tracking_controller.robot.X[3]} | Deadlock Threshold: {deadlock_threshold} | Deadlock time: {deadlock_time}")
                
            # If collision occurs, handle the exception
            except InfeasibleError:
                plt.close(fig)  

        plt.close(fig)  

    except InfeasibleError:
        plt.close(fig) 



if __name__ == "__main__":
    # Example to visualize the safety loss function grid of lambda_1 and delta_theta
    safety_loss_function_example()
    
    # Example to simulate a scenario and check for deadlocks
    dead_lock_example()