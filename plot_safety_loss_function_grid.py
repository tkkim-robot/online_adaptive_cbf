import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(project_root, 'safe_control'),
])

import numpy as np
from safe_control.utils import plotting, env
from safe_control.tracking import LocalTrackingController, InfeasibleError
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


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
        Phi = lambda_j / (beta_j * (np.linalg.norm(robot_pos - obs_pos) - robot_rad - obs_rad)**2 + 1)
        return Phi

def save_subplot_svg(lambda_1, delta_theta, Z, Z_obs, X, Y, color_scale, output_dir):
    single_fig = go.Figure()

    # Add safety loss function surface
    single_fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale=color_scale, showscale=True, opacity=0.8, name='Safety Loss'))

    # Add obstacle surface
    single_fig.add_trace(go.Surface(z=Z_obs, x=X, y=Y, colorscale='Reds', showscale=True, opacity=0.9, name='Obstacle'))

    # Update layout for single subplot
    single_fig.update_layout(
        template='plotly_white',
        title=f'λ₁ = {lambda_1}, Δθ = {delta_theta}',
        scene=dict(
            camera=dict(
                eye=dict(x=1.7, y=1.7, z=1.7)
            )
        ),
        autosize=False,
        width=640,
        height=480,
    )

    # Log before saving
    print(f"Saving subplot λ₁ = {lambda_1}, Δθ = {delta_theta}...")

    # Save as SVG
    filename = f'lambda_{lambda_1}_delta_theta_{delta_theta}.svg'
    filepath = os.path.join(output_dir, filename)
    
    # Save the subplot as SVG
    try:
        single_fig.write_image(filepath)
        print(f"Saved subplot: {filepath}")
    except Exception as e:
        print(f"Failed to save subplot λ₁ = {lambda_1}, Δθ = {delta_theta}: {str(e)}")

    return filepath


def plot_safety_loss_function_grid(tracking_controller, safety_metric, save_svg=False, output_dir='svg_subplots'):
    '''
    Plot the safety loss function grid for different lambda_1 and delta_theta values
    We assume a zero control input in this plot
    '''
    lambda_1_values = [0.6, 0.4, 0.2]
    delta_theta_values = [-0.1, -1.5, -2.9]
    
    # Create output directory for SVGs if saving is enabled
    if save_svg:
        os.makedirs(output_dir, exist_ok=True)

    # Create subplots for each combination of lambda_1 and delta_theta
    fig = make_subplots(rows=3, cols=3, specs=[[{'type': 'surface'}]*3]*3, 
                        subplot_titles=[f'λ₁ = {lambda_1}, Δθ = {delta_theta}' 
                                        for lambda_1 in lambda_1_values for delta_theta in delta_theta_values],
                        horizontal_spacing=0.02,  # Adjusted for better spacing
                        vertical_spacing=0.05,     # Adjusted for better spacing
                        )

    # Set the overall theme to 'plotly_white'
    fig.update_layout(template='plotly_white',
                      height=960, width=1920, 
                      title_text="Safety Loss Function Visualization",
                      )

    x_range = np.linspace(0, 8, 22)
    y_range = np.linspace(0, 8, 22)
    # x_range = np.linspace(0, 8, 2)
    # y_range = np.linspace(0, 8, 2)
    X, Y = np.meshgrid(x_range, y_range)

    # Update the detected obstacle
    nearest_obs = tracking_controller.obs.flatten()
    obs_x, obs_y, obs_r = nearest_obs

    # Set CBF parameters
    cbf_alpha1 = 0.15
    cbf_alpha2 = 0.15

    # Define the color scale
    color_scale = 'RdYlBu_r'  # Changed color scale

    # Initialize a list to store save tasks
    save_tasks = []

    # Initialize ThreadPoolExecutor for parallel saving
    if save_svg:
        executor = ThreadPoolExecutor(max_workers=4)  # Adjust based on CPU cores

    # Iterate over each combination to create subplots
    for i, lambda_1 in enumerate(lambda_1_values):
        for j, delta_theta in enumerate(delta_theta_values):
            Z = np.zeros_like(X)
            # Using tqdm to show progress for inner loops
            for m in tqdm(range(X.shape[0]), desc=f'Processing λ₁={lambda_1}, Δθ={delta_theta}'):
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
                    cbf_constraint_value = dd_h + (cbf_alpha1 + cbf_alpha2) * d_h + cbf_alpha1 * cbf_alpha2 * h_k
                    safety_metric.lambda_1 = lambda_1
                    Z[m, n] = safety_metric.compute_safety_loss_function(robot_pos, obs_pos, robot_rad, obs_rad, cbf_constraint_value, relative_theta)

            Z_obs = np.where((X - obs_x) ** 2 + (Y - obs_y) ** 2 <= obs_r ** 2, Z, np.nan)

            # Calculate subplot position
            row = i + 1
            col = j + 1

            # Add safety loss function surface to the main figure
            fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale=color_scale, showscale=False, opacity=0.8), row=row, col=col)
            # Add obstacle surface to the main figure
            fig.add_trace(go.Surface(z=Z_obs, x=X, y=Y, colorscale='Reds', showscale=False), row=row, col=col)

            if save_svg:
                # Schedule saving the subplot as SVG in parallel
                save_tasks.append(executor.submit(save_subplot_svg, lambda_1, delta_theta, Z, Z_obs, X, Y, color_scale, output_dir))

    # Wait for all saving tasks to complete
    if save_svg:
        for future in tqdm(save_tasks, desc='Saving SVGs'):
            filepath = future.result()  # This will raise exceptions if any occurred
            # print(f"Saved subplot: {filepath}")  # Already printed in save_subplot_svg

        executor.shutdown()

    # Update camera for all subplots
    camera = dict(
        eye=dict(x=1.7, y=1.7, z=1.7)  # Adjusted for better zoom
    )
    
    for i in range(1, 4):
        for j in range(1, 4):
            fig.update_scenes(camera=camera, row=i, col=j)

    # Display the combined figure
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

    known_obs = np.array([[4, 4, 0.2]])

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
        'cam_range': 0.0
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
    safety_metric = SafetyLossFunction()

        
    # Plot safety loss function grid
    plot_safety_loss_function_grid(tracking_controller, safety_metric)


if __name__ == "__main__":
    # Example to visualize the safety loss function grid of lambda_1 and delta_theta
    safety_loss_function_example()
