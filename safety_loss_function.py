import numpy as np
import matplotlib.pyplot as plt
from tracking import LocalTrackingController, CollisionError
from utils import plotting
from utils import env
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class SafetyLossFunction:
    def __init__(self, alpha_1=0.2, alpha_2=0.1, beta_1=7.0, beta_2=2.5, epsilon=0.07):
        '''Initialize parameters for the safety loss function'''
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def compute_alpha_k(self, zeta_k):
        '''Compute the alpha component based on the control barrier function constraint value (zeta_k)'''
        return self.alpha_1 * np.exp(-self.alpha_2 * zeta_k)

    def compute_beta_k(self, delta_theta):
        '''Compute the beta component based on the change in angle (delta_theta)'''
        return self.beta_1 * np.exp(-self.beta_2 * (np.cos(delta_theta) + 1))

    def compute_safety_loss_function(self, robot_pos, obs_pos, cbf_constraint_value, delta_theta):
        '''
        Compute the safety loss function value based on robot position, obstacle position,
        control barrier function constraint value, and change in angle
        '''
        alpha_k = self.compute_alpha_k(cbf_constraint_value)
        beta_k = self.compute_beta_k(delta_theta)
        phi = alpha_k / (beta_k * np.linalg.norm(robot_pos - obs_pos) ** 2 + self.epsilon)
        return phi




def plot_safety_loss_function_grid(tracking_controller):
    '''
    Plot the safety loss function grid for different alpha_1 and delta_theta values
    We assume a zero control input in this plot
    '''
    if not tracking_controller.data_generation:
        return

    alpha_1_values = [0.4, 0.2, 0.1]
    delta_theta_values = [-0.1, -0.7, -1.4, -2.1]
    
    # Create subplots for each combination of alpha_1 and delta_theta
    fig = make_subplots(rows=3, cols=4, specs=[[{'type': 'surface'}]*4]*3, 
                        subplot_titles=[f'alpha_1 = {alpha_1}, delta_theta = {delta_theta}' 
                                        for delta_theta in delta_theta_values for alpha_1 in alpha_1_values])

    x_range = np.linspace(0, 6, 50)
    y_range = np.linspace(0, 6, 50)
    X, Y = np.meshgrid(x_range, y_range)

    obs_x, obs_y, obs_r = tracking_controller.near_obs.flatten()

    # Calculate and plot safety loss function for each grid point
    for j, delta_theta in enumerate(delta_theta_values):
        for i, alpha_1 in enumerate(alpha_1_values):
            Z = np.zeros_like(X)
            for m in range(X.shape[0]):
                for n in range(X.shape[1]):
                    robot_pos = np.array([X[m, n], Y[m, n]])
                    robot_state = np.zeros_like(tracking_controller.robot.X)
                    robot_state[0, 0] = X[m, n]
                    robot_state[1, 0] = Y[m, n]
                    obs_pos = tracking_controller.near_obs[:2].flatten()
                    tracking_controller.robot.set_cbf_params(gamma1=tracking_controller.gamma1, gamma2=tracking_controller.gamma2)
                    cbf_constraint_value = tracking_controller.robot.agent_barrier(
                        robot_state, np.array([0, 0]), tracking_controller.robot.robot_radius, tracking_controller.near_obs
                    )
                    tracking_controller.safety_metric.alpha_1 = alpha_1
                    Z[m, n] = tracking_controller.safety_metric.compute_safety_loss_function(robot_pos, obs_pos, cbf_constraint_value, delta_theta)

            Z_obs = np.where((X - obs_x) ** 2 + (Y - obs_y) ** 2 <= obs_r ** 2, Z, np.nan)

            fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', showscale=False, opacity=0.8), row=i+1, col=j+1)
            fig.add_trace(go.Surface(z=Z_obs, x=X, y=Y, colorscale='Reds', showscale=False), row=i+1, col=j+1)

    fig.update_layout(height=1080, width=1920, title_text="Safety Loss Function Visualization")
    fig.show()

def safety_loss_function_example():
    '''
    Example function to visualize the safety loss function grid of alpha_1 and delta_theta
    '''
    dt = 0.05

    # Define waypoints for the robot to follow
    waypoints = [
            [0.5, 3, 0.1, 0],
            [5.5, 3, 0, 0]
    ]
    waypoints = np.array(waypoints, dtype=np.float64)

    x_init = waypoints[0]
    x_goal = waypoints[-1]

    # Initialize environment and plotting handler
    env_handler = env.Env(width=6.0, height=6.0)
    plot_handler = plotting.Plotting(env_handler)
    ax, fig = plot_handler.plot_grid("Local Tracking Controller")

    # Initialize tracking controller with DynamicUnicycle2D model
    type = 'DynamicUnicycle2D'
    tracking_controller = LocalTrackingController(x_init, type=type, dt=dt,
                                         show_animation=True,
                                         save_animation=False,
                                         ax=ax, fig=fig,
                                         env=env_handler,
                                         waypoints=waypoints,
                                         data_generation=True)

    # Define obstacle
    unknown_obs = np.array([[3, 3, 0.5]])
    tracking_controller.set_unknown_obs(unknown_obs)
    tracking_controller.set_waypoints(waypoints)
    tracking_controller.near_obs = unknown_obs.reshape(-1, 1)
    
    # Plot safety loss function grid
    plot_safety_loss_function_grid(tracking_controller)
    
def dead_lock_example(deadlock_threshold=0.1, max_sim_time=15):
    '''
    Example function to simulate a scenario and check for deadlocks
    '''
    distance = 0.5
    velocity = 0.0
    theta = 0.001
    gamma1 = 0.1
    gamma2 = 0.1
    
    try:
        dt = 0.05

        # Define waypoints and unknown obstacles based on sampled parameters
        waypoints = np.array([
            [1, 3, theta+0.01, velocity],
            [11, 3, 0, 0]
        ], dtype=np.float64)

        x_init = waypoints[0]
        x_goal = waypoints[-1]

        # Initialize environment and plotting handler
        env_handler = env.Env(width=12.0, height=6.0)
        plot_handler = plotting.Plotting(env_handler)
        ax, fig = plot_handler.plot_grid("Local Tracking Controller")

        # Initialize tracking controller with DynamicUnicycle2D model
        tracking_controller = LocalTrackingController(
            x_init, type='DynamicUnicycle2D', dt=dt,
            show_animation=True, save_animation=False,
            ax=ax, fig=fig, env=env_handler,
            waypoints=waypoints, data_generation=True
        )

        # Set gamma values for control barrier function
        tracking_controller.gamma1 = gamma1
        tracking_controller.gamma2 = gamma2
        
        # Set unknown obstacles
        unknown_obs = np.array([[1 + distance, 3, 0.1]])
        tracking_controller.set_unknown_obs(unknown_obs)

        tracking_controller.set_waypoints(waypoints)
        tracking_controller.set_init_state()

        # Run simulation to check for deadlocks
        unexpected_beh = 0
        deadlock_time = 0.0
        sim_time = 0.0
        safety_loss = 0.0

        for _ in range(int(max_sim_time / dt)):
            try:
                ret = tracking_controller.control_step()
                unexpected_beh += ret
                sim_time += dt

                # Check for deadlock
                if np.abs(tracking_controller.robot.X[3]) < deadlock_threshold:
                    deadlock_time += dt

                # Store max safety metric
                if tracking_controller.safety_loss > safety_loss:
                    safety_loss = tracking_controller.safety_loss[0]

                print(f"Current Velocity: {tracking_controller.robot.X[3]} | Deadlock Threshold: {deadlock_threshold} | Deadlock time: {deadlock_time} | Safety Loss Function Value: {tracking_controller.safety_loss}")
                
            # If collision occurs, handle the exception
            except CollisionError:
                plt.close(fig)  
                return distance, velocity, theta, gamma1, gamma2, False, safety_loss, deadlock_time, sim_time

        plt.close(fig)  
        return distance, velocity, theta, gamma1, gamma2, True, safety_loss, deadlock_time, sim_time

    except CollisionError:
        plt.close(fig) 
        return distance, velocity, theta, gamma1, gamma2, False, safety_loss, deadlock_time, sim_time



if __name__ == "__main__":
    # Example to visualize the safety loss function grid of alpha_1 and delta_theta
    safety_loss_function_example()
    
    # Example to simulate a scenario and check for deadlocks
    dead_lock_example()
    