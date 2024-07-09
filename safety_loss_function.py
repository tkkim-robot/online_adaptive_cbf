import numpy as np
import matplotlib.pyplot as plt
from tracking import LocalTrackingController, angle_normalize
from utils import plotting
from utils import env
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SafetyLossFunction:
    def __init__(self, alpha_1=0.2, alpha_2=0.1, beta_1=7.0, beta_2=2.5, epsilon=0.07):
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def compute_alpha_k(self, zeta_k):
        return self.alpha_1 * np.exp(-self.alpha_2 * zeta_k)

    def compute_beta_k(self, delta_theta):
        return self.beta_1 * np.exp(-self.beta_2 * (np.cos(delta_theta) + 1))

    def compute_safety_loss_function(self, robot_pos, obs_pos, cbf_constraint_value, delta_theta):
        alpha_k = self.compute_alpha_k(cbf_constraint_value)
        beta_k = self.compute_beta_k(delta_theta)
        phi = alpha_k / (beta_k * np.linalg.norm(robot_pos - obs_pos) ** 2 + self.epsilon)
        return phi

def plot_safety_loss_function_grid(tracking_controller):
    # We assume a zero control input in this plot
    if not tracking_controller.data_generation:
        return

    alpha_1_values = [0.4, 0.2, 0.1]
    delta_theta_values = [-0.1, -0.7, -1.4, -2.1]
    
    fig = make_subplots(rows=3, cols=4, specs=[[{'type': 'surface'}]*4]*3, 
                        subplot_titles=[f'alpha_1 = {alpha_1}, delta_theta = {delta_theta}' 
                                        for delta_theta in delta_theta_values for alpha_1 in alpha_1_values])

    x_range = np.linspace(0, 6, 50)
    y_range = np.linspace(0, 6, 50)
    X, Y = np.meshgrid(x_range, y_range)

    obs_x, obs_y, obs_r = tracking_controller.near_obs.flatten()

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

if __name__ == "__main__":
    dt = 0.05

    # temporal
    waypoints = [
            [0.5, 3, 0.1, 0],
            [5.5, 3, 0, 0]
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

    unknown_obs = np.array([[3, 3, 0.5]])

    tracking_controller.set_unknown_obs(unknown_obs)
    tracking_controller.set_waypoints(waypoints)
    
    tracking_controller.near_obs = unknown_obs.reshape(-1, 1)
    
    plot_safety_loss_function_grid(tracking_controller)
