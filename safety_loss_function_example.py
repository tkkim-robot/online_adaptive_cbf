import numpy as np
from casadi import *
from tracking import LocalTrackingController
from utils import plotting
from utils import env
import matplotlib.pyplot as plt


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

def plot_safety_loss_function(tracking_controller):
    if not tracking_controller.data_generation:
        return

    # Create a grid of points to evaluate the safety loss function
    x_range = np.linspace(tracking_controller.robot.X[0] - 10, tracking_controller.robot.X[0] + 5, 100)
    y_range = np.linspace(tracking_controller.robot.X[1] - 5, tracking_controller.robot.X[1] + 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)

    # Compute the safety loss function for each point in the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            z = np.array([X[i, j], Y[i, j]])
            obs = {
                'z': tracking_controller.near_obs[:2].flatten(),
                'h': tracking_controller.robot.agent_barrier(tracking_controller.robot.X, np.array([0, 0]), tracking_controller.gamma1, tracking_controller.gamma2, tracking_controller.dt, tracking_controller.robot.robot_radius, tracking_controller.near_obs),
                'd': tracking_controller.angle_normalize(tracking_controller.robot.X[2])
            }
            Z[i, j] = compute_safety_loss_function(obs, tracking_controller.alpha_obs, tracking_controller.beta_obs, tracking_controller.gamma_loss, tracking_controller.lambda_loss, z)

    # Plot the safety loss function
    plt.figure()
    cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(cp)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Safety Loss Function')
    plt.scatter(tracking_controller.near_obs[0], tracking_controller.near_obs[1], color='red', label='Obstacle')
    plt.legend()
    plt.show()



def plot_safety_loss_function_grid(tracking_controller):
    if not tracking_controller.data_generation:
        return

    alpha_obs_values = [1.5, 1.0, 0.5]
    delta_theta_values = [-0.1, -0.7, -1.4, -2.1]
    beta_obs = 0.5
    fig, axs = plt.subplots(3, 4, figsize=(15, 15), subplot_kw={'projection': '3d'})
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    x_range = np.linspace(tracking_controller.robot.X[0] - 10, tracking_controller.robot.X[0] + 5, 30)
    y_range = np.linspace(tracking_controller.robot.X[1] - 5, tracking_controller.robot.X[1] + 5, 30)
    X, Y = np.meshgrid(x_range, y_range)

    for j, delta_theta in enumerate(delta_theta_values):
        for i, alpha_obs in enumerate(alpha_obs_values):
            Z = np.zeros_like(X)
            for m in range(X.shape[0]):
                for n in range(X.shape[1]):
                    z = np.array([X[m, n], Y[m, n]])
                    obs = {
                        'z': tracking_controller.near_obs[:2].flatten(),
                        'h': tracking_controller.robot.agent_barrier(tracking_controller.robot.X, np.array([0, 0]), tracking_controller.gamma1, tracking_controller.gamma2, tracking_controller.dt, tracking_controller.robot.robot_radius, tracking_controller.near_obs),
                        'd': delta_theta
                    }
                    Z[m, n] = compute_safety_loss_function(obs, alpha_obs, beta_obs, tracking_controller.gamma_loss, tracking_controller.lambda_loss, z)

            ax = axs[i, j]
            ax.plot_surface(X, Y, Z, cmap='viridis')
            ax.set_title(f'alpha_obs = {alpha_obs}, delta_theta = {delta_theta}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Safety Loss Function')

    plt.show()

def plot_safety_loss_function_grid_2d(tracking_controller):
    if not tracking_controller.data_generation:
        return

    alpha_obs_values = [1.5, 1.0, 0.5]
    delta_theta_values = [-0.1, -0.7, -1.4, -2.1]
    beta_obs = 0.5
    fig, axs = plt.subplots(3, 4, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    x_range = np.linspace(tracking_controller.robot.X[0] - 5, tracking_controller.robot.X[0], 10)
    y_range = np.linspace(tracking_controller.robot.X[1] - 3, tracking_controller.robot.X[1] + 3, 10)
    X, Y = np.meshgrid(x_range, y_range)

    for j, delta_theta in enumerate(delta_theta_values):
        for i, alpha_obs in enumerate(alpha_obs_values):
            Z = np.zeros_like(X)
            for m in range(X.shape[0]):
                for n in range(X.shape[1]):
                    z = np.array([X[m, n], Y[m, n]])
                    obs = {
                        'z': tracking_controller.near_obs[:2].flatten(),
                        'h': tracking_controller.robot.agent_barrier(tracking_controller.robot.X, np.array([0, 0]), tracking_controller.gamma1, tracking_controller.gamma2, tracking_controller.dt, tracking_controller.robot.robot_radius, tracking_controller.near_obs),
                        'd': delta_theta
                    }
                    Z[m, n] = compute_safety_loss_function(obs, alpha_obs, beta_obs, tracking_controller.gamma_loss, tracking_controller.lambda_loss, z)

            ax = axs[i, j]
            cp = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
            fig.colorbar(cp, ax=ax)
            ax.set_title(f'alpha_obs = {alpha_obs}, delta_theta = {delta_theta}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

    plt.show()



def example():
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

    unknown_obs = np.array([[3, 3, 0.1]])

    tracking_controller.set_unknown_obs(unknown_obs)
    tracking_controller.set_waypoints(waypoints)
    tracking_controller.set_init_state()
    unexpected_beh = tracking_controller.run_all_steps(tf=30)
    
    # Plot the safety loss function after reaching the goal
    plot_safety_loss_function_grid(tracking_controller)    


if __name__ == "__main__":
    example()
    