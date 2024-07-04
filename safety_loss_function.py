import numpy as np
import matplotlib.pyplot as plt
from tracking import LocalTrackingController, angle_normalize
from utils import plotting
from utils import env


class SafetyLossFunction:
    def __init__(self, alpha_1=0.2, alpha_2=0.1, beta_1=7.0, beta_2=2.5, epsilon=0.07):
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def compute_alpha_k(self, h_k):
        return self.alpha_1 * np.exp(-self.alpha_2 * h_k)

    def compute_beta_k(self, delta_theta):
        return self.beta_1 * np.exp(-self.beta_2 * (np.cos(delta_theta) + 1))

    def compute_safety_loss_function(self, robot_pos, obs_pos, cbf_constraint_value, delta_theta):
        alpha_k = self.compute_alpha_k(cbf_constraint_value)
        beta_k = self.compute_beta_k(delta_theta)
        phi = alpha_k / (beta_k * np.linalg.norm(robot_pos - obs_pos) ** 2 + self.epsilon)
        return phi

    def plot_safety_loss_function_grid(self, tracking_controller):
        # We assume a zero control input in this plot
        if not tracking_controller.data_generation:
            return

        alpha_1_values = [0.4, 0.2, 0.1]
        delta_theta_values = [-0.1, -0.7, -1.4, -2.1]
        fig, axs = plt.subplots(3, 4, figsize=(8, 6), subplot_kw={'projection': '3d'})
        plt.subplots_adjust(hspace=0.4, wspace=0.4)

        x_range = np.linspace(0, 6, 20)
        y_range = np.linspace(0, 6, 20)
        X, Y = np.meshgrid(x_range, y_range)

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
                        self.alpha_1 = alpha_1
                        Z[m, n] = self.compute_safety_loss_function(robot_pos, obs_pos, cbf_constraint_value, delta_theta)

                ax = axs[i, j]
                ax.plot_surface(X, Y, Z, cmap='viridis')
                ax.set_title(f'alpha_1 = {alpha_1}, delta_theta = {delta_theta}', fontsize=10)
                ax.set_xlabel('X', fontsize=8)
                ax.set_ylabel('Y', fontsize=8)
                ax.set_zlabel('Safety Loss Function', fontsize=8)
                ax.set_zlim([0, 5])
                ax.tick_params(axis='both', which='major', labelsize=6)

                # Plotting the circle
                obs_x, obs_y, obs_r = tracking_controller.near_obs.flatten()
                robot_r = tracking_controller.robot.robot_radius
                angle = np.linspace(0, 2 * np.pi, 100)
                circle_x = obs_x + (obs_r + robot_r) * np.cos(angle)
                circle_y = obs_y + (obs_r + robot_r) * np.sin(angle)
                circle_z = np.full_like(circle_x, np.max(Z) + 0.2)  # Set height slightly above max Z value
                
                ax.plot(circle_x, circle_y, circle_z, 'r--')  # Plot the circle
                ax.scatter(obs_x, obs_y, np.max(Z) + 0.2, color='red')  # Mark the obstacle center at the same height

        plt.show()



if __name__ == "__main__":
    alpha_1 = 1.0
    alpha_2 = 0.1
    beta_1 = 1.0
    beta_2 = 1.7 
    epsilon = 0.25
    safety_loss = SafetyLossFunction(alpha_1, alpha_2, beta_1, beta_2, epsilon)
    
    robot_state = np.array([1.0, 3.0, 0.3, 0.0]).reshape(-1,1)
    robot_pos = robot_state[:2]
    obs = np.array([2.0, 3.0, 0.1]).reshape(-1,1)
    obs_pos = obs[:2]
    robot_theta = robot_state[2]
    relative_angle = np.arctan2(obs_pos[1] - robot_pos[1], obs_pos[0] - robot_pos[0]) - robot_theta
    delta_theta = angle_normalize(relative_angle)
    
    from robots.robot import BaseRobot
    ax = plt.axes()
    robot = BaseRobot(robot_state, dt=0.05, ax=ax, data_generation=True)
    
    control_input = np.array([0.3, 0.3])
    gamma1 = 0.05
    gamma2 = 0.05
    robot.set_cbf_params(gamma1=gamma1, gamma2=gamma2)
    cbf_constraint_value = robot.agent_barrier(x_k=robot_state, u_k=control_input, robot_radius=robot.robot_radius, obs=obs)
    phi = safety_loss.compute_safety_loss_function(robot_pos, obs_pos, cbf_constraint_value, delta_theta)
    print(f"Safety Loss Function Value: {phi}")





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
    # unexpected_beh = tracking_controller.run_all_steps(tf=30)
    
    plt.show()
    plt.ioff()
    plt.close()
    
    tracking_controller.near_obs = unknown_obs.reshape(-1, 1)
    # Plot the safety loss function
    tracking_controller.safety_loss.plot_safety_loss_function_grid(tracking_controller)

