import numpy as np
from casadi import *
from tracking import LocalTrackingController
from utils import plotting
from utils import env
import matplotlib.pyplot as plt


# FIXME: safety density loss function should be a class. And if we just run that script containing the class, it should not run the plotting functions just as this script does.
# FIXME: then, when you generate data, you should import that function. 
# FIXME: for example, safety_density_loss.py has a class, and a direct running example
# FIXME: then, tracking.py has a class, and a direct running example
# FIXME: finally, data_generation.py has a class that import safety_density_loss.py and tracking.py, and a direct running example. That example is the final "Data Generating" function that we will use.
# FIXME:: the other direct examples are very helpful for debugging and for readers to understand, so they are the must-have.

def compute_alpha_k(h_k, alpha_obs, gamma): # FIXME: don't term it as gamma and lambda. they are too common. instead, rather do alpha_obs -> alpha_1, gama -> alpha 2, beta_obs -> beta_1, lambda_ -> beta_2
    return alpha_obs * np.exp(-gamma * h_k)

def compute_beta_k(delta_theta, beta_obs, lambda_):
    print("delta_theta: ", delta_theta)
    print(np.exp(-lambda_ * (cos(delta_theta)+1)) )
    return beta_obs * np.exp(-lambda_ * (cos(delta_theta)+1)) 


#FIXME: I intentionally leave this ugly code here. You can see and erase
# FIXME: things to tune: lambda_, and the 0.25 that you see below. You should define it as a variable like self.epsilon
# FIXME: also, try to change the variable name that I mentinoed in another place.
lambda_ = 3.0
compute_beta_k(0.0, 1.0, lambda_)
compute_beta_k(-1, 1.0, lambda_)
compute_beta_k(-2, 1.0, lambda_)
compute_beta_k(-3, 1.0, lambda_)

def compute_safety_loss_function(obs, alpha_obs, beta_obs, gamma, lambda_, z): #FIXME: use obs as a dict is weird. and variable z should named as robot_pos or something like that. it's very confusing
    z_k, h_k, delta_theta = obs['z'], obs['h'], obs['d']
    alpha_k = compute_alpha_k(h_k, alpha_obs, gamma) #FIXME: those parameters should not be passed to the function, instead, define as a local instances of a class
    beta_k = compute_beta_k(delta_theta, beta_obs, lambda_)
    phi = alpha_k / (beta_k * np.linalg.norm(z - z_k)**2 + 0.25)
    return phi


# FIXME: if not using this function, and the 2d plot grid function, then remove them.
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
    beta_obs = 1.0
    fig, axs = plt.subplots(3, 4, figsize=(8, 6), subplot_kw={'projection': '3d'})
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
                    robot_state = np.zeros_like(tracking_controller.robot.X)
                    robot_state[0,0] = X[m, n]
                    robot_state[1,0] = Y[m, n]
                    # FIXME: don't call the output of the agent barrier as h, it's fundamentally different from the value of h. Let's use zeta, or cbf_constraint_value in this work (the latter is more claer).
                    obs = {
                        'z': np.array([[3, 3]]), # FIXME: i changed it to a obstacle position directly, but should be changed
                        'h': tracking_controller.robot.agent_barrier(robot_state, np.array([0, 0]), tracking_controller.gamma1, tracking_controller.gamma2, tracking_controller.dt, tracking_controller.robot.robot_radius, tracking_controller.near_obs), # FIXME: need to make a comment that in this visualization, we assume a zero control iinput
                        'd': delta_theta
                    }
                    Z[m, n] = compute_safety_loss_function(obs, alpha_obs, beta_obs, tracking_controller.gamma_loss, tracking_controller.lambda_loss, z)

            ax = axs[i, j]
            ax.plot_surface(X, Y, Z, cmap='viridis')
            ax.set_title(f'alpha_obs = {alpha_obs}, delta_theta = {delta_theta}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Safety Loss Function')
            # set z lim from 0 to 1
            ax.set_zlim([0, 5])
            # FIXME: would be good to draw a circle, with the same height of that concave, with the radius of (robot_r + obs_r) and with the position of the obstacle (in this case, 3,3 but need to be a variable not a magic number)

    plt.show()

def plot_safety_loss_function_grid_2d(tracking_controller):
    if not tracking_controller.data_generation:
        return

    alpha_obs_values = [1.5, 1.0, 0.5]
    delta_theta_values = [-0.1, -0.7, -1.4, -2.1]
    beta_obs = 0.5
    fig, axs = plt.subplots(3, 4, figsize=(5, 8))
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
            # set y lim from 0 to 1
            ax.set_ylim([0, 1])

    plt.show()

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

    unknown_obs = np.array([[3, 3, 0.1]])

    tracking_controller.set_unknown_obs(unknown_obs)
    tracking_controller.set_waypoints(waypoints)
    #tracking_controller.set_init_state() # FIXME: already in the init function of tracking controller
    #unexpected_beh = tracking_controller.run_all_steps(tf=30)
    
    plt.ioff()
    plt.close()
    
    # Plot the safety loss function after reaching the goal
    plot_safety_loss_function_grid(tracking_controller)    

    