import numpy as np
from tracking import LocalTrackingController, CollisionError
from utils import plotting
from utils import env
import matplotlib.pyplot as plt



def single_agent_simulation(distance, velocity, theta, gamma1, gamma2, deadlock_threshold=0.1, max_sim_time=5):
    try:
        dt = 0.05

        # Define waypoints and unknown obstacles based on sampled parameters
        waypoints = np.array([
            [1, 3, theta+0.01, velocity],
            [11, 3, 0, 0]
        ], dtype=np.float64)

        x_init = waypoints[0]
        x_goal = waypoints[-1]

        env_handler = env.Env(width=12.0, height=6.0)
        plot_handler = plotting.Plotting(env_handler)
        ax, fig = plot_handler.plot_grid("Local Tracking Controller")

        tracking_controller = LocalTrackingController(
            x_init, type='DynamicUnicycle2D', dt=dt,
            show_animation=True, save_animation=False,
            ax=ax, fig=fig, env=env_handler,
            waypoints=waypoints, data_generation=True
        )

        # Set gamma values
        tracking_controller.gamma1 = gamma1
        tracking_controller.gamma2 = gamma2
        
        # Set unknown obstacles
        unknown_obs = np.array([[1 + distance, 3, 0.1]])
        tracking_controller.set_unknown_obs(unknown_obs)

        tracking_controller.set_waypoints(waypoints)
        tracking_controller.set_init_state()

        # Run simulation
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
                current_safety_loss = float(tracking_controller.safety_loss)
                if current_safety_loss > safety_loss:
                    safety_loss = current_safety_loss

                print(f"Current Velocity: {tracking_controller.robot.X[3]} | Deadlock Threshold: {deadlock_threshold} | Deadlock time: {deadlock_time}")
                
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
    distance = 1.0
    velocity = 0.0
    theta = 0.001
    gamma1 = 0.99
    gamma2 = 0.99
    result = single_agent_simulation(distance, velocity, theta, gamma1, gamma2)
    print(result)