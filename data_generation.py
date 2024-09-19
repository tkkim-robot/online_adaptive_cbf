import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'safe_control'))

import numpy as np
import pandas as pd
import tqdm
from multiprocessing import Pool
import matplotlib
import matplotlib.pyplot as plt
from safe_control.utils import plotting, env
from safe_control.tracking import LocalTrackingController, InfeasibleError
from safety_loss_function import SafetyLossFunction

# Use a non-interactive backend for matplotlib to avoid display issues
matplotlib.use('Agg')

# Suppress print statements during simulations
class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_safety_loss_from_controller(tracking_controller, safety_metric):
    '''
    Calculate the safety loss from the tracking controller's current state
    '''
    def angle_normalize(x):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)

    # Retrieve gamma parameters
    gamma0 = tracking_controller.pos_controller.cbf_param['alpha1']
    gamma1 = tracking_controller.pos_controller.cbf_param['alpha2']
    
    robot_state = tracking_controller.robot.X
    robot_rad = tracking_controller.robot.robot_radius
    obs_state = tracking_controller.nearest_obs.flatten()
    relative_angle = np.arctan2(obs_state[1] - robot_state[1], obs_state[0] - robot_state[0]) - robot_state[2]
    delta_theta = angle_normalize(relative_angle)
    
    # Compute the Control Barrier Function (CBF) values
    h_k, d_h, dd_h = tracking_controller.robot.agent_barrier_dt(robot_state, np.array([0, 0]), obs_state)
    cbf_constraint_value = dd_h + (gamma0 + gamma1) * d_h + gamma0 * gamma1 * h_k
    
    # Compute the safety loss
    safety_loss = safety_metric.compute_safety_loss_function(robot_state[:2], obs_state[:2], robot_rad, obs_state[2], cbf_constraint_value, delta_theta)
    
    return safety_loss

def single_agent_simulation(distance, velocity, theta, gamma0, gamma1, deadlock_threshold=0.2, max_sim_time=25):
    '''
    Run a single agent simulation to evaluate safety loss and deadlock
    '''
    try:
        dt = 0.05

        # Define waypoints for the robot's path
        waypoints = np.array([
            [1, 2, theta],
            [8, 2, 0]
        ], dtype=np.float64)
        x_init = np.append(waypoints[0], velocity)
        
        # Define known obstacles
        obstacles = [
            [1 + distance, 2, 0.2],        
        ]
        
        # Initialize plot and environment handlers
        plot_handler = plotting.Plotting(width=10, height=4, known_obs=obstacles)
        ax, fig = plot_handler.plot_grid("Local Tracking Controller")
        env_handler = env.Env()

        # Set up the robot specifications
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
                                                    show_animation=False,
                                                    save_animation=False,
                                                    ax=ax, fig=fig,
                                                    env=env_handler)

        # Adjust distance considering the radii of the robot and obstacle
        distance = distance - obstacles[0][2] - tracking_controller.robot.robot_radius

        # Set gamma values for CBF
        tracking_controller.pos_controller.cbf_param['alpha1'] = gamma0
        tracking_controller.pos_controller.cbf_param['alpha2'] = gamma1

        # Set known obstacles
        tracking_controller.obs = np.array(obstacles)
        tracking_controller.set_waypoints(waypoints)

        # Initialize safety loss function
        safety_metric = SafetyLossFunction()

        # Run the simulation loop
        unexpected_beh = 0
        deadlock_time = 0.0
        sim_time = 0.0
        safety_loss = 0.0
        max_safety_loss = 1.0 # tuned to be twice amount of the maximum safety loss without collision

        for _ in range(int(max_sim_time / dt)):
            try:
                ret = tracking_controller.control_step()
                tracking_controller.draw_plot()

                unexpected_beh += ret
                sim_time += dt

                if ret == -1:
                    break
                
                # Check for deadlock
                if np.abs(tracking_controller.robot.X[3]) < deadlock_threshold:
                    deadlock_time += dt

                # Calculate safety loss and store the maximum safety metric encountered
                safety_loss_new = get_safety_loss_from_controller(tracking_controller, safety_metric)
                if safety_loss_new > safety_loss:
                    safety_loss = safety_loss_new[0]

            # Handle InfeasibleError and Collision
            except InfeasibleError:
                plt.ioff()
                plt.close()
                return (distance, velocity, theta, gamma0, gamma1, False, max_safety_loss, deadlock_time, sim_time)
                
        plt.ioff()
        plt.close()
        return (distance, velocity, theta, gamma0, gamma1, True, safety_loss, deadlock_time, sim_time)

    except InfeasibleError:
        plt.ioff()
        plt.close()
        return (distance, velocity, theta, gamma0, gamma1, False, max_safety_loss, deadlock_time, sim_time)

def worker(params):
    '''
    Worker function for parallel processing
    '''
    distance, velocity, theta, gamma0, gamma1 = params
    with SuppressPrints():  # Suppress output during the simulation
        result = single_agent_simulation(distance, velocity, theta, gamma0, gamma1)
    return result

def generate_data(samples_per_dimension=5, num_processes=8, batch_size=6):
    '''
    Generate simulation data by running simulations in parallel
    '''
    # Define ranges for the parameter space
    distance_range = np.linspace(0.55, 3.0, samples_per_dimension)
    velocity_range = np.linspace(0.01, 1.0, samples_per_dimension)
    theta_range = np.linspace(0, np.pi, samples_per_dimension)
    gamma0_range = np.linspace(0.01, 0.18, samples_per_dimension)
    gamma1_range = np.linspace(0.01, 0.18, samples_per_dimension)
    parameter_space = [(d, v, theta, g1, g2) for d in distance_range
                       for v in velocity_range
                       for theta in theta_range
                       for g1 in gamma0_range
                       for g2 in gamma1_range]

    # Calculate total number of batches
    total_batches = len(parameter_space) // batch_size + (1 if len(parameter_space) % batch_size != 0 else 0)

    # Run simulations in batches using multiprocessing
    for batch_index in range(total_batches):
        batch_parameters = parameter_space[batch_index * batch_size:(batch_index + 1) * batch_size]

        pool = Pool(processes=num_processes)
        results = []
        # Collect results using multiprocessing
        for result in tqdm.tqdm(pool.imap(worker, batch_parameters), total=len(batch_parameters)):
            results.append(result)
        pool.close()
        pool.join()

        # Store results in a DataFrame and save to a CSV file
        columns = ['Distance', 'Velocity', 'Theta', 'gamma0', 'gamma1', 'No Collision', 'Safety Loss', 'Deadlock Time', 'Simulation Time']
        df = pd.DataFrame(results, columns=columns)
        df.to_csv(f'data_generation_results_batch_{batch_index + 1}.csv', index=False)

def concatenate_csv_files(output_filename, total_batches):
    '''
    Concatenate multiple CSV files generated from the simulation batches
    '''
    all_data = []

    # Read and concatenate batch files
    for batch_index in range(total_batches):
        batch_file = f'data_generation_results_batch_{batch_index + 1}.csv'
        batch_data = pd.read_csv(batch_file)
        all_data.append(batch_data)

    # Concatenate all data into a single DataFrame and save
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_filename, index=False)
    print(f"All batch files have been concatenated into {output_filename}")


if __name__ == "__main__":
    samples_per_dimension = 7   # Number of samples per dimension
    batch_size = 6**5           # Specify the batch size
    num_processes = 6           # Change based on the number of cores available

    total_datapoints = samples_per_dimension ** 5
    total_batches = total_datapoints // batch_size + (1 if total_datapoints % batch_size != 0 else 0)

    # Generate simulation data and concatenate results
    generate_data(samples_per_dimension, num_processes, batch_size)
    concatenate_csv_files(f'data_generation_results_{samples_per_dimension}datapoint_0907.csv', total_batches)

    print("Data generation complete.")
