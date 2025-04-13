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

# Use a non-interactive backend to avoid display issues
matplotlib.use('Agg')


# Robot-specific configurations
ROBOT_SPECS = {
    "DynamicUnicycle2D": {
        "spec": {
            "model": "DynamicUnicycle2D",
            "w_max": 0.5,
            "a_max": 0.5,
            "fov_angle": 70.0,
            "cam_range": 3.0,
            "radius": 0.3
        },
        "param_ranges": {
            # distance, velocity, theta, gamma0, gamma1
            "distance_range":  (0.55, 3.0),
            "velocity_range":  (0.01, 1.0),
            "theta_range":     (0.0,  np.pi),
            "gamma0_range":    (0.01, 0.18),
            "gamma1_range":    (0.01, 0.18)
        }
    },

    "KinematicBicycle2D": {
        "spec": {
            "model": "KinematicBicycle2D",
            "a_max": 0.5,
            "fov_angle": 170.0,
            "cam_range": 0.01,
            "radius": 0.5
        },
        "param_ranges": {
            # distance, velocity, theta, gamma0, gamma1
            "distance_range":  (0.9,  3.0),
            "velocity_range":  (0.01, 1.0),
            "theta_range":     (0.0,  np.pi/6),
            "gamma0_range":    (0.01, 3.0),
            "gamma1_range":    (0.01, 3.0)
        }
    },

    "Quad2D": {
        "spec": {
            "model": "Quad2D",
            "f_min": 3.0,
            "f_max": 10.0,
            "sensor": "rgbd",
            "radius": 0.25
        },
        "param_ranges": {
            # distance, velocity_x, velocity_z, theta, gamma0, gamma1
            "distance_range":   (0.62, 3.0),
            "velocity_x_range": (0.01, 1.0),
            "velocity_z_range": (0.01, 1.0),
            "theta_range":      (-np.pi/6, np.pi/6),
            "gamma0_range":     (0.01, 1.1),
            "gamma1_range":     (0.01, 1.1)
        }
    },

    "VTOL2D": {
        "spec": {
            "model": "VTOL2D",
            'radius': 0.6,
            'v_max': 20.0,
            'reached_threshold': 1.0 # meter
        },
        "param_ranges": {
            # x, velocity_x, velocity_z, theta, gamma0, gamma1
            "x_range":   (2.0, 50.0), # transition point is fixed, varying init x
            "z_range":   (8, 10),
            "theta_range":   (-np.pi/8, np.pi/8),
            "velocity_x_range": (5.0, 20.0), # init velocity
            "gamma0_range":     (0.05, 0.35),
            "gamma1_range":     (0.05, 0.35)
        }
    },

}



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
    # TODO:

    u = tracking_controller.get_control_input()
    h_k, d_h, dd_h = tracking_controller.robot.agent_barrier_dt(robot_state, u, obs_state)
    cbf_constraint_value = dd_h + (gamma0 + gamma1) * d_h + gamma0 * gamma1 * h_k
    
    # Compute the safety loss
    safety_loss = safety_metric.compute_safety_loss_function(robot_state[:2], obs_state[:2], robot_rad, obs_state[2], cbf_constraint_value, delta_theta)
    
    return safety_loss

def single_agent_simulation(robot_model, distance, velocity_x, velocity_z, theta, gamma0, gamma1,
                            deadlock_threshold=0.2, max_sim_time=15):
    """
    Run a single agent simulation to evaluate safety loss and deadlock
    """
    try:
        dt = 0.05

        # Set up the robot specifications
        robot_spec = ROBOT_SPECS[robot_model]["spec"]

        # Define waypoints for the robot's path
        waypoints = np.array([
            [1, 2, theta],
            [8, 2, 0]
        ], dtype=np.float64)


        if robot_model == "VTOL2D":
                        # overwrite waypoints
            waypoints = np.array([
                            [70, 10],
                            [70, 0.5]
                        ], dtype=np.float64)
            pillar_1_x = 67.0
            pillar_2_x = 73.0
            obstacles = np.array([
                [pillar_1_x, 6.0, 0.5],
                [pillar_1_x, 7.0, 0.5],
                [pillar_1_x, 8.0, 0.5],
                [pillar_1_x, 9.0, 0.5],
                [pillar_2_x, 1.0, 0.5],
                [pillar_2_x, 2.0, 0.5],
                [pillar_2_x, 3.0, 0.5],
                [pillar_2_x, 4.0, 0.5],
                [pillar_2_x, 5.0, 0.5],
                [pillar_2_x, 6.0, 0.5],
                [pillar_2_x, 7.0, 0.5],
                [pillar_2_x, 8.0, 0.5],
                [pillar_2_x, 9.0, 0.5],
                [pillar_2_x, 10.0, 0.5],
                [pillar_2_x, 11.0, 0.5],
                [pillar_2_x, 12.0, 0.5],
                [pillar_2_x, 13.0, 0.5],
                [pillar_2_x, 14.0, 0.5],
                [pillar_2_x, 15.0, 0.5],
                [60.0, 12.0, 1.5]
            ])

            env_width = 75.0
            env_height = 15.0
            plt.rcParams['figure.figsize'] = [12, 8]
        else:
            # Define known obstacles
            obstacles = np.array([
                [1 + distance, 2, 0.2],
            ])

            env_width = 10.0
            env_height = 4.0
            # Adjust distance considering the radii of the robot and obstacle
            distance_adjusted = distance - obstacles[0][2] - robot_spec["radius"]

        if obstacles.shape[1] != 5:
            obstacles = np.hstack((obstacles, np.zeros((obstacles.shape[0], 2)))) # Set static obs velocity 0.0 at (5, 5)
            
        if robot_model == "Quad2D":
            x_init = np.append(waypoints[0], [velocity_x, velocity_z, 0])
        elif robot_model == "VTOL2D":
            # abuse variable names
            # distance: an array of x and z
            # x_init: x, y, theta, velocity_x, velocity_z, velocity_theta
            x_init = np.array([distance[0], distance[1], theta, velocity_x, 0, 0])
            dist_to_obs = np.linalg.norm(x_init[:2] - obstacles[-1][:2])
            distance_adjusted = dist_to_obs - obstacles[-1][2] - robot_spec["radius"]

        else:
            x_init = np.append(waypoints[0], velocity_x)

        # Initialize plot and environment handlers
        plot_handler = plotting.Plotting(width=env_width, height=env_height, known_obs=obstacles)
        ax, fig = plot_handler.plot_grid("Data Generation")
        env_handler = env.Env()


        control_type = 'mpc_cbf'
        tracking_controller = LocalTrackingController(x_init, robot_spec,
                                                    control_type=control_type,
                                                    dt=dt,
                                                    show_animation=False,
                                                    save_animation=False,
                                                    raise_error=True, # Raise error if infeasible
                                                    ax=ax, fig=fig,
                                                    env=env_handler)


        # Set gamma values for CBF
        tracking_controller.pos_controller.cbf_param['alpha1'] = gamma0
        tracking_controller.pos_controller.cbf_param['alpha2'] = gamma1

        # Set known obstacles and waypoints
        tracking_controller.obs = np.array(obstacles)
        tracking_controller.set_waypoints(waypoints)

        # Initialize safety loss function
        safety_metric = SafetyLossFunction()

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
                if robot_model in ["Quad2D", "VTOL2D"]:
                    # For Quad2D, index 3 is x-dot, index 4 is z-dot
                    vx = tracking_controller.robot.X[3]
                    vz = tracking_controller.robot.X[4]
                    if np.hypot(vx, vz) < deadlock_threshold:
                        deadlock_time += dt
                else:
                    # For ground robots, index 3 is the single velocity
                    if abs(tracking_controller.robot.X[3]) < deadlock_threshold:
                        deadlock_time += dt

                # Calculate safety loss and store the maximum safety metric encountered
                new_safety_loss = get_safety_loss_from_controller(tracking_controller, safety_metric)
                if new_safety_loss > safety_loss:
                    safety_loss = new_safety_loss[0]

            except InfeasibleError:
                plt.ioff()
                plt.close()
                return (distance_adjusted, velocity_x, velocity_z, theta, gamma0, gamma1, False, max_safety_loss, deadlock_time, sim_time)

        plt.ioff()
        plt.close()
        return (distance_adjusted, velocity_x, velocity_z, theta, gamma0, gamma1, True, safety_loss, deadlock_time, sim_time)

    except InfeasibleError:
        plt.ioff()
        plt.close()
        return (distance, velocity_x, velocity_z, theta, gamma0, gamma1, False, max_safety_loss, 0.0, 0.0)

def worker(params):
    '''
    Worker function for parallel processing
    '''
    with SuppressPrints(): # Suppress output during the simulation
        return single_agent_simulation(*params)

def generate_data_for_model(robot_model, samples_per_dimension=5, num_processes=8, batch_size=6):
    '''
    Generate simulation data by running simulations in parallel
    '''
    ranges = ROBOT_SPECS[robot_model]["param_ranges"]

    if robot_model in ("DynamicUnicycle2D", "KinematicBicycle2D"):
        # 5D parameter space
        dist_vals    = np.linspace(ranges["distance_range"][0],  ranges["distance_range"][1],  samples_per_dimension)
        vel_vals     = np.linspace(ranges["velocity_range"][0], ranges["velocity_range"][1], samples_per_dimension)
        theta_vals   = np.linspace(ranges["theta_range"][0],    ranges["theta_range"][1],    samples_per_dimension)
        gamma0_vals  = np.linspace(ranges["gamma0_range"][0],   ranges["gamma0_range"][1],   samples_per_dimension)
        gamma1_vals  = np.linspace(ranges["gamma1_range"][0],   ranges["gamma1_range"][1],   samples_per_dimension)

        parameter_space = []
        for d in dist_vals:
            for v in vel_vals:
                for th in theta_vals:
                    for g0 in gamma0_vals:
                        for g1 in gamma1_vals:
                            parameter_space.append((
                                # robot_model, distance, velocity_x=v, velocity_z=0.0
                                robot_model, d, v, 0.0, th, g0, g1
                            ))

        columns = ["Distance", "Velocity", "Theta", "gamma0", "gamma1",
                   "No Collision", "Safety Loss", "Deadlock Time", "Simulation Time"]

    elif robot_model == "Quad2D":
        # Quad2D => 6D parameter space
        dist_vals     = np.linspace(ranges["distance_range"][0],   ranges["distance_range"][1],   samples_per_dimension)
        velx_vals     = np.linspace(ranges["velocity_x_range"][0], ranges["velocity_x_range"][1], samples_per_dimension)
        velz_vals     = np.linspace(ranges["velocity_z_range"][0], ranges["velocity_z_range"][1], samples_per_dimension)
        theta_vals    = np.linspace(ranges["theta_range"][0],      ranges["theta_range"][1],      samples_per_dimension)
        gamma0_vals   = np.linspace(ranges["gamma0_range"][0],     ranges["gamma0_range"][1],     samples_per_dimension)
        gamma1_vals   = np.linspace(ranges["gamma1_range"][0],     ranges["gamma1_range"][1],     samples_per_dimension)

        parameter_space = []
        for d in dist_vals:
            for vx in velx_vals:
                for vz in velz_vals:
                    for th in theta_vals:
                        for g0 in gamma0_vals:
                            for g1 in gamma1_vals:
                                parameter_space.append((
                                    robot_model, d, vx, vz, th, g0, g1
                                ))

        columns = ["Distance", "VelocityX", "VelocityZ", "Theta", "gamma0", "gamma1",
                   "No Collision", "Safety Loss", "Deadlock Time", "Simulation Time"]
        
    elif robot_model == "VTOL2D":
        x_vals     = np.linspace(ranges["x_range"][0],             ranges["x_range"][1],          10)
        z_vals     = np.linspace(ranges["z_range"][0],             ranges["z_range"][1],          3)
        theta_vals     = np.linspace(ranges["theta_range"][0],     ranges["theta_range"][1],      3)
        velx_vals     = np.linspace(ranges["velocity_x_range"][0], ranges["velocity_x_range"][1], 5)
        gamma0_vals   = np.linspace(ranges["gamma0_range"][0],     ranges["gamma0_range"][1],     7)
        gamma1_vals   = np.linspace(ranges["gamma1_range"][0],     ranges["gamma1_range"][1],     7)
        
        parameter_space = []
        for x in x_vals:
            for z in z_vals:
                for th in theta_vals:
                    for vx in velx_vals:
                        for g0 in gamma0_vals:
                            for g1 in gamma1_vals:
                                parameter_space.append((
                                    robot_model, [x, z], vx, 0.0, th, g0, g1
                                ))

    # Number of total points
    total_points = len(parameter_space)
    total_batches = total_points // batch_size + (1 if total_points % batch_size else 0)

    # Run simulations in batches
    for batch_idx in range(total_batches):
        batch_params = parameter_space[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        pool = Pool(processes=num_processes)
        results = []
        # Collect results in parallel
        for res in tqdm.tqdm(pool.imap(worker, batch_params), total=len(batch_params)):
            results.append(res)
        pool.close()
        pool.join()

        df_raw = pd.DataFrame(results,
                              columns=[
                                  "Distance", "VelX_Returned", "VelZ_Returned", "Theta",
                                  "gamma0", "gamma1", "No Collision", "Safety Loss",
                                  "Deadlock Time", "Simulation Time"
                              ])
        # Transform df_raw to the final columns based on robot model
        if robot_model in ["DynamicUnicycle2D", "KinematicBicycle2D", "VTOL2D"]:
            # For ground robots, use "Velocity" instead of VelX
            df_final = pd.DataFrame()
            df_final["Distance"] = df_raw["Distance"]
            df_final["Velocity"] = df_raw["VelX_Returned"]  # single velocity
            df_final["Theta"] = df_raw["Theta"]
            df_final["gamma0"] = df_raw["gamma0"]
            df_final["gamma1"] = df_raw["gamma1"]
            df_final["No Collision"] = df_raw["No Collision"]
            df_final["Safety Loss"] = df_raw["Safety Loss"]
            df_final["Deadlock Time"] = df_raw["Deadlock Time"]
            df_final["Simulation Time"] = df_raw["Simulation Time"]
        else:
            # Quad2D => rename columns to VelocityX, VelocityZ
            df_final = pd.DataFrame()
            df_final["Distance"] = df_raw["Distance"]
            df_final["VelocityX"] = df_raw["VelX_Returned"]
            df_final["VelocityZ"] = df_raw["VelZ_Returned"]
            df_final["Theta"] = df_raw["Theta"]
            df_final["gamma0"] = df_raw["gamma0"]
            df_final["gamma1"] = df_raw["gamma1"]
            df_final["No Collision"] = df_raw["No Collision"]
            df_final["Safety Loss"] = df_raw["Safety Loss"]
            df_final["Deadlock Time"] = df_raw["Deadlock Time"]
            df_final["Simulation Time"] = df_raw["Simulation Time"]

        df_final.to_csv(f"data_results_{robot_model}_batch_{batch_idx + 1}.csv", index=False)


def concatenate_csv_files(robot_model, total_batches, output_filename):
    """
    Concatenate multiple CSV files generated from the simulation batches
    for the given robot_model.
    """
    all_data = []
    for batch_index in range(total_batches):
        batch_file = f"data_results_{robot_model}_batch_{batch_index + 1}.csv"
        df = pd.read_csv(batch_file)
        all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_filename, index=False)
    print(f"All batch files have been concatenated into {output_filename}")



if __name__ == "__main__":
    robot_model_list = [
        "DynamicUnicycle2D",
        "KinematicBicycle2D",
        "Quad2D",
        "VTOL2D"
    ]
    robot_model = robot_model_list[3]

    samples_per_dimension = 4   # Number of samples per dimension
    num_processes = 6           # Change based on the number of cores available

    if robot_model in ("DynamicUnicycle2D", "KinematicBicycle2D"):
        total_datapoints = samples_per_dimension ** 5
        batch_size = (samples_per_dimension-1) ** 5
    elif robot_model == "Quad2D":
        total_datapoints = samples_per_dimension ** 6
        batch_size = (samples_per_dimension-1) ** 6
    elif robot_model == "VTOL2D":
        total_datapoints = 10*3*3*5*7*7
        batch_size = 10*3*3*5*7

    total_batches = total_datapoints // batch_size + (1 if total_datapoints % batch_size else 0)

    generate_data_for_model(robot_model,
                            samples_per_dimension=samples_per_dimension,
                            num_processes=num_processes,
                            batch_size=batch_size)

    output_csv = f"data_generation_{robot_model}_{samples_per_dimension}datapoint.csv"
    concatenate_csv_files(robot_model, total_batches, output_csv)

    # single_agent_simulation("VTOL2D", [2.0, 10.0], 15.0, 0.0, 0.0, 0.05, 0.05, max_sim_time=15) # success
    # single_agent_simulation("VTOL2D", [2.0, 10.0], 15.0, 0.0, 0.0, 0.35, 0.35, max_sim_time=15) # fail
    # single_agent_simulation("VTOL2D", [50.0, 10.0], 5.0, 0.0, 0.0, 0.35, 0.35, max_sim_time=15) #success
    #single_agent_simulation("Quad2D", 1.0, 5.0, 0.0, 0.0, 0.05, 0.05, max_sim_time=15)

    print("Data generation complete!")
