import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'safe_control'))

import numpy as np
import pickle
import tqdm
import torch
from torch.multiprocessing import Pool
import matplotlib
import matplotlib.pyplot as plt

from safe_control.utils import plotting, env
from safe_control.tracking import LocalTrackingController, InfeasibleError
from safety_loss_function import SafetyLossFunction
from nn_model.penn.gat import GATModule


# sample gamma 0.01-0.35 MPC
# 0.1-3.0 QP need to check

# Robot-specific configurations
ROBOT_SPECS = {
    "DynamicUnicycle2D": {
        "spec": {
            "model": "DynamicUnicycle2D",
            "w_max": 0.5,
            "a_max": 0.5,
            "radius": 0.3
        },
        "param_ranges": {
            "theta_range":     (-np.pi/2,  np.pi/2),
            "gamma0_range":    (0.01, 0.35),
            "gamma1_range":    (0.01, 0.35)
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
            "theta_range":     (-np.pi/2,  np.pi/2),
            "gamma0_range":    (0.01, 1.0),
            "gamma1_range":    (0.01, 1.0)
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
            "theta_range":      (-np.pi/6, np.pi/6),
            "gamma0_range":     (0.01, 1.0),
            "gamma1_range":     (0.01, 1.0)
        }
    }
}



class SuppressPrints:  # Suppress print statements during simulations
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_safety_loss_from_controller(tracking_controller, safety_metric):
    """
    Calculate the safety loss from the tracking controller's current state
    """
    def angle_normalize(x):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)

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
    safety_loss = safety_metric.compute_safety_loss_function(
        robot_state[:2],
        obs_state[:2],
        robot_rad,
        obs_state[2],
        cbf_constraint_value,
        delta_theta
    )
    return safety_loss

def single_agent_simulation_gat(
        robot_model, controller_name,
        gamma0, gamma1, theta,
        num_obstacles=5,
        max_sim_time=20.0,
        deadlock_threshold=0.2,
        show_animation=False
    ):
    """
    Run a single agent simulation with multiple random obstacles to evaluate
    maximum safety loss and deadlock time, returning the constructed graph (PyG graph).
    """
    # 1) Time step
    dt = 0.05

    # 2) Waypoints for the robot's path
    waypoints = np.array([
        [1, 2, theta],
        [8, 2, 0]
    ], dtype=np.float64)

    # Robot initial state
    if robot_model == "Quad2D": # Quad2D => need (x, z) velocities
        vx_init = np.random.uniform(0.0, 1.0)
        vz_init = np.random.uniform(0.0, 1.0)
        x_init = np.append(waypoints[0], [vx_init, vz_init, 0.0])
    else:
        velocity_init = np.random.uniform(0.0, 1.0)
        x_init = np.append(waypoints[0], velocity_init)

    # 3) Create random known obstacles, For each obstacle => random (x, y, radius)
    robot_spec = ROBOT_SPECS[robot_model]["spec"]
    robot_radius = robot_spec["radius"]
    min_gap = robot_radius * 2.0  # Required clearance between obstacles
    obstacles = []
    max_attempts = 500
    attempts = 0

    while len(obstacles) < num_obstacles and attempts < max_attempts:
        ox = np.random.uniform(2.0, 6.0)
        oy = np.random.uniform(0.5, 3.5)
        radius = np.random.uniform(0.2, 0.4)

        # Skip if too close to start or goal
        dist_robot = np.hypot(ox - 1.0, oy - 2.0)
        dist_goal  = np.hypot(ox - 8.0, oy - 2.0)
        if dist_robot < 0.1 or dist_goal < 0.1:
            attempts += 1
            continue

        # Ensure this obstacle has enough gap to all existing ones
        valid = True
        for existing in obstacles:
            ex, ey, er = existing
            center_dist = np.hypot(ox - ex, oy - ey)
            min_clearance = er + radius + min_gap
            if center_dist < min_clearance:
                valid = False
                break
        if valid:
            obstacles.append([ox, oy, radius])
        attempts += 1

    # Initialize plot and environment handlers
    plot_handler = plotting.Plotting(width=10, height=4, known_obs=np.array(obstacles))
    ax, fig = plot_handler.plot_grid("Local Tracking Controller")
    env_handler = env.Env()

    # Set up the robot specifications
    robot_spec = ROBOT_SPECS[robot_model]["spec"]

    if controller_name == "cbf_qp":
        enable_rotation = False
    else:
        enable_rotation = True

    tracking_controller = LocalTrackingController(
        x_init, robot_spec,
        control_type=controller_name,
        dt=dt,
        show_animation=show_animation,
        save_animation=False,
        enable_rotation=enable_rotation,
        ax=ax, fig=fig, env=env_handler,
    )

    # Set the obstacles and waypoints
    tracking_controller.obs = np.array(obstacles)
    tracking_controller.set_waypoints(waypoints)

    # Set the gamma parameters for CBF
    tracking_controller.pos_controller.cbf_param['alpha1'] = gamma0
    tracking_controller.pos_controller.cbf_param['alpha2'] = gamma1

    # 5) Simulate
    safety_metric = SafetyLossFunction()
    sim_time = 0.0
    deadlock_time = 0.0
    safety_loss_upper_bound = 1.0 # tuned to be twice amount of the maximum safety loss without collision
    max_safety_loss = 0.0
    success = True

    for _ in range(int(max_sim_time / dt)):
        try:
            ret = tracking_controller.control_step()
            tracking_controller.draw_plot() 

            sim_time += dt

            if ret == -1:
                print("Arrived to goal successfully.")
                success = True
                break
            if ret == -2:
                print("Collision detected.")
                success = False
                break

            # Check deadlock
            if robot_model == "Quad2D":
                # For Quad2D, index 3 is x-dot, index 4 is z-dot
                vx = tracking_controller.robot.X[3]
                vz = tracking_controller.robot.X[4]
                if np.hypot(vx, vz) < deadlock_threshold:
                    deadlock_time += dt
            else:
                # for ground robots, index 3 is linear velocity
                if abs(tracking_controller.robot.X[3]) < deadlock_threshold:
                    deadlock_time += dt

            # Calculate safety loss
            new_safety_loss = get_safety_loss_from_controller(tracking_controller, safety_metric)
            if new_safety_loss[0] > safety_loss_upper_bound:
                max_safety_loss = new_safety_loss[0]
            if new_safety_loss[0] > max_safety_loss:
                max_safety_loss = new_safety_loss[0]
            # print(new_safety_loss, max_safety_loss, deadlock_time)

        except InfeasibleError:
            success = False
            max_safety_loss = safety_loss_upper_bound
            break

    if ret != -1:
        success = False
        max_safety_loss = safety_loss_upper_bound

    plt.ioff()
    plt.close()

    
    # 6) Construct a graph for the final scenario using GATModule
    # The "robot" should be the initial state and the "goal" should be the second waypoint.
    module = GATModule() 
    if robot_model == "Quad2D":
        # [rx, ry, rtheta, vx, vz]
        rx, ry, rtheta, vx_init, vz_init, _ = x_init
        robot_state = [rx, ry, vx_init, vz_init]
    else:
        # [rx, ry, rtheta, vx]
        rx, ry, rtheta, velocity_init = x_init
        # Convert heading + velocity => vx, vy
        vx_init = velocity_init * np.cos(rtheta)
        vy_init = velocity_init * np.sin(rtheta)
        robot_state = [rx, ry, vx_init, vy_init]

    goal_state = [8.0, 2.0]
    graph_data = module.create_graph(robot=robot_state, obstacles=obstacles, goal=goal_state, deadlock=deadlock_time, risk=max_safety_loss)
    graph_data.gamma = [[gamma0, gamma1]]
    
    return {
        "graph_data": graph_data,
    }

def worker(params):
    with SuppressPrints():  
        result = single_agent_simulation_gat(*params)

    # Ensure all necessary PyG fields are included
    graph_data = result["graph_data"]

    # Convert gamma to a PyTorch tensor if it's a list or NumPy array
    if isinstance(graph_data.gamma, list) or isinstance(graph_data.gamma, np.ndarray):
        graph_data.gamma = torch.tensor(graph_data.gamma, dtype=torch.float)

    # Convert tensors to numpy before returning
    result["graph_data"] = {
        "x": graph_data.x.cpu().numpy(),  
        "edge_index": graph_data.edge_index.cpu().numpy(),
        "edge_attr": graph_data.edge_attr.cpu().numpy(),
        "y": graph_data.y.cpu().numpy() if hasattr(graph_data, 'y') else None, 
        "gamma": graph_data.gamma.cpu().numpy() if hasattr(graph_data, 'gamma') else None
    }
    
    return result


def generate_data_for_model_gat(
    robot_model, controller_name,
    num_samples=10,
    num_processes=1,
    obstacles_range=(2, 10),
    output_prefix="gat_datagen"
):
    """
    Randomly samples multiple obstacles (2~10), random robot initial states,
    random gamma0, gamma1, runs single_agent_simulation_gat, and saves data in .pkl.
    """
    param_ranges = ROBOT_SPECS[robot_model]["param_ranges"]
    th_min, th_max = param_ranges["theta_range"]
    g0_min, g0_max = param_ranges["gamma0_range"]
    g1_min, g1_max = param_ranges["gamma1_range"]

    parameter_space = []
    for _ in range(num_samples):
        gamma0 = np.random.uniform(g0_min, g0_max)
        gamma1 = np.random.uniform(g1_min, g1_max)
        theta = np.random.uniform(th_min, th_max)
        n_obs  = np.random.randint(obstacles_range[0], obstacles_range[1] + 1)
        parameter_space.append((robot_model, controller_name, gamma0, gamma1, theta, n_obs))

    # Use a multiprocessing pool
    pool = Pool(processes=num_processes)
    results = []
    for res in tqdm.tqdm(pool.imap(worker, parameter_space), total=len(parameter_space)):
        results.append(res)
    pool.close()
    pool.join()

    print(f"Finished simulation. Saving simulation results.")
    
    # Save them to a pickle file
    output_file = f"{output_prefix}_{num_samples}_{robot_model}_{controller_name}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved {len(results)} simulation results to {output_file}.")


def single_simulation_example(robot_model, controller_name, gamma0=0.5, gamma1=0.5, theta=0.01):
    """
    Demonstrates running a single simulation with random obstacles, printing the result.
    """
    num_obstacles = np.random.randint(2, 10)
    result = single_agent_simulation_gat(
        robot_model=robot_model,
        controller_name=controller_name,
        gamma0=gamma0, 
        gamma1=gamma1,  
        theta=theta,
        num_obstacles=num_obstacles,
        max_sim_time=20.0,
        show_animation=True
    )
    print("Single Simulation Result:")
    print("Graph Data:", result["graph_data"])
    print("Gamma0:", result["graph_data"].gamma[0][0])
    print("Gamma1:", result["graph_data"].gamma[0][1])
    print("Max Risk:", result["graph_data"].y[0][1])
    print("Deadlock Time:", result["graph_data"].y[0][0])




if __name__ == "__main__":
    controller_list = [
        "cbf_qp", 
        "mpc_cbf",
        ]
    robot_model_list = [
        "DynamicUnicycle2D", 
        "KinematicBicycle2D", 
        "Quad2D"
        ]
    controller_name = controller_list[1]
    robot_model = robot_model_list[0]
    
    TESTMODE = False
    
    if TESTMODE:
        single_simulation_example(robot_model, controller_name, 
                                  gamma0=0.05, gamma1=0.07, theta=0.01)

    else:
        matplotlib.use('Agg') # Use a non-interactive backend to avoid display issues

        generate_data_for_model_gat(
            robot_model=robot_model,
            controller_name=controller_name,
            num_samples=1000,       
            num_processes=5,        # Change based on the number of cores available
            obstacles_range=(2, 10),
            output_prefix="gat_datagen"
        )
        print("Data generation complete!")

