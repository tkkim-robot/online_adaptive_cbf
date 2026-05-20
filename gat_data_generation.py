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
from safe_control.dynamic_env.main import LocalTrackingControllerDyn
from safety_loss_function import SafetyLossFunction
from nn_model.penn.gat import GATModule



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
            # "theta_range":     (-np.pi/2,  np.pi/2),
            "theta_range":     (-0.01,  0.01),
            "gamma0_range":    (0.01, 0.35),
            "gamma1_range":    (0.01, 0.35)
        }
    },
    "KinematicBicycle2D_DPCBF": {
        "spec": {
            "model": "KinematicBicycle2D_DPCBF",
            # "a_max": 0.3,
            "a_max": 5.0,
            # "v_max": 1.0,
            "radius": 0.3,
        },
        "param_ranges": {
            "theta_range":     (-0.01,  0.01),
            "gamma0_range":    (0.1, 15.0),
            "gamma1_range":    (0.1, 15.0)
        }
    },
    "Quad2D": {
        "spec": {
            "model": "Quad2D",
            "f_min": 2.5,
            "f_max": 5.5,
            "inertia": 0.05,
            "sensor": "rgbd",
            "radius": 0.3
        },
        "param_ranges": {
            "theta_range":      (-np.pi/6, np.pi/6),
            "gamma0_range":     (0.01, 0.99),
            "gamma1_range":     (0.01, 0.99)
        }
    },
    "Quad3D": {
        "spec": {
            "model": "Quad3D",
            "u_min": -2.0,
            "u_max": 5.0,
            "radius": 0.3
        },
        "param_ranges": {
            "theta_range":      (-np.pi/6, np.pi/6),
            "gamma0_range":     (0.01, 0.99),
            "gamma1_range":     (0.01, 0.99)
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
    
    robot_state = tracking_controller.robot.X
    robot_rad = tracking_controller.robot.robot_radius
    obs_state = tracking_controller.nearest_obs.flatten()
    relative_angle = np.arctan2(obs_state[1] - robot_state[1], obs_state[0] - robot_state[0]) - robot_state[2]
    delta_theta = angle_normalize(relative_angle)
    
    # Compute the Control Barrier Function (CBF) values
    model = tracking_controller.robot_spec['model']
    if model in ["KinematicBicycle2D_DPCBF", "Quad3D"]:
        gamma0 = tracking_controller.pos_controller.cbf_param['alpha']
        gamma1 = None
    else:
        gamma0 = tracking_controller.pos_controller.cbf_param['alpha1']
        gamma1 = tracking_controller.pos_controller.cbf_param['alpha2']
    if model in ["KinematicBicycle2D_DPCBF"]:
        h_k, d_h = tracking_controller.robot.agent_barrier_dt(robot_state, np.array([0, 0]), obs_state)
        cbf_constraint_value = d_h + gamma0 * h_k
    elif tracking_controller.robot_spec['model'] in ['Quad3D']:
        h_k, d_h = tracking_controller.robot.agent_barrier_dt(robot_state, np.array([0, 0, 0, 0]), obs_state)
        cbf_constraint_value = d_h + gamma0 * h_k
    else:
        h_k, d_h, dd_h = tracking_controller.robot.agent_barrier_dt(robot_state, np.array([0, 0]), obs_state)
        cbf_constraint_value = dd_h + (gamma0 + gamma1) * d_h + gamma0 * gamma1 * h_k
    
    # gamma0 = tracking_controller.pos_controller.cbf_param['alpha1']
    # if tracking_controller.robot_spec['model'] in ['KinematicBicycle2D_relaxedC3BF']:
    #     h_k, d_h = tracking_controller.robot.agent_barrier_dt(robot_state, np.array([0, 0]), obs_state)
    #     cbf_constraint_value = d_h + gamma0 * h_k
    # elif tracking_controller.robot_spec['model'] in ['Quad3D']:
    #     h_k, d_h = tracking_controller.robot.agent_barrier_dt(robot_state, np.array([0, 0, 0, 0]), obs_state)
    #     cbf_constraint_value = d_h + gamma0 * h_k
    # else:
    #     gamma1 = tracking_controller.pos_controller.cbf_param['alpha2']
    #     h_k, d_h, dd_h = tracking_controller.robot.agent_barrier_dt(robot_state, np.array([0, 0]), obs_state)
    #     cbf_constraint_value = dd_h + (gamma0 + gamma1) * d_h + gamma0 * gamma1 * h_k
    
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
        gamma0, gamma1=None, theta=0.01,
        num_obstacles=5,
        max_sim_time=30.0,
        deadlock_threshold=0.2,
        show_animation=False,
        retry_limit=10,
        _attempt=0,
        save_cccp_traj=False,
    ):
    """
    Run a single agent simulation with multiple random obstacles to evaluate
    maximum safety loss and deadlock time, returning the constructed graph (PyG graph).
    
    Args:
        save_cccp_traj: If True, also stores per-time-step graphs for CCCP calibration.
    """
    if robot_model not in ["KinematicBicycle2D_DPCBF", "Quad3D"] and gamma1 is None:
        raise ValueError("Selected model needs gamma1.")
    
    
    # 1) Time step
    dt = 0.05
    stuck_steps = int(5.0 / dt)
    move_tol = 1.0

    # 2) Waypoints for the robot's path
    waypoints = np.array([
        [2.5, 2, theta],
        [9.5, 2, 0]
    ], dtype=np.float64)

    # Robot initial state
    if robot_model == "Quad2D": # Quad2D => need (x, z) velocities
        vx_init = np.random.uniform(0.0, 1.0)
        vz_init = np.random.uniform(0.0, 1.0)
        x_init = np.append(waypoints[0], [vx_init, vz_init, 0.0])
    elif robot_model == "Quad3D": # Quad3D => [x, y, z, θ, φ, ψ, vx, vy, vz, q, p, r]
        x = waypoints[0][0]
        y = waypoints[0][1]
        z = 0.0
        theta, phi, psi = 0.0, 0.0, 0.0
        vx_init = np.random.uniform(0.0, 1.0)
        vy_init = 0.0
        vz_init = np.random.uniform(0.0, 1.0)
        q, p, r = 0.0, 0.0, 0.0
        x_init = np.array([x, y, z, theta, phi, psi, vx_init, vy_init, vz_init, q, p, r])
    else:
        # DPCBF requires minimum velocity >= 0.2
        if robot_model == "KinematicBicycle2D_DPCBF":
            velocity_init = np.random.uniform(0.2, 1.0)
        else:
            velocity_init = np.random.uniform(0.0, 1.0)
        # velocity_init = 0.4
        x_init = np.append(waypoints[0], velocity_init)

    # 3) Create random known obstacles, For each obstacle => random (x, y, radius)
    robot_spec = ROBOT_SPECS[robot_model]["spec"]
    robot_radius = robot_spec["radius"]
    min_gap = robot_radius * 2.0  # Required clearance between obstacles
    obstacles = []
    max_attempts = 500
    attempts = 0

    while len(obstacles) < num_obstacles and attempts < max_attempts:
        ox = np.random.uniform(0.5, 7.5)
        oy = np.random.uniform(0.5, 3.5)
        radius = np.random.uniform(0.2, 0.4)

        # Skip if too close to start or goal - must account for obstacle radius and robot radius
        dist_robot = np.hypot(ox - 2.5, oy - 2.0)
        dist_goal  = np.hypot(ox - 9.5, oy - 2.0)
        min_clearance_start = radius + robot_radius + 0.5  # Safety margin
        min_clearance_goal = radius + robot_radius + 0.5   # Safety margin
        if dist_robot < min_clearance_start or dist_goal < min_clearance_goal:
            attempts += 1
            continue

        # Ensure this obstacle has enough gap to all existing ones
        valid = True
        for existing in obstacles:
            ex, ey, er = existing[:3]  # Only compare position and radius
            center_dist = np.hypot(ox - ex, oy - ey)
            min_clearance = er + radius + min_gap
            if center_dist < min_clearance*1.2: 
                valid = False
                break
        
        # For tracking, obstacles should follow 7-field format:
        # [x, y, r, vx, vy, y_min_or_theta, flag] where flag=0 for circles
        # Use zeros for velocities and extra field for static circular obstacles
        obstacles.append([ox, oy, radius, 0.0, 0.0, 0.0, 0])
        attempts += 1

    # print(obstacles)
    # obstacles = np.array([[2.7, 2.2, 0.372],
    #                       [6.4, 3.3, 0.2]] ,
    #                      dtype=np.float64)
    # obstacles = np.array([[3.557, 2.102, 0.372], 
    #                       [4.973, 1.2556, 0.2754], 
    #                       [6.407, 3.2399, 0.2066]], 
    #                      dtype=np.float64)
    
    # Initialize plot and environment handlers
    # For plotting, use base obstacle format [x, y, r, type]
    # Extract position, radius, and type flag for visualization
    plot_obstacles = np.array([[obs[0], obs[1], obs[2], 0] for obs in obstacles])
    plot_handler = plotting.Plotting(width=10, height=4, known_obs=plot_obstacles)
    ax, fig = plot_handler.plot_grid("Local Tracking Controller")
    env_handler = env.Env()

    # Set up the robot specifications
    robot_spec = ROBOT_SPECS[robot_model]["spec"]

    if controller_name == "cbf_qp":
        enable_rotation = False
    else:
        enable_rotation = True

    # Use LocalTrackingControllerDyn for KinematicBicycle2D_DPCBF
    if robot_model == "KinematicBicycle2D_DPCBF":
        tracking_controller = LocalTrackingControllerDyn(
            x_init, robot_spec,
            controller_type={'pos': controller_name},
            dt=dt,
            show_animation=show_animation,
            save_animation=False,
            enable_rotation=enable_rotation,
            ax=ax, fig=fig, env=env_handler,
        )
    else:
        tracking_controller = LocalTrackingController(
            x_init, robot_spec,
            controller_type={'pos': controller_name},
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
    if robot_model in ["KinematicBicycle2D_DPCBF", "Quad3D"]:
        tracking_controller.pos_controller.cbf_param['alpha'] = gamma0
    else:
        tracking_controller.pos_controller.cbf_param['alpha1'] = gamma0
        tracking_controller.pos_controller.cbf_param['alpha2'] = gamma1
        
    # tracking_controller.pos_controller.cbf_param['alpha1'] = gamma0
    # if robot_model not in ["KinematicBicycle2D_relaxedC3BF", "Quad3D"]:
    #     tracking_controller.pos_controller.cbf_param['alpha2'] = gamma1
        
    # 5) Simulate
    safety_metric = SafetyLossFunction()
    sim_time = 0.0
    deadlock_time = 0.0
    safety_loss_upper_bound = 1.0 # tuned to be twice amount of the maximum safety loss without collision
    max_safety_loss = 0.0
    success = True

    # Initialize CCCP trajectory graph collection
    cccp_traj_graphs = [] if save_cccp_traj else None
    goal_state = [9.5, 2.0]  # Goal state for graph creation
    module = GATModule() if save_cccp_traj else None

    init_xy = None
    ret = None  # Initialize ret before loop
    for step_idx in range(int(max_sim_time / dt)):
        if init_xy is None:
            init_xy = tracking_controller.robot.X[:2].copy()
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
            elif robot_model == "Quad3D":
                # For Quad3D, index 6 is x-dot, index 8 is z-dot
                vx = tracking_controller.robot.X[6]
                vz = tracking_controller.robot.X[8]
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
            
            # CCCP: Create per-time-step graph for calibration 
            # This implements the trajectory-level nonconformity score collection
            # Each graph represents state X_t at time step t
            if save_cccp_traj:
                # Extract current robot state and convert to [x, y, vx, vy] format
                current_state = np.asarray(tracking_controller.robot.X).flatten()
                if robot_model == "Quad2D":
                    rx, ry = current_state[0], current_state[1]
                    vx, vz = current_state[3], current_state[4]
                    robot_state = [rx, ry, vx, vz]
                elif robot_model == "Quad3D":
                    rx, ry = current_state[0], current_state[1]
                    vx, vz = current_state[6], current_state[8]
                    robot_state = [rx, ry, vx, vz]
                else:
                    # Ground robots: [x, y, theta, v]
                    rx, ry, rtheta = current_state[0], current_state[1], current_state[2]
                    velocity = current_state[3]
                    vx = velocity * np.cos(rtheta)
                    vy = velocity * np.sin(rtheta)
                    robot_state = [rx, ry, vx, vy]
                
                # Create per-step graph with dummy labels (CCCP only needs inputs for JRD)
                step_graph = module.create_graph(
                    robot=robot_state,
                    obstacles=obstacles,
                    goal=goal_state,
                    deadlock=0.0,  # Dummy value, CCCP ignores labels
                    risk=0.0       # Dummy value, CCCP ignores labels
                )
                # Attach gamma parameters (same as final graph)
                step_graph.gamma = [[gamma0]] if gamma1 is None else [[gamma0, gamma1]]
                cccp_traj_graphs.append(step_graph)
            
            if step_idx == stuck_steps:
                moved = np.linalg.norm(tracking_controller.robot.X[:2] - init_xy)
                if moved < move_tol:
                    ret = 'stuck'
                    success = False
                    max_safety_loss = safety_loss_upper_bound
                    break

        except InfeasibleError:
            if _attempt < retry_limit:
                return single_agent_simulation_gat(
                    robot_model, controller_name,
                    gamma0, gamma1, theta,
                    num_obstacles, max_sim_time,
                    deadlock_threshold, show_animation,
                    retry_limit, _attempt + 1,
                    save_cccp_traj
                )
            success = False
            max_safety_loss = safety_loss_upper_bound
            break
        
        except Exception as e:
            # Handle solver errors or any other exceptions during control_step
            if _attempt < retry_limit:
                return single_agent_simulation_gat(
                    robot_model, controller_name,
                    gamma0, gamma1, theta,
                    num_obstacles, max_sim_time,
                    deadlock_threshold, show_animation,
                    retry_limit, _attempt + 1,
                    save_cccp_traj
                )
            success = False
            max_safety_loss = safety_loss_upper_bound
            print(f"Solver/Control error: {type(e).__name__}: {e}")
            break

    if ret != -1:
        success = False
        max_safety_loss = safety_loss_upper_bound

    if ret == 'stuck' and _attempt < retry_limit:
        return single_agent_simulation_gat(
            robot_model, controller_name,
            gamma0, gamma1, theta,
            num_obstacles, max_sim_time,
            deadlock_threshold, show_animation,
            retry_limit, _attempt + 1,
            save_cccp_traj
        )
        
    # tracking_controller.export_video()  # Disabled for memory efficiency
    plt.ioff()
    plt.close('all')  # Close all figures
    plt.clf()  # Clear current figure
    plt.cla()  # Clear current axes

    
    # 6) Construct a graph for the final scenario using GATModule
    # The "robot" should be the initial state and the "goal" should be the second waypoint.
    # Note: This graph is used for training (deadlock/risk prediction), not for CCCP
    if not save_cccp_traj:
        module = GATModule()
    
    if robot_model == "Quad2D":
        # [rx, ry, rtheta, vx, vz]
        rx, ry, rtheta, vx_init, vz_init, _ = x_init
        robot_state = [rx, ry, vx_init, vz_init]
    elif robot_model == "Quad3D":
        # [x, y, z, theta, phi, psi, vx_init, vy_init, vz_init, q, p, r]
        rx, ry, _, _, _, _, vx_init, _, vz_init, _, _, _ = x_init
        robot_state = [rx, ry, vx_init, vz_init]
    else:
        # [rx, ry, rtheta, vx]
        rx, ry, rtheta, velocity_init = x_init
        # Convert heading + velocity => vx, vy
        vx_init = velocity_init * np.cos(rtheta)
        vy_init = velocity_init * np.sin(rtheta)
        robot_state = [rx, ry, vx_init, vy_init]

    graph_data = module.create_graph(robot=robot_state, obstacles=obstacles, goal=goal_state, deadlock=deadlock_time, risk=max_safety_loss)
    graph_data.gamma = [[gamma0]] if gamma1 is None else [[gamma0, gamma1]]
    
    # Construct result: always include graph_data, optionally include cccp_traj_graphs
    result = {
        "graph_data": graph_data,
    }
    if save_cccp_traj:
        result["cccp_traj_graphs"] = cccp_traj_graphs
        
    return result

def worker(params):
    with SuppressPrints():  
        result = single_agent_simulation_gat(*params)

    # Ensure all necessary PyG fields are included
    graph_data = result["graph_data"]

    # Convert gamma to a PyTorch tensor if it's a list or NumPy array
    if isinstance(graph_data.gamma, list) or isinstance(graph_data.gamma, np.ndarray):
        graph_data.gamma = torch.tensor(graph_data.gamma, dtype=torch.float)

    # Convert tensors to numpy before returning (for training graph)
    result["graph_data"] = {
        "x": graph_data.x.cpu().numpy(),  
        "edge_index": graph_data.edge_index.cpu().numpy(),
        "edge_attr": graph_data.edge_attr.cpu().numpy(),
        "y": graph_data.y.cpu().numpy() if hasattr(graph_data, 'y') else None, 
        "gamma": graph_data.gamma.cpu().numpy() if hasattr(graph_data, 'gamma') else None
    }
    
    # Convert CCCP trajectory graphs to serializable format
    if "cccp_traj_graphs" in result:
        cccp_traj_dicts = []
        for step_graph in result["cccp_traj_graphs"]:
            # Convert each per-step graph to dict with NumPy arrays
            step_dict = {
                "x": step_graph.x.cpu().numpy(),
                "edge_index": step_graph.edge_index.cpu().numpy(),
                "edge_attr": step_graph.edge_attr.cpu().numpy(),
                "gamma": step_graph.gamma.cpu().numpy() if hasattr(step_graph, 'gamma') else None,
            }
            # Include y if present (though CCCP will ignore it)
            if hasattr(step_graph, 'y') and step_graph.y is not None:
                step_dict["y"] = step_graph.y.cpu().numpy()
            cccp_traj_dicts.append(step_dict)
        result["cccp_traj_graphs"] = cccp_traj_dicts
    
    return result


def generate_data_for_model_gat(
    robot_model, controller_name,
    num_samples=10,
    num_processes=1,
    obstacles_range=(2, 10),
    output_prefix="gat_datagen",
    save_cccp_traj=False
    ):
    """
    Randomly samples multiple obstacles (2~10), random robot initial states,
    random gamma0, gamma1, runs single_agent_simulation_gat, and saves data in .pkl.
    
    Args:
        save_cccp_traj: If True, also saves per-time-step graphs for CCCP calibration.
    """
    param_ranges = ROBOT_SPECS[robot_model]["param_ranges"]
    th_min, th_max = param_ranges["theta_range"]
    g0_min, g0_max = param_ranges["gamma0_range"]
    g1_min, g1_max = param_ranges["gamma1_range"]

    parameter_space = []
    for _ in range(num_samples):
        gamma0 = np.random.uniform(g0_min, g0_max)
        if robot_model in ["KinematicBicycle2D_DPCBF", "Quad3D"]:
            gamma1 = None               
        else:
            gamma1 = np.random.uniform(g1_min, g1_max)        
        theta = np.random.uniform(th_min, th_max)
        n_obs  = np.random.randint(obstacles_range[0], obstacles_range[1] + 1)
        parameter_space.append((robot_model, controller_name, gamma0, gamma1, theta, n_obs, save_cccp_traj))

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
        show_animation=False,
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
        "KinematicBicycle2D_DPCBF", 
        "Quad2D",
        "Quad3D"
        ]
    controller_name = controller_list[1]
    robot_model = robot_model_list[0]
    
    # CCCP trajectory collection flag (set to True to enable CCCP calibration data)
    save_cccp_traj = False    
    TESTMODE = False
    np.random.seed(42)
    
    if TESTMODE:
        np.random.seed(46)
        single_simulation_example(robot_model, controller_name,
                                  gamma0=12.5, gamma1=0.99, theta=0.01)

        # np.random.seed(46)
        # single_simulation_example(robot_model, controller_name,
        #                           gamma0=0.9, gamma1=0.9, theta=0.01)
        # np.random.seed(46)
        # single_simulation_example(robot_model, controller_name,
        #                           gamma0=0.8, gamma1=0.8, theta=0.01)
        # np.random.seed(46)
        # single_simulation_example(robot_model, controller_name,
        #                           gamma0=0.7, gamma1=0.7, theta=0.01)
        # np.random.seed(46)
        # single_simulation_example(robot_model, controller_name,
        #                           gamma0=0.6, gamma1=0.6, theta=0.01)
        # np.random.seed(46)
        # single_simulation_example(robot_model, controller_name,
        #                           gamma0=0.5, gamma1=0.5, theta=0.01)
        # np.random.seed(46)
        # single_simulation_example(robot_model, controller_name,
        #                           gamma0=0.4, gamma1=0.4, theta=0.01)
        # np.random.seed(46)
        # single_simulation_example(robot_model, controller_name,
        #                           gamma0=0.3, gamma1=0.3, theta=0.01)
        # np.random.seed(46)
        # single_simulation_example(robot_model, controller_name,
        #                           gamma0=0.2, gamma1=0.2, theta=0.01)
        # np.random.seed(46)
        # single_simulation_example(robot_model, controller_name,
        #                           gamma0=0.1, gamma1=0.1, theta=0.01)
        # np.random.seed(46)
        # single_simulation_example(robot_model, controller_name,
        #                           gamma0=0.01, gamma1=0.01, theta=0.01)
        
        
        

    else: 
        matplotlib.use('Agg') # Use a non-interactive backend to avoid display issues

        generate_data_for_model_gat(
            robot_model=robot_model,
            controller_name=controller_name,
            num_samples=200000,       
            num_processes=28,        # Change based on the number of cores available
            obstacles_range=(2, 10),
            output_prefix="gat_datagen_1112",
            save_cccp_traj=save_cccp_traj
        ) 
        print("Data generation complete!")
        
        