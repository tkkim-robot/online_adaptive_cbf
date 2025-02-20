import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'safe_control'))

import numpy as np
import pickle
import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt

from safe_control.utils import plotting, env
from safe_control.tracking import LocalTrackingController, InfeasibleError
from safety_loss_function import SafetyLossFunction
from gnn_gcbf import GCBFModule

# Use a non-interactive backend to avoid display issues
# matplotlib.use('Agg')


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
    },
    "KinematicBicycle2D": {
        "spec": {
            "model": "KinematicBicycle2D",
            "a_max": 0.5,
            "fov_angle": 170.0,
            "cam_range": 0.01,
            "radius": 0.5
        },
    },
    "Quad2D": {
        "spec": {
            "model": "Quad2D",
            "f_min": 3.0,
            "f_max": 10.0,
            "sensor": "rgbd",
            "radius": 0.25
        },
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

def single_agent_simulation_gnn(
        robot_model, controller_name,
        gamma0, gamma1, theta,
        num_obstacles=5,
        max_sim_time=60.0,
        deadlock_threshold=0.2,
        show_animation=False
    ):
    """
    Run a single agent simulation with multiple random obstacles to evaluate
    maximum safety loss and deadlock time, returning the constructed graph.

    Returns:
        A dictionary containing:
          - "graph_data": PyG graph
          - "gamma0": float
          - "gamma1": float
          - "max_risk": float (the maximum safety loss encountered)
          - "deadlock_time": float
          - "success": bool
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
    obstacles = []
    for _ in range(num_obstacles): #FIXME: should not be within the robot and goal initial position
        ox = np.random.uniform(2.0, 6.0)
        oy = np.random.uniform(1.0, 3.0)
        radius = np.random.uniform(0.2, 0.4)
        obstacles.append([ox, oy, radius])

    # Initialize plot and environment handlers
    plot_handler = plotting.Plotting(width=10, height=4, known_obs=obstacles)
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
    max_safety_loss = 1.0 # tuned to be twice amount of the maximum safety loss without collision
    success = True

    for _ in range(int(max_sim_time / dt)):
        try:
            ret = tracking_controller.control_step()
            tracking_controller.draw_plot() 

            sim_time += dt

            if ret == -1:
                success = True
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
            if new_safety_loss[0] > max_safety_loss:
                max_safety_loss = new_safety_loss[0]

        except InfeasibleError:
            success = False
            break

    plt.ioff()
    plt.close()

    #FIXME: Need to figure out if this is right or wrong!!!
    
    # 6) Construct a graph for the final scenario using GCBFModule
    # The "robot" can be: [x, y, vx, vy] from the final or initial state
    # or some combined notion. Let's just use the initial for demonstration.
    # The "goal" can be the second waypoint. Example below:
    module = GCBFModule()  # or re-use a global instance if you prefer
    # robot => [rx, ry, vx, vy]
    if robot_model == "Quad2D":
        # in that case, x_init => [rx, ry, rtheta, vx, vz, something...]
        # We'll do a simplified approach: just store the 2D velocity as if (vx, vy).
        # This is an approximation for demonstration.
        rx, ry, rtheta, vx_init, vz_init, _ = x_init
        robot_state = [rx, ry, vx_init, vz_init]
    else:
        rx, ry, rtheta, velocity_init = x_init
        # Convert heading + velocity => vx, vy
        vx_init = velocity_init * np.cos(rtheta)
        vy_init = velocity_init * np.sin(rtheta)
        robot_state = [rx, ry, vx_init, vy_init]

    goal = [8.0, 2.0]

    graph_data = module.create_graph(robot=robot_state, obstacles=obstacles, goal=goal, risk=max_safety_loss)

    return {
        "graph_data": graph_data,
        "gamma0": gamma0,
        "gamma1": gamma1,
        "max_risk": max_safety_loss,
        "deadlock_time": deadlock_time,
        "success": success
    }



def generate_data_for_model_gnn(
    robot_model, controller_name,
    num_samples=10,
    num_processes=1,
    obstacles_range=(2, 10),
    output_prefix="gnn_datagen"
):
    """
    Randomly samples multiple obstacles (2~10), random robot initial states,
    random gamma0, gamma1, runs single_agent_simulation_gnn, and saves data in .pkl.
    """
    #FIXME: Need to get the gamma, theta range for each robot dynamics just like the original data_generation script
    parameter_space = []
    for _ in range(num_samples):
        gamma0 = np.random.uniform(0.5, 3.0)
        gamma1 = np.random.uniform(0.5, 3.0)
        theta = np.random.uniform(0.01, np.pi/2)
        n_obs  = np.random.randint(obstacles_range[0], obstacles_range[1]+1)
        parameter_space.append((robot_model, controller_name, gamma0, gamma1, theta, n_obs))

    def worker(params):
        '''
        Worker function for parallel processing
        '''
        with SuppressPrints(): # Suppress output during the simulation
            return single_agent_simulation_gnn(*params)

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
    result = single_agent_simulation_gnn(
        robot_model=robot_model,
        controller_name=controller_name,
        gamma0=gamma0, 
        gamma1=gamma1,  
        theta=theta,
        num_obstacles=2,
        show_animation=True
    )
    print("Single Simulation Result:")
    print("Gamma0:", result["gamma0"])
    print("Gamma1:", result["gamma1"])
    print("Max Risk:", result["max_risk"])
    print("Deadlock Time:", result["deadlock_time"])
    print("Success?", result["success"])
    print("Graph Data:", result["graph_data"])




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

    # generate_data_for_model_gnn(
    #     robot_model=robot_model,
    #     controller_name=controller_name,
    #     num_samples=20,       
    #     num_processes=2,        # Change based on the number of cores available
    #     obstacles_range=(2, 10),
    #     output_prefix="gnn_datagen"
    # )
    # print("Data generation complete!")


    single_simulation_example(robot_model, controller_name, 
                              gamma0=0.1, gamma1=0.1, theta=0.01)

