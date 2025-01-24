import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'safe_control'))
sys.path.append(os.path.join(project_root, 'DistributionallyRobustCVaR'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from safe_control.utils import plotting, env
from safe_control.tracking import LocalTrackingController
from nn_model.penn.nn_iccbf_predict import ProbabilisticEnsembleNN
from DistributionallyRobustCVaR.distributionally_robust_cvar import DistributionallyRobustCVaR
from online_cbf_config import ALL_DEFAULTS, ADAPTIVE_MODELS



class OnlineCBFAdapter:
    def __init__(self, model_name, scaler_name, d_min=0.075, step_size=0.05, epistemic_threshold=0.2, lower_bound=0.01, upper_bound=1.0, robot_model=None):
        '''
        Initialize the adaptive CBF parameter selector
        '''
        self.robot_model = robot_model
        if self.robot_model == 'Quad2D':
            self.extra_state = 1
        else:
            self.extra_state = 0
            
        self.penn = ProbabilisticEnsembleNN(n_states=6+self.extra_state)
        self.penn.load_model(model_name)
        self.penn.load_scaler(scaler_name)
        self.lower_bound = lower_bound  # Lower bound for CBF parameter sampling, Conservative
        self.upper_bound = upper_bound  # Upper bound for CBF parameter sampling, Aggressive
        self.d_min = d_min  # Closest allowable distance to obstacles
        self.step_size = step_size  # Step size for sampling caldidate CBF parameters
        self.epistemic_threshold = epistemic_threshold  # Threshold for filtering predictions based on epistemic uncertainty

    def sample_cbf_parameters(self, current_gamma0, current_gamma1):
        '''
        Sample CBF parameters (gamma0 and gamma1) within a specified range
        '''
        gamma0_range = np.arange(max(self.lower_bound, current_gamma0 - 2.5), min(self.upper_bound, current_gamma0 + 2.5 + self.step_size), self.step_size)
        gamma1_range = np.arange(max(self.lower_bound, current_gamma1 - 2.5), min(self.upper_bound, current_gamma1 + 2.5 + self.step_size), self.step_size)
        return gamma0_range, gamma1_range

    def get_rel_state_wt_obs(self, tracking_controller):
        '''
        Get the relative state of the robot with respect to the nearest obstacle
        '''
        robot_pos = tracking_controller.robot.X[:2, 0].flatten()
        robot_theta = tracking_controller.robot.X[2, 0]
        robot_radius = tracking_controller.robot.robot_radius
        try:
            near_obs = tracking_controller.nearest_obs.flatten()
        except:
            near_obs = [100, 100, 0.2]  # Default obstacle in case of no nearby obstacle
        
        # Calculate distance, velocity, and relative angle with the obstacle
        distance = np.linalg.norm(robot_pos - near_obs[:2]) - 0.45 + robot_radius + near_obs[2]
        delta_theta = np.arctan2(near_obs[1] - robot_pos[1], near_obs[0] - robot_pos[0]) - robot_theta
        delta_theta = ((delta_theta + np.pi) % (2 * np.pi)) - np.pi
        gamma0 = tracking_controller.pos_controller.cbf_param['alpha1']
        gamma1 = tracking_controller.pos_controller.cbf_param['alpha2']
        
        if self.robot_model == 'Quad2D':
            velocity_x = tracking_controller.robot.X[3, 0]
            velocity_z = tracking_controller.robot.X[4, 0]
            return [distance, velocity_x, velocity_z, delta_theta, gamma0, gamma1]
        else:
            velocity = tracking_controller.robot.X[3, 0]
            return [distance, velocity, delta_theta, gamma0, gamma1]

    def predict_with_penn(self, current_state, gamma0_range, gamma1_range):
        '''
        Predict safety loss, deadlock time, and epistemic uncertainty using the Probabilistic Ensemble Neural Network
        '''
        batch_input = []
        for gamma0 in gamma0_range:
            for gamma1 in gamma1_range:
                state = current_state.copy()
                state[3+self.extra_state] = gamma0
                state[4+self.extra_state] = gamma1
                batch_input.append(state)
        
        batch_input = np.array(batch_input)
        y_pred_safety_loss, y_pred_deadlock_time, epistemic_uncertainty = self.penn.predict(batch_input)
        predictions = []

        for i, (gamma0, gamma1) in enumerate(zip(gamma0_range.repeat(len(gamma1_range)), np.tile(gamma1_range, len(gamma0_range)))):
            predictions.append((gamma0, gamma1, y_pred_safety_loss[i], y_pred_deadlock_time[i][0], epistemic_uncertainty[i]))

        return predictions

    def filter_by_epistemic_uncertainty(self, predictions):
        '''
        Filter predictions based on epistemic uncertainty
        We employ Jensen-Renyi Divergence (JRD) with quadratic Renyi entropy, which has a closed-form expression of the divergence of a GMM
        If the JRD D(X) of the prediction of a given input X is greater than the predefined threshold, it is deemed to be out-of-distribution
        '''
        epistemic_uncertainties = [pred[4] for pred in predictions]
        if all(pred > 1.0 for pred in epistemic_uncertainties):
            filtered_predictions = []  # If all uncertainties are high, return an empty list
        else:
            scaler = MinMaxScaler()
            normalized_epistemic_uncertainties = scaler.fit_transform(np.array(epistemic_uncertainties).reshape(-1, 1)).flatten()
            filtered_predictions = [pred for pred, norm_uncert in zip(predictions, normalized_epistemic_uncertainties) if norm_uncert <= self.epistemic_threshold]
        return filtered_predictions

    def calculate_cvar_boundary(self):
        '''
        Calculate the boundary where that the class K functions are locally valid
        '''
        lambda_1 = 0.4  # Same value used from the data generation step
        beta_1 = 100.0  # Same value used from the data generation step
        d_min = self.d_min
        cvar_boundary = lambda_1 / (beta_1 * d_min**2 + 1)
        return cvar_boundary

    def filter_by_aleatoric_uncertainty(self, filtered_predictions):
        '''
        Filter predictions (GMM distribution due to ensemble predictions) based on aleatoric uncertainty 
        using the distributionally robust Conditional Value at Risk (CVaR) boundary
        '''
        final_predictions = []
        cvar_boundary = self.calculate_cvar_boundary()
        for pred in filtered_predictions:
            _, _, y_pred_safety_loss, _, _ = pred
            gmm = self.penn.create_gmm(y_pred_safety_loss)
            cvar_filter = DistributionallyRobustCVaR(gmm)

            if cvar_filter.is_within_boundary(cvar_boundary):
                final_predictions.append(pred)
        return final_predictions

    def select_best_parameters(self, final_predictions, tracking_controller):
        '''
        Select the best CBF parameters based on filtered predictions.
        '''
        # If no predictions were selected, gradually decrease the parameter
        if not final_predictions:
            current_gamma0 = tracking_controller.pos_controller.cbf_param['alpha1']
            current_gamma1 = tracking_controller.pos_controller.cbf_param['alpha2']
            gamma0 = max(self.lower_bound, current_gamma0 - self.step_size)
            gamma1 = max(self.lower_bound, current_gamma1 - self.step_size)
            return gamma0, gamma1
        min_deadlock_time = min(final_predictions, key=lambda x: x[3])[3]
        best_predictions = [pred for pred in final_predictions if pred[3][0] < 1e-3]
        # If no predictions under 1e-3, use the minimum deadlock time
        if not best_predictions:
            best_predictions = [pred for pred in final_predictions if pred[3] == min_deadlock_time]
        # If there are multiple best predictions, use harmonic mean to select the best one
        if len(best_predictions) != 1:
            best_prediction = max(best_predictions, key=lambda x: 2 * (x[0] * x[1]) / (x[0] + x[1]) if (x[0] + x[1]) != 0 else 0)
            return best_prediction[0], best_prediction[1]
        return best_predictions[0][0], best_predictions[0][1]

    def cbf_param_adaptation(self, tracking_controller):
        '''
        Perform adaptive CBF parameter selection based on the prediction from the PENN model 
        which is both confident and satisfies the local validity condition
        '''
        current_state = self.get_rel_state_wt_obs(tracking_controller)
        gamma0_range, gamma1_range = self.sample_cbf_parameters(current_state[3+self.extra_state], current_state[4+self.extra_state])
        predictions = self.predict_with_penn(current_state, gamma0_range, gamma1_range)
        filtered_predictions = self.filter_by_epistemic_uncertainty(predictions)
        final_predictions = self.filter_by_aleatoric_uncertainty(filtered_predictions)
        best_gamma0, best_gamma1 = self.select_best_parameters(final_predictions, tracking_controller)
        if best_gamma0 is not None and best_gamma1 is not None:
            print(f"CBF parameters updated to: {best_gamma0:.2f}, {best_gamma1:.2f} | Total prediction count: {len(predictions)} | Filtered {len(predictions)-len(filtered_predictions)} with Epistemic | Filtered {len(filtered_predictions)-len(final_predictions)} with Aleatoric DR-CVaR")        
        else:
            print(f"CBF parameters updated to: NONE, NONE | Total prediction count: {len(predictions)} | Filtered {len(predictions)-len(filtered_predictions)} with Epistemic | Filtered {len(filtered_predictions)-len(final_predictions)} with Aleatoric DR-CVaR")        
            
        return best_gamma0, best_gamma1


def get_robot_spec_and_obs(robot_model):
    '''
    Returns (robot_spec, default_obs) for a given robot_model.
    '''
    if robot_model not in ALL_DEFAULTS:
        raise ValueError(f"Unknown robot_model '{robot_model}'")

    entry = ALL_DEFAULTS[robot_model]
    return entry["robot_spec"], entry["default_obs"]

def get_controller_defaults(robot_model, controller_name):
    '''
    Returns (controller_type, gamma0, gamma1) for the given robot_model 
    and high-level controller_name.
    '''
    
    if robot_model not in ALL_DEFAULTS:
        raise ValueError(f"Unknown robot_model '{robot_model}'")

    controller_params = ALL_DEFAULTS[robot_model]["controller_params"].get(controller_name)
    if controller_params is None:
        raise ValueError(
            f"Unknown controller_name '{controller_name}' "
            f"for robot_model '{robot_model}'"
        )

    return (
        controller_params["type"],
        controller_params["gamma0"],
        controller_params["gamma1"]
    )

def get_online_cbf_adapter(robot_model):
    '''
    Returns an OnlineCBFAdapter instance for the given robot_model.
    '''
    if robot_model not in ADAPTIVE_MODELS:
        raise ValueError(f"No online adapter config found for '{robot_model}'")

    cfg = ADAPTIVE_MODELS[robot_model]
    return OnlineCBFAdapter(
        model_name=cfg["model_path"],
        scaler_name=cfg["scaler_path"],
        step_size=cfg["step_size"],
        lower_bound=cfg["lower_bound"],
        upper_bound=cfg["upper_bound"],
        robot_model=robot_model
    )


def single_agent_simulation(velocity,
                            waypoints,
                            controller_name,
                            robot_model,
                            max_sim_time=30,
                            dt=0.05):
    """
    Run a single-agent trajectory simulation using the specified 
    robot_model, controller strategy, initial velocity, and waypoints.
    """
 
    # Get the robot spec & default obstacles & default controller type & gamma values
    robot_spec, default_obs = get_robot_spec_and_obs(robot_model)
    ctrl_type, gamma0, gamma1 = get_controller_defaults(robot_model, controller_name)
    
    print(robot_spec, default_obs)
    print(ctrl_type, gamma0, gamma1)

    # Set initial state
    if robot_model == "Quad2D":
        # velocity should be [vx, vz] for Quad2D
        x_init = np.append(waypoints[0], [velocity[0], velocity[1], 0])
    else:
        # velocity is a single scalar for 2D ground vehicles
        x_init = np.append(waypoints[0], velocity)

    # Set plotting and environment
    plot_handler = plotting.Plotting(width=11.0, height=3.8, known_obs=default_obs)
    ax, fig = plot_handler.plot_grid(f"{controller_name} controller")
    env_handler = env.Env()

    # Create the tracking controller
    tracking_controller = LocalTrackingController(
        x_init,
        robot_spec,
        control_type=ctrl_type,
        dt=dt,
        show_animation=True,
        save_animation=False,
        ax=ax,
        fig=fig,
        env=env_handler
    )

    # Initialize the CBF parameters
    tracking_controller.pos_controller.cbf_param['alpha1'] = gamma0
    tracking_controller.pos_controller.cbf_param['alpha2'] = gamma1

    # Load default obstacles & set waypoints
    tracking_controller.obs = default_obs
    tracking_controller.set_waypoints(waypoints)

    # If controller is 'Online Adaptive CBF', get adapter
    if controller_name == 'Online Adaptive CBF':
        online_cbf_adapter = get_online_cbf_adapter(robot_model)
    else:
        online_cbf_adapter = None

    # Main simulation loop
    n_steps = int(max_sim_time / dt)
    for _ in range(n_steps):
        ret = tracking_controller.control_step()
        tracking_controller.draw_plot()

        # Check if we've reached the goal or collided
        if ret == -1:
            # Dist to final waypoint
            dist_to_goal = np.linalg.norm(tracking_controller.robot.X[:2, 0] - waypoints[-1][:2])
            if dist_to_goal < tracking_controller.reached_threshold:
                print("Goal point reached.")
            else:
                print("Collided.")
            break

        # Adapt the CBF parameters if using an online approach
        if online_cbf_adapter is not None:
            best_gamma0, best_gamma1 = online_cbf_adapter.cbf_param_adaptation(tracking_controller)
            if best_gamma0 is not None and best_gamma1 is not None:
                tracking_controller.pos_controller.cbf_param['alpha1'] = best_gamma0
                tracking_controller.pos_controller.cbf_param['alpha2'] = best_gamma1

    tracking_controller.export_video()
    plt.ioff()
    plt.close()




if __name__ == "__main__":
    controller_list = [
        "MPC-CBF low fixed param",
        "MPC-CBF high fixed param",
        "Optimal Decay CBF-QP",
        "Optimal Decay MPC-CBF",
        "Online Adaptive CBF"
    ]
    robot_model_list = [
        "DynamicUnicycle2D",
        "KinematicBicycle2D",
        "Quad2D"
    ]

    # Pick a specific controller and robot model
    controller_name = controller_list[-1]   
    robot_model = robot_model_list[1]       

    # Define waypoints for the simulation
    waypoints = np.array([
        [0.75, 2.0, 0.01],
        [10.0, 1.5, 0.0]
    ], dtype=np.float64)

    # For ground vehicles, velocity is a single scalar
    if robot_model == "Quad2D":
        init_vel = [0.4, 0.2]
    else:
        init_vel = 0.4

    # Run the simulation
    single_agent_simulation(init_vel, waypoints, controller_name, robot_model)
