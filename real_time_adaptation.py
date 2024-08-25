import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'cbf_tracking'))
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'DistributionallyRobustCVaR'))

import numpy as np
import matplotlib.pyplot as plt
from cbf_tracking.utils import plotting, env
from cbf_tracking.tracking import LocalTrackingController
from real_time_plot import RealTimePlotter
from evidential_deep_regression import EvidentialDeepRegression
from DistributionallyRobustCVaR.distributionally_robust_cvar import DistributionallyRobustCVaR
from sklearn.preprocessing import MinMaxScaler


class AdaptiveCBFParameterSelector:
    def __init__(self, edr_model, edr_scaler, lower_bound=0.05, upper_bound=1.0, step_size=0.03, epistemic_threshold=1e-1, cvar_boundary=2.0):
        self.edr = EvidentialDeepRegression()
        self.edr.load_saved_model(edr_model)
        self.edr.load_saved_scaler(edr_scaler)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.step_size = step_size
        self.epistemic_threshold = epistemic_threshold
        self.cvar_boundary = cvar_boundary

    def sample_cbf_parameters(self, current_gamma1, current_gamma2):
        gamma1_range = np.arange(max(self.lower_bound, current_gamma1 - 0.3), min(self.upper_bound, current_gamma1 + 0.1 + self.step_size), self.step_size)
        gamma2_range = np.arange(max(self.lower_bound, current_gamma2 - 0.3), min(self.upper_bound, current_gamma2 + 0.1 + self.step_size), self.step_size)
        return gamma1_range, gamma2_range

    def get_rel_state_wt_obs(self, tracking_controller):
        robot_pos = tracking_controller.robot.X[:2, 0].flatten()
        near_obs_pos = tracking_controller.nearest_obs[:2].flatten()
        
        distance = np.linalg.norm(robot_pos - near_obs_pos)
        velocity = tracking_controller.robot.X[3, 0]
        theta = np.arctan2(near_obs_pos[1] - robot_pos[1], near_obs_pos[0] - robot_pos[0])
        gamma1 = tracking_controller.controller.cbf_param['alpha1']
        gamma2 = tracking_controller.controller.cbf_param['alpha2']
        
        return [distance, velocity, theta, gamma1, gamma2]
        # return [distance, 7, 9, velocity, theta, gamma1, gamma2]

    def predict_with_edr(self, current_state, gamma1_range, gamma2_range):
        batch_input = []
        for gamma1 in gamma1_range:
            for gamma2 in gamma2_range:
                state = current_state.copy()
                state[3] = gamma1
                state[4] = gamma2
                # state[5] = gamma1
                # state[6] = gamma2
                batch_input.append(state)
        
        batch_input = np.array(batch_input)
        y_pred_safety_loss, y_pred_deadlock_time = self.edr.predict(batch_input)
        predictions = []

        for i, (gamma1, gamma2) in enumerate(zip(gamma1_range.repeat(len(gamma2_range)), np.tile(gamma2_range, len(gamma1_range)))):
            gamma, aleatoric_uncertainty, epistemic_uncertainty = self.edr.calculate_uncertainties(y_pred_safety_loss[i])
            predictions.append((gamma1, gamma2, y_pred_safety_loss[i], y_pred_deadlock_time[i][0], aleatoric_uncertainty, epistemic_uncertainty))

        return predictions

    def filter_by_epistemic_uncertainty(self, predictions):
        epistemic_uncertainties = [pred[5] for pred in predictions]
        scaler = MinMaxScaler()
        normalized_epistemic_uncertainties = scaler.fit_transform(np.array(epistemic_uncertainties).reshape(-1, 1)).flatten()
        filtered_predictions = [pred for pred, norm_uncert in zip(predictions, normalized_epistemic_uncertainties) if norm_uncert <= self.epistemic_threshold]
        return filtered_predictions

    def filter_by_aleatoric_uncertainty(self, filtered_predictions):
        final_predictions = []
        for pred in filtered_predictions:
            gamma1, gamma2, y_pred_safety_loss, _, aleatoric_uncertainty, _ = pred
            gmm = self.edr.create_gmm(y_pred_safety_loss)
            cvar_filter = DistributionallyRobustCVaR(gmm)
            if cvar_filter.is_within_boundary(self.cvar_boundary):
                final_predictions.append(pred)
        return final_predictions

    def select_best_parameters(self, filtered_predictions):
        if not filtered_predictions:
            return None, None
        min_deadlock_time = min(filtered_predictions, key=lambda x: x[3])[3]
        best_predictions = [pred for pred in filtered_predictions if pred[3] < 1e-3]
        # If no predictions under 1e-3, use the minimum deadlock time
        if not best_predictions:
            best_predictions = [pred for pred in filtered_predictions if pred[3] == min_deadlock_time]
        # If there are multiple best predictions, use harmonic mean to select the best one
        if len(best_predictions) != 1:
            best_prediction = max(best_predictions, key=lambda x: 2 * (x[0] * x[1]) / (x[0] + x[1]) if (x[0] + x[1]) != 0 else 0)
            return best_prediction[0], best_prediction[1]
        return best_predictions[0][0], best_predictions[0][1]

    def adaptive_parameter_selection(self, tracking_controller):
        current_state = self.get_rel_state_wt_obs(tracking_controller)
        gamma1_range, gamma2_range = self.sample_cbf_parameters(current_state[3], current_state[4])
        # gamma1_range, gamma2_range = self.sample_cbf_parameters(current_state[5], current_state[6])
        predictions = self.predict_with_edr(current_state, gamma1_range, gamma2_range)
        filtered_predictions = self.filter_by_epistemic_uncertainty(predictions)
        final_predictions = self.filter_by_aleatoric_uncertainty(filtered_predictions)
        best_gamma1, best_gamma2 = self.select_best_parameters(final_predictions)
        if best_gamma1 is not None and best_gamma2 is not None:
            print(f"CBF parameters updated to: {best_gamma1:.2f}, {best_gamma2:.2f} | Total prediction count: {len(predictions)} | Filtered {len(predictions)-len(filtered_predictions)} with Epistemic | Filtered {len(filtered_predictions)-len(final_predictions)} with Aleatoric DR-CVaR")        
        else:
            print(f"CBF parameters updated to: NONE, NONE | Total prediction count: {len(predictions)} | Filtered {len(predictions)-len(filtered_predictions)} with Epistemic | Filtered {len(filtered_predictions)-len(final_predictions)} with Aleatoric DR-CVaR")        
            
        return best_gamma1, best_gamma2






def single_agent_simulation(distance, velocity, theta, gamma1, gamma2, max_sim_time=20):
    dt = 0.05

    waypoints = np.array([
        [1, 3, theta],
        [11, 3, 0]
    ], dtype=np.float64)

    x_init = np.append(waypoints[0], velocity)

    plot_handler = plotting.Plotting()
    ax, fig = plot_handler.plot_grid("Local Tracking Controller")
    env_handler = env.Env()
    
    # Set robot with controller 
    robot_spec = {
        'model': 'DynamicUnicycle2D',
        'w_max': 0.5,
        'a_max': 0.5,
        'fov_angle': 70.0,
        'cam_range': 7.0
    }
    control_type = 'mpc_cbf'
    tracking_controller = LocalTrackingController(x_init, robot_spec,
                                                control_type=control_type,
                                                dt=dt,
                                                show_animation=True,
                                                save_animation=False,
                                                ax=ax, fig=fig,
                                                env=env_handler)

    # Initialize AdaptiveCBFParameterSelector
    adaptive_selector = AdaptiveCBFParameterSelector('edr_model_9datapoint_tuned.h5', 'scaler_9datapoint_tuned.save')

    # Set gamma values
    tracking_controller.controller.cbf_param['alpha1'] = gamma1
    tracking_controller.controller.cbf_param['alpha2'] = gamma2
    
    # Set known obstacles
    tracking_controller.obs = np.array([[1 + distance, 3, 0.4]])
    tracking_controller.unknown_obs = np.array([[1 + distance, 3, 0.4]])
    tracking_controller.set_waypoints(waypoints)

    # Run simulation
    unexpected_beh = 0    

    for _ in range(int(max_sim_time / dt)):
        ret = tracking_controller.control_step()
        tracking_controller.draw_plot()
        unexpected_beh += ret
        if ret == -1:
            break
        
        # Adapt CBF parameters
        best_gamma1, best_gamma2 = adaptive_selector.adaptive_parameter_selection(tracking_controller)
        if best_gamma1 is not None and best_gamma2 is not None:
            tracking_controller.controller.cbf_param['alpha1'] = best_gamma1
            tracking_controller.controller.cbf_param['alpha2'] = best_gamma2
    
    tracking_controller.export_video()

    plt.ioff()
    plt.close()


def single_agent_simulation_traj(distance, velocity, theta, gamma1, gamma2, unknown_obs, max_sim_time=20, adapt_cbf=False):
    dt = 0.05

    waypoints = np.array([
        [1, 3, theta],
        [11, 3, 0]
    ], dtype=np.float64)

    x_init = np.append(waypoints[0], velocity)

    plot_handler = plotting.Plotting()
    ax, fig = plot_handler.plot_grid("Local Tracking Controller")
    env_handler = env.Env(width=12.5, height=6.0)
    
    # Set robot with controller 
    robot_spec = {
        'model': 'DynamicUnicycle2D',
        'w_max': 0.5,
        'a_max': 0.5,
        'fov_angle': 70.0,
        'cam_range': 4.0
    }
    control_type = 'mpc_cbf'
    tracking_controller = LocalTrackingController(x_init, robot_spec,
                                                control_type=control_type,
                                                dt=dt,
                                                show_animation=True,
                                                save_animation=False,
                                                ax=ax, fig=fig,
                                                env=env_handler)

    # Initialize AdaptiveCBFParameterSelector if adaptation is enabled
    if adapt_cbf:
        adaptive_selector = AdaptiveCBFParameterSelector('edr_model_8datapoint_single.h5', 'scaler_8datapoint_single.save')

    # Set gamma values
    tracking_controller.controller.cbf_param['alpha1'] = gamma1
    tracking_controller.controller.cbf_param['alpha2'] = gamma2
    
    # Set known obstacles
    tracking_controller.obs = unknown_obs   
    tracking_controller.unknown_obs = unknown_obs  
    tracking_controller.set_waypoints(waypoints)

    # Run simulation and collect trajectory
    trajectory = []  
    gamma_history = []
    for _ in range(int(max_sim_time / dt)):
        trajectory.append(tracking_controller.robot.X[:2, 0].flatten())  # Save current position
        ret = tracking_controller.control_step()
        tracking_controller.draw_plot()
        if ret == -1:
            break
        
        # Adapt CBF parameters if enabled
        if adapt_cbf:
            best_gamma1, best_gamma2 = adaptive_selector.adaptive_parameter_selection(tracking_controller)
            if best_gamma1 is not None and best_gamma2 is not None:
                tracking_controller.controller.cbf_param['alpha1'] = best_gamma1
                tracking_controller.controller.cbf_param['alpha2'] = best_gamma2
                gamma_history.append([best_gamma1, best_gamma2])
    
    tracking_controller.export_video()
    plt.ioff()
    plt.close()

    return np.array(trajectory), np.array(gamma_history)



def plot_traj(trajectory_1, trajectory_2, trajectory_3):
    plot_handler = plotting.Plotting()
    ax, fig = plot_handler.plot_grid("Robot Trajectories")

    import matplotlib.patches as patches
    for obs in unknown_obs:
        ox, oy, r = obs
        ax.add_patch(
            patches.Circle(
                (ox, oy), r,
                edgecolor='black',
                facecolor='gray',
                fill=True
            )
        )

    ax.plot(trajectory_1[:, 0], trajectory_1[:, 1], label="Without Adaptation (gamma1=0.05, gamma2=0.05)", linestyle="--")
    ax.plot(trajectory_2[:, 0], trajectory_2[:, 1], label="Without Adaptation (gamma1=0.2, gamma2=0.2)", linestyle="--")
    ax.plot(trajectory_3[:, 0], trajectory_3[:, 1], label="With Adaptation", linestyle="-")
    ax.scatter(1, 3, c='green', marker='o', label='Start Point')
    ax.scatter(11, 3, c='red', marker='x', label='Goal Point')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Robot Trajectories')
    ax.legend()
    ax.grid(True)

    plt.show()




if __name__ == "__main__":
    unknown_obs = np.array([[1 + 3.0, 3, 0.2], 
                            [4, 2.2, 0.2], [4, 3.0, 0.2], [4, 4.2, 0.2],
                            [6, 2.4, 0.2], [6, 3.4, 0.2], [6, 4.4, 0.2],
                            [8, 1.9, 0.2], [8, 2.9, 0.2], [8, 3.9, 0.2],
                            [10, 2.6, 0.2], [10, 3.6, 0.2], [10, 4.6, 0.2]])
    
    # single_agent_simulation(3.0, 0.5, 0.001, 0.1, 0.2)
    trajectory_1, _ = single_agent_simulation_traj(3.0, 0.5, 0.01, 0.05, 0.05, unknown_obs = unknown_obs, adapt_cbf=False)
    trajectory_2, _ = single_agent_simulation_traj(3.0, 0.5, 0.01, 0.20, 0.20, unknown_obs = unknown_obs, adapt_cbf=False)
    trajectory_3, gamma_history = single_agent_simulation_traj(3.0, 0.5, 0.01, 0.05, 0.05, unknown_obs = unknown_obs, adapt_cbf=True)
    
    plot_traj(trajectory_1, trajectory_2, trajectory_3)


