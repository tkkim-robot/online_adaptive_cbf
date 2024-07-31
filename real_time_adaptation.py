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
    def __init__(self, edr_model, edr_scaler, lower_bound=0.05, upper_bound=1.0, step_size=0.05, epistemic_threshold=0.5, cvar_boundary=1.0):
        self.edr = EvidentialDeepRegression()
        self.edr.load_saved_model(edr_model)
        self.edr.load_saved_scaler(edr_scaler)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.step_size = step_size
        self.epistemic_threshold = epistemic_threshold
        self.cvar_boundary = cvar_boundary

    def sample_cbf_parameters(self, current_gamma1, current_gamma2):
        gamma1_range = np.arange(max(self.lower_bound, current_gamma1 - 0.2), min(self.upper_bound, current_gamma1 + 0.1) + self.step_size, self.step_size)
        gamma2_range = np.arange(max(self.lower_bound, current_gamma2 - 0.2), min(self.upper_bound, current_gamma2 + 0.1) + self.step_size, self.step_size)
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

    def predict_with_edr(self, current_state, gamma1_range, gamma2_range):
        batch_input = []
        for gamma1 in gamma1_range:
            for gamma2 in gamma2_range:
                state = current_state.copy()
                state[3] = gamma1
                state[4] = gamma2
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
        best_predictions = [pred for pred in filtered_predictions if pred[3] == min_deadlock_time]
        if len(best_predictions) != 1:
            best_prediction = max(best_predictions, key=lambda x: 2 * (x[4] * x[5]) / (x[4] + x[5]) if (x[4] + x[5]) != 0 else 0)
            return best_prediction[0], best_prediction[1]
        return best_predictions[0][0], best_predictions[0][1]

    def adaptive_parameter_selection(self, tracking_controller):
        current_state = self.get_rel_state_wt_obs(tracking_controller)
        gamma1_range, gamma2_range = self.sample_cbf_parameters(current_state[3], current_state[4])
        predictions = self.predict_with_edr(current_state, gamma1_range, gamma2_range)
        filtered_predictions = self.filter_by_epistemic_uncertainty(predictions)
        final_predictions = self.filter_by_aleatoric_uncertainty(filtered_predictions)
        best_gamma1, best_gamma2 = self.select_best_parameters(final_predictions)
        return best_gamma1, best_gamma2



def single_agent_simulation(distance, velocity, theta, gamma1, gamma2, max_sim_time=20, plot_deadlock=False):
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
            print("CBF parameters updated to: alpha1 = {}, alpha2 = {}".format(best_gamma1, best_gamma2))
            tracking_controller.controller.cbf_param['alpha1'] = best_gamma1
            tracking_controller.controller.cbf_param['alpha2'] = best_gamma2
    
    tracking_controller.export_video()

    plt.ioff()
    plt.close()

if __name__ == "__main__":
    single_agent_simulation(3.0, 0.5, 0.001, 0.1, 0.2, plot_deadlock=False)
