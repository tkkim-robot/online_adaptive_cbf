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
from penn.dynamics.nn_vehicle import ProbabilisticEnsembleNN
from DistributionallyRobustCVaR.distributionally_robust_cvar import DistributionallyRobustCVaR
from sklearn.preprocessing import MinMaxScaler


class AdaptiveCBFParameterSelector:
    def __init__(self, model_name, scaler_name, distance_margin=0.07, step_size=0.02, epistemic_threshold=0.2):
        self.penn = ProbabilisticEnsembleNN()
        self.penn.load_model(model_name)
        self.penn.load_scaler(scaler_name)
        self.lower_bound = 0.01
        self.upper_bound = 0.2
        self.distance_margin = distance_margin
        self.step_size = step_size
        self.epistemic_threshold = epistemic_threshold

    def sample_cbf_parameters(self, current_gamma1, current_gamma2):
        gamma1_range = np.arange(max(self.lower_bound, current_gamma1 - 0.2), min(self.upper_bound, current_gamma1 + 0.2 + self.step_size), self.step_size)
        gamma2_range = np.arange(max(self.lower_bound, current_gamma2 - 0.2), min(self.upper_bound, current_gamma2 + 0.2 + self.step_size), self.step_size)
        return gamma1_range, gamma2_range

    def get_rel_state_wt_obs(self, tracking_controller):
        robot_pos = tracking_controller.robot.X[:2, 0].flatten()
        robot_theta = tracking_controller.robot.X[2, 0]
        robot_radius = tracking_controller.robot.robot_radius
        try:
            near_obs = tracking_controller.nearest_obs.flatten()
        except:
            near_obs = [100, 100, 0.2]
        
        distance = np.linalg.norm(robot_pos - near_obs[:2]) - 0.45 + robot_radius + near_obs[2]
        velocity = tracking_controller.robot.X[3, 0]
        theta = np.arctan2(near_obs[1] - robot_pos[1], near_obs[0] - robot_pos[0]) - robot_theta
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        gamma1 = tracking_controller.pos_controller.cbf_param['alpha1']
        gamma2 = tracking_controller.pos_controller.cbf_param['alpha2']
        
        return [distance, velocity, theta, gamma1, gamma2]

    def predict_with_penn(self, current_state, gamma1_range, gamma2_range):
        batch_input = []
        for gamma1 in gamma1_range:
            for gamma2 in gamma2_range:
                state = current_state.copy()
                state[3] = gamma1
                state[4] = gamma2
                batch_input.append(state)
        
        batch_input = np.array(batch_input)
        y_pred_safety_loss, y_pred_deadlock_time, epistemic_uncertainty = self.penn.predict(batch_input)
        predictions = []

        for i, (gamma1, gamma2) in enumerate(zip(gamma1_range.repeat(len(gamma2_range)), np.tile(gamma2_range, len(gamma1_range)))):
            predictions.append((gamma1, gamma2, y_pred_safety_loss[i], y_pred_deadlock_time[i][0], epistemic_uncertainty[i]))

        return predictions

    def filter_by_epistemic_uncertainty(self, predictions):
        epistemic_uncertainties = [pred[4] for pred in predictions]
        if all(pred > 1.0 for pred in epistemic_uncertainties):
            filtered_predictions = []
        else:
            scaler = MinMaxScaler()
            normalized_epistemic_uncertainties = scaler.fit_transform(np.array(epistemic_uncertainties).reshape(-1, 1)).flatten()
            filtered_predictions = [pred for pred, norm_uncert in zip(predictions, normalized_epistemic_uncertainties) if norm_uncert <= self.epistemic_threshold]
        return filtered_predictions

    def calculate_cvar_boundary(self):
        alpha_1 = 0.4
        beta_1 = 100.0 
        min_distance = self.distance_margin
        cvar_boundary = alpha_1 / (beta_1 * min_distance**2 + 1)
        return cvar_boundary

    def filter_by_aleatoric_uncertainty(self, filtered_predictions):
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
        # If no predictions were selected, gradually decrease the parameter
        if not final_predictions:
            current_gamma1 = tracking_controller.pos_controller.cbf_param['alpha1']
            current_gamma2 = tracking_controller.pos_controller.cbf_param['alpha2']
            gamma1 = max(self.lower_bound, current_gamma1 - 0.02)
            gamma2 = max(self.lower_bound, current_gamma2 - 0.02)
            return gamma1, gamma2
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

    def adaptive_parameter_selection(self, tracking_controller):
        current_state = self.get_rel_state_wt_obs(tracking_controller)
        gamma1_range, gamma2_range = self.sample_cbf_parameters(current_state[3], current_state[4])
        predictions = self.predict_with_penn(current_state, gamma1_range, gamma2_range)
        filtered_predictions = self.filter_by_epistemic_uncertainty(predictions)
        final_predictions = self.filter_by_aleatoric_uncertainty(filtered_predictions)
        best_gamma1, best_gamma2 = self.select_best_parameters(final_predictions, tracking_controller)
        if best_gamma1 is not None and best_gamma2 is not None:
            print(f"CBF parameters updated to: {best_gamma1:.2f}, {best_gamma2:.2f} | Total prediction count: {len(predictions)} | Filtered {len(predictions)-len(filtered_predictions)} with Epistemic | Filtered {len(filtered_predictions)-len(final_predictions)} with Aleatoric DR-CVaR")        
        else:
            print(f"CBF parameters updated to: NONE, NONE | Total prediction count: {len(predictions)} | Filtered {len(predictions)-len(filtered_predictions)} with Epistemic | Filtered {len(filtered_predictions)-len(final_predictions)} with Aleatoric DR-CVaR")        
            
        return best_gamma1, best_gamma2



def single_agent_simulation_traj(velocity, waypoints, known_obs, controller_name, max_sim_time=150):
    dt = 0.05
    
    adapt_cbf = False
    if controller_name == 'MPC-CBF low fixed param':
        controller = 'mpc_cbf'
        gamma1 = 0.01
        gamma2 = 0.01
    elif controller_name == 'MPC-CBF high fixed param':
        controller = 'mpc_cbf'
        gamma1 = 0.2
        gamma2 = 0.2
    elif controller_name == 'Optimal Decay CBF-QP':
        controller = 'optimal_decay_cbf_qp'
        gamma1 = 0.5 
        gamma2 = 0.5 
    elif controller_name == 'Optimal Decay MPC-CBF':
        controller = 'optimal_decay_mpc_cbf'
        gamma1 = 0.01
        gamma2 = 0.01   
    elif controller_name == 'Online Adaptive CBF':
        adapt_cbf = True
        controller = 'mpc_cbf'
        gamma1 = 0.01
        gamma2 = 0.01
        
    plot_handler = plotting.Plotting(width=11.0, height=3.8, known_obs=known_obs)
    x_init = np.append(waypoints[0], velocity)

    ax, fig = plot_handler.plot_grid(f"{controller_name} controller")
    env_handler = env.Env()
    
    # Set robot with controller 
    robot_spec = {
        'model': 'DynamicUnicycle2D',
        'w_max': 0.5,
        'a_max': 0.5,
        'fov_angle': 70.0,
        'cam_range': 3.0
    }
    tracking_controller = LocalTrackingController(x_init, robot_spec,
                                                control_type=controller,
                                                dt=dt,
                                                show_animation=True,
                                                save_animation=False,
                                                ax=ax, fig=fig,
                                                env=env_handler)

    tracking_controller.robot.robot_radius = 0.3

    # Initialize AdaptiveCBFParameterSelector if adaptation is enabled
    if adapt_cbf:
        adaptive_selector = AdaptiveCBFParameterSelector('checkpoint/penn_model_0907.pth', 'checkpoint/scaler_0907.save')

    # Set gamma values
    tracking_controller.pos_controller.cbf_param['alpha1'] = gamma1
    tracking_controller.pos_controller.cbf_param['alpha2'] = gamma2
    
    # Set known obstacles
    tracking_controller.obs = known_obs   
    tracking_controller.set_waypoints(waypoints)

    # Run simulation and collect trajectory
    for _ in range(int(max_sim_time / dt)):
        ret = tracking_controller.control_step()
        tracking_controller.draw_plot()
        if ret == -1 and np.linalg.norm(tracking_controller.robot.X[:2, 0].flatten() - waypoints[1][:2]) < tracking_controller.reached_threshold:
            print("Goal point reached")
            break
        elif ret == -1:
            print("Collided")
            break
        
        # Adapt CBF parameters if enabled
        if adapt_cbf:
            best_gamma1, best_gamma2 = adaptive_selector.adaptive_parameter_selection(tracking_controller)
            if best_gamma1 is not None and best_gamma2 is not None:
                tracking_controller.pos_controller.cbf_param['alpha1'] = best_gamma1
                tracking_controller.pos_controller.cbf_param['alpha2'] = best_gamma2
    
    tracking_controller.export_video()
    plt.ioff()
    plt.close()



if __name__ == "__main__":
    controller_list = ['MPC-CBF low fixed param', 'MPC-CBF high fixed param', 'Optimal Decay CBF-QP', 'Optimal Decay MPC-CBF', 'Online Adaptive CBF']
    controller_name = controller_list[4]
    
    
    waypoints = np.array([
        [0.75, 2.0, 0.01],
        [10, 1.5, 0]
    ], dtype=np.float64)   
    init_vel = 0.4
    known_obs = np.array([[4.0, 0.3, 0.3], [3.5, 0.5, 0.4], [3.5, 2.4, 0.5],
                            [6.5, 2.6, 1.05], [8.5, 0.4, 0.2],
                            [8, 0.6, 0.35], [7.5, 2.3, 0.45],])
    
    single_agent_simulation_traj(init_vel, waypoints, known_obs = known_obs, controller_name = controller_name)
