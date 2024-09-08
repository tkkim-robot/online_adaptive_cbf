import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'cbf_tracking'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from cbf_tracking.utils import plotting, env
from cbf_tracking.tracking import LocalTrackingController
from probabilistic_ensemble_nn.dynamics.nn_vehicle import ProbabilisticEnsembleNN

class RealTimePlotter:
    def __init__(self, model_name, scaler_name, plot_deadlock=False):
        self.model_name = model_name
        self.scaler_name = scaler_name
        self.plot_deadlock = plot_deadlock
        self.penn = ProbabilisticEnsembleNN()
        self.penn.load_model(model_name)
        self.penn.load_scaler(scaler_name)
        self.gamma_pairs = [(0.05, 0.05), (0.1, 0.1), (0.2, 0.2)]
        self.fig = None
        self.gs = None
        self.ax_gmm_safety = None
        self.ax_gmm_deadlock = None
        self.lines_safety = None
        self.lines_deadlock = None

    def get_rel_state_wt_obs(self, tracking_controller):
        robot_pos = tracking_controller.robot.X[:2, 0].flatten()
        robot_rad = tracking_controller.robot.robot_radius
        near_obs_pos = tracking_controller.nearest_obs[:2].flatten()
        near_obs_rad = tracking_controller.nearest_obs[2].flatten()
        
        distance = np.linalg.norm(robot_pos - near_obs_pos) - robot_rad - near_obs_rad[0]
        velocity = tracking_controller.robot.X[3, 0]
        theta = np.arctan2(near_obs_pos[1] - robot_pos[1], near_obs_pos[0] - robot_pos[0])
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        gamma1 = tracking_controller.controller.cbf_param['alpha1']
        gamma2 = tracking_controller.controller.cbf_param['alpha2']
        
        return [distance, velocity, theta, gamma1, gamma2]

    def update_real_time_gmm(self, gmms_safety, gmms_deadlock=None):
        plt.ion()  
        x = np.linspace(0, 1.2, 300).reshape(-1, 1)
        for gmm, line_set in zip(gmms_safety, self.lines_safety):
            logprob = gmm.score_samples(x)
            responsibilities = gmm.predict_proba(x)
            pdf = np.exp(logprob)
            pdf_individual = responsibilities * pdf[:, np.newaxis]

            line_set[0].set_ydata(pdf)
            for i, line in enumerate(line_set[1]):
                line.set_ydata(pdf_individual[:, i])

        if self.plot_deadlock:
            for gmm, line_set in zip(gmms_deadlock, self.lines_deadlock):
                logprob = gmm.score_samples(x)
                responsibilities = gmm.predict_proba(x)
                pdf = np.exp(logprob)
                pdf_individual = responsibilities * pdf[:, np.newaxis]

                line_set[0].set_ydata(pdf)
                for i, line in enumerate(line_set[1]):
                    line.set_ydata(pdf_individual[:, i])

        plt.draw()

    def setup_gmm_plots(self):
        plt.ion()  # Enable interactive mode for faster plotting
        right_gs = self.gs[0, 1].subgridspec(3, 2, hspace=0.3)
        
        gmm_ax1_safety = self.fig.add_subplot(right_gs[0, 0])
        gmm_ax2_safety = self.fig.add_subplot(right_gs[1, 0], sharex=gmm_ax1_safety, sharey=gmm_ax1_safety)
        gmm_ax3_safety = self.fig.add_subplot(right_gs[2, 0], sharex=gmm_ax1_safety, sharey=gmm_ax1_safety)
        
        self.ax_gmm_safety = [gmm_ax1_safety, gmm_ax2_safety, gmm_ax3_safety]
        
        if self.plot_deadlock:
            gmm_ax1_deadlock = self.fig.add_subplot(right_gs[0, 1])
            gmm_ax2_deadlock = self.fig.add_subplot(right_gs[1, 1], sharex=gmm_ax1_deadlock, sharey=gmm_ax1_deadlock)
            gmm_ax3_deadlock = self.fig.add_subplot(right_gs[2, 1], sharex=gmm_ax1_deadlock, sharey=gmm_ax1_deadlock)
            
            self.ax_gmm_deadlock = [gmm_ax1_deadlock, gmm_ax2_deadlock, gmm_ax3_deadlock]

        x = np.linspace(0, 1.2, 300).reshape(-1, 1)
        self.lines_safety = []
        self.lines_deadlock = []

        for ax, gamma_pair in zip(self.ax_gmm_safety, self.gamma_pairs):
            main_line, = ax.plot(x, np.zeros_like(x), '-k', label='GMM')
            comp_lines = [ax.plot(x, np.zeros_like(x), '--', label=f'GMM Component {i+1}', linewidth=2)[0] for i in range(3)]
            self.lines_safety.append([main_line, comp_lines])
            ax.set_xlim([0, 1.2])
            ax.set_ylim([0, 5])
            ax.set_title(f'Safety Loss - Gamma Pair: {gamma_pair}', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=10)

            ax.xaxis.set_major_locator(FixedLocator(np.linspace(0, 1.2, 7)))
            ax.yaxis.set_major_locator(FixedLocator(np.linspace(0, 5, 5)))
            
        if self.plot_deadlock:
            for ax, gamma_pair in zip(self.ax_gmm_deadlock, self.gamma_pairs):
                main_line, = ax.plot(x, np.zeros_like(x), '-k', label='GMM')
                comp_lines = [ax.plot(x, np.zeros_like(x), '--', label=f'GMM Component {i+1}')[0] for i in range(3)]
                self.lines_deadlock.append([main_line, comp_lines])
                ax.set_xlim([-0.05, 0.30])
                ax.set_ylim([0, 10])
                ax.set_title(f'Deadlock Loss - Gamma Pair: {gamma_pair}', fontsize=10)
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                ax.xaxis.set_major_locator(FixedLocator(np.arange(0, 0.3, 0.05)))  # Custom X-axis ticks
                ax.yaxis.set_major_locator(FixedLocator(np.linspace(0, 10, 5)))  # Custom Y-axis ticks

        # for ax in self.ax_gmm_safety + (self.ax_gmm_deadlock if self.plot_deadlock else []):
    
        # Set xlabel and ylabel
        self.ax_gmm_safety[-1].set_xlabel('Safety Loss GMM Distribution Prediction', fontsize=12)
        
        if self.plot_deadlock:
            self.ax_gmm_deadlock[-1].set_xlabel('Deadlock Loss GMM Distribution Prediction', fontsize=12)
        
        self.ax_gmm_safety[len(self.ax_gmm_safety) // 2].set_ylabel('Density', fontsize=12)

        # Add a legend to the first subplot only
        self.ax_gmm_safety[0].legend(loc='upper right', fontsize=10)

    def predict_and_update_gmm(self, tracking_controller):
        try:
            current_state_wt_obs = self.get_rel_state_wt_obs(tracking_controller)
            batch_input = []
            for g_pair in self.gamma_pairs:
                state = current_state_wt_obs.copy()
                state[3] = g_pair[0]
                state[4] = g_pair[1]
                batch_input.append(state)

            batch_input = np.array(batch_input)
            y_pred_safety_loss, y_pred_deadlock_loss, _ = self.penn.predict(batch_input)

            # Create GMM for safety loss predictions
            gmms_safety = [self.penn.create_gmm(y_pred_safety_loss[i]) for i in range(len(self.gamma_pairs))]
            
            if self.plot_deadlock:
                gmms_deadlock = [self.penn.create_gmm(y_pred_deadlock_loss[i]) for i in range(len(self.gamma_pairs))]
                self.update_real_time_gmm(gmms_safety, gmms_deadlock)
            else:
                self.update_real_time_gmm(gmms_safety)
        except:
            pass

    def initialize_plots(self, plot_handler, env_handler):
        (ax_main, right_ax, self.gs), self.fig = plot_handler.plot_grid("Local Tracking Controller", with_right_subplot=True)
        self.setup_gmm_plots()
        return ax_main, env_handler


def single_agent_simulation(distance, velocity, theta, gamma1, gamma2, max_sim_time=20, plot_deadlock=False):
    dt = 0.05

    waypoints = np.array([
        [1, 3, theta],
        [11, 3, 0]
    ], dtype=np.float64)

    x_init = np.append(waypoints[0], velocity)

    known_obs = np.array([[1 + distance, 3, 0.2], [7, 3.5, 0.2]])

    # Set plot with env
    plot_handler = plotting.Plotting(width=12.5, height=6, known_obs=known_obs)
    env_handler = env.Env()
    
    real_time_plotter = RealTimePlotter('penn_model_0907.pth', 'scaler_0907.save', plot_deadlock)
    ax_main, env_handler = real_time_plotter.initialize_plots(plot_handler, env_handler)
    
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
                                                save_animation=True,
                                                ax=ax_main, fig=real_time_plotter.fig,
                                                env=env_handler)

    # Set gamma values
    tracking_controller.controller.cbf_param['alpha1'] = gamma1
    tracking_controller.controller.cbf_param['alpha2'] = gamma2
    
    # Set known obstacles
    tracking_controller.obs = known_obs
    tracking_controller.set_waypoints(waypoints)

    # Run simulation
    unexpected_beh = 0    

    for _ in range(int(max_sim_time / dt)):
        ret = tracking_controller.control_step()
        tracking_controller.draw_plot()
        unexpected_beh += ret
        if ret == -1:
            break
        
        # Predict safety loss distribution with current robot and obs state
        real_time_plotter.predict_and_update_gmm(tracking_controller)
    
    tracking_controller.export_video()

    plt.ioff()
    plt.close()


if __name__ == "__main__":
    single_agent_simulation(3.0, 0.5, 0.001, 0.1, 0.1, plot_deadlock=False)
