import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'safe_control'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from safe_control.utils import plotting, env
from safe_control.tracking import LocalTrackingController
from penn.dynamics.nn_vehicle import ProbabilisticEnsembleNN

class RealTimePlotter:
    def __init__(self, model_name, scaler_name, plot_deadlock=False):
        '''
        Initialize the real-time predicted GMM plotter
        '''
        self.model_name = model_name
        self.scaler_name = scaler_name
        self.plot_deadlock = plot_deadlock
        self.penn = ProbabilisticEnsembleNN()
        self.penn.load_model(model_name)
        self.penn.load_scaler(scaler_name)
        self.gamma_pairs = [(0.03, 0.03), (0.1, 0.1), (0.18, 0.18)] # Gamma pairs for safety loss function evaluation
        self.fig = None
        self.gs = None
        self.ax_gmm_safety = None
        self.ax_gmm_deadlock = None
        self.lines_safety = None
        self.lines_deadlock = None

    def get_rel_state_wt_obs(self, tracking_controller):
        '''
        Get the relative state between the robot and the nearest obstacle
        '''
        robot_pos = tracking_controller.robot.X[:2, 0].flatten()
        robot_theta = tracking_controller.robot.X[2,0]
        robot_rad = tracking_controller.robot.robot_radius
        near_obs_pos = tracking_controller.nearest_obs[:2].flatten()
        near_obs_rad = tracking_controller.nearest_obs[2].flatten()

        # Calculate distance, velocity, and change in angle
        distance = np.linalg.norm(robot_pos - near_obs_pos) - robot_rad - near_obs_rad[0]
        velocity = tracking_controller.robot.X[3, 0]
        delta_theta = np.arctan2(near_obs_pos[1] - robot_pos[1], near_obs_pos[0] - robot_pos[0]) - robot_theta
        delta_theta = ((delta_theta + np.pi) % (2 * np.pi)) - np.pi
        gamma0 = tracking_controller.pos_controller.cbf_param['alpha1']
        gamma1 = tracking_controller.pos_controller.cbf_param['alpha2']
        
        return [distance, velocity, delta_theta, gamma0, gamma1]

    def update_real_time_gmm(self, gmms_safety, gmms_deadlock=None):
        '''
        Update the real-time plots of Gaussian Mixture Models (GMMs) for safety
        '''
        plt.ion()  # Enable interactive mode
        x = np.linspace(0, 1.2, 300).reshape(-1, 1)
        # Update safety GMM plots
        for gmm, line_set in zip(gmms_safety, self.lines_safety):
            logprob = gmm.score_samples(x)
            responsibilities = gmm.predict_proba(x)
            pdf = np.exp(logprob)
            pdf_individual = responsibilities * pdf[:, np.newaxis]

            # Update lines in the plot
            line_set[0].set_ydata(pdf)
            for i, line in enumerate(line_set[1]):
                line.set_ydata(pdf_individual[:, i])

        # Update deadlock GMM plots if enabled
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
        '''
        Set up the plot for GMMs for safety and optionally for deadlock
        '''
        plt.ion()  # Enable interactive mode for faster plotting
        right_gs = self.gs[0, 1].subgridspec(3, 2, hspace=0.3)
        
        # Create subplots for safety GMMs
        gmm_ax1_safety = self.fig.add_subplot(right_gs[0, 0])
        gmm_ax2_safety = self.fig.add_subplot(right_gs[1, 0], sharex=gmm_ax1_safety, sharey=gmm_ax1_safety)
        gmm_ax3_safety = self.fig.add_subplot(right_gs[2, 0], sharex=gmm_ax1_safety, sharey=gmm_ax1_safety)
        
        self.ax_gmm_safety = [gmm_ax1_safety, gmm_ax2_safety, gmm_ax3_safety]
        
        # Create subplots for deadlock GMMs if enabled
        if self.plot_deadlock:
            gmm_ax1_deadlock = self.fig.add_subplot(right_gs[0, 1])
            gmm_ax2_deadlock = self.fig.add_subplot(right_gs[1, 1], sharex=gmm_ax1_deadlock, sharey=gmm_ax1_deadlock)
            gmm_ax3_deadlock = self.fig.add_subplot(right_gs[2, 1], sharex=gmm_ax1_deadlock, sharey=gmm_ax1_deadlock)
            
            self.ax_gmm_deadlock = [gmm_ax1_deadlock, gmm_ax2_deadlock, gmm_ax3_deadlock]

        # Set up the plot lines for GMMs
        x = np.linspace(0, 1.2, 300).reshape(-1, 1)
        self.lines_safety = []
        self.lines_deadlock = []

        # Initialize safety GMM plots with placeholder lines
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
        
        # Initialize deadlock GMM plots if enabled
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

        self.ax_gmm_safety[-1].set_xlabel('Safety Loss GMM Distribution Prediction', fontsize=12)
        
        if self.plot_deadlock:
            self.ax_gmm_deadlock[-1].set_xlabel('Deadlock Loss GMM Distribution Prediction', fontsize=12)
        
        self.ax_gmm_safety[len(self.ax_gmm_safety) // 2].set_ylabel('Density', fontsize=12)

        self.ax_gmm_safety[0].legend(loc='upper right', fontsize=10)

    def predict_and_update_gmm(self, tracking_controller):
        '''
        Predict safety loss distributions and update GMM plots in real-time
        '''
        try:
            # Get the current state with respect to the obstacle
            current_state_wt_obs = self.get_rel_state_wt_obs(tracking_controller)
            batch_input = []
            for g_pair in self.gamma_pairs:
                state = current_state_wt_obs.copy()
                state[3] = g_pair[0]
                state[4] = g_pair[1]
                batch_input.append(state)

            batch_input = np.array(batch_input)
            y_pred_safety_loss, y_pred_deadlock_loss, _ = self.penn.predict(batch_input)

            # Create GMMs for safety and deadlock loss predictions
            gmms_safety = [self.penn.create_gmm(y_pred_safety_loss[i]) for i in range(len(self.gamma_pairs))]
            
            if self.plot_deadlock:
                gmms_deadlock = [self.penn.create_gmm(y_pred_deadlock_loss[i]) for i in range(len(self.gamma_pairs))]
                self.update_real_time_gmm(gmms_safety, gmms_deadlock)
            else:
                self.update_real_time_gmm(gmms_safety)
        except:
            pass

    def initialize_plots(self, plot_handler, env_handler):
        '''
        Initialize the plots
        '''
        (ax_main, right_ax, self.gs), self.fig = plot_handler.plot_grid("Local Tracking Controller", with_right_subplot=True)
        self.setup_gmm_plots()
        return ax_main, env_handler

def test_plot_example(max_sim_time=20):
    '''
    Example function to visualize the predicted safety loss GMMs in real-time
    '''
    dt = 0.05

    # Define waypoints for the robot to follow
    waypoints = np.array([
        [1, 1, np.pi/2],
        [2.5, 6, 0]
    ], dtype=np.float64)
    x_init = np.append(waypoints[0], 0.5)

    # Define known obstacles in the environment
    known_obs = np.array([[1.2, 3.5, 0.2]])

    # Initialize plot and environment handlers
    plot_handler = plotting.Plotting(width=3, height=6.5, known_obs=known_obs)
    env_handler = env.Env()
    
    # Initialize the real-time plotter and tracking controller
    real_time_plotter = RealTimePlotter('checkpoint/penn_model_0907.pth', 'checkpoint/scaler_0907.save')
    ax_main, env_handler = real_time_plotter.initialize_plots(plot_handler, env_handler)
    
    # Set up the robot's specifications
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
                                                show_animation=True,
                                                save_animation=False,
                                                ax=ax_main, fig=real_time_plotter.fig,
                                                env=env_handler)

    # Set gamma values for the CBF
    tracking_controller.pos_controller.cbf_param['alpha1'] = 0.1
    tracking_controller.pos_controller.cbf_param['alpha2'] = 0.1
    
    # Set known obstacles
    tracking_controller.obs = known_obs
    tracking_controller.set_waypoints(waypoints)

    # Run the simulation loop
    unexpected_beh = 0    
    for _ in range(int(max_sim_time / dt)):
        ret = tracking_controller.control_step()
        tracking_controller.draw_plot()
        unexpected_beh += ret
        if ret == -1:
            break
        
        # Predict and update the safety loss distribution in real-time
        real_time_plotter.predict_and_update_gmm(tracking_controller)
    
    tracking_controller.export_video()

    plt.ioff()
    plt.close()


if __name__ == "__main__":
    test_plot_example()
