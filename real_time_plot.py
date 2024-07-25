import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'cbf_tracking'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from cbf_tracking.utils import plotting, env
from cbf_tracking.tracking import LocalTrackingController
from evidential_deep_regression import EvidentialDeepRegression


def get_rel_state_wt_obs(tracking_controller):
    robot_pos = tracking_controller.robot.X[:2,0].flatten()
    near_obs_pos = tracking_controller.nearest_obs[:2].flatten()
    
    distance = np.linalg.norm(robot_pos - near_obs_pos)
    velocity = tracking_controller.robot.X[3,0]
    theta = np.arctan2(near_obs_pos[1] - robot_pos[1], near_obs_pos[0] - robot_pos[0])
    gamma1 = tracking_controller.controller.cbf_param['alpha1']
    gamma2 = tracking_controller.controller.cbf_param['alpha2']
    
    return [distance, velocity, theta, gamma1, gamma2]

def update_real_time_gmm(gmms, lines):
    plt.ion()  
    x = np.linspace(-1, 7, 1000).reshape(-1, 1)
    for gmm, line_set in zip(gmms, lines):
        logprob = gmm.score_samples(x)
        responsibilities = gmm.predict_proba(x)
        pdf = np.exp(logprob)
        pdf_individual = responsibilities * pdf[:, np.newaxis]

        line_set[0].set_ydata(pdf)
        for i, line in enumerate(line_set[1]):
            line.set_ydata(pdf_individual[:, i])

    plt.draw()

def setup_gmm_plots(fig, gs):
    plt.ion()  # Enable interactive mode for faster plotting
    right_gs = gs[0, 1].subgridspec(3, 1, hspace=0.3)
    gmm_ax1 = fig.add_subplot(right_gs[0, 0])
    gmm_ax2 = fig.add_subplot(right_gs[1, 0], sharex=gmm_ax1, sharey=gmm_ax1)
    gmm_ax3 = fig.add_subplot(right_gs[2, 0], sharex=gmm_ax1, sharey=gmm_ax1)
    ax_gmm = [gmm_ax1, gmm_ax2, gmm_ax3]

    gamma_pairs = [(0.1, 0.1), (0.5, 0.5), (0.99, 0.99)]
    x = np.linspace(-1, 7, 1000).reshape(-1, 1)
    lines = []

    for ax, gamma_pair in zip(ax_gmm, gamma_pairs):
        main_line, = ax.plot(x, np.zeros_like(x), '-k', label='GMM')
        comp_lines = [ax.plot(x, np.zeros_like(x), '--', label=f'GMM Component {i+1}')[0] for i in range(3)]
        lines.append([main_line, comp_lines])
        ax.set_xlim([-1, 7])
        ax.set_ylim([0, 1])
        ax.set_title(f'Gamma Pair: {gamma_pair}', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)

    for ax in ax_gmm:
        ax.xaxis.set_major_locator(FixedLocator(np.arange(0, 7, 1)))
        ax.yaxis.set_major_locator(FixedLocator(np.linspace(0, 1, 5)))

    # Set xlabel and ylabel
    ax_gmm[-1].set_xlabel('Safety Loss GMM Distribution Prediction', fontsize=12)
    ax_gmm[len(ax_gmm) // 2].set_ylabel('Density', fontsize=12)

    # Add a legend to the first subplot only
    ax_gmm[0].legend(loc='upper right', fontsize=10)

    return ax_gmm, lines, gamma_pairs


def single_agent_simulation(distance, velocity, theta, gamma1, gamma2, deadlock_threshold=0.1, max_sim_time=20):
    dt = 0.05

    waypoints = np.array([
        [1, 3, theta],
        [11, 3, 0]
    ], dtype=np.float64)

    x_init = np.append(waypoints[0], velocity)

    # Set plot with env
    plot_handler = plotting.Plotting()
    (ax_main, right_ax, gs), fig = plot_handler.plot_grid("Local Tracking Controller", with_right_subplot=True)
    env_handler = env.Env()
    
    # Set GMM plots
    ax_gmm, lines, gamma_pairs = setup_gmm_plots(fig, gs)
    
    # Set EDR model
    model_name = 'edr_model_0725.h5'
    scaler_name = 'scaler.save'    
    edr = EvidentialDeepRegression()
    edr.load_saved_model(model_name)
    edr.load_saved_scaler(scaler_name)

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
                                                ax=ax_main, fig=fig,
                                                env=env_handler)

    # Set gamma values
    tracking_controller.controller.cbf_param['alpha1'] = gamma1
    tracking_controller.controller.cbf_param['alpha2'] = gamma2
    
    # Set known obstacles
    tracking_controller.obs = np.array([[1 + distance, 3, 0.1]])
    tracking_controller.unknown_obs = np.array([[1 + distance, 3, 0.1]])
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
        current_state_wt_obs = get_rel_state_wt_obs(tracking_controller)
        batch_input = []
        for g_pair in gamma_pairs:
            state = current_state_wt_obs.copy()
            state[3] = g_pair[0]
            state[4] = g_pair[1]
            batch_input.append(state)

        batch_input = np.array(batch_input)
        y_pred_safety_loss, y_pred_deadlock_loss = edr.predict(batch_input)

        # Create GMM for safety loss predictions
        gmms = [edr.create_gmm(y_pred_safety_loss[i]) for i in range(len(gamma_pairs))]
        update_real_time_gmm(gmms, lines)
    
    
    plt.ioff()
    plt.close()



if __name__ == "__main__":
    single_agent_simulation(1.5, 0.5, 0.001, 0.1, 0.2)