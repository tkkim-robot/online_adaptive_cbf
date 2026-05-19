import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
# Make sure the repository root is on sys.path (so `import safe_control` resolves locally),
# and add cvar_gmm_filter. Avoid adding the `safe_control/` directory itself because that
# can cause confusing resolution in multiprocessing contexts.
if project_root not in sys.path:
    sys.path.insert(0, project_root)
_cvar_path = os.path.join(project_root, 'cvar_gmm_filter')
if _cvar_path not in sys.path:
    sys.path.insert(0, _cvar_path)

import csv
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from safe_control.utils import plotting, env
from safe_control.tracking import LocalTrackingController
from safe_control.dynamic_env.main import LocalTrackingControllerDyn
from nn_model.penn.nn_iccbf_predict import ProbabilisticEnsembleNN
from cvar_gmm_filter.distributionally_robust_cvar import DistributionallyRobustCVaR
from online_cbf_config import ALL_DEFAULTS, ADAPTIVE_MODELS

# torch_geometric is only required for GAT-based models. Make it optional so
# non-GAT experiments (including BarrierNet rollouts) can run without it.
try:
    import torch_geometric  # noqa: F401
    from torch_geometric.data import Batch  # noqa: F401
    _TORCH_GEOMETRIC_AVAILABLE = True
except Exception:
    Batch = None
    _TORCH_GEOMETRIC_AVAILABLE = False

class OnlineCBFAdapter:
    def __init__(self, model_name, scaler_name=None, d_min=0.075, step_size=0.05,
                 epistemic_threshold=0.2, lower_bound=0.01, upper_bound=1.0,
                 robot_model=None, use_gat=False, print_info=True):
        """
        Initialize the adaptive CBF parameter selector
        """
        self.print_info = print_info
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        print(f"Using device {self.device} for OnlineCBFAdapter")
        self.robot_model = robot_model
        if self.robot_model == 'Quad2D': #TODO: make state dic
            self.extra_state = 1
            self.gamma_dim = 2
            self.theta_index = 3  # [distance, velocity_x, velocity_z, delta_theta, gamma0, gamma1]
            self.gamma0_index = 4
            self.gamma1_index = 5
        elif self.robot_model == 'Quad3D':
            self.extra_state = 0
            self.gamma_dim = 1
            self.theta_index = 3  # [distance, velocity_x, velocity_z, delta_theta, gamma0]
            self.gamma0_index = 4
            self.gamma1_index = None
        elif self.robot_model == 'KinematicBicycle2D_DPCBF':
            self.extra_state = -1
            self.gamma_dim = 1
            self.theta_index = 2  # [distance, velocity, delta_theta, gamma0]
            self.gamma0_index = 3
            self.gamma1_index = None
        else:
            self.extra_state = 0
            self.gamma_dim = 2
            self.theta_index = 2  # [distance, velocity, delta_theta, gamma0, gamma1]
            self.gamma0_index = 3
            self.gamma1_index = 4

        self.use_gat = use_gat
        self.n_states = 18 if self.use_gat else 6 + self.extra_state

        self.gat_module = None
        if self.use_gat:
            if not _TORCH_GEOMETRIC_AVAILABLE:
                raise ModuleNotFoundError(
                    "torch_geometric is required for GAT-based online adaptation "
                    "(e.g., 'Online Adaptive ... GAT'). Install torch_geometric or use the MLP controller."
                )
            # Local imports to avoid hard dependency when not using GAT
            from nn_model.penn.gat import GATModule
            from nn_model.penn.nn_gat_iccbf_predict import ProbabilisticEnsembleGAT

            self.gat_module = GATModule().to(self.device)
            self.penn = ProbabilisticEnsembleGAT(self.gat_module, device=self.device, gamma_dim=self.gamma_dim)
        else:
            self.penn = ProbabilisticEnsembleNN(n_states=self.n_states, device=self.device, theta_index=self.theta_index)
            if scaler_name:
                self.penn.load_scaler(scaler_name)

        self.penn.load_model(model_name)
        self.lower_bound = lower_bound  # Lower bound for CBF parameter sampling, Conservative
        self.upper_bound = upper_bound  # Upper bound for CBF parameter sampling, Aggressive
        self.d_min = d_min  # Closest allowable distance to obstacles
        self.step_size = step_size  # Step size for sampling caldidate CBF parameters
        self.epistemic_threshold = epistemic_threshold  # Threshold for filtering predictions based on epistemic uncertainty

    def sample_cbf_parameters(self, current_gamma0, current_gamma1=None):
        '''
        Sample CBF parameters (gamma0 and gamma1) within a specified range
        '''
        # gamma0_range = np.arange(max(self.lower_bound, current_gamma0 - 2.5), min(self.upper_bound, current_gamma0 + 2.5 + self.step_size), self.step_size)
        # if self.gamma_dim == 2:
        #     gamma1_range = np.arange(max(self.lower_bound, current_gamma1 - 2.5), min(self.upper_bound, current_gamma1 + 2.5 + self.step_size), self.step_size)
        #     return gamma0_range, gamma1_range
        # else:
        #     return gamma0_range, None
        eps = 1e-9                       # make upper bound inclusive
        gamma0_range = np.arange(self.lower_bound,
                                self.upper_bound + eps,
                                self.step_size)

        if self.gamma_dim == 2:
            gamma1_range = np.arange(self.lower_bound,
                                    self.upper_bound + eps,
                                    self.step_size)
            return gamma0_range, gamma1_range
        else:
            return gamma0_range, None

    def get_rel_state_wt_obs(self, tracking_controller):
        """
        Get the relative state of the robot with respect to the nearest obstacle
        """
        robot_pos = tracking_controller.robot.X[:2, 0].flatten()
        robot_theta = tracking_controller.robot.X[2, 0]
        robot_radius = tracking_controller.robot.robot_radius
        try:
            near_obs = tracking_controller.nearest_obs.flatten()
        except:
            near_obs = [100, 100, 0.2]  # Default obstacle in case of no nearby obstacle
        
        # Calculate distance, velocity, and relative angle with the obstacle
        if self.robot_model == 'VTOL2D':
            distance = np.linalg.norm(robot_pos - near_obs[:2]) - robot_radius - near_obs[2]
            #distance = np.linalg.norm(robot_pos - near_obs[:2]) - 0.45 + robot_radius + near_obs[2] # correct the distance for mistake in the training data
            # abuse variable name: it's just pitch angle in this scenario (since all obs are fixed)
            delta_theta = robot_theta 
        else:
            distance = np.linalg.norm(robot_pos - near_obs[:2]) - robot_radius - near_obs[2]
            #distance = np.linalg.norm(robot_pos - near_obs[:2]) - 0.45 + robot_radius + near_obs[2]
            delta_theta = np.arctan2(near_obs[1] - robot_pos[1], near_obs[0] - robot_pos[0]) - robot_theta
            delta_theta = ((delta_theta + np.pi) % (2 * np.pi)) - np.pi  
        
        
        if self.gamma_dim == 1:
            gamma0 = tracking_controller.pos_controller.cbf_param['alpha']
            gamma1 = None
        else:
            gamma0 = tracking_controller.pos_controller.cbf_param['alpha1']
            gamma1 = tracking_controller.pos_controller.cbf_param['alpha2']

        # gamma0 = tracking_controller.pos_controller.cbf_param['alpha1']
        # if self.gamma_dim == 2:
        #     gamma1 = tracking_controller.pos_controller.cbf_param['alpha2']

        if self.robot_model == 'Quad2D': # If Quad2D => velocity_x, velocity_z
            velocity_x = tracking_controller.robot.X[3, 0]
            velocity_z = tracking_controller.robot.X[4, 0]
            return [distance, velocity_x, velocity_z, delta_theta, gamma0, gamma1]
        elif self.robot_model == 'Quad3D': # If Quad3D => velocity_x, velocity_z
            velocity_x = tracking_controller.robot.X[6, 0]
            velocity_z = tracking_controller.robot.X[8, 0]
            return [distance, velocity_x, velocity_z, delta_theta, gamma0]
        elif self.robot_model == 'KinematicBicycle2D_DPCBF':
            velocity = tracking_controller.robot.X[3, 0]
            return [distance, velocity, delta_theta, gamma0]
        else:
            # for vtol, also put x_vel only in this particular scenario (same setting for training)            
            # 2D ground => velocity is single scalar
            velocity = tracking_controller.robot.X[3, 0]
            return [distance, velocity, delta_theta, gamma0, gamma1]

    def predict_with_penn(self, current_state, gamma0_range, gamma1_range):
        """
        Predict safety loss, deadlock time, and epistemic uncertainty using PENN (MLP-based)
        """
        if self.gamma_dim == 2:
            g0_grid, g1_grid = np.meshgrid(gamma0_range, gamma1_range, indexing='ij')
            gamma_flat = np.stack([g0_grid.flatten(), g1_grid.flatten()], axis=1)
        else:
            gamma_flat = gamma0_range.reshape(-1, 1)

        num_samples = gamma_flat.shape[0]
        state_repeated = np.tile(current_state, (num_samples, 1))

        # Use correct gamma indices based on robot model
        if self.gamma_dim == 2:
            state_repeated[:, self.gamma0_index] = gamma_flat[:, 0]
            state_repeated[:, self.gamma1_index] = gamma_flat[:, 1]
        else:
            state_repeated[:, self.gamma0_index] = gamma_flat[:, 0]

        # Predict using vectorized PENN
        y_pred_safety_loss, y_pred_deadlock_time, epistemic_uncertainty = self.penn.predict(state_repeated)

        # Repackage predictions
        predictions = []
        for i in range(num_samples):
            g0 = gamma_flat[i, 0]
            g1 = gamma_flat[i, 1] if self.gamma_dim == 2 else 0.0
            predictions.append((g0, g1, y_pred_safety_loss[i], y_pred_deadlock_time[i][0], epistemic_uncertainty[i]))

        return predictions
    
    def build_graph_from_env(self, tracking_controller):
        """
        Build a PyG graph from the current environment
        """
        rx, ry = tracking_controller.robot.X[0, 0], tracking_controller.robot.X[1, 0]
        rtheta = tracking_controller.robot.X[2, 0]

        # Convert heading+velocity to vx,vy if ground robot
        if self.robot_model == 'Quad2D':
            vx = tracking_controller.robot.X[3, 0]
            vz = tracking_controller.robot.X[4, 0]
            robot_state = [rx, ry, vx, vz]
        elif self.robot_model == 'Quad3D':
            vx = tracking_controller.robot.X[6, 0]
            vz = tracking_controller.robot.X[8, 0]
            robot_state = [rx, ry, vx, vz]
        else:
            vel = tracking_controller.robot.X[3, 0]
            vx = vel * np.cos(rtheta)
            vy = vel * np.sin(rtheta)
            robot_state = [rx, ry, vx, vy]

        obstacles = tracking_controller.nearest_multi_obs
        if isinstance(obstacles, np.ndarray):
            if obstacles.size == 0:
                obstacles = [[100., 100., 0.2]]
        else:
            if not obstacles:
                obstacles = [[100., 100., 0.2]]

        final_waypoint = tracking_controller.waypoints[-1]
        goal = [final_waypoint[0], final_waypoint[1]]

        # Build the graph using the GATModule
        gdata = self.gat_module.create_graph(
            robot=robot_state,
            obstacles=obstacles,
            goal=goal,
            deadlock=0.0,  # placeholders
            risk=0.0
        )
        return gdata
    
    def predict_with_gat_penn(self, tracking_controller, gamma0_range, gamma1_range):
        """
        Predict safety loss, deadlock time, and epistemic uncertainty 
        using the Probabilistic Ensemble Neural Network
        """
        if not _TORCH_GEOMETRIC_AVAILABLE or Batch is None:
            raise ModuleNotFoundError(
                "torch_geometric is required for GAT-based online adaptation. "
                "Install torch_geometric or disable use_gat."
            )
        base_graph = self.build_graph_from_env(tracking_controller)

        # Generate all gamma combinations
        if self.gamma_dim == 2:
            g0_grid, g1_grid = torch.meshgrid(
                torch.tensor(gamma0_range, dtype=torch.float32),
                torch.tensor(gamma1_range, dtype=torch.float32),
                indexing='ij'
            )
            gamma_comb = torch.stack([g0_grid.flatten(), g1_grid.flatten()], dim=1).to(self.device)
        else:
            gamma_comb = torch.tensor(gamma0_range, dtype=torch.float32).reshape(-1, 1).to(self.device)

        num_samples = gamma_comb.shape[0]
        graph_list = [base_graph.clone() for _ in range(num_samples)]
        for i in range(num_samples):
            graph_list[i].gamma = gamma_comb[i].unsqueeze(0)

        batched_graph = Batch.from_data_list(graph_list).to(self.device)

        # Predict with vectorized PENN
        y_pred_safety_list, y_pred_deadlock_list, div_list = self.penn.predict([batched_graph])
        predictions = []
        for i in range(num_samples):
            g0 = gamma_comb[i, 0].item()
            g1 = gamma_comb[i, 1].item() if self.gamma_dim == 2 else 0.0
            predictions.append((g0, g1, y_pred_safety_list[i], y_pred_deadlock_list[i], div_list[i]))

        return predictions

    def filter_by_epistemic_uncertainty(self, predictions):
        '''
        Filter predictions based on epistemic uncertainty
        We employ Jensen-Renyi Divergence (JRD) with quadratic Renyi entropy, which has a closed-form expression of the divergence of a GMM
        If the JRD D(X) of the prediction of a given input X is greater than the predefined threshold, it is deemed to be out-of-distribution
        
        Uses raw JRD values directly (no normalization) for consistent comparison with CCCP-calibrated thresholds
        '''                

        if not predictions:
            return []
        epi = np.asarray([p[4] for p in predictions], dtype=np.float32)          # (N,)
        # If all uncertainties are high, return an empty list
        if np.all(epi > 100.0):
            return []
        
        # Use raw JRD values directly - no normalization needed
        # The epistemic_threshold is now a raw JRD value calibrated by CCCP
        keep_mask = epi <= self.epistemic_threshold                               # (N,) bool
        
        return [pred for pred, keep in zip(predictions, keep_mask) if keep]

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
        if not filtered_predictions:
            return []

        # y_pred_safety_loss[i] == list_of_ensembles  => [ [mu, var], ... ]
        N, E = len(filtered_predictions), self.penn.n_ensemble
        mu_mat   = np.zeros((N, E), dtype=np.float64)
        sig2_mat = np.zeros((N, E), dtype=np.float64)

        for i, pred in enumerate(filtered_predictions):
            ens = pred[2]  # y_pred_safety_loss  => list[[mu,var],...]
            mu_mat[i]   = [e[0] for e in ens]
            sig2_mat[i] = [e[1] for e in ens]

        boundary  = self.calculate_cvar_boundary()
        # print(f"CVaR boundary: {boundary:.4f}")
        keep_mask = DistributionallyRobustCVaR.batch_within_boundary(mu_mat, sig2_mat, boundary, alpha=0.99)

        return [pred for pred, keep in zip(filtered_predictions, keep_mask) if keep]

    def select_best_parameters(self, final_predictions, tracking_controller):
        '''
        Select the best CBF parameters based on filtered predictions.
        '''
        if self.gamma_dim == 1:
            current_gamma0 = tracking_controller.pos_controller.cbf_param['alpha']
        elif self.gamma_dim == 2:
            current_gamma0 = tracking_controller.pos_controller.cbf_param['alpha1']
            current_gamma1 = tracking_controller.pos_controller.cbf_param['alpha2']

        # If no predictions were selected, degrade conservatively
        if not final_predictions:
            gamma0 = max(self.lower_bound, current_gamma0 - self.step_size)
            gamma1 = max(self.lower_bound, current_gamma1 - self.step_size) if self.gamma_dim == 2 else 0.0
            return gamma0, gamma1

        # Use harmonic mean only if gamma1 exists
        if self.gamma_dim == 2:
            best_prediction = max(
                final_predictions,
                key=lambda x: 2.0 * (x[0] * x[1]) / (x[0] + x[1]) if (x[0] + x[1]) != 0 else 0.0
            )
            return best_prediction[0], best_prediction[1]
        else:
            # gamma1 is unused → use max gamma0
            best_prediction = max(final_predictions, key=lambda x: x[0])
            return best_prediction[0], 0.0  # gamma0, dummy gamma1

    def cbf_param_adaptation(self, tracking_controller):
        '''
        Perform adaptive CBF parameter selection based on the prediction from the PENN model 
        which is both confident and satisfies the local validity condition
        '''
        current_state = self.get_rel_state_wt_obs(tracking_controller)
        gamma0 = current_state[self.gamma0_index]
        gamma1 = current_state[self.gamma1_index] if self.gamma_dim == 2 else None
        gamma0_range, gamma1_range = self.sample_cbf_parameters(gamma0, gamma1)

        if self.use_gat:
            predictions = self.predict_with_gat_penn(tracking_controller, gamma0_range, gamma1_range)
        else:
            predictions = self.predict_with_penn(current_state, gamma0_range, gamma1_range)
        # print(f"INITIAL PREDICTIONS@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@{predictions}")

        filtered_predictions = self.filter_by_epistemic_uncertainty(predictions)
        # print(f"EPISTEMIC PREDICTIONS@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@{filtered_predictions}")

        final_predictions = self.filter_by_aleatoric_uncertainty(filtered_predictions)
        # print(f"FINAL PREDICTIONS@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@{final_predictions}")
        best_gamma0, best_gamma1 = self.select_best_parameters(final_predictions, tracking_controller)

        if self.print_info:
            if self.gamma_dim == 2:
                print(f"CBF parameters updated to: {best_gamma0:.2f}, {best_gamma1:.2f}"
                    f" | Total predictions: {len(predictions)}"
                    f" | Filtered {len(predictions)-len(filtered_predictions)} with Epistemic"
                    f" | Filtered {len(filtered_predictions)-len(final_predictions)} with Aleatoric")
            else:
                print(f"CBF parameter updated to: {best_gamma0:.2f}"
                    f" | Total predictions: {len(predictions)}"
                    f" | Filtered {len(predictions)-len(filtered_predictions)} with Epistemic"
                    f" | Filtered {len(filtered_predictions)-len(final_predictions)} with Aleatoric")

        return best_gamma0, best_gamma1


def get_robot_spec_and_obs(robot_model):
    """
    Returns (robot_spec, default_obs) for a given robot_model.
    """
    if robot_model not in ALL_DEFAULTS:
        raise ValueError(f"Unknown robot_model '{robot_model}'")
    
    entry = ALL_DEFAULTS[robot_model]
    return entry["robot_spec"], entry["default_obs"]

def get_controller_defaults(robot_model, controller_name):
    """
    Returns (controller_type, gamma0, gamma1) for the given robot_model
    and high-level controller_name.
    """
    # BarrierNet is handled as a direct controller in safe_control/tracking.py.
    # It does not use gamma parameters here.
    if controller_name == "BarrierNet":
        return ("barriernet", 0.0, 0.0)

    if robot_model not in ALL_DEFAULTS:
        raise ValueError(f"Unknown robot_model '{robot_model}'")
    
    controller_params = ALL_DEFAULTS[robot_model]["controller_params"].get(controller_name)
    if controller_params is None:
        raise ValueError(f"Unknown controller_name '{controller_name}' for robot_model '{robot_model}'")
    
    return (
        controller_params["type"],
        controller_params["gamma0"],
        controller_params["gamma1"]
    )

def get_env_defaults(robot_model):
    '''
    Returns environment configurations
    '''
    
    if robot_model not in ALL_DEFAULTS:
        raise ValueError(f"Unknown robot_model '{robot_model}'")

    entry = ALL_DEFAULTS[robot_model]

    env_width = entry.setdefault("env_width", 11.0)
    env_height = entry.setdefault("env_height", 3.8)

    return env_width, env_height

def get_online_cbf_adapter(robot_model, controller_name, print_info=True):
    """
    Returns an OnlineCBFAdapter instance for the given robot_model
    """
    if robot_model not in ADAPTIVE_MODELS:
        raise ValueError(f"No online adapter config found for '{robot_model}'")

    controller_subkey_map = {
        "Online Adaptive CBF-QP":  "online_cbf_qp",
        "Online Adaptive CBF-QP MLP": "online_cbf_qp_mlp",
        "Online Adaptive CBF-QP GAT": "online_cbf_qp_gat",
        "Online Adaptive MPC-CBF MLP": "online_mpc_cbf_mlp",
        "Online Adaptive MPC-CBF GAT": "online_mpc_cbf_gat",
    }
    # Determine if controller uses GAT
    if controller_name in ["Online Adaptive MPC-CBF GAT", "Online Adaptive CBF-QP GAT"]: 
        use_gat = True
    else:
        use_gat = False
    
    if controller_name not in controller_subkey_map:
        raise ValueError(f"Controller '{controller_name}' not recognized for online adaptation.")
    subkey = controller_subkey_map[controller_name]
    if subkey not in ADAPTIVE_MODELS[robot_model]:
        raise ValueError(f"No config for subkey '{subkey}' in '{robot_model}'")

    cfg = ADAPTIVE_MODELS[robot_model][subkey]
    # Allow environment variable override for checkpoint and scaler paths
    model_override = os.environ.get("CHECKPOINT_FILE")
    scaler_override = os.environ.get("SCALER_FILE")
    threshold_override = os.environ.get("RAW_EPISTEMIC_THRESHOLD")
    if threshold_override:
        epistemic_threshold = float(threshold_override)
    else:
        epistemic_threshold = cfg.get("raw_epistemic_threshold", cfg.get("epistemic_threshold", 0.20))

    return OnlineCBFAdapter(
        model_name=model_override if model_override else cfg["model_path"],
        scaler_name=scaler_override if scaler_override else cfg["scaler_path"],
        step_size=cfg["step_size"],
        lower_bound=cfg["lower_bound"],
        upper_bound=cfg["upper_bound"],
        epistemic_threshold=epistemic_threshold,
        robot_model=robot_model,
        use_gat=use_gat,
        print_info=print_info,
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
    env_width, env_height = get_env_defaults(robot_model)

    print(robot_spec, default_obs)
    print(controller_name, ctrl_type, gamma0, gamma1)

    # For DPCBF, obstacles need 7 elements: [x, y, r, vx, vy, y_min, flag]
    # For other robots, obstacles need 5 elements: [x, y, r, vx, vy]
    if robot_model == "KinematicBicycle2D_DPCBF":
        if default_obs.shape[1] != 7:
            # Pad to 7 elements: [x, y, r, vx, vy, y_min, flag]
            # flag=0 for circles, flag=1 for superellipsoids
            if default_obs.shape[1] == 3:
                # Start with [x, y, r], add [vx, vy, y_min, flag]
                default_obs = np.hstack((default_obs, np.zeros((default_obs.shape[0], 4))))
            elif default_obs.shape[1] == 5:
                # Already has [x, y, r, vx, vy], add [y_min, flag]
                default_obs = np.hstack((default_obs, np.zeros((default_obs.shape[0], 2))))
            # If already 7, do nothing
    else:
        if default_obs.shape[1] != 5:
            default_obs = np.hstack((default_obs, np.zeros((default_obs.shape[0], 2)))) # Set static obs velocity 0.0 at (5, 5)
    
    # Set initial state
    if robot_model == "Quad2D":
        # velocity should be [vx, vz] for Quad2D
        x_init = np.append(waypoints[0], [velocity[0], velocity[1], 0])
    elif robot_model == "Quad3D":
        # x_init = np.append(waypoints[0], [velocity[0], velocity[1], 0])
        x, y = waypoints[0][:2]
        z       = 0.0
        theta   = phi = psi = 0.0
        vx      = velocity[0]
        vy      = 0.0
        vz      = velocity[1]
        q = p = r = 0.0
        x_init = np.array([x, y, z, theta, phi, psi, vx, vy, vz, q, p, r])
    elif robot_model == "VTOL2D":
        x_init = np.hstack((2.0, 10.0, 0.0, velocity, 0.0, 0.0))
        plt.rcParams['figure.figsize'] = [12, 5]
    else:
        # velocity is a single scalar for 2D ground vehicles
        x_init = np.append(waypoints[0], velocity)

    # Set plotting and environment
    plot_handler = plotting.Plotting(width=env_width, height=env_height, known_obs=default_obs)
    ax, fig = plot_handler.plot_grid("")
    env_handler = env.Env()

    # Create the tracking controller - use LocalTrackingControllerDyn for KinematicBicycle2D_DPCBF
    if robot_model == "KinematicBicycle2D_DPCBF":
        tracking_controller = LocalTrackingControllerDyn(
            x_init,
            robot_spec,
            controller_type={'pos': ctrl_type},
            dt=dt,
            show_animation=True,
            save_animation=True,
            ax=ax,
            fig=fig,
            env=env_handler
        )
    else:
        tracking_controller = LocalTrackingController(
            x_init,
            robot_spec,
            controller_type={'pos': ctrl_type},
            dt=dt,
            show_animation=True,
            save_animation=True,
            ax=ax,
            fig=fig,
            env=env_handler
        )

    # Initialize the CBF parameters
    if robot_model not in ["KinematicBicycle2D_DPCBF", "Quad3D"]:
        tracking_controller.pos_controller.cbf_param['alpha1'] = gamma0
        tracking_controller.pos_controller.cbf_param['alpha2'] = gamma1
    else:
        tracking_controller.pos_controller.cbf_param['alpha'] = gamma0
        tracking_controller.pos_controller.cbf_param['alpha2'] = 0.0  # dummy placeholder
    

    # Load obstacles & set waypoints
    tracking_controller.obs = default_obs
    tracking_controller.set_waypoints(waypoints)

    # If controller is 'Online Adaptive', get adapter
    if controller_name in ['Online Adaptive CBF-QP', 'Online Adaptive CBF-QP MLP', 'Online Adaptive CBF-QP GAT', 'Online Adaptive MPC-CBF MLP', 'Online Adaptive MPC-CBF GAT']:
        online_cbf_adapter = get_online_cbf_adapter(robot_model, controller_name)
    else:
        online_cbf_adapter = None

    import csv
    # create a csv file to record the states, control inputs, and CBF parameters
    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['states', 'control_inputs', 'alpha1', 'alpha2'])

    # Main simulation loop
    n_steps = int(max_sim_time / dt)
    import time
    for _ in range(n_steps):
        
        ret = tracking_controller.control_step()
        tracking_controller.draw_plot()

        # Check if we've reached the goal or collided
        if ret == -1 or ret == -2:
            dist_to_goal = np.linalg.norm(tracking_controller.robot.X[:2, 0] - waypoints[-1][:2])
            print("Goal point reached." if dist_to_goal < tracking_controller.reached_threshold else "Collided.")
            break

        # Adapt the CBF parameters if using an online approach
        if online_cbf_adapter is not None:
            start = time.time()
            best_gamma0, best_gamma1 = online_cbf_adapter.cbf_param_adaptation(tracking_controller)
            if best_gamma0 is not None:
                if robot_model not in ["KinematicBicycle2D_DPCBF", "Quad3D"]:
                    tracking_controller.pos_controller.cbf_param['alpha1'] = best_gamma0
                    tracking_controller.pos_controller.cbf_param['alpha2'] = best_gamma1
                else:
                    tracking_controller.pos_controller.cbf_param['alpha'] = best_gamma0
                          
            end = time.time()
            print(f"Time taken for pure adaptation step: {end - start:.4f} seconds")

        # get states of the robot
        robot_state = tracking_controller.robot.X[:,0].flatten()
        control_input = tracking_controller.get_control_input().flatten()
        # print(f"Robot state: {robot_state}")
        # print(f"Control input: {control_input}")

        # append the states, control inputs, and CBF parameters by appending to csv
        with open('output.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if robot_model not in ["KinematicBicycle2D_DPCBF", "Quad3D"]:
                writer.writerow(np.append(robot_state, np.append(control_input, 
                    [tracking_controller.pos_controller.cbf_param['alpha1'], 
                    tracking_controller.pos_controller.cbf_param['alpha2']])))
            else:
                writer.writerow(np.append(robot_state, np.append(control_input, 
                    [tracking_controller.pos_controller.cbf_param['alpha'], 
                    tracking_controller.pos_controller.cbf_param['alpha2']])))

    tracking_controller.export_video()
    plt.ioff()
    plt.close()


if __name__ == "__main__":
    controller_list = [
        "CBF-QP low fixed param",      # 0
        "CBF-QP high fixed param",     # 1
        "MPC-CBF low fixed param",     # 2
        "MPC-CBF high fixed param",    # 3
        "Optimal Decay CBF-QP",        # 4
        "Optimal Decay MPC-CBF",       # 5
        "Online Adaptive CBF-QP",      # 6
        "Online Adaptive CBF-QP MLP",  # 7
        "Online Adaptive CBF-QP GAT",  # 8
        "Online Adaptive MPC-CBF MLP", # 9
        "Online Adaptive MPC-CBF GAT", # 10
    ]
    robot_model_list = [
        "DynamicUnicycle2D",           # 0
        "KinematicBicycle2D_DPCBF",    # 1
        "Quad2D",                      # 2
        "Quad3D",                      # 3
        "VTOL2D",                      # 4
    ]

    # Pick a specific controller and robot model
    controller_name = controller_list[8]   
    robot_model = robot_model_list[1]       
    
    # Define waypoints for the simulation
    if robot_model == "VTOL2D":
        waypoints = np.array([
                    [70, 10],
                    [70, 0.5]
                ], dtype=np.float64) 
    else:
        waypoints = np.array([
                    [0.75, 2.0, 0.01],
                    [10.0, 1.5, 0.0]
                ], dtype=np.float64)

    # For ground vehicles, velocity is a single scalar
    if robot_model in ["Quad2D", "Quad3D"]:
        init_vel = [0.4, 0.2]
    elif robot_model == "VTOL2D":
        init_vel = 20.0
    else:
        init_vel = 0.4

    # Run the simulation
    single_agent_simulation(init_vel, waypoints, controller_name, robot_model)
