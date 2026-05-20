import json
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import joblib

from train_data import load_graph_dataset as load_graph_dataset_train
from penn.gat import GATModule
from penn.nn_gat_iccbf_predict import ProbabilisticEnsembleGAT
from penn.nn_iccbf_predict import ProbabilisticEnsembleNN


def load_traj_graph_dataset(pickle_file):
    """
    Load trajectory-level graph dataset for CCCP calibration.
    
    Each entry in the pickle file should contain:
        - "graph_data": single graph for training (ignored here)
        - "cccp_traj_graphs": list of per-time-step graph dicts for this trajectory
    
    Returns:
        List of trajectories, where each trajectory is a list of PyG Data objects
        representing states X_t at each time step t.
    """
    with open(pickle_file, 'rb') as f:
        results = pickle.load(f)

    traj_list = []
    for entry in results:
        if "cccp_traj_graphs" not in entry:
            # Skip entries without CCCP trajectory data
            continue
            
        traj_graphs = []
        for graph_dict in entry["cccp_traj_graphs"]:
            # Reconstruct PyG Data object from dict
            graph_data = Data(
                x=torch.tensor(graph_dict["x"], dtype=torch.float),
                edge_index=torch.tensor(graph_dict["edge_index"], dtype=torch.long),
                edge_attr=torch.tensor(graph_dict["edge_attr"], dtype=torch.float)
            )
            
            if graph_dict.get("gamma") is not None:
                gamma_array = graph_dict["gamma"]
                # Ensure gamma is 2D: [1, gamma_dim] or [gamma_dim]
                # Handle both numpy arrays and lists
                if isinstance(gamma_array, np.ndarray):
                    if gamma_array.ndim == 1:
                        gamma_array = gamma_array.reshape(1, -1)
                    elif gamma_array.ndim == 2 and gamma_array.shape[0] != 1:
                        gamma_array = gamma_array.reshape(1, -1)
                elif isinstance(gamma_array, (list, tuple)):
                    gamma_array = np.array(gamma_array)
                    if gamma_array.ndim == 1:
                        gamma_array = gamma_array.reshape(1, -1)
                graph_data.gamma = torch.tensor(gamma_array, dtype=torch.float)
            
            # Include y if present (though CCCP will ignore it)
            if graph_dict.get("y") is not None:
                graph_data.y = torch.tensor(graph_dict["y"], dtype=torch.float)
            
            traj_graphs.append(graph_data)
        
        if len(traj_graphs) > 0:
            traj_list.append(traj_graphs)
    
    return traj_list


class ClassConditionedConformalPrediction:
    """
    Compute a CCCP threshold for in-distribution recall.
    
    Two modes available:
    1. Trajectory-level (use_trajectory_level=True, default):
       Implements the Class-Conditioned Conformal Prediction (CCCP) method as specified:
       - Per-time-step nonconformity: D(X_t) (JRD from the ensemble)
       - Trajectory-level nonconformity: Q_i = max_t D(X_t)  [Equation (51)]
       - Threshold: D_thr = Quantile({Q_i}; alpha_cal)  [Equation (52)]
       - Coverage guarantee: P(Q_test <= D_thr | tau_test in D) >= alpha_cal
       where alpha_cal = 1 - delta_cal is the coverage probability.
    
    2. Graph-level (use_trajectory_level=False):
       Implementation that processes individual graphs:
       - Computes D(X) (JRD) for each graph independently
       - Threshold: D_thr = Quantile({D(X)}; alpha_cal)
       - No trajectory aggregation
    """

    def __init__(
        self,
        pickle_path: str = None,
        csv_path: str = None,
        model_path: str = None,
        scaler_path: str = None,
        robot_model: str = None,
        gamma_dim: int = None,
        alpha_cal: float = 0.95,
        use_trajectory_level: bool = True,
        use_mlp: bool = False,
        device: str = "cpu",
        n_output: int = 2,
        n_hidden: int = 40,
        n_ensemble: int = 3,
    ):
        """
        Initialize CCCP calibration.
        
        Args:
            pickle_path: Path to pickle file (for GAT mode)
            csv_path: Path to CSV file (for MLP mode)
            model_path: Path to model checkpoint
            scaler_path: Path to scaler file (required for MLP mode)
            robot_model: Robot model name (required for MLP mode)
            gamma_dim: Gamma dimension (required for GAT mode, inferred for MLP)
            alpha_cal: Coverage probability = 1 - delta_cal
            use_trajectory_level: Whether to use trajectory-level calibration (GAT only)
            use_mlp: Whether to use MLP mode (True) or GAT mode (False)
            device: Device to use
            n_output: Number of outputs
            n_hidden: Hidden layer size
            n_ensemble: Number of ensemble members
        """
        self.use_mlp = use_mlp
        self.pickle_path = pickle_path
        self.csv_path = csv_path
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.robot_model = robot_model
        self.alpha_cal = alpha_cal  # Coverage probability = 1 - delta_cal
        self.use_trajectory_level = use_trajectory_level
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if self.use_mlp:
            # MLP mode: Load CSV data
            if csv_path is None:
                raise ValueError("csv_path is required for MLP mode")
            if scaler_path is None:
                raise ValueError("scaler_path is required for MLP mode")
            if robot_model is None:
                raise ValueError("robot_model is required for MLP mode")
            
            # Load CSV dataset (we need raw data for MLP predictor, which handles transformation internally)
            # But we still need to extract the raw features for batch processing
            dataset = pd.read_csv(csv_path)
            
            # Prepare X depending on robot model (raw features before transformation)
            if robot_model == 'Quad2D':
                X = dataset[['Distance', 'VelocityX', 'VelocityZ', 'Theta', 'gamma0', 'gamma1']].values
            elif robot_model == 'Quad3D':
                X = dataset[['Distance', 'VelocityX', 'VelocityZ', 'Theta', 'gamma0']].values
            elif robot_model in ['KinematicBicycle2D_C3BF', 'KinematicBicycle2D_DPCBF']:
                X = dataset[['Distance', 'Velocity', 'Theta', 'gamma0']].values
            else:  # DynamicUnicycle2D, etc.
                X = dataset[['Distance', 'Velocity', 'Theta', 'gamma0', 'gamma1']].values
            
            self.csv_data = X  # Store raw data (predictor will handle transformation)
            print(f"[INFO] Loaded {len(self.csv_data):,} samples from {csv_path}")
            
            # Determine n_states and theta_index based on robot model
            if robot_model == 'Quad2D':
                n_states = 7  # Distance, VelocityX, VelocityZ, sin(Theta), cos(Theta), gamma0, gamma1
                theta_index = 3 
            elif robot_model == 'Quad3D':
                n_states = 6  # Distance, VelocityX, VelocityZ, sin(Theta), cos(Theta), gamma0
                theta_index = 3  
            elif robot_model in ['KinematicBicycle2D_C3BF', 'KinematicBicycle2D_DPCBF']:
                n_states = 5  # Distance, Velocity, sin(Theta), cos(Theta), gamma0
                theta_index = 2 
            else:  # DynamicUnicycle2D, etc.
                n_states = 6  # Distance, Velocity, sin(Theta), cos(Theta), gamma0, gamma1
                theta_index = 2  
            
            # Build MLP model
            self.predictor = ProbabilisticEnsembleNN(
                n_states=n_states,
                n_output=n_output,
                n_hidden=n_hidden,
                n_ensemble=n_ensemble,
                device=str(self.device),
                theta_index=theta_index,
            )
            self.predictor.load_scaler(scaler_path)
            self.predictor.load_model(model_path)
            print(f"[INFO] Loaded MLP weights from {model_path}")
            
            # MLP mode only supports graph-level (non-trajectory) calibration
            self.use_trajectory_level = False
            self.traj_list = None
            self.data_list = None
            
        else:
            # GAT mode: Load pickle data
            if pickle_path is None:
                raise ValueError("pickle_path is required for GAT mode")
            
            # Load dataset based on mode
            if self.use_trajectory_level:
                # Load trajectory dataset (each trajectory is a list of graphs)
                self.traj_list = load_traj_graph_dataset(self.pickle_path)
                print(f"[INFO] Loaded {len(self.traj_list):,} trajectories from {self.pickle_path}")
                self.data_list = None  # Not used in trajectory mode
            else:
                # Load individual graphs
                # Use load_graph_dataset from train_data which handles the standard pickle format
                self.data_list = load_graph_dataset_train(self.pickle_path)
                print(f"[INFO] Loaded {len(self.data_list):,} graphs from {self.pickle_path}")
                self.traj_list = None  # Not used in graph mode

            # Build GAT model
            gat_mod = GATModule(device=self.device).to(self.device)
            gat_net = gat_mod.gat
            self.predictor = ProbabilisticEnsembleGAT(
                gat_net,
                n_output=n_output,
                n_hidden=n_hidden,
                n_ensemble=n_ensemble,
                gamma_dim=gamma_dim,
                device=self.device,
            ).to(self.device)
            self.predictor.load_model(self.model_path)
            print(f"[INFO] Loaded GAT weights from {self.model_path}")

        # Storage for scores (trajectory-level or graph-level depending on mode)
        if self.use_trajectory_level:
            # Q_vals corresponds to {Q_i}_{i=1}^{N_traj} from equation (51)
            self.Q_vals = None
        else:
            # jrd_vals stores individual JRD values for graph-level calibration
            self.jrd_vals = None
        # threshold corresponds to D_thr from equation (52) or graph-level quantile
        self.threshold = None

    def _collect_divergences(self):
        """
        Run the model across the calibration dataset and compute scores.
        
        Mode 1 (use_trajectory_level=True):
        For each trajectory tau_i = {X_t}_{t=1}^{T_i}:
        - Computes D(X_t) (JRD) for each state X_t
        - Computes Q_i = max_t D(X_t) (trajectory-level nonconformity score)
        Stores Q_i values in self.Q_vals, which corresponds to {Q_i}_{i=1}^{N_traj}
        from equation (51) in the LaTeX spec.
        
        Mode 2 (use_trajectory_level=False):
        For each graph:
        - Computes D(X) (JRD) for each individual graph
        Stores JRD values in self.jrd_vals
        """
        if self.use_trajectory_level:
            # Trajectory-level processing
            Q_list = []
            total_trajectories = len(self.traj_list)
            print(f"[INFO] Starting trajectory-level divergence collection for {total_trajectories:,} trajectories...")
            
            for i, traj_graphs in enumerate(self.traj_list):
                # Predict on all graphs in this trajectory
                # predictor.predict expects a list of Data objects
                _, _, divs = self.predictor.predict(traj_graphs)
                divs = np.asarray(divs, dtype=np.float64)
                
                # Compute trajectory-level nonconformity: Q_i = max_t D(X_t)
                # This implements equation (51): Q_i := max_t D(X_t)
                Q_i = float(divs.max())
                Q_list.append(Q_i)
                
                # Show progress every 1000 trajectories or at the end
                if (i + 1) % 1000 == 0 or (i + 1) == total_trajectories:
                    progress = (i + 1) / total_trajectories * 100
                    print(f"[PROGRESS] Processed {i + 1:,}/{total_trajectories:,} trajectories ({progress:.1f}%)")

            self.Q_vals = np.asarray(Q_list, dtype=np.float64)
            print(f"[INFO] Collected {self.Q_vals.size:,} trajectory-level scores (Q_i values).")
        else:
            # Graph-level processing
            jrd_list = []
            
            if self.use_mlp:
                # MLP mode: Process CSV data
                total_samples = len(self.csv_data)
                print(f"[INFO] Starting graph-level divergence collection for {total_samples:,} samples (MLP mode)...")
                
                # Process in batches for efficiency
                batch_size = 10000
                for batch_start in range(0, total_samples, batch_size):
                    batch_end = min(batch_start + batch_size, total_samples)
                    batch_data = self.csv_data[batch_start:batch_end]
                    
                    # Predict on batch (predictor handles transformation and scaling internally)
                    _, _, divs = self.predictor.predict(batch_data)
                    jrd_list.extend(divs)
                    
                    # Show progress
                    if batch_end % 50000 == 0 or batch_end == total_samples:
                        progress = batch_end / total_samples * 100
                        print(f"[PROGRESS] Processed {batch_end:,}/{total_samples:,} samples ({progress:.1f}%)")
            else:
                # GAT mode: Process graph data
                total_samples = len(self.data_list)
                print(f"[INFO] Starting graph-level divergence collection for {total_samples:,} graphs...")
                
                for i, data in enumerate(self.data_list):
                    # predictor.predict expects a *list* of Data objects
                    _, _, divs = self.predictor.predict([data])
                    jrd_list.extend(divs)
                    
                    # Show progress every 50,000 samples or at the end
                    if (i + 1) % 50000 == 0 or (i + 1) == total_samples:
                        progress = (i + 1) / total_samples * 100
                        print(f"[PROGRESS] Processed {i + 1:,}/{total_samples:,} graphs ({progress:.1f}%)")

            self.jrd_vals = np.asarray(jrd_list, dtype=np.float64)
            print(f"[INFO] Collected {self.jrd_vals.size:,} JRD values.")

    def calibrate(self):
        """
        Compute threshold based on selected mode.
        
        Mode 1 (use_trajectory_level=True):
        D_thr = Quantile({Q_i}; alpha_cal) as per equation (52).
        - alpha_cal = 1 - delta_cal is the coverage probability
        - D_thr is the (alpha_cal)-quantile of trajectory-level scores {Q_i}
        - Coverage guarantee: P(Q_test <= D_thr | tau_test in D) >= alpha_cal
        
        Mode 2 (use_trajectory_level=False):
        D_thr = Quantile({D(X)}; alpha_cal)
        - Computes threshold on individual graph JRD values
        """
        if self.use_trajectory_level:
            if self.Q_vals is None:
                self._collect_divergences()

            # alpha_cal = 1 - delta_cal (coverage probability)
            # D_thr = Quantile({Q_i}; alpha_cal) as per equation (52)
            coverage = self.alpha_cal  # e.g., 0.95 means 95% coverage
            self.threshold = float(
                np.quantile(self.Q_vals, coverage, interpolation="higher")
            )
            
            print(f"[RESULT] CCCP threshold D_thr (coverage={coverage:.3f}) = {self.threshold:.6f}")
            print(f"[INFO] Trajectory-level score (Q_i) statistics:")
            print(f"  Min: {self.Q_vals.min():.6f}, Max: {self.Q_vals.max():.6f}")
            print(f"  Mean: {self.Q_vals.mean():.6f}, Std: {self.Q_vals.std():.6f}")
        else:
            if self.jrd_vals is None:
                self._collect_divergences()

            # Graph-level threshold computation
            coverage = self.alpha_cal
            self.threshold = float(
                np.quantile(self.jrd_vals, coverage, interpolation="higher")
            )
            
            print(f"[RESULT] CCCP threshold D_thr (coverage={coverage:.3f}) = {self.threshold:.6f}")
            print(f"[INFO] JRD statistics:")
            print(f"  Min: {self.jrd_vals.min():.6f}, Max: {self.jrd_vals.max():.6f}")
            print(f"  Mean: {self.jrd_vals.mean():.6f}, Std: {self.jrd_vals.std():.6f}")
        
        return self.threshold

    def save_threshold(self, out_json: str):
        """
        Save the computed CCCP threshold to JSON.
        
        The threshold format depends on the mode:
        - Trajectory-level: D_thr = Quantile({Q_i}; alpha_cal) per equation (52)
        - Graph-level: D_thr = Quantile({D(X)}; alpha_cal)
        """
        if self.threshold is None:
            raise RuntimeError("Call calibrate() before saving.")

        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        
        if self.use_trajectory_level:
            stats = {
                "min": float(self.Q_vals.min()),
                "max": float(self.Q_vals.max()),
                "mean": float(self.Q_vals.mean()),
                "std": float(self.Q_vals.std()),
                "num_trajectories": int(self.Q_vals.size)
            }
            description = {
                "alpha_cal": "Coverage probability = 1 - delta_cal",
                "cccp_threshold": "D_thr = Quantile({Q_i}; alpha_cal) per equation (52)",
                "Q_i": "Trajectory-level nonconformity score = max_t D(X_t) per equation (51)",
                "mode": "trajectory-level"
            }
        else:
            stats = {
                "min": float(self.jrd_vals.min()),
                "max": float(self.jrd_vals.max()),
                "mean": float(self.jrd_vals.mean()),
                "std": float(self.jrd_vals.std()),
                "num_graphs": int(self.jrd_vals.size)
            }
            description = {
                "alpha_cal": "Coverage probability = 1 - delta_cal",
                "cccp_threshold": "D_thr = Quantile({D(X)}; alpha_cal)",
                "mode": "graph-level"
            }
        
        # Prepare metadata based on mode
        if self.use_mlp:
            metadata = {
                "csv_path": self.csv_path,
                "model_path": self.model_path,
                "scaler_path": self.scaler_path,
                "robot_model": self.robot_model,
                "alpha_cal": self.alpha_cal,
                "use_trajectory_level": False,  # MLP always uses graph-level
                "use_mlp": True,
                "cccp_threshold": self.threshold,
                "score_statistics": stats,
                "description": description
            }
        else:
            metadata = {
                "pickle_path": self.pickle_path,
                "model_path": self.model_path,
                "alpha_cal": self.alpha_cal,
                "use_trajectory_level": self.use_trajectory_level,
                "use_mlp": False,
                "cccp_threshold": self.threshold,
                "score_statistics": stats,
                "description": description
            }
        
        with open(out_json, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[INFO] Threshold saved to {out_json}")


if __name__ == "__main__":
    CONFIG = {
        # "pickle_path": "data/gat_datagen_300000_Quad3D_mpc_cbf.pkl",
        # "model_path": "checkpoint/Quad3D_0807_gat_0230.pth",
        "pickle_path": "data/gat_datagen_300000_DynamicUnicycle2D_mpc_cbf.pkl",
        "model_path": "checkpoint/DynamicUnicycle2D_0731_gat_2130.pth",
        "gamma_dim": 2,  # 1 for KinematicBicycle2D/Quad3D, 2 for DynamicUnicycle2D/Quad2D, etc.
        "alpha_cal": 0.95,  # Coverage probability = 1 - delta_cal
        "use_trajectory_level": True,  # Set to False for graph-level calibration
        "device": "cuda",  
        "out_json": "checkpoint/jrd_cccp_threshold.json",
    }

    cccp = ClassConditionedConformalPrediction(
        pickle_path=CONFIG["pickle_path"],
        model_path=CONFIG["model_path"],
        gamma_dim=CONFIG["gamma_dim"],
        alpha_cal=CONFIG["alpha_cal"],
        use_trajectory_level=CONFIG["use_trajectory_level"],
        device=CONFIG["device"],
    )
    cccp.calibrate()
    # cccp.save_threshold(CONFIG["out_json"])
