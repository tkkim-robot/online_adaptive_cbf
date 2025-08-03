import json
import os
import numpy as np
import torch

from train_data import load_graph_dataset
from penn.gat import GATModule
from penn.nn_gat_iccbf_predict import ProbabilisticEnsembleGAT


class ClassConditionedConformalPrediction:
    """
    Compute a CCCP threshold for in‑distribution recall.
    """

    def __init__(
        self,
        pickle_path: str,
        model_path: str,
        gamma_dim: int,
        alpha_cal: float = 0.05,
        normalized_thresholds: list = [0.05, 0.1, 0.15, 0.2],  # Multiple thresholds to calculate
        device: str = "cpu",
        n_output: int = 2,
        n_hidden: int = 40,
        n_ensemble: int = 3,
    ):
        self.pickle_path = pickle_path
        self.model_path = model_path
        self.alpha_cal = alpha_cal
        self.normalized_thresholds = normalized_thresholds
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load dataset
        self.data_list = load_graph_dataset(self.pickle_path)
        print(f"[INFO] Loaded {len(self.data_list):,} graphs from {self.pickle_path}")

        # Build model
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
        print(f"[INFO] Loaded weights from {self.model_path}")

        # Storage
        self.jrd_vals = None
        self.threshold = None
        self.raw_thresholds_for_normalized = {}

    def _collect_divergences(self):
        """Run the model across the calibration dataset and store JRD values."""
        jrd_list = []
        total_samples = len(self.data_list)
        print(f"[INFO] Starting divergence collection for {total_samples:,} samples...")
        
        for i, data in enumerate(self.data_list):
            # predictor.predict expects a *list* of Data objects
            _, _, divs = self.predictor.predict([data])
            jrd_list.extend(divs)
            
            # Show progress every 1000 samples or at the end
            if (i + 1) % 10000 == 0 or (i + 1) == total_samples:
                progress = (i + 1) / total_samples * 100
                print(f"[PROGRESS] Processed {i + 1:,}/{total_samples:,} samples ({progress:.1f}%)")

        self.jrd_vals = np.asarray(jrd_list, dtype=np.float64)
        print(f"[INFO] Collected {self.jrd_vals.size:,} JRD values.")

    def calibrate(self):
        """Compute ε̂ such that P(ID ≤ ε̂) ≥ 1 − α_cal."""
        if self.jrd_vals is None:
            self._collect_divergences()

        # Calculate the normalized threshold (equivalent to epistemic_threshold=0.2 in OnlineCBFAdapter)
        jrd_normalized = (self.jrd_vals - self.jrd_vals.min()) / (self.jrd_vals.max() - self.jrd_vals.min() + 1e-8)
        
        # Calculate raw JRD values for multiple normalized thresholds
        print(f"[INFO] Calculating raw JRD thresholds for normalized thresholds: {self.normalized_thresholds}")
        for norm_thresh in self.normalized_thresholds:
            # Find the raw JRD value that corresponds to the normalized threshold
            # We want to find the value where normalized JRD <= norm_thresh
            # This means we keep the bottom norm_thresh% of samples (lowest JRD values)
            raw_thresh = float(np.quantile(self.jrd_vals, norm_thresh, interpolation="higher"))
            self.raw_thresholds_for_normalized[norm_thresh] = raw_thresh
            print(f"[RESULT] Raw JRD threshold for normalized threshold {norm_thresh} = {raw_thresh:.6f}")
        
        # Also calculate the original CCCP threshold
        q = 1.0 - self.alpha_cal  # e.g. 0.05 for α=0.95
        self.threshold = float(np.quantile(self.jrd_vals, q, interpolation="higher"))
        
        print(f"[RESULT] CCCP threshold ε̂ (quantile {q:.3f}) = {self.threshold:.6f}")
        print(f"[INFO] JRD statistics - Min: {self.jrd_vals.min():.6f}, Max: {self.jrd_vals.max():.6f}, Mean: {self.jrd_vals.mean():.6f}")
        
        return self.threshold

    def save_threshold(self, out_json: str):
        if self.threshold is None:
            raise RuntimeError("Call calibrate() before saving.")

        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(
                {
                    "pickle_path": self.pickle_path,
                    "model_path": self.model_path,
                    "alpha_cal": self.alpha_cal,
                    "normalized_thresholds": self.normalized_thresholds,
                    "cccp_threshold": self.threshold,
                    "raw_thresholds_for_normalized": self.raw_thresholds_for_normalized,
                    "jrd_statistics": {
                        "min": float(self.jrd_vals.min()),
                        "max": float(self.jrd_vals.max()),
                        "mean": float(self.jrd_vals.mean()),
                        "std": float(self.jrd_vals.std())
                    }
                },
                f,
                indent=2,
            )
        print(f"[INFO] Thresholds saved to {out_json}")


if __name__ == "__main__":
    CONFIG = {
        "pickle_path": "data/gat_datagen_200000_DynamicUnicycle2D_mpc_cbf.pkl",
        "model_path": "checkpoint/DynamicUnicycle2D_0731_gat_2130.pth",
        "gamma_dim": 2,  # 1 for KinematicBicycle2D/Quad3D, 2 for DynamicUnicycle2D/Quad2D, etc.
        "alpha_cal": 0.95,
        "normalized_thresholds": [0.05, 0.1, 0.15, 0.2],  # Multiple thresholds to calculate
        "device": "cuda",  
        "out_json": "checkpoint/jrd_cccp_threshold.json",
    }

    cccp = ClassConditionedConformalPrediction(
        pickle_path=CONFIG["pickle_path"],
        model_path=CONFIG["model_path"],
        gamma_dim=CONFIG["gamma_dim"],
        alpha_cal=CONFIG["alpha_cal"],
        normalized_thresholds=CONFIG["normalized_thresholds"],
        device=CONFIG["device"],
    )
    cccp.calibrate()
    # cccp.save_threshold(CONFIG["out_json"])
