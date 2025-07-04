import argparse
import json
import os
import numpy as np
import torch

from nn_model.train_data import load_graph_dataset
from nn_model.penn.gat import GATModule         
from nn_model.penn.nn_gat_iccbf_predict import ProbabilisticEnsembleGAT


class ClassConditionedConformalPrediction:
    """
    Compute a CCCP threshold for in-distribution recall
    """
    def __init__(self, pickle_path: str, model_path: str, gamma_dim: int, alpha_cal: float = 0.05,
                 device: str = "cpu", n_output: int = 2, n_hidden: int = 40, n_ensemble: int = 3,
                ):
        self.pickle_path = pickle_path
        self.model_path = model_path
        self.alpha_cal = alpha_cal
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

    def _collect_divergences(self):
        """Run the model across the calibration dataset and store JRD values."""
        jrd_list = []
        for data in self.data_list:
            # predictor.predict expects a *list* of Data objects
            _, _, divs = self.predictor.predict([data])
            jrd_list.extend(divs)

        self.jrd_vals = np.asarray(jrd_list, dtype=np.float64)
        print(f"[INFO] Collected {self.jrd_vals.size:,} JRD values.")

    def calibrate(self):
        """Compute ε̂ such that P(ID ≤ ε̂) ≥ 1 − α_cal."""
        if self.jrd_vals is None:
            self._collect_divergences()

        q = 1.0 - self.alpha_cal          # e.g. 0.95 for α=0.05
        self.threshold = float(np.quantile(self.jrd_vals, q, interpolation="higher"))
        print(
            f"[RESULT] CCCP threshold ε̂ (quantile {q:.3f}) = {self.threshold:.6f}"
        )
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
                    "threshold": self.threshold,
                },
                f,
                indent=2,
            )
        print(f"[INFO] Threshold saved to {out_json}")


def parse_args():
    p = argparse.ArgumentParser(description="CCCP threshold calibration for JRD")
    p.add_argument("--pickle", required=True, help=".pkl dataset path")
    p.add_argument("--model", required=True, help="trained .pth model path")
    p.add_argument(
        "--gamma-dim",
        type=int,
        required=True,
        help="dimension of gamma (1 for C3BF, 2 for ICCBF, etc.)",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="α_cal (default 0.05)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="device")
    p.add_argument(
        "--out-json",
        default="cccp_threshold.json",
        help="file to store calibrated threshold",
    )
    return p.parse_args()


if __name__ == "__main__":
    """
    Input:
    python cccp_calibrate_jrd.py \
        --pickle data/gat_datagen_10000_KinematicBicycle2D_C3BF_mpc_cbf.pkl \
        --model  checkpoint/KinematicBicycle2D_C3BF_gat.pth \
        --gamma-dim 1 \
        --alpha 0.05 \
        --device cuda \
        --out  checkpoint/jrd_cccp_threshold.json

    Output:
    [INFO] Loaded 10000 calibration graphs from gat_datagen_…pkl
    [INFO] Restored weights   → KinematicBicycle2D_C3BF_gat.pth
    [INFO] Collected 10000 JRD scores
    [RESULT] CCCP threshold ε̂ @ 0.950-quantile = 0.xxxxxx
    [INFO] Saved ε̂ to /…/checkpoint/jrd_cccp_threshold.json
    """

    args = parse_args()
    cccp = ClassConditionedConformalPrediction(
        pickle_path=args.pickle,
        model_path=args.model,
        gamma_dim=args.gamma_dim,
        alpha_cal=args.alpha,
        device=args.device,
    )
    cccp.calibrate()
    cccp.save_threshold(args.out_json)
