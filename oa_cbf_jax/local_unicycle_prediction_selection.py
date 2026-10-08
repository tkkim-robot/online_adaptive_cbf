"""Optional validation checkpoint selection; optimization losses stay unchanged."""
import numpy as np


def contract():
    return dict(schema='local_unicycle_prediction_checkpoint_selection_v1',
        score='mean(normalized_mae) + mean(event_brier)',
        targets=['normalized_clearance','normalized_conditional_progress'],
        events=['collision','controller_stop'],
        population='Original live-query and target masks, with equal physical-parent validation weights.',
        optimization='Unchanged Gaussian NLL and event BCE; no new objective or gradient.',
        selection='Minimum validation score with original 1e-4 improvement tolerance and patience.',
        calibration='Refit every model-dependent calibration after selection; likelihood is still recorded.',
        forward_or_navigation_selection=False)


def score(metrics):
    mae=np.asarray(metrics['normalized_mae'],dtype=float)
    brier=np.asarray(metrics['brier'],dtype=float)
    if (mae.shape!=(2,) or brier.shape!=(2,) or not np.isfinite(mae).all()
            or not np.isfinite(brier).all() or (mae<0).any() or (brier<0).any() or (brier>1).any()):
        raise ValueError('Two finite nonnegative mean errors and binary-event Brier scores required')
    return float(mae.mean()+brier.mean())
