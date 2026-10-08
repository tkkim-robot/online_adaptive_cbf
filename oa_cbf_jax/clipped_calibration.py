"""Reserved-parent calibration of the explicit clipped-mixture predictor.

No training, validation, benchmark or reflected duplicate is a calibration unit.
The observation-level screen is not a physical trajectory or tail guarantee.
"""


from pathlib import Path


import numpy as np


from .censored_risk import contract, mixture_tail
from .dataset import load_dataset, sha256


SCHEMA='quad2d_clipped_mixture_prediction_calibration_v1'
DISAGREEMENT='CS with respect to unit atom at physical risk -2 plus Lebesgue above -2; exact component overlaps.'


def validate_calibration(info,metadata,bundle):
    """Require the actual matched distribution; never inherit Gaussian scales."""
    if (info.get('schema') not in (SCHEMA,'oa_cbf_frozen_trajectory_gate_v1')
            or info.get('risk_distribution_contract')!=contract(2)
            or metadata.get('risk_distribution_contract')!=contract(2)
            or info.get('disagreement_contract')!=DISAGREEMENT
            or info.get('event_statistic')!='maximum_member_probability'
            or info.get('bundle_manifest_sha256')!=sha256(Path(bundle)/'manifest.json')
            or info.get('weights_sha256')!=metadata['weights_sha256']):
        raise ValueError('Explicit matched clipped-mixture calibration required')
    scale=np.asarray(info['variance_scale'],float)
    if scale.shape!=(2,) or not np.isfinite(scale).all() or np.any(scale<1):
        raise ValueError('Invalid non-shrinking component variance calibration')
    for path,digest in info['bindings'].items():
        if sha256(path)!=digest:raise ValueError('Changed clipped calibration input')
    if info.get('schema')=='oa_cbf_frozen_trajectory_gate_v1' and info.get('predictive_calibration_schema')!=SCHEMA:
        raise ValueError('Different trajectory-gate predictive distribution')
