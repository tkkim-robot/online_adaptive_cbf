"""Reserved shared-policy calibration coverage; never neural weight fitting.

Both frozen encoders visit every predictive-fit/audit parent. Eight stratified
uniform live-query samples per actor retain all outcomes and causal memories.
The held labels still span eight seconds. No final-forward parent is used.
"""


import json


from pathlib import Path


import numpy as np
from .dataset import sha256


from .local_unicycle_calibration import validate_calibration_data
from .local_unicycle_observer import contract


from .local_unicycle_prediction_attribution import reviewed_navigation,check_bindings

SCHEMA='local_unicycle_reserved_policy_coverage_v1'
MODES=('gat','nearest_fc')
VISITS=8


def read(path):return json.loads(Path(path).read_text())


def validate_source(dataset):
    root=Path(dataset);m=read(root/'manifest.json');base=Path(m['base_dataset']);validate_calibration_data(base)
    check_bindings(m['bindings'])
    if (m['schema']!=SCHEMA or m['modes']!=list(MODES) or m['visits']!=VISITS or m['position_observer']!=contract()
        or m['weight_fit_authorized'] or m['forward_parents_used'] or m['trajectory_gate_parents_used']
        or m['collector_sha256']!=sha256(__file__) or m['numpy_version']!=np.__version__):raise ValueError('Changed reserved collection contract')
    for name,digest in m['helper_sha256'].items():
        if sha256(Path(__file__).with_name(name))!=digest:raise ValueError('Changed physical helper')
    records=read(base/'records.json')
    if len(set(m['indices']))!=len(m['indices']) or any(records[i]['partition'] not in ('predictive_fit','predictive_audit') for i in m['indices']):
        raise ValueError('Duplicate or forbidden parent')
    if m['pilot'] and any(records[i]['partition']!='predictive_fit' for i in m['indices']):raise ValueError('Pilot cannot select on audit parents')
    return m,base,records
