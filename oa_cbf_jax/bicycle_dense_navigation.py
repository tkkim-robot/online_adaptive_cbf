"""Frozen readout policies on a complete fresh dense moving-obstacle cohort.

Only the reserved seed/namespace changes. Neither failures nor route rejections
are excluded. This experiment compares GAT and nearest FC; native baselines
require matching-parent runs before any new all-method benchmark is claimed.
"""


from collections import Counter


from pathlib import Path


import numpy as np
from .bicycle_experiment import read,control_config
from .bicycle_random_density import geometry,geometry_metrics,GROUPS,FAMILIES,COUNTS,NOISE


from .dataset import sha256


SEED=3010044000
SCHEMA='bicycle_dense_readout_fresh_navigation'


def validate_source(source):
    from .bicycle_observation import BASE_NOISE
    source=Path(source);m=read(source/'manifest.json');parents=read(source/'scenes.json')
    if (m.get('schema')!=SCHEMA or m.get('groups')!=GROUPS or m.get('data_role')!='frozen_policy_margin_recovery_development'
            or m.get('training_use') is not False or m.get('weight_fit_authorized') is not False or m.get('final_test') is not False
            or m['reservation_sha256']!=sha256(m['reservation']) or m['scenes_sha256']!=sha256(source/'scenes.json')):
        raise ValueError('Changed fresh readout navigation reservation')
    s=read(m['reservation']);info=s['models'][m['encoder']]
    if (s['seed']!=SEED or s['parents']!=GROUPS or len(parents)!=GROUPS
            or m['phase_order']!={'policy_audit':np.random.default_rng(SEED+144).permutation(GROUPS).tolist()}
            or any(m[k]!=s[k] for k in ('config','controller','runtime_guidance'))
            or m['policy_config']!=info['policy_config'] or m['prediction_fit_sha256']!=sha256(info['prediction_fit'])
            or m['gate_sha256']!=sha256(info['gate']) or m['weights_sha256']!=sha256(Path(info['bundle'])/'weights.msgpack')):
        raise ValueError('Changed fresh readout policy/controller')
    for path,digest in s['bindings'].items():
        if sha256(path)!=digest:raise ValueError('Changed frozen readout navigation dependency')
    for i,p in enumerate(parents):
        if p['partition']!='policy_audit' or p['calibration_role']!='none' or p['weight_fit_allowed']:
            raise ValueError('Navigation parents cannot fit weights or gates')
        for k,v in geometry(i,seed=SEED,schema=SCHEMA).items():np.testing.assert_array_equal(p[k],v)
        np.testing.assert_array_equal(p['noise'],BASE_NOISE*np.float32(p['noise_level']))
    if len(set(p['group_id'] for p in parents))!=GROUPS:raise ValueError('Duplicate navigation parent')
    counts=Counter((p['family'],p['obstacle_count'],p['noise_level']) for p in parents)
    if len(counts)!=64 or set(counts.values())!={12}:raise ValueError('Incomplete navigation strata')
    return m,parents
