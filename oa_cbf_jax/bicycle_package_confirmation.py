"""One fixed, prospective confirmation of the best reviewed package per encoder.

Choose packages on the complete development comparison, before new geometry.
Evaluate exactly two new balanced 768-parent cohorts; no outcome-dependent
expansion, early efficacy stopping, scene selection or parameter tuning.
"""


from collections import Counter


from pathlib import Path


import numpy as np
from .bicycle_experiment import read,control_config
from .bicycle_random_density import geometry,geometry_metrics


from .dataset import sha256


SCHEMA='bicycle_selected_package_fresh_confirmation'
SEEDS=(3110054000,3210054000)
GROUPS=1536


def fresh_geometry(index):
    if type(index) is not int or not 0<=index<GROUPS:raise ValueError('Fixed1536-parent reservation required')
    cohort,local=divmod(index,768)
    return dict(geometry(local,seed=SEEDS[cohort],schema=SCHEMA),confirmation_cohort=cohort)


def validate_source(source):
    from .bicycle_observation import BASE_NOISE
    source=Path(source);m=read(source/'manifest.json');parents=read(source/'scenes.json')
    if (m.get('schema')!=SCHEMA or m.get('groups')!=GROUPS or m.get('training_use') is not False
            or m.get('weight_fit_authorized') is not False or m.get('final_test') is not False
            or sha256(m['reservation'])!=m['reservation_sha256'] or sha256(source/'scenes.json')!=m['scenes_sha256']):
        raise ValueError('Changed fixed confirmation reservation')
    s=read(m['reservation']);info=s['models'][m['encoder']]
    if (s['seeds']!=list(SEEDS) or s['parents']!=GROUPS or len(parents)!=GROUPS or not s['fixed_sample_size']
            or s['outcome_dependent_expansion'] or m['phase_order']!={'policy_audit':np.random.default_rng(SEEDS[0]+144).permutation(GROUPS).tolist()}
            or any(m[k]!=s[k] for k in ('config','controller','runtime_guidance')) or m['policy_config']!=info['policy_config']
            or m['weights_sha256']!=sha256(Path(info['bundle'])/'weights.msgpack') or m['prediction_fit_sha256']!=sha256(info['prediction_fit'])
            or m['gate_sha256']!=sha256(info['gate'])):raise ValueError('Changed selected package or denominator')
    for path,digest in s['bindings'].items():
        if sha256(path)!=digest:raise ValueError('Changed frozen confirmation dependency')
    for i,p in enumerate(parents):
        if p['partition']!='policy_audit' or p['calibration_role']!='none' or p['weight_fit_allowed']:raise ValueError('No confirmation fitting')
        for k,v in fresh_geometry(i).items():np.testing.assert_array_equal(p[k],v)
        np.testing.assert_array_equal(p['noise'],BASE_NOISE*np.float32(p['noise_level']))
    counts=Counter((p['family'],p['obstacle_count'],p['noise_level']) for p in parents)
    if len({p['group_id'] for p in parents})!=GROUPS or len(counts)!=64 or set(counts.values())!={24}:raise ValueError('Incomplete confirmation cells')
    return m,parents
