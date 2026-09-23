"""Shared prediction and observed-history learning contracts."""
import copy
from .dataset import sha256


def prediction_contract(manifest, reference):
    if (manifest['schema'] != 'oa_cbf_bicycle_acquired_history_hurdle_v67'
            or reference['architecture']['encoder'] != 'gat'
            or manifest.get('weight_fit_authorized') is not True):
        raise ValueError('Audited acquired bicycle data and OA reference required')
    for key in ('config', 'sensor_schema', 'graph_schema', 'horizon_steps', 'capacity',
                'acquisition_mode', 'snapshot_ticks', 'replicas'):
        if manifest[key] != reference['bicycle_contract'][key]:
            raise ValueError('Changed bicycle acquisition/prediction contract: ' + key)
    for key in ('controller', 'gain_domain', 'targets', 'events', 'graph_features'):
        if manifest[key] != reference[key]:
            raise ValueError('Changed bicycle shared learning/controller treatment: ' + key)
    if manifest['gain_dimension'] != 1 or manifest['queries'] != 8:
        raise ValueError('Expected the shared eight-candidate scalar-gain design')
    return True


def verify_motion_only_settings(before,after):
    from .bicycle_motion_features import contract,SCHEMA
    left,right=copy.deepcopy(before),copy.deepcopy(after)
    if left['graph_features']!=35 or right.pop('graph_features')!=39:
        raise ValueError('Motion study requires original35 to observed39 columns')
    left.pop('graph_features')
    if right['architecture'].pop('bicycle_motion_history') is not True:
        raise ValueError('Missing explicit motion feature treatment')
    if left['architecture'].pop('bicycle_motion_history',False) is not False:
        raise ValueError('Original model already has motion treatment')
    for side in (left,right):
        if side['architecture'].get('bicycle_affine_gain',False) is not False:
            raise ValueError('Rejected affine-gain feature must remain disabled')
        side['architecture'].pop('bicycle_affine_gain',None)
    if right.pop('bicycle_motion_history_contract')!=contract() or right.pop('offline_motion_history_pilot') is not True:
        raise ValueError('Missing observed-only feature contract')
    if right['bicycle_contract']['graph_schema']!=SCHEMA:raise ValueError('Wrong history graph schema')
    right['bicycle_contract']['graph_schema']=right['bicycle_contract'].pop('source_graph_schema')
    source=right.pop('bicycle_motion_history_source')
    if right.pop('bicycle_motion_history_source_sha256')!=sha256(source):raise ValueError('Changed history source')
    right.pop('motion_features_sha256')
    for side in (left,right):
        for key in ('device','source_fingerprint'):side.pop(key)
    if left!=right:
        bad=[k for k in left.keys()|right.keys() if left.get(k)!=right.get(k)]
        raise ValueError('Motion pilot changed other learning treatment: '+', '.join(sorted(bad)))
    return True
