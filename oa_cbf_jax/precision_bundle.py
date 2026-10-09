"""Shared precision bundle implementation."""

import copy

import json

from pathlib import Path

from .io import sha256

SCHEMA='oa_cbf_graph_network64_raw32_derivation_v75'

FLIGHT_SCHEMA='oa_cbf_flight_network64_raw32_derivation_v1'

def derivation_schema(trained):
    architecture=trained['architecture']
    if trained.get('graph_features')==40:
        if (trained.get('dataset_schema')!='oa_cbf_quad2d_guided_hurdle_v1'
                or trained.get('gain_dimension')!=2 or architecture['encoder']!='gat'
                or not architecture.get('flight_history_invariant')
                or not architecture.get('flight_obstacle_pooling')
                or architecture.get('flight_local_residual')
                or not trained.get('controller',{}).get('config',{}).get('stationary_obstacles')):
            raise ValueError('Flight precision derivation requires the static guided pooled graph40 GAT')
        return FLIGHT_SCHEMA
    motion=architecture.get('bicycle_motion_history',False)
    if architecture['encoder'] not in ('gat','matched_fc','nearest_fc') or trained.get('graph_features')!=(39 if motion else 35) or trained.get('gain_dimension')!=1:
        raise ValueError('Only observed bicycle or static guided flight derivations are supported')
    return SCHEMA

def validate_derivation(root,metadata):
    root=Path(root);proof=metadata.get('numerical_derivation',{})
    if proof.get('schema') not in (SCHEMA,FLIGHT_SCHEMA) or proof.get('new_training') is not False:
        raise ValueError('FP64 inference requires an explicit genuine-weight derivation')
    original=root/'trained_manifest.json'
    if sha256(original)!=proof['source_manifest_sha256']:
        raise ValueError('Changed original training manifest')
    trained=json.loads(original.read_text())
    if proof['schema']!=derivation_schema(trained):
        raise ValueError('Numerical derivation scope does not match the training contract')
    if trained.get('numerical_derivation') or trained['architecture'].get('compute_dtype','float32')!='float32':
        raise ValueError('Derive directly from the original FP32 training artifact')
    expected=copy.deepcopy(trained);expected['architecture']['compute_dtype']='float64'
    expected.update(numerical_derivation=proof,production_eligible=False,calibration=None,
        limitation='Derived numerical inference of unchanged FP32-trained weights. Requires its own prediction fit, trajectory gate and physical validation. No new training or final promotion.')
    if metadata!=expected or proof['source_weights_sha256']!=trained['weights_sha256'] or sha256(root/'weights.msgpack')!=trained['weights_sha256']:
        raise ValueError('Numerical port changed trained parameters or model semantics')
    if trained['architecture']['encoder'] == 'nearest_fc':
        from .nearest_fc import validate_metadata
        validate_metadata(trained)
    motion=trained['architecture'].get('bicycle_motion_history',False)
    if motion:
        from .bicycle_policy import validate_metadata
        validate_metadata(trained)
    return proof
