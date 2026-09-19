"""Lossless, explicitly derived numerical inference of authentic trained GATs.

The source weight bytes and full training manifest stay intact. A precision port
is neither a new training run nor permission to reuse the source calibration.
"""
import argparse
import copy
import json
from pathlib import Path
from .dataset import sha256
from .io import write_json

SCHEMA='oa_cbf_graph_network64_raw32_derivation_v75'


def validate_derivation(root,metadata):
    root=Path(root);proof=metadata.get('numerical_derivation',{})
    if proof.get('schema')!=SCHEMA or proof.get('new_training') is not False:
        raise ValueError('FP64 inference requires an explicit genuine-weight derivation')
    original=root/'trained_manifest.json'
    if sha256(original)!=proof['source_manifest_sha256']:
        raise ValueError('Changed original training manifest')
    trained=json.loads(original.read_text())
    if trained.get('numerical_derivation') or trained['architecture'].get('compute_dtype','float32')!='float32':
        raise ValueError('Derive directly from the original FP32 training artifact')
    expected=copy.deepcopy(trained);expected['architecture']['compute_dtype']='float64'
    expected.update(numerical_derivation=proof,production_eligible=False,calibration=None,
        limitation='Derived numerical inference of unchanged FP32-trained weights. Requires its own prediction fit, trajectory gate and physical validation. No new training or final promotion.')
    if metadata!=expected or proof['source_weights_sha256']!=trained['weights_sha256'] or sha256(root/'weights.msgpack')!=trained['weights_sha256']:
        raise ValueError('Numerical port changed trained parameters or model semantics')
    if trained['architecture']['encoder']!='gat' or trained.get('graph_features')!=35 or trained.get('gain_dimension')!=1:
        raise ValueError('Only the observed bicycle GAT port is qualified here')
    return proof


def derive(source,output):
    source=Path(source);root=Path(output)
    from .inference import ResearchPredictor
    predictor=ResearchPredictor(source,allow_uncalibrated=True)
    trained=predictor.metadata
    if predictor.model.config.compute_dtype!='float32' or trained.get('numerical_derivation'):
        raise ValueError('Source must be an original FP32-trained bundle')
    import jax
    import numpy as np
    if any(np.asarray(v).dtype!=np.float32 or not np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(predictor.params)):
        raise ValueError('Expected finite genuine FP32 parameter leaves')
    root.mkdir(parents=True,exist_ok=False)
    (root/'weights.msgpack').write_bytes((source/'weights.msgpack').read_bytes())
    (root/'trained_manifest.json').write_bytes((source/'manifest.json').read_bytes())
    metadata=copy.deepcopy(trained);metadata['architecture']['compute_dtype']='float64'
    metadata.update(numerical_derivation=dict(schema=SCHEMA,source=str(source.resolve()),source_manifest_sha256=sha256(source/'manifest.json'),
        source_weights_sha256=sha256(source/'weights.msgpack'),new_training=False,
        arithmetic='Observed graph geometry and weight arithmetic FP64; gain-bank/log basis FP32; raw neural outputs rounded FP32 before original calibration/selection.',
        observed_inputs_only=True,global_jax_x64_enabled=False),production_eligible=False,calibration=None,
        limitation='Derived numerical inference of unchanged FP32-trained weights. Requires its own prediction fit, trajectory gate and physical validation. No new training or final promotion.')
    validate_derivation(root,metadata);write_json(root/'manifest.json',metadata)
    print(json.dumps(dict(stage='precision_bundle_derived',bundle=str(root),weights_sha256=metadata['weights_sha256'],new_training=False)),flush=True)
    return root


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);derive(**vars(p.parse_args()))
