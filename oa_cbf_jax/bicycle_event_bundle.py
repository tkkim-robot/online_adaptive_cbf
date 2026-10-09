"""Bicycle event bundle functions and shared contracts."""

from copy import deepcopy

from pathlib import Path

import numpy as np

from flax import serialization

from .bicycle_control import read

from .io import sha256

from .io import write_json

SCHEMA='bicycle_frozen_encoder_adverse_readout_v1'

LIMITATION=('Only the final adverse-event readout was refitted on reserved TRAIN parents and selected on validation. '
            'Original encoder/member provenance is retained separately from this refit. '
            'Requires its own reviewed prediction calibration, trajectory gate and physical navigation evaluation.')

def replace_readouts(source,heads):
    """Retain serialized source dtypes and all leaves except output column 5."""
    heads=np.asarray(heads)
    kernel=np.asarray(source['output']['kernel']);bias=np.asarray(source['output']['bias'])
    if (kernel.ndim!=3 or kernel.shape[0]!=4 or kernel.shape[-1]!=6 or bias.shape!=(4,6)
            or heads.shape!=(4,kernel.shape[1]+1) or heads.dtype!=np.float32
            or not np.isfinite(heads).all()):raise ValueError('Invalid four-member adverse readouts')
    result=deepcopy(source)
    result['output']['kernel'][...,5]=heads[:,:-1]
    result['output']['bias'][...,5]=heads[:,-1]
    return result

def exact_tree(actual,expected):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError('Changed exported parameter tree')
        for key in expected:exact_tree(actual[key],expected[key])
    else:
        a,b=np.asarray(actual),np.asarray(expected)
        if a.dtype!=b.dtype or a.shape!=b.shape or not np.isfinite(a).all() or not np.array_equal(a,b):
            raise ValueError('Export changed frozen weights or selected readout')

def expected_metadata(original,contract,digest):
    m=deepcopy(original);m.pop('numerical_derivation',None)
    m.update(event_readout_refit=contract,weights_sha256=digest,calibration=None,production_eligible=False,limitation=LIMITATION)
    return m

def reviewed(refit,name):
    refit=Path(refit);r=read(refit/'report.json');proof=read(refit/'independent_completion_review.json')
    if (name not in ('gat','nearest_fc') or not proof['review_passed'] or proof['report_sha256']!=sha256(refit/'report.json')
            or proof['calibration_parents_used'] or r['calibration_parents_used'] or r['model_promoted']):
        raise ValueError('Reviewed TRAIN/validation-only refit required')
    p=read(refit/'protocol.json')
    if r['protocol_sha256']!=sha256(refit/'protocol.json'):raise ValueError('Changed refit contract')
    for path,digest in p['bindings'].items():
        if sha256(path)!=digest:raise ValueError('Changed refit source evidence')
    m=r['methods'][name];source=Path(m['source_bundle']);original=read(source/'manifest.json')
    if (original.get('event_readout_refit') or original['architecture']['encoder']!=name
            or original.get('graph_features')!=35 or original.get('gain_dimension')!=1
            or m['source_manifest_sha256']!=sha256(source/'manifest.json')
            or m['source_weights_sha256']!=sha256(source/'weights.msgpack')
            or original['weights_sha256']!=m['source_weights_sha256']):raise ValueError('Changed original bicycle bundle')
    if original['architecture'].get('compute_dtype','float32')=='float64':
        from .precision_bundle import validate_derivation
        validate_derivation(source,original)
    if m['candidate_sha256']!=sha256(refit/name/'candidate.npz'):raise ValueError('Changed selected readouts')
    with np.load(refit/name/'candidate.npz') as z:heads=z['readouts']
    raw=serialization.msgpack_restore((source/'weights.msgpack').read_bytes())
    return source,original,heads,raw

def export(refit,name,output):
    refit=Path(refit).resolve();root=Path(output)
    source,original,heads,raw=reviewed(refit,name)
    updated=replace_readouts(raw,heads)
    root.mkdir(parents=True,exist_ok=False)
    (root/'source_manifest.json').write_bytes((source/'manifest.json').read_bytes())
    (root/'weights.msgpack').write_bytes(serialization.msgpack_serialize(updated))
    contract=dict(schema=SCHEMA,method=name,refit=str(refit),report_sha256=sha256(refit/'report.json'),
        review_sha256=sha256(refit/'independent_completion_review.json'),candidate_sha256=sha256(refit/name/'candidate.npz'),
        source_bundle=str(source.resolve()),source_manifest_sha256=sha256(source/'manifest.json'),
        source_weights_sha256=sha256(source/'weights.msgpack'),new_training=True,
        unchanged_encoder=True,changed_leaves='output.kernel[...,5] and output.bias[...,5] only',
        compute_dtype=original['architecture'].get('compute_dtype','float32'),calibration_reuse_allowed=False)
    metadata=expected_metadata(original,contract,sha256(root/'weights.msgpack'))
    write_json(root/'manifest.json',metadata);validate(root,metadata)
    return root

def validate(root,metadata):
    root=Path(root);c=metadata.get('event_readout_refit',{})
    if (c.get('schema')!=SCHEMA or c.get('new_training') is not True or not c.get('unchanged_encoder')
            or c.get('calibration_reuse_allowed') is not False):raise ValueError('Explicit readout refit lineage required')
    refit=Path(c['refit']);name=c['method'];source,original,heads,raw=reviewed(refit,name)
    for path,digest in [(refit/'report.json',c['report_sha256']),
            (refit/'independent_completion_review.json',c['review_sha256']),
            (refit/name/'candidate.npz',c['candidate_sha256']),
            (source/'manifest.json',c['source_manifest_sha256']),
            (source/'weights.msgpack',c['source_weights_sha256']),
            (root/'source_manifest.json',c['source_manifest_sha256']),
            (root/'weights.msgpack',metadata['weights_sha256'])]:
        if sha256(path)!=digest:raise ValueError('Changed readout bundle binding')
    if (Path(c['source_bundle']).resolve()!=source.resolve()
            or c['compute_dtype']!=original['architecture'].get('compute_dtype','float32')
            or metadata!=expected_metadata(original,c,metadata['weights_sha256'])):
        raise ValueError('Readout refit changed encoder, physical or numerical semantics')
    exact_tree(serialization.msgpack_restore((root/'weights.msgpack').read_bytes()),replace_readouts(raw,heads))
    return c


def validate_fit(fit, bundle):
    """Authenticate the completed calibration, without rewriting frozen fits.

    This permits research inference. A separate trajectory gate remains required
    for adaptive navigation; neither this binding nor calibration grants safety.
    """
    from .bicycle_event_bundle import validate
    bundle=Path(bundle).resolve();meta=read(bundle/'manifest.json')
    contract=validate(bundle,meta);name=contract['method'];root=bundle.parent.parent
    recorded=root/name/'prediction_fit.json';report=read(root/'report.json')
    proof=read(root/'independent_completion_review.json');frozen=read(root/'both_fits_frozen.json')
    flags=('review_passed','all_bundle_refit_and_label_bindings_checked',
           'all_roles_and_saved_query_orders_checked','all_density_noise_scores_independently_recomputed',
           'all_moment_scales_and_event_fit_losses_recomputed','frozen_nonadverse_predictions_exact_on_audit')
    if (not all(proof.get(k) is True for k in flags)
            or proof['report_sha256']!=sha256(root/'report.json')
            or proof['both_fits_frozen_sha256']!=sha256(root/'both_fits_frozen.json')
            or report['both_fits_frozen_sha256']!=sha256(root/'both_fits_frozen.json')
            or not report['audit_read_after_both_fits_frozen'] or report['network_training_performed']):
        raise ValueError('Completed independent readout calibration review required')
    if (set(frozen)!={'gat','nearest_fc'} or fit!=read(recorded)
            or any(sha256(root/n/'prediction_fit.json')!=h for n,h in frozen.items())):
        raise ValueError('Changed frozen readout prediction fit')
    protocol=read(root/'protocol.json');qualification=read(root/'numerical_qualification.json')
    if (report['protocol_sha256']!=sha256(root/'protocol.json')
            or qualification['protocol_sha256']!=report['protocol_sha256']
            or report['numerical_qualification_sha256']!=sha256(root/'numerical_qualification.json')
            or fit['runtime_qualification_sha256']!=report['numerical_qualification_sha256']):
        raise ValueError('Changed readout numerical/calibration protocol')
    for path,digest in protocol['bindings'].items():
        if sha256(path)!=digest:raise ValueError('Changed readout calibration evidence')
    q=qualification[name]
    if (not qualification['qualification_passed'] or q['backend']!='cpu' or q['calibration_parents_used']
            or q['queries']!=1787 or not all(q[k] for k in ('all_observed_graphs_checked',
               'all_nonadverse_outputs_exact','all_refit_logits_reproduced','all_frozen_weights_exact'))
            or q['prediction_sha256']!=sha256(root/name/'qualification_predictions.npz')
            or q['bundle_manifest_sha256']!=sha256(bundle/'manifest.json')
            or q['weights_sha256']!=meta['weights_sha256'] or fit['event_readout_refit']!=contract
            or fit['bundle_manifest_sha256']!=q['bundle_manifest_sha256']
            or fit['weights_sha256']!=q['weights_sha256'] or not fit['dense_readout_calibration']
            or fit['runtime_ready'] or fit['trajectory_gate_ready'] or fit['production_eligible']):
        raise ValueError('Unqualified or changed readout inference bundle')
    roles=fit['group_ids'];a=set(roles['prediction_fit']);b=set(roles['prediction_audit'])
    if len(a)!=64 or len(b)!=64 or a&b:raise ValueError('Readout calibration roles overlap')
    return dict(calibration_root=str(root),method=name,review_sha256=sha256(root/'independent_completion_review.json'))

def validate_gate(path, fit, bundle):
    """Bind an adaptive refit to its independently reviewed reference gate."""
    path=Path(path).resolve();root=path.parent.parent
    proof=read(root/'independent_completion_review.json');report=read(root/'report.json');gate=read(path)
    name=read(Path(bundle)/'manifest.json')['event_readout_refit']['method']
    flags=('review_passed','all_source_fit_and_model_bindings_checked',
           'all_parent_roles_and_disjointness_checked','all_physical_replay_bindings_checked',
           'all_raw_maxima_and_family_ranks_recomputed','all_reference_actions_fixed',
           'identical_reference_physics_across_models')
    if (not all(proof.get(k) is True for k in flags) or proof['report_sha256']!=sha256(root/'report.json')
            or proof['protocol_sha256']!=sha256(root/'inputs/protocol.json')
            or report['protocol_sha256']!=proof['protocol_sha256']
            or path!=Path(report['methods'][name]['gate']).resolve()
            or proof['methods'][name]['gate_sha256']!=sha256(path)
            or report['methods'][name]['gate_sha256']!=sha256(path)
            or gate['runtime_guidance']!=fit['runtime_guidance']
            or gate['weights_sha256']!=fit['weights_sha256']
            or gate['prediction_fit_sha256']!=sha256(Path(bundle).parent/'prediction_fit.json')):
        raise ValueError('Changed or unreviewed readout trajectory gate')
    return gate
