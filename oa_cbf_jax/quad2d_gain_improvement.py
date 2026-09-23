"""Calibrate learned progress differences using development parents only.

This is a proposal admission rule, not an analytic gain search. The target is
mean recorded progress across the paired physical replicas. Observation-level
calibration does not imply an adaptive-trajectory or safety guarantee.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .dataset import load_dataset, sha256
from .io import write_json
from .uncertainty import conformal_threshold

SCHEMA = 'quad2d_paired_progress_admission_v1'
SCALE_FLOOR = 1e-3


def difference_statistics(means, variances, anchor, xp=np):
    """Ensemble axis first; a shared anchor preserves memberwise differences."""
    delta = means-means[:, anchor:anchor+1]
    center = xp.mean(delta, axis=0)
    scale = xp.sqrt(xp.mean(variances+variances[:, anchor:anchor+1], axis=0)
                    +xp.var(delta, axis=0)+SCALE_FLOOR**2)
    return center, scale


def admission(means, variances, quantile):
    """Last pool entry is the exact previous gain; all others need evidence."""
    center, scale = difference_statistics(means, variances, means.shape[1]-1, jnp)
    lower = center-quantile*scale
    accepted = jnp.isfinite(lower)&(lower>0)
    return accepted.at[-1].set(True), lower


def calibrate(bundle, dataset, base_calibration, output, coverage=.95):
    from .inference import ResearchPredictor
    root, dest = Path(dataset), Path(output)
    if dest.exists(): raise ValueError('Fresh calibration output required')
    manifest = json.loads((root/'manifest.json').read_text())
    base = json.loads(Path(base_calibration).read_text())
    if (manifest.get('data_role') != 'training' or not manifest.get('weight_fit_authorized')
            or manifest['schema'] != 'oa_cbf_quad2d_guided_hurdle_v1'
            or not manifest['config']['stationary_obstacles']
            or base['schema'] != 'oa_cbf_development_calibration_v1'
            or base['dataset_manifest_sha256'] != sha256(root/'manifest.json')):
        raise ValueError('Audited training development partition and base calibration required')
    audited = json.loads((root/'independent_replay.json').read_text())
    if (not audited['audit_passed'] or audited['manifest_sha256'] != sha256(root/'manifest.json')
            or audited['index_sha256'] != sha256(root/'index.json')):
        raise ValueError('Changed independent physical label audit')
    predictor = ResearchPredictor(bundle, allow_uncalibrated=True)
    if (predictor.metadata['weights_sha256'] != base['weights_sha256']
            or predictor.metadata['dataset_manifest_sha256'] != base['dataset_manifest_sha256']
            or predictor.metadata['controller'] != base['controller']):
        raise ValueError('Changed predictive model or controller')
    data = load_dataset(root, 'development_calibration')
    ids = data['group_id'].tolist(); replicas = manifest['replicas']
    if len(ids) != len(set(ids)) or not np.all(data['target_mask'][..., 1]):
        raise ValueError('Complete recorded progress for every reserved parent required')
    banks = data['gains'][:, ::replicas]
    if not np.array_equal(data['gains'], np.repeat(banks, replicas, axis=1)):
        raise ValueError('Unpaired gain replicas')
    anchors = np.all(banks == data['previous_gain'][:, None], axis=-1)
    if not np.all(anchors.sum(axis=-1) == 1):
        raise ValueError('Every parent needs one exactly labeled previous gain')
    anchor = anchors.argmax(axis=-1)
    target = data['target'][..., 1].reshape(len(ids), banks.shape[1], replicas).mean(axis=-1)
    print(json.dumps(dict(stage='predict_development', parents=len(ids))), flush=True)
    predictor.warm_features(len(ids), data['features'].shape[1], banks.shape[1])
    prediction = jax.tree.map(np.asarray, predictor.predict_features(data['features'], data['node_mask'], banks))
    mean = prediction['mean'][..., 1].astype(np.float64)
    variance = prediction['variance'][..., 1].astype(np.float64)*base['variance_scale'][1]
    centers, scales = zip(*(difference_statistics(mean[:, i], variance[:, i], int(a)) for i,a in enumerate(anchor)))
    centers, scales = np.stack(centers), np.stack(scales)
    actual = target-target[np.arange(len(ids)), anchor, None]
    scores = np.max((centers-actual)/scales, axis=-1)
    positions = {g:i for i,g in enumerate(ids)}
    gate_ids, audit_ids = base['group_ids']['gate'], base['group_ids']['audit']
    if (len(set(gate_ids)) != len(gate_ids) or len(set(audit_ids)) != len(audit_ids)
            or set(gate_ids)&set(audit_ids) or not (set(gate_ids)|set(audit_ids)) <= set(ids)):
        raise ValueError('Disjoint reserved development gate/audit parents required')
    gate = np.array([positions[g] for g in gate_ids]); audit = np.array([positions[g] for g in audit_ids])
    quantile = conformal_threshold(scores[gate], coverage)
    if quantile['status'] != 'calibrated': raise ValueError('Insufficient independent development parents')
    q = max(0., quantile['threshold'])
    lower = centers-q*scales
    admitted = lower>0
    actual_good = actual>0
    diagnostic = dict(audit_parents=len(audit),
        simultaneous_lower_coverage=float(np.mean(np.all(lower[audit] <= actual[audit]+1e-7, axis=-1))),
        parents_with_admitted_change=int(admitted[audit].any(axis=-1).sum()),
        admitted_candidates=int(admitted[audit].sum()),
        admitted_nonpositive_actual_candidates=int((admitted[audit]&~actual_good[audit]).sum()))
    bound = {str(Path(p).resolve()): sha256(p) for p in (root/'manifest.json', root/'index.json',
        root/'independent_replay.json', Path(bundle)/'manifest.json', Path(bundle)/'weights.msgpack', Path(base_calibration))}
    info = deepcopy(base)
    info['gain_improvement'] = dict(schema=SCHEMA, threshold=q, scale_floor=SCALE_FLOOR,
        coverage=coverage, gate=quantile, gate_group_ids=gate_ids, audit_group_ids=audit_ids,
        dataset=str(root.resolve()), weights_sha256=base['weights_sha256'], frozen_files=bound,
        diagnostic=diagnostic, actual_target='Mean recorded progress difference across common physical replicas, proposal minus previous gain.',
        statistic='Maximum normalized overprediction across all proposed gains per independent parent.',
        scope='Development observation admission only; not policy improvement, trajectory coverage, or a safety guarantee.')
    dest.mkdir(parents=True)
    np.savez_compressed(dest/'paired_scores.npz', group_id=data['group_id'], previous_gain=data['previous_gain'],
        prediction_difference=centers, scale=scales, actual_difference=actual, group_score=scores,
        lower=lower, anchor=anchor)
    info['gain_improvement']['scores_file'] = str((dest/'paired_scores.npz').resolve())
    info['gain_improvement']['scores_sha256'] = sha256(dest/'paired_scores.npz')
    write_json(dest/'calibration.json', info)
    validate(info)
    print(json.dumps(dict(stage='calibrated', threshold=q, diagnostic=diagnostic)), flush=True)


def validate(info):
    """Recompute the finite-sample rank from bound development-only evidence."""
    c = info.get('gain_improvement', {})
    if (c.get('schema') != SCHEMA or c.get('scale_floor') != SCALE_FLOOR
            or c.get('weights_sha256') != info['weights_sha256']):
        raise ValueError('Matched progress improvement calibration required')
    for file,digest in c['frozen_files'].items():
        if sha256(file) != digest: raise ValueError('Changed progress calibration evidence')
    bases = [file for file in c['frozen_files'] if Path(file).name=='calibration.json']
    if len(bases)!=1:raise ValueError('One frozen original predictive calibration required')
    base = json.loads(Path(bases[0]).read_text())
    for key in ('weights_sha256','dataset_manifest_sha256','targets','events','robot','controller',
                'variance_scale','event_calibration','gain_domain','horizon_steps'):
        if info[key]!=base[key]:raise ValueError('Changed original progress prediction transform')
    if (c['gate_group_ids']!=base['group_ids']['gate']
            or c['audit_group_ids']!=base['group_ids']['audit']):
        raise ValueError('Changed reserved improvement calibration/audit parents')
    if sha256(c['scores_file']) != c['scores_sha256']: raise ValueError('Changed paired calibration scores')
    with np.load(c['scores_file'], allow_pickle=False) as z:
        ids = z['group_id'].tolist(); positions = {g:i for i,g in enumerate(ids)}
        expected = [g for rows in base['group_ids'].values() for g in rows]
        if len(ids)!=len(set(ids)) or set(ids)!=set(expected):
            raise ValueError('Changed complete development calibration inventory')
        scores = np.max((z['prediction_difference']-z['actual_difference'])/z['scale'], axis=-1)
        np.testing.assert_allclose(scores, z['group_score'], rtol=0, atol=1e-12)
    q = conformal_threshold(scores[[positions[g] for g in c['gate_group_ids']]], c['coverage'])
    if q != c['gate'] or c['threshold'] != max(0., q['threshold']):
        raise ValueError('Changed paired progress threshold')
    return c


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('bundle','dataset','base-calibration','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--coverage',type=float,default=.95)
    calibrate(**vars(p.parse_args()))
