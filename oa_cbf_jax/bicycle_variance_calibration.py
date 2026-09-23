"""Reserved-parent likelihood calibration of frozen ensemble variances.

Only each head's positive aleatoric scale is fitted. Member means, ensemble
disagreement, event probabilities, physical targets and control thresholds stay
unchanged. These development fits require new model-bound trajectory gates.
"""
import argparse
import copy
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .dataset import sha256
from .io import write_json

SCALE_BOUNDS = (.01, 100.)
GRID_POINTS = 49
METHOD = 'parent_weighted_gaussian_mixture_likelihood_scale'
RUNTIME_SCHEMA = 'oa_cbf_bicycle_variance_calibration_runtime'


def validate_variant(original, candidate):
    """A calibration-only variant cannot silently change the learned method."""
    added = {'variance_calibration','derived_from_prediction_fit',
             'derived_from_prediction_fit_sha256','source_prediction_fit_cache_sha256'}
    if set(candidate) != set(original)|added:
        raise ValueError('Unexpected calibration variant fields')
    for key,value in original.items():
        if key not in ('variance_scale','limitation') and candidate[key] != value:
            raise ValueError('Changed frozen prediction semantics: '+key)
    method = candidate['variance_calibration']
    scales = np.asarray(candidate['variance_scale'])
    if (method['method'] != METHOD or method['scale_bounds'] != list(SCALE_BOUNDS)
            or method['grid_points'] != GRID_POINTS or method['physical_safety_guarantee'] is not False
            or scales.shape != (2,) or not np.isfinite(scales).all()
            or np.any(scales < np.float32(SCALE_BOUNDS[0])) or np.any(scales > np.float32(SCALE_BOUNDS[1]))):
        raise ValueError('Wrong common mixture calibration procedure')


def runtime_models(training, spec, base):
    """Bind new fits to genuine original checkpoints without claiming retraining."""
    from .bicycle_experiment import read
    if (spec['schema'] != RUNTIME_SCHEMA or spec['training_performed'] is not False
            or Path(spec['training']).resolve() != Path(training).resolve()
            or sha256(spec['base_training_complete']) != spec['base_training_complete_sha256']
            or sha256(spec['review']) != spec['review_sha256']):
        raise ValueError('Changed calibration-only runtime lineage')
    proof = read(spec['review']); root = Path(spec['calibration'])
    if (proof.get('status') != 'passed' or proof['report_sha256'] != sha256(root/'report.json')
            or not proof['all_source_model_prediction_reservation_bindings_verified']
            or not proof['independent_mixture_likelihood_minima_and_heldout_diagnostics_verified']
            or not proof['original_weights_means_event_fits_thresholds_unchanged']):
        raise ValueError('Independent variance-calibration review required')
    report = read(root/'report.json'); result = {}
    for encoder,item in base.items():
        path = root/encoder/'prediction_fit.json'; new = read(path); old = item['calibration']
        validate_variant(old,new)
        if (new['derived_from_prediction_fit_sha256'] != item['fit_sha256']
                or Path(new['derived_from_prediction_fit']).resolve() != Path(item['fit']).resolve()
                or new['weights_sha256'] != item['weights_sha256']
                or sha256(path) != report['methods'][encoder]['prediction_fit_sha256']
                or sha256(path) != read(root/encoder/'complete.json')['prediction_fit_sha256']
                or sha256(root/encoder/'prediction_audit.json') != read(root/encoder/'complete.json')['prediction_audit_sha256']):
            raise ValueError('Changed frozen variance-fit binding')
        result[encoder] = dict(item,fit=str(path.resolve()),fit_sha256=sha256(path),calibration=new)
    return result


def inputs(prediction, data, head):
    means = np.asarray(prediction['mean'], float)
    variance = np.asarray(prediction['variance'], float)
    target = np.asarray(data['target'], float)
    valid = np.asarray(data['target_mask'], bool)
    ids = np.asarray(data['group_id'])
    if (means.ndim != 4 or means.shape != variance.shape or means.shape[1:] != target.shape
            or valid.shape != target.shape or ids.shape != (len(target),)
            or means.shape[0] < 1 or not 0 <= head < target.shape[-1]
            or not np.isfinite(means).all() or not np.isfinite(variance).all()
            or np.any(variance <= 0)):
        raise ValueError('Finite positive ensemble predictions and aligned parent labels required')
    mask = valid[..., head]
    if not mask.any() or not np.isfinite(target[..., head][mask]).all():
        raise ValueError('No finite observed head targets')
    _, inverse = np.unique(ids, return_inverse=True)
    counts = np.bincount(inverse, weights=mask.sum(1))
    weights = np.broadcast_to(1/np.maximum(counts[inverse, None], 1), mask.shape)[mask]
    weights /= np.count_nonzero(counts)
    return means[..., head][:, mask], variance[..., head][:, mask], target[..., head][mask], weights


def score_from_inputs(values, log_scale):
    mean, variance, target, weights = values
    log_variance = np.log(variance)+log_scale
    log_density = -.5*((target[None]-mean)**2*np.exp(-log_variance)+log_variance+np.log(2*np.pi))
    nll = -(logsumexp(log_density, axis=0)-np.log(len(mean)))
    return float(np.dot(weights,nll))


def scores(prediction, data, scales):
    if len(scales) != data['target'].shape[-1] or not np.isfinite(scales).all() or np.any(np.asarray(scales) <= 0):
        raise ValueError('One finite positive scale per head required')
    return [score_from_inputs(inputs(prediction,data,h),np.log(scale)) for h,scale in enumerate(scales)]


def fit_scales(prediction, data):
    """Fit only observed labels, with equal total mass per physical parent.

    Scaling component variance inside the mixture leaves epistemic mean spread
    unchanged. A fixed log grid plus bounded refinements handles nonconvex
    scalar mixture likelihoods; scale one is always a candidate. Neither
    validation/audit targets nor navigation outcomes select any parameter.
    """
    if 'calibration_role' in data and not np.all(data['calibration_role'] == 'prediction_fit'):
        raise ValueError('Variance fitting requires the reserved prediction_fit role')
    lower, upper = np.log(SCALE_BOUNDS); grid = np.linspace(lower,upper,GRID_POINTS)
    scales, details = [], []
    for head in range(data['target'].shape[-1]):
        values = inputs(prediction,data,head)
        objective = lambda value: score_from_inputs(values,float(value))
        curve = np.array([objective(x) for x in grid])
        candidates = [(float(y),float(x)) for x,y in zip(grid,curve)]
        candidates.append((objective(0.),0.))
        for i in range(1,len(grid)-1):
            if curve[i] <= curve[i-1] and curve[i] <= curve[i+1]:
                fit = minimize_scalar(objective,bounds=(grid[i-1],grid[i+1]),method='bounded',
                                      options={'xatol':1e-8,'maxiter':150})
                if not fit.success or not np.isfinite(fit.fun):
                    raise ValueError('Finite bounded mixture-likelihood fit required')
                candidates.append((float(fit.fun),float(fit.x)))
        _, optimum = min(candidates)
        scale = float(np.float32(np.exp(optimum))); value = objective(np.log(scale))
        if not np.isfinite(value) or value > objective(0.)+1e-8:
            raise ValueError('Likelihood calibration worsened its reference on fit parents')
        scales.append(scale)
        details.append(dict(scale=scale,fit_nll=value,unit_scale_nll=objective(0.),
            observed_labels=len(values[2]),observed_parents=int(len(np.unique(data['group_id'][data['target_mask'][...,head].any(1)]))),
            scale_at_search_boundary=bool(np.isclose(scale,SCALE_BOUNDS[0]) or np.isclose(scale,SCALE_BOUNDS[1])),
            log_scale_grid=grid.tolist(),nll_grid=curve.tolist(),candidate_minima=len(candidates)))
    return dict(variance_scale=scales,variance_calibration=dict(method=METHOD,scale_bounds=list(SCALE_BOUNDS),
        grid_points=GRID_POINTS,heads=details,unit='one equal-weight original physical parent',
        uncertainty='Only component variances scale; member means and their disagreement remain unchanged.',
        status='fitted_development',physical_safety_guarantee=False))


def check_files(files):
    for p,digest in files.items():
        if sha256(p) != digest:raise ValueError('Changed frozen input: '+p)


def load_role(dataset, role):
    from .bicycle_experiment import read
    dataset = Path(dataset); manifest = read(dataset/'manifest.json')
    source = read(Path(manifest['source'])/'scenes.json')
    reserved = {r['group_id']:r for r in source if r['partition']=='development_calibration'}
    ids = {g for g,r in reserved.items() if r['calibration_role']==role}
    parts = []
    keys = ('group_id','query_tick','calibration_role','partition','target','target_mask','events','event_mask')
    for entry in read(dataset/'index.json'):
        if entry['group_id'] not in ids:continue
        path = dataset/entry['file']
        if sha256(path) != entry['sha256']:raise ValueError('Changed reserved label shard')
        with np.load(path) as z:
            if not (z['calibration_role']==role).all() or not (z['partition']=='development_calibration').all():
                raise ValueError('Changed prediction role')
            parts.append({k:z[k] for k in keys})
    result = {k:np.concatenate([p[k] for p in parts]) for k in keys}
    if set(result['group_id']) != ids or not ids:raise ValueError('Incomplete reserved parent set')
    return result


def prediction(path, data):
    with np.load(path) as z:
        for key in ('group_id','query_tick'):np.testing.assert_array_equal(z[key],data[key])
        result = {k:z[k] for k in ('mean','variance','event_logits')}
    # Stored candidate replicas must carry the identical network prediction.
    for key,value in result.items():np.testing.assert_array_equal(value[:,:,::2],value[:,:,1::2])
    return result


def run(protocol, output, directory):
    from .bicycle_experiment import read
    from .bicycle_predictive_calibration import fit_parameters, diagnostics
    protocol_path = Path(protocol); spec = read(protocol_path); out = Path(output); job = Path(directory)
    out.mkdir(parents=True,exist_ok=False); start = time.monotonic();check_files(spec['bound_files'])
    review = read(spec['training_review']); decision_review = read(spec['selection_review'])
    if (review.get('status')!='passed' or decision_review.get('status')!='passed'
            or not decision_review['independent_parent_weighted_strata_and_paired_changes_verified']):
        raise ValueError('Completed independent learning and selection reviews required')
    dataset = Path(spec['dataset'])
    for name,key in (('manifest','dataset_manifest_sha256'),('index','dataset_index_sha256'),('independent_replay','dataset_audit_sha256')):
        if sha256(dataset/(name+'.json'))!=review[key]:raise ValueError('Wrong reviewed calibration source')
    write_json(job/'progress.json',dict(stage='reserved_fit_parents',estimated_remaining_seconds=300))
    fit_data = load_role(dataset,'prediction_fit'); fitted = {}; report = {}
    # Freeze BOTH model-specific fits before opening either audit prediction.
    for encoder,item in spec['models'].items():
        base = Path(item['calibration']); original = read(base/'prediction_fit.json')
        raw = prediction(base/'prediction_fit_predictions.npz',fit_data)
        old = fit_parameters(raw,fit_data)
        for key in old:
            if old[key] != original[key]:raise ValueError('Original fitted parameters do not reproduce')
        candidate = copy.deepcopy(original); candidate.update(fit_scales(raw,fit_data))
        candidate.update(derived_from_prediction_fit=str(base/'prediction_fit.json'),
            derived_from_prediction_fit_sha256=sha256(base/'prediction_fit.json'),
            source_prediction_fit_cache_sha256=sha256(base/'prediction_fit_predictions.npz'),
            trajectory_gate_ready=False,production_eligible=False,
            limitation='Reserved-parent empirical mixture-likelihood component-variance fit. All means, event fits and control thresholds unchanged. '
            'Censored targets remain missing. New trajectory calibration and prospective physical evaluation required; no unconditional safety guarantee.')
        path = out/encoder/'prediction_fit.json'; write_json(path,candidate); fitted[encoder] = (candidate,sha256(path))
        report[encoder] = dict(original_scales=original['variance_scale'],new_scales=candidate['variance_scale'],
            fit=dict(original_nll=scores(raw,fit_data,original['variance_scale']),new_nll=scores(raw,fit_data,candidate['variance_scale'])),
            prediction_fit_path=str(path.resolve()),prediction_fit_sha256=sha256(path))
        print(dict(encoder=encoder,phase='fit_frozen',scales=candidate['variance_scale'],fit=report[encoder]['fit']),flush=True)
    write_json(job/'progress.json',dict(stage='reserved_audit_parents_fits_frozen',estimated_remaining_seconds=180))
    audit_data = load_role(dataset,'prediction_audit')
    if set(fit_data['group_id']) & set(audit_data['group_id']):raise ValueError('Fit and audit parents overlap')
    for encoder,item in spec['models'].items():
        base = Path(item['calibration']); original = read(base/'prediction_fit.json'); candidate,digest = fitted[encoder]
        assert sha256(out/encoder/'prediction_fit.json')==digest
        raw = prediction(base/'prediction_audit_predictions.npz',audit_data)
        old_diagnostics = diagnostics(raw,audit_data,original)
        if old_diagnostics != read(base/'prediction_audit.json')['audit']:raise ValueError('Original held-out audit does not reproduce')
        new_diagnostics = diagnostics(raw,audit_data,candidate)
        report[encoder]['audit'] = dict(original_nll=scores(raw,audit_data,original['variance_scale']),
            new_nll=scores(raw,audit_data,candidate['variance_scale']),original=old_diagnostics,candidate=new_diagnostics)
        write_json(out/encoder/'prediction_audit.json',dict(prediction_fit_sha256=digest,comparison=report[encoder],whole_goal_complete=False))
        write_json(out/encoder/'complete.json',dict(status='completed',prediction_fit_sha256=digest,
            prediction_audit_sha256=sha256(out/encoder/'prediction_audit.json'),training_performed=False,whole_goal_complete=False))
    check_files(spec['bound_files'])
    result = dict(method=METHOD,methods=report,protocol_sha256=sha256(protocol_path),
        fit_parents=len(np.unique(fit_data['group_id'])),audit_parents=len(np.unique(audit_data['group_id'])),
        fit_queries=len(fit_data['group_id']),audit_queries=len(audit_data['group_id']),
        elapsed_seconds=time.monotonic()-start,training_performed=False,benchmark_parents_used=False,
        safety_thresholds_changed=False,event_calibration_changed=False,old_artifacts_unchanged=True,
        new_trajectory_gates_required=True,model_promoted=False,whole_goal_complete=False)
    write_json(out/'report.json',result);write_json(job/'progress.json',dict(stage='completed',elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('protocol','output','directory'):p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))
