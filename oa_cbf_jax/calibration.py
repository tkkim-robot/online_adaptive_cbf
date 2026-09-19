"""Development calibration, with independent fit/gate/audit parent groups.

These data contain one observation per parent, not whole adaptive trajectories.
The resulting CS gate is an observation-level development screen. It is not a
trajectory-coverage, OOD-detection, CVaR-tail or collision-probability guarantee.
Continuous targets censored by controller failure stay censored throughout.
"""

import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .dataset import load_dataset, sha256
from .inference import ResearchPredictor
from .io import write_json
from .uncertainty import cs_disagreement, conformal_threshold


def grouped_mean(values, valid):
    count=valid.sum(axis=1)
    mean=np.where(valid,values,0.).sum(axis=1)/np.maximum(count,1)
    return mean[count>0].mean()


def fit_event_calibration(logits,events,valid):
    """One affine logit transform per head, fitted on equally weighted groups."""
    settings=[]
    for head in range(events.shape[-1]):
        z=logits[...,head];y=events[...,head];mask=valid[...,head]
        def objective(theta):
            probability=expit(z/np.exp(theta[0])+theta[1]).mean(axis=0)
            probability=np.clip(probability,1e-10,1-1e-10)
            bce=-y*np.log(probability)-(1-y)*np.log1p(-probability)
            return grouped_mean(bce,mask)
        result=minimize(objective,np.zeros(2),method='L-BFGS-B',bounds=[(np.log(.2),np.log(5.)),(-5.,5.)])
        if not result.success or not np.isfinite(result.fun):raise ValueError(f'Event calibration failed: {result.message}')
        settings.append(dict(temperature=float(np.exp(result.x[0])),bias=float(result.x[1]),fit_bce=float(result.fun)))
    return settings


def calibrate(bundle,dataset,output,coverage=.95,seed=71891):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    model=ResearchPredictor(bundle,allow_uncalibrated=True)
    manifest=json.loads((Path(dataset)/'manifest.json').read_text())
    if model.metadata.get('controller',dict(sensor_margin_scale=0.))!=manifest.get('controller',dict(sensor_margin_scale=0.)):
        raise ValueError('Calibration controller/target contract mismatch')
    history_schema='oa_cbf_quad2d_motion_history_hurdle_v1'
    flight_schemas=('oa_cbf_quad2d_initial_flight_v1','oa_cbf_quad2d_initial_hurdle_v2','oa_cbf_quad2d_guided_hurdle_v1',history_schema)
    if manifest['schema'] not in ('oa_cbf_route_sensor_v4','oa_cbf_route_continuation_v5','oa_cbf_route_continuation_v6',*flight_schemas) or model.metadata['dataset_manifest_sha256']!=sha256(Path(dataset)/'manifest.json'):
        raise ValueError('Calibration requires an exact supported observation-conditioned training contract')
    if manifest['schema']==history_schema:
        from .quad2d_history_contract import validate_dataset
        validate_dataset(dataset)
        if model.metadata.get('graph_features')!=50 or model.metadata.get('dataset_schema')!=history_schema:
            raise ValueError('Matched observed-history graph50 model required')
    elif manifest['schema'] in flight_schemas:
        replay=json.loads((Path(dataset)/'independent_replay.json').read_text())
        if not replay['audit_passed'] or not replay.get('all_collision_bound_branches_audited') or not (replay.get('all_initial_graph_features_independently_checked') or replay.get('all_observed_graph_features_independently_checked')) or replay['manifest_sha256']!=sha256(Path(dataset)/'manifest.json') or replay['index_sha256']!=sha256(Path(dataset)/'index.json'):
            raise ValueError('Independently audited flight data required')
        if manifest['schema']=='oa_cbf_quad2d_guided_hurdle_v1' and not (replay.get('all_observed_graph_features_independently_checked') and replay.get('all_guidance_approvals_checked')):
            raise ValueError('Guided observation and approval audit required')
    data=load_dataset(dataset,'development_calibration');n=len(data['group_id'])
    if len(set(data['group_id']))!=n:raise ValueError('Calibration units must be unique independent parent groups')
    permutation=np.random.default_rng(seed).permutation(n)
    fit,gate,audit=np.split(permutation,[n//2,3*n//4])
    if min(len(fit),len(gate),len(audit))<50:raise ValueError('Insufficient independent calibration groups')
    # One explicit warmed signature; all ensemble parameters remain runtime inputs.
    model.warm_features(n,data['features'].shape[1],data['gains'].shape[1])
    prediction=jax.tree.map(np.asarray,model.predict_features(data['features'],data['node_mask'],data['gains']))
    means=prediction['mean'].astype(np.float64);variances=prediction['variance'].astype(np.float64)
    mixture_mean=means.mean(axis=0)
    total_variance=variances.mean(axis=0)+np.var(means,axis=0)
    errors=data['target']-mixture_mean
    scale=[]
    for head in range(2):
        ratio=errors[fit,:,head]**2/np.maximum(total_variance[fit,:,head],1e-8)
        scale.append(max(1.,float(grouped_mean(ratio,data['target_mask'][fit,:,head]))))
    probability=np.clip(prediction['event_probability'].astype(np.float64),1e-8,1-1e-8)
    logits=np.log(probability)-np.log1p(-probability)
    event=fit_event_calibration(logits[:,fit],data['events'][fit],data['event_mask'][fit])
    calibrated_variance=variances*np.asarray(scale)
    # Gate groups were not used to fit the predictive variance/probabilities.
    risk_mean=np.transpose(means[...,:1],(1,2,0,3))
    risk_variance=np.transpose(calibrated_variance[...,:1],(1,2,0,3))
    scores=np.asarray(cs_disagreement(jnp.asarray(risk_mean),jnp.asarray(risk_variance)))
    maximum=scores.max(axis=1)
    threshold=conformal_threshold(maximum[gate],coverage)
    if threshold['status']!='calibrated':raise ValueError('Insufficient gate calibration; refusing usable artifact')
    calibrated_probability=expit(logits/np.array([e['temperature'] for e in event])+np.array([e['bias'] for e in event])).mean(axis=0)
    diag=dict(audit_groups=len(audit),gate_acceptance=float(np.mean(maximum[audit]<=threshold['threshold'])),
              first_event_brier=[float(grouped_mean((calibrated_probability[audit,:,h]-data['events'][audit,:,h])**2,data['event_mask'][audit,:,h])) for h in range(2)],
              marginal_observed_upper_95_coverage=[])
    # Group-average marginal coverage is a diagnostic. Censored outcomes are
    # excluded explicitly; this does not assert an unconditional safety bound.
    calibrated_total=calibrated_variance.mean(axis=0)+np.var(means,axis=0)
    for h in range(2):
        upper=mixture_mean[audit,:,h]+1.6448536269514722*np.sqrt(calibrated_total[audit,:,h])
        diag['marginal_observed_upper_95_coverage'].append(float(grouped_mean((data['target'][audit,:,h]<=upper).astype(float),data['target_mask'][audit,:,h])))
    result=dict(schema='oa_cbf_development_calibration_v1',stage='development_observation_screen',production_eligible=False,
                 weights_sha256=model.metadata['weights_sha256'],dataset_manifest_sha256=model.metadata['dataset_manifest_sha256'],
                 targets=model.metadata['targets'],events=model.metadata['events'],robot=manifest['config'],horizon_steps=manifest['horizon_steps'],
                 gain_domain=model.metadata.get('gain_domain',dict(lower=.3,upper=4.)),
                 controller=model.metadata.get('controller',dict(sensor_margin_scale=0.)),
                 variance_scale=scale,event_calibration=event,cs_gate=threshold,seed=seed,
                 group_ids={name:data['group_id'][idx].tolist() for name,idx in [('fit',fit),('gate',gate),('audit',audit)]},
                 diagnostics=diag,
                 interpretation='CS maximum over the recorded candidate set at ONE observation per independent parent. Continuous calibration is conditional on target observability. No adaptive-trajectory, OOD, physical-CVaR or collision-probability guarantee.',
                 required_before_final_claim='Fresh predictive calibration and independent trajectory-level gate calibration after policy selection, then locked ID/OOD evaluation.')
    if 'scene_distribution' in manifest:
        result.update(obstacle_capacity=manifest['capacity'],scene_distribution=manifest['scene_distribution'])
    if manifest['schema'] in flight_schemas:
        result.update(dynamics='Quad2D',graph_features=50 if manifest['schema']==history_schema else 40,dataset_schema=manifest['schema'])
    write_json(root/'calibration.json',result)
    np.savez_compressed(root/'diagnostics.npz',group_id=data['group_id'],cs=scores,group_maximum=maximum,
                        calibrated_mean=mixture_mean,calibrated_variance=calibrated_total,calibrated_event_probability=calibrated_probability)
    print(json.dumps(dict(variance_scale=scale,cs_gate=threshold,diagnostics=diag,event_calibration=event)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',required=True);parser.add_argument('--dataset',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--coverage',type=float,default=.95);parser.add_argument('--seed',type=int,default=71891)
    calibrate(**vars(parser.parse_args()))
