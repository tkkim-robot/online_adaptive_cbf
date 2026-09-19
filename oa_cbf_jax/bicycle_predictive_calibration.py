"""Reserved-parent prediction fit/audit; trajectory gating is a separate stage."""
import argparse
from pathlib import Path
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .bicycle_data import validate_training_dataset,SCHEMA as DATA_SCHEMA
from .bicycle_features import SCHEMA as GRAPH_SCHEMA
from .bicycle_observation import SCHEMA as SENSOR_SCHEMA
from .bicycle_experiment import read
from .metrics import parent_mean
from .dataset import load_dataset,sha256
from .inference import ResearchPredictor
from .models import predict_ensemble
from .io import write_json

SCHEMA='oa_cbf_bicycle_reserved_predictive_v70'


def reserved_role_groups(ids,roles,reserved):
    """Require every preassigned parent; derive counts from frozen reservation."""
    if set(ids)!=set(reserved):raise ValueError('Missing reserved calibration parent')
    for group,role in zip(ids,roles,strict=True):
        if role!=reserved[group]['calibration_role']:raise ValueError('Changed prediction fit/audit reservation')
    selections={role:np.flatnonzero(roles==role) for role in ('prediction_fit','prediction_audit')}
    groups={role:sorted(set(ids[idx])) for role,idx in selections.items()}
    expected={role:sorted(g for g,p in reserved.items() if p['calibration_role']==role) for role in groups}
    if groups!=expected or any(not g for g in groups.values()) or set(groups['prediction_fit'])&set(groups['prediction_audit']) or sum(map(len,groups.values()))!=len(reserved):
        raise ValueError('Expected complete disjoint prespecified prediction roles')
    return selections,groups


def validate_model(metadata,manifest):
    if metadata.get('dataset_schema')!=DATA_SCHEMA or metadata.get('gain_dimension')!=1 or metadata.get('graph_features')!=35 or metadata['architecture']['encoder'] not in ('gat','full_fc'):
        raise ValueError('Observed scalar-gain bicycle GAT or complete-input FC required')
    if metadata.get('bicycle_contract',{}).get('sensor_schema')!=SENSOR_SCHEMA or metadata['bicycle_contract'].get('graph_schema')!=GRAPH_SCHEMA:
        raise ValueError('Wrong bicycle sensing/feature contract')
    for field in ('config','sensor_schema','graph_schema','horizon_steps','capacity','replicas','snapshot_ticks','acquisition_mode'):
        if metadata['bicycle_contract'].get(field)!=manifest.get(field):raise ValueError('Bicycle contract mismatch: '+field)
    for field in ('targets','events','controller','gain_domain'):
        if metadata[field]!=manifest[field]:raise ValueError('Changed bicycle prediction semantics: '+field)


def fit_parameters(prediction,data):
    means=prediction['mean'].astype(float);variances=prediction['variance'].astype(float);logits=prediction['event_logits'].astype(float)
    ids=data['group_id'];average=means.mean(0);total=variances.mean(0)+means.var(0)
    scales=[]
    for h in range(2):
        ratio=(data['target'][...,h]-average[...,h])**2/np.maximum(total[...,h],1e-8)
        value=parent_mean(ratio,ids,data['target_mask'][...,h])
        if value is None or not np.isfinite(value):raise ValueError('No finite observed prediction calibration labels')
        scales.append(float(np.float32(max(1.,value))))
    event=[]
    for h in range(2):
        z=logits[...,h].max(0);y=data['events'][...,h];valid=data['event_mask'][...,h]
        if not valid.any() or not np.isfinite(z).all() or not np.isin(y[valid],[0.,1.]).all():raise ValueError('Invalid event calibration labels')
        positive=len(np.unique(ids[np.any(valid&(y>0),axis=1)]));negative=len(np.unique(ids[np.any(valid&(y==0),axis=1)]))
        info=dict(positive_parents=positive,negative_parents=negative,observed_targets=int(valid.sum()),positive_targets=int(np.sum(y[valid])),statistic='maximum_member_probability')
        if min(positive,negative)<2:
            # An all-negative collision head is not made "calibrated" by moving
            # its intercept toward -infinity. It is excluded from failure budgeting.
            event.append(dict(**info,status='insufficient_event_support',temperature=1.,bias=0.,usable_for_failure_budget=False));continue
        def objective(theta):
            value=z/np.exp(theta[0])+theta[1]
            return parent_mean(np.logaddexp(0.,value)-y*value,ids,valid)
        fit=minimize(objective,np.zeros(2),method='L-BFGS-B',bounds=[(np.log(.2),np.log(5.)),(-5.,5.)])
        if not fit.success or not np.isfinite(fit.fun):raise ValueError('Parent-weighted event fit failed: '+fit.message)
        event.append(dict(**info,status='fitted_development',temperature=float(np.float32(np.exp(fit.x[0]))),bias=float(np.float32(fit.x[1])),fit_bce=float(fit.fun),usable_for_failure_budget=h==1))
    if not event[1]['usable_for_failure_budget']:raise ValueError('Adverse-event head lacks development support')
    return dict(variance_scale=scales,event_calibration=event)


def diagnostics(prediction,data,fit):
    means=prediction['mean'].astype(float);variances=prediction['variance'].astype(float)*np.array(fit['variance_scale']);ids=data['group_id']
    mu=means.mean(0);variance=variances.mean(0)+means.var(0)
    temperature=np.array([e['temperature'] for e in fit['event_calibration']]);bias=np.array([e['bias'] for e in fit['event_calibration']])
    probability=expit(prediction['event_logits'].astype(float)/temperature+bias).max(0)
    return dict(parents=len(np.unique(ids)),queries=len(ids),observed_upper95_coverage=[parent_mean(data['target'][...,h]<=mu[...,h]+1.6448536269514722*np.sqrt(variance[...,h]),ids,data['target_mask'][...,h]) for h in range(2)],
        event_brier=[parent_mean((probability[...,h]-data['events'][...,h])**2,ids,data['event_mask'][...,h]) for h in range(2)],
        adverse_observed_rate=parent_mean(data['events'][...,1],ids,data['event_mask'][...,1]),
        limitation='Parent-weighted observed-label diagnostics. Censored futures stay missing. Not unconditional tail/physical CVaR, OOD, trajectory or rare-collision confidence.')


def calibrate(bundle,dataset,output,batch=64):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);dataset=Path(dataset);m=validate_training_dataset(dataset)
    model=ResearchPredictor(bundle,allow_uncalibrated=True);validate_model(model.metadata,m)
    if sha256(dataset/'manifest.json')!=model.metadata['dataset_manifest_sha256']:raise ValueError('Wrong weight-training lineage')
    data=load_dataset(dataset,'development_calibration');ids=data['group_id'];roles=data['calibration_role']
    graph_proof=dict(mode='original_stored_fp32',compiled_graph_signatures=0)
    if model.model.config.compute_dtype=='float64':
        # Reconstruct from actual saved observations/history, never the adjacent
        # latent physical state or obstacle arrays. Same numeric path as runtime.
        from .bicycle_features import bicycle_inference_graph
        from .bicycle_experiment import control_config
        c=control_config(m['config'])
        fields=('observed_state','goal','observed_obstacles','obstacle_mask','points','route_mask','cursor','previous_control','previous_gain','noise')
        def graph(*a):return jax.vmap(lambda *v:bicycle_inference_graph(*v,config=c,compute_dtype='float64'))(*a)
        graph_fn=jax.jit(graph)
        def graph_args(start):
            return tuple(jnp.asarray(np.pad(data[k][start:start+batch],((0,max(0,batch-len(ids[start:start+batch]))),)+((0,0),)*(data[k].ndim-1)),dtype=bool if k in ('obstacle_mask','route_mask') else jnp.float32) for k in fields)
        start=time.monotonic();graph_exe=graph_fn.lower(*graph_args(0)).compile();cold_graph=time.monotonic()-start
        features=[];masks=[]
        for i in range(0,len(ids),batch):
            f,mask=jax.device_get(graph_exe(*graph_args(i)));n=min(batch,len(ids)-i);features.append(f[:n]);masks.append(mask[:n])
        regenerated=np.concatenate(features);np.testing.assert_array_equal(np.concatenate(masks),data['node_mask'])
        graph_proof=dict(mode='observed_context_fp64_reconstruction',input_fields=list(fields),compiled_graph_signatures=1,compile_seconds=cold_graph,
            implicit_jit_cache_entries=graph_fn._cache_size(),maximum_change_from_stored_fp32=float(np.max(np.abs(regenerated-data['features']))))
        if graph_fn._cache_size():raise ValueError('Unexpected graph JIT during calibration')
        data['features']=regenerated
    source=read(Path(m['source'])/'scenes.json');reserved={r['group_id']:r for r in source if r['partition']=='development_calibration'}
    selections,groups=reserved_role_groups(ids,roles,reserved)
    replica=m['replicas'];bank=np.geomspace(.5,8,8).astype(np.float32)[:,None]
    np.testing.assert_array_equal(data['gains'],np.broadcast_to(np.repeat(bank,replica,axis=0),data['gains'].shape))
    mean=jnp.asarray(model.metadata['normalization']['target_mean'],jnp.float32);scale=jnp.asarray(model.metadata['normalization']['target_scale'],jnp.float32)
    def raw(params,features,mask,gains):
        out=predict_ensemble(model.model,params,features,mask,gains)
        return dict(mean=out['mean']*scale+mean,variance=jnp.exp(out['log_variance'])*scale**2,event_logits=out['event_logits'])
    feature_dtype=jnp.float64 if model.model.config.compute_dtype=='float64' else jnp.float32
    arguments=(jnp.zeros((batch,66,35),feature_dtype),jnp.ones((batch,66),bool),jnp.ones((batch,8,1),jnp.float32))
    start=time.monotonic();execute=jax.jit(raw).lower(model.params,*arguments).compile();jax.block_until_ready(execute(model.params,*arguments));cold=time.monotonic()-start
    def predict(indices):
        result=[]
        for start in range(0,len(indices),batch):
            index=indices[start:start+batch];n=len(index)
            f=np.pad(data['features'][index],((0,batch-n),(0,0),(0,0)));mask=np.pad(data['node_mask'][index],((0,batch-n),(0,0)))
            p=jax.tree.map(np.asarray,execute(model.params,jnp.asarray(f,dtype=feature_dtype),jnp.asarray(mask),jnp.asarray(np.broadcast_to(bank,(batch,8,1)))))
            result.append({k:np.repeat(v[:,:n],replica,axis=2) for k,v in p.items()})
        return {k:np.concatenate([v[k] for v in result],axis=1) for k in result[0]}
    subset=lambda idx:{k:v[idx] for k,v in data.items()}
    fit_prediction=predict(selections['prediction_fit']);parameters=fit_parameters(fit_prediction,subset(selections['prediction_fit']))
    fit=dict(schema=SCHEMA,stage='prediction_fit_only',bundle=str(Path(bundle).resolve()),weights_sha256=model.metadata['weights_sha256'],bundle_manifest_sha256=sha256(Path(bundle)/'manifest.json'),
        dataset=str(dataset.resolve()),dataset_manifest_sha256=sha256(dataset/'manifest.json'),dataset_index_sha256=sha256(dataset/'index.json'),dataset_audit_sha256=sha256(dataset/'independent_replay.json'),source_manifest_sha256=m['source_manifest_sha256'],
        bicycle_contract=model.metadata['bicycle_contract'],controller=m['controller'],targets=m['targets'],events=m['events'],gain_domain=m['gain_domain'],candidates=bank.tolist(),group_ids=groups,
        **parameters,prediction_batch=batch,prediction_candidates=8,compiled_prediction_signatures=1,compile_seconds=cold,
        event_budget_statistic='Maximum-member calibrated any_adverse_termination only. Includes collision/physical bounds/CBF/QP/planner rejection; censored collision_first head is not unconditional collision probability.',
        inference_graph=graph_proof,trajectory_gate_ready=False,production_eligible=False,whole_goal_complete=False,
        limitation='Development variance/event fit under actual fixed-gain acquired histories. Empirical conditional prediction adjustment; no posterior, physical CVaR, adaptive-trajectory, OOD or rare-collision guarantee. Fresh trajectory gate and final-policy physical audit required.')
    # Freeze the fitted artifact before predicting or evaluating the held role.
    write_json(root/'prediction_fit.json',fit);fit_sha=sha256(root/'prediction_fit.json')
    audit_prediction=predict(selections['prediction_audit'])
    report=dict(prediction_fit_sha256=fit_sha,fit=diagnostics(fit_prediction,subset(selections['prediction_fit']),fit),audit=diagnostics(audit_prediction,subset(selections['prediction_audit']),fit),model_promoted=False,whole_goal_complete=False)
    for role,prediction in [('prediction_fit',fit_prediction),('prediction_audit',audit_prediction)]:
        idx=selections[role];np.savez_compressed(root/(role+'_predictions.npz'),group_id=ids[idx],query_tick=data['query_tick'][idx],**prediction)
    write_json(root/'prediction_audit.json',report);write_json(root/'complete.json',dict(status='completed',prediction_fit_sha256=fit_sha,prediction_audit_sha256=sha256(root/'prediction_audit.json'),whole_goal_complete=False))
    print(json.dumps(dict(stage='predictive_fit_audit_completed',variance_scale=fit['variance_scale'],event_calibration=fit['event_calibration'],diagnostics=report)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('bundle','dataset','output'):p.add_argument('--'+field,required=True)
    p.add_argument('--batch',type=int,default=64);calibrate(**vars(p.parse_args()))
