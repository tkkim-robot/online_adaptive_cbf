"""Bicycle predictive calibration functions and shared contracts."""

import argparse

from pathlib import Path

import json

import time

import jax

import jax.numpy as jnp

import numpy as np

from scipy.optimize import minimize

from scipy.special import expit

from .bicycle_data import validate_training_dataset, SCHEMA as DATA_SCHEMA

from .bicycle_features import SCHEMA as GRAPH_SCHEMA

from .bicycle_observation import SCHEMA as SENSOR_SCHEMA

from .bicycle_control import read

from .io import parent_mean

from .io import load_dataset, sha256

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

def validate_model(metadata,manifest,qualified_motion=False,qualified_candidate=False,qualified_nearest=False):
    from .bicycle_gain_contract import validate_target_metadata
    validate_target_metadata(metadata);validate_target_metadata(manifest)
    if metadata.get('bicycle_task_progress_contract')!=manifest.get('bicycle_task_progress_contract'):
        raise ValueError('Changed task-progress calibration semantics')
    from .bicycle_gain_contract import TRAIN_SCHEMA, validate_manifest
    wide=metadata.get('offline_wide_gain_pilot',False)
    nearest=metadata.get('architecture',{}).get('encoder')=='nearest_fc'
    if nearest:
        from .nearest_fc import validate_metadata
        validate_metadata(metadata)
        if not qualified_nearest:raise ValueError('Nearest-FC requires its own numerical qualification')
    if wide:
        validate_manifest(manifest)
        if not (qualified_candidate or nearest and qualified_nearest) or metadata.get('bicycle_gain_contract')!=manifest['bicycle_gain_contract']:
            raise ValueError('Reviewed wide-gain runtime qualification required')
    candidate=metadata.get('architecture',{}).get('bicycle_candidate_encoding',False)
    if candidate and not qualified_candidate:
        raise ValueError('Candidate-encoding pilot requires separate runtime qualification before calibration')
    if candidate:
        from .bicycle_features import validate_metadata
        validate_metadata(metadata)
    motion=metadata.get('architecture',{}).get('bicycle_motion_history',False)
    if motion and not qualified_motion:
        raise ValueError('Observed motion-history pilot requires separate runtime qualification before calibration')
    if motion:
        from .bicycle_policy import validate_metadata
        validate_metadata(metadata)
    if metadata.get('architecture',{}).get('bicycle_affine_gain'):
        raise ValueError('Affine-gain pilot requires separate runtime qualification before calibration')
    if metadata.get('architecture',{}).get('bicycle_constraint_features'):
        from .bicycle_features import constraint_features_contract as contract
        if metadata.get('bicycle_constraint_features_contract') != contract():
            raise ValueError('Missing or changed observed constraint feature contract')
    if metadata.get('offline_reserve_auxiliary_pilot') or metadata.get('architecture',{}).get('bicycle_reserve_auxiliary'):
        raise ValueError('Reserve auxiliary pilot requires explicit reviewed runtime qualification before calibration')
    if metadata.get('dataset_schema')!=(TRAIN_SCHEMA if wide else DATA_SCHEMA) or metadata.get('gain_dimension')!=1 or metadata.get('graph_features')!=(39 if motion else 35) or metadata['architecture']['encoder'] not in ('gat','full_fc','matched_fc','nearest_fc'):
        raise ValueError('Observed scalar-gain bicycle GAT or complete-input FC required')
    if metadata.get('bicycle_contract',{}).get('sensor_schema')!=SENSOR_SCHEMA or metadata['bicycle_contract'].get('source_graph_schema' if motion else 'graph_schema')!=GRAPH_SCHEMA:
        raise ValueError('Wrong bicycle sensing/feature contract')
    for field in ('config','sensor_schema','graph_schema','horizon_steps','capacity','replicas','snapshot_ticks','acquisition_mode'):
        actual=metadata['bicycle_contract'].get('source_graph_schema' if motion and field=='graph_schema' else field)
        if actual!=manifest.get(field):raise ValueError('Bicycle contract mismatch: '+field)
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

def calibrated_parameters(prediction,data,variance_method='moment'):
    if variance_method not in ('moment','mixture_likelihood'):
        raise ValueError('Unknown reserved-parent variance procedure')
    result=fit_parameters(prediction,data)
    if variance_method=='mixture_likelihood':
        from .bicycle_predictive_calibration import fit_scales
        result.update(fit_scales(prediction,data))
    return result

def calibrate(bundle,dataset,output,batch=64,variance_method='moment',runtime_qualification=None,phase='full'):
    if phase not in ('full','fit','audit'):raise ValueError('Unknown calibration phase')
    root=Path(output)
    if phase=='audit':
        if not (root/'prediction_fit.json').is_file() or (root/'prediction_audit.json').exists():
            raise ValueError('Audit requires a frozen fit and a fresh audit artifact')
    else:root.mkdir(parents=True,exist_ok=False)
    print(json.dumps(dict(stage='validate_original_training_sources',phase=phase)),flush=True)
    dataset=Path(dataset);m=validate_training_dataset(dataset)
    model=ResearchPredictor(bundle,allow_uncalibrated=True)
    motion=model.model.config.bicycle_motion_history
    candidate=model.model.config.bicycle_candidate_encoding
    nearest=model.model.config.encoder=='nearest_fc'
    if nearest:
        from .nearest_fc import validate_qualification
        validate_qualification(runtime_qualification,bundle,dataset)
    if candidate:
        from .bicycle_predictive_calibration import validate_qualification
        validate_qualification(runtime_qualification,bundle,dataset)
    if motion:
        from .bicycle_predictive_calibration import motion_calibration_validate_qualification as validate_qualification
        validate_qualification(runtime_qualification,bundle,dataset)
    validate_model(model.metadata,m,qualified_motion=motion,qualified_candidate=candidate,qualified_nearest=nearest)
    if model.model.config.bicycle_constraint_features and not (motion or candidate):
        from .bicycle_policy import validate_qualification
        validate_qualification(runtime_qualification,bundle,dataset)
    if sha256(dataset/'manifest.json')!=model.metadata['dataset_manifest_sha256']:raise ValueError('Wrong weight-training lineage')
    print(json.dumps(dict(stage='load_reserved_queries',phase=phase)),flush=True)
    data=load_dataset(dataset,'development_calibration');ids=data['group_id'];roles=data['calibration_role']
    motion_proof=None
    if motion:
        from .bicycle_predictive_calibration import observed_history
        print(json.dumps(dict(stage='reconstruct_causal_reserved_history',phase=phase,queries=len(ids))),flush=True)
        past,elapsed,motion_proof=observed_history(data,dataset)
        data.update(motion_past_positions=past,motion_elapsed=elapsed)
        history_path=root/'motion_history_sources.json'
        if history_path.exists():
            if read(history_path)!=motion_proof:raise ValueError('Reserved motion-history source changed')
        else:write_json(history_path,motion_proof)
    graph_proof=dict(mode='original_stored_fp32',compiled_graph_signatures=0)
    if model.model.config.compute_dtype=='float64' or motion:
        # Reconstruct from actual saved observations/history, never the adjacent
        # latent physical state or obstacle arrays. Same numeric path as runtime.
        from .bicycle_features import bicycle_inference_graph
        from .bicycle_control import control_config
        c=control_config(m['config'])
        fields=('observed_state','goal','observed_obstacles','obstacle_mask','points','route_mask','cursor','previous_control','previous_gain','noise')
        if motion:
            from .bicycle_policy import graph as history_graph
            fields+=('motion_past_positions','motion_elapsed')
        def graph(*a):
            if motion:return jax.vmap(lambda *v:history_graph(*v,config=c,compute_dtype=model.model.config.compute_dtype))(*a)
            return jax.vmap(lambda *v:bicycle_inference_graph(*v,config=c,compute_dtype='float64'))(*a)
        graph_fn=jax.jit(graph)
        def graph_args(start):
            return tuple(jnp.asarray(np.pad(data[k][start:start+batch],((0,max(0,batch-len(ids[start:start+batch]))),)+((0,0),)*(data[k].ndim-1)),dtype=bool if k in ('obstacle_mask','route_mask') else jnp.float64 if k=='motion_elapsed' else jnp.float32) for k in fields)
        start=time.monotonic();graph_exe=graph_fn.lower(*graph_args(0)).compile();cold_graph=time.monotonic()-start
        features=[];masks=[]
        for i in range(0,len(ids),batch):
            f,mask=jax.device_get(graph_exe(*graph_args(i)));n=min(batch,len(ids)-i);features.append(f[:n]);masks.append(mask[:n])
        regenerated=np.concatenate(features);np.testing.assert_array_equal(np.concatenate(masks),data['node_mask'])
        graph_proof=dict(mode='observed_context_fp64_reconstruction',input_fields=list(fields),compiled_graph_signatures=1,compile_seconds=cold_graph,
            implicit_jit_cache_entries=graph_fn._cache_size(),maximum_change_from_stored_fp32=float(np.max(np.abs(regenerated[...,:35]-data['features']))))
        if motion:
            from .bicycle_features import numpy_features
            history_error=0.
            for i,feature in enumerate(regenerated):
                expected=numpy_features(feature[:,:35],data['node_mask'][i],data['observed_state'][i],
                    data['observed_obstacles'][i],past[i],data['noise'][i],elapsed[i])
                np.testing.assert_allclose(feature,expected,atol=3e-6,rtol=2e-6)
                history_error=max(history_error,float(np.max(np.abs(feature-expected))))
            graph_proof.update(mode='observed_context_and_causal_motion_history',history_sources_sha256=sha256(root/'motion_history_sources.json'),
                graph_features=39,physical_truth_loaded=False,every_history_feature_independently_verified=True,
                independent_history_max_error=history_error)
        if graph_fn._cache_size():raise ValueError('Unexpected graph JIT during calibration')
        data['features']=regenerated
    source=read(Path(m['source'])/'scenes.json');reserved={r['group_id']:r for r in source if r['partition']=='development_calibration'}
    selections,groups=reserved_role_groups(ids,roles,reserved)
    from .bicycle_gain_contract import model_bank
    replica=m['replicas'];bank=model_bank(model.metadata)
    np.testing.assert_array_equal(data['gains'],np.broadcast_to(np.repeat(bank,replica,axis=0),data['gains'].shape))
    mean=jnp.asarray(model.metadata['normalization']['target_mean'],jnp.float32);scale=jnp.asarray(model.metadata['normalization']['target_scale'],jnp.float32)
    def raw(params,features,mask,gains):
        out=predict_ensemble(model.model,params,features,mask,gains)
        return dict(mean=out['mean']*scale+mean,variance=jnp.exp(out['log_variance'])*scale**2,event_logits=out['event_logits'])
    feature_dtype=jnp.float64 if model.model.config.compute_dtype=='float64' else jnp.float32
    arguments=(jnp.zeros((batch,66,model.metadata['graph_features']),feature_dtype),jnp.ones((batch,66),bool),jnp.ones((batch,len(bank),1),jnp.float32))
    start=time.monotonic();raw_fn=jax.jit(raw);execute=raw_fn.lower(model.params,*arguments).compile();jax.block_until_ready(execute(model.params,*arguments));cold=time.monotonic()-start
    def predict(indices):
        result=[]
        for start in range(0,len(indices),batch):
            index=indices[start:start+batch];n=len(index)
            f=np.pad(data['features'][index],((0,batch-n),(0,0),(0,0)));mask=np.pad(data['node_mask'][index],((0,batch-n),(0,0)))
            p=jax.tree.map(np.asarray,execute(model.params,jnp.asarray(f,dtype=feature_dtype),jnp.asarray(mask),jnp.asarray(np.broadcast_to(bank,(batch,len(bank),1)))))
            result.append({k:np.repeat(v[:,:n],replica,axis=2) for k,v in p.items()})
        return {k:np.concatenate([v[k] for v in result],axis=1) for k in result[0]}
    subset=lambda idx:{k:v[idx] for k,v in data.items()}
    print(json.dumps(dict(stage='fit_reserved_prediction_parameters',phase=phase,queries=len(selections['prediction_fit']))),flush=True)
    fit_prediction=predict(selections['prediction_fit']);parameters=calibrated_parameters(fit_prediction,subset(selections['prediction_fit']),variance_method)
    audit_path=(dataset/'independent_replay.json' if 'task_progress_derivative' in m else
        Path(m['reviewed_training_view']['review']) if model.metadata.get('offline_wide_gain_pilot') else dataset/'independent_replay.json')
    fit=dict(schema=SCHEMA,stage='prediction_fit_only',bundle=str(Path(bundle).resolve()),weights_sha256=model.metadata['weights_sha256'],bundle_manifest_sha256=sha256(Path(bundle)/'manifest.json'),
        dataset=str(dataset.resolve()),dataset_manifest_sha256=sha256(dataset/'manifest.json'),dataset_index_sha256=sha256(dataset/'index.json'),dataset_audit_sha256=sha256(audit_path),source_manifest_sha256=m['source_manifest_sha256'],
        bicycle_contract=model.metadata['bicycle_contract'],controller=m['controller'],targets=m['targets'],events=m['events'],gain_domain=m['gain_domain'],candidates=bank.tolist(),group_ids=groups,
        **parameters,prediction_batch=batch,prediction_candidates=len(bank),compiled_prediction_signatures=1,compile_seconds=cold,
        event_budget_statistic='Maximum-member calibrated any_adverse_termination only. Includes collision/physical bounds/CBF/QP/planner rejection; censored collision_first head is not unconditional collision probability.',
        inference_graph=graph_proof,trajectory_gate_ready=False,production_eligible=False,whole_goal_complete=False,
        limitation='Development variance/event fit under actual fixed-gain acquired histories. Empirical conditional prediction adjustment; no posterior, physical CVaR, adaptive-trajectory, OOD or rare-collision guarantee. Fresh trajectory gate and final-policy physical audit required.')
    if runtime_qualification is not None:
        fit.update(runtime_qualification=str(Path(runtime_qualification).resolve()),runtime_qualification_sha256=sha256(runtime_qualification))
    if nearest:
        fit['nearest_fc_contract']=model.metadata['nearest_fc_contract']
    if model.metadata.get('offline_wide_gain_pilot'):
        fit.update(bicycle_gain_contract=model.metadata['bicycle_gain_contract'],dataset_audit_file=str(audit_path.resolve()))
    if 'bicycle_task_progress_contract' in m:
        fit['bicycle_task_progress_contract']=m['bicycle_task_progress_contract']
    # Freeze the fitted artifact before predicting or evaluating the held role.
    if phase=='audit':
        frozen=read(root/'prediction_fit.json')
        for key in ('weights_sha256','bundle_manifest_sha256','dataset_manifest_sha256','dataset_index_sha256','dataset_audit_sha256','group_ids','candidates','prediction_batch',*parameters):
            if frozen[key]!=fit[key]:raise ValueError('Frozen prediction fit does not reproduce: '+key)
        if runtime_qualification is not None and frozen['runtime_qualification_sha256']!=sha256(runtime_qualification):
            raise ValueError('Changed runtime qualification')
        with np.load(root/'prediction_fit_predictions.npz') as cache:
            for key,value in fit_prediction.items():np.testing.assert_array_equal(value,cache[key])
        fit=frozen
    else:
        write_json(root/'prediction_fit.json',fit)
        idx=selections['prediction_fit']
        np.savez_compressed(root/'prediction_fit_predictions.npz',group_id=ids[idx],query_tick=data['query_tick'][idx],**fit_prediction)
    fit_sha=sha256(root/'prediction_fit.json')
    if phase=='fit':
        if raw_fn._cache_size():raise ValueError('Unexpected prediction JIT during fitting')
        print(json.dumps(dict(stage='prediction_fit_frozen',prediction_fit_sha256=fit_sha)),flush=True)
        return
    audit_prediction=predict(selections['prediction_audit'])
    if raw_fn._cache_size():raise ValueError('Unexpected prediction JIT during calibration')
    report=dict(prediction_fit_sha256=fit_sha,fit=diagnostics(fit_prediction,subset(selections['prediction_fit']),fit),audit=diagnostics(audit_prediction,subset(selections['prediction_audit']),fit),model_promoted=False,whole_goal_complete=False)
    for role,prediction in [('prediction_fit',fit_prediction),('prediction_audit',audit_prediction)]:
        if role=='prediction_fit':continue
        idx=selections[role];np.savez_compressed(root/(role+'_predictions.npz'),group_id=ids[idx],query_tick=data['query_tick'][idx],**prediction)
    write_json(root/'prediction_audit.json',report);write_json(root/'complete.json',dict(status='completed',prediction_fit_sha256=fit_sha,prediction_audit_sha256=sha256(root/'prediction_audit.json'),whole_goal_complete=False))
    print(json.dumps(dict(stage='predictive_fit_audit_completed',variance_scale=fit['variance_scale'],event_calibration=fit['event_calibration'],diagnostics=report)),flush=True)



def validate_qualification(review_path,bundle,dataset):
    if review_path is None:raise ValueError('Independent candidate runtime qualification is required')
    review=read(review_path);report=read(review['report']);meta=read(Path(bundle)/'manifest.json')
    from .bicycle_features import validate_metadata
    from .bicycle_predictive_calibration import CANDIDATE_QUALIFICATION_SCHEMA as SCHEMA, WIDE_SCHEMA
    validate_metadata(meta);encoder=meta['architecture']['encoder']
    target=meta.get('bicycle_task_progress_contract')
    manifest=read(Path(dataset)/'manifest.json')
    if any(record.get('bicycle_task_progress_contract')!=target for record in (manifest,report,review)):
        raise ValueError('Qualification changed the task-progress target contract')
    wide=meta.get('offline_wide_gain_pilot',False)
    if wide:
        from .bicycle_gain_contract import contract
        if (review.get('wide_gain_contract_and_all_candidates_verified') is not True
                or report.get('bicycle_gain_contract')!=contract()
                or review.get('bicycle_gain_contract')!=contract()):
            raise ValueError('Independent wide-gain qualification required')
    flags=('all_source_and_checkpoint_bindings_verified','all_candidate_and_observed_features_verified',
        'exported_weights_equal_selected_trained_members','cpu_gpu_and_live_selector_parity_verified',
        'independent_selection_statistics_verified','warrants_reserved_calibration')
    if (review.get('schema')!=('independent_bicycle_wide_candidate_runtime_review' if wide else 'independent_bicycle_candidate_runtime_review') or review.get('status')!='passed'
            or any(review.get(k) is not True for k in flags) or review['report_sha256']!=sha256(review['report'])
            or review.get('protocol_sha256')!=report.get('protocol_sha256')
            or report.get('schema')!=(WIDE_SCHEMA if wide else SCHEMA) or report.get('status')!='passed'
            or report['dataset_manifest_sha256']!=sha256(Path(dataset)/'manifest.json')
            or meta.get('dataset_manifest_sha256')!=report['dataset_manifest_sha256']
            or report['dataset_index_sha256']!=sha256(Path(dataset)/'index.json')
            or report['models'][encoder]['bundle_manifest_sha256']!=sha256(Path(bundle)/'manifest.json')
            or report['models'][encoder]['weights_sha256']!=sha256(Path(bundle)/'weights.msgpack')
            or not report['models'][encoder]['parity_passed']
            or any(x.get(k) is not False for x in (review,report) for k in ('calibration_fitted','reserved_calibration_parents_used','benchmark_parents_used','model_promoted'))):
        raise ValueError('Changed or incomplete independent candidate runtime qualification')
    return report

def validate_fitted_model(fit,bundle):
    """A real fit must remain bound to independent numerical qualification."""
    path=fit.get('runtime_qualification');dataset=fit.get('dataset')
    if (path is None or dataset is None or fit.get('diagnostic_identity_only')
            or not Path(path).is_file() or fit.get('runtime_qualification_sha256')!=sha256(path)):
        raise ValueError('Candidate policy requires a bound independently reviewed runtime qualification')
    report=validate_qualification(path,bundle,dataset)
    from .bicycle_gain_contract import validate_target_metadata
    validate_target_metadata(fit)
    meta=read(Path(bundle)/'manifest.json')
    if fit.get('bicycle_task_progress_contract')!=meta.get('bicycle_task_progress_contract'):
        raise ValueError('Fitted model changed task-progress semantics')
    if 'bicycle_gain_contract' in report:
        from .bicycle_gain_contract import contract, candidate_bank
        if fit.get('bicycle_gain_contract')!=contract() or fit.get('candidates')!=candidate_bank()[:,None].tolist():
            raise ValueError('Changed wide-gain calibrated candidate contract')
    if (fit.get('dataset_manifest_sha256')!=report['dataset_manifest_sha256']
            or fit.get('dataset_index_sha256')!=report['dataset_index_sha256']):
        raise ValueError('Candidate calibration dataset differs from runtime qualification')
    return report


CANDIDATE_QUALIFICATION_SCHEMA='bicycle_candidate_encoding_runtime_qualification'

WIDE_SCHEMA='bicycle_wide_candidate_encoding_runtime_qualification'


from concurrent.futures import ThreadPoolExecutor


def motion_calibration_validate_qualification(review_path,bundle,dataset):
    if review_path is None:raise ValueError('Independent motion runtime qualification is required')
    review=read(review_path);report=read(review['report']);meta=read(Path(bundle)/'manifest.json')
    from .bicycle_policy import validate_metadata
    from .bicycle_predictive_calibration import MOTION_QUALIFICATION_SCHEMA as SCHEMA
    validate_metadata(meta);encoder=meta['architecture']['encoder']
    flags=('all_source_and_checkpoint_bindings_verified','all_causal_history_and_observed_features_verified',
        'exported_weights_equal_selected_trained_members','cpu_gpu_and_live_selector_parity_verified',
        'independent_selection_statistics_verified','warrants_reserved_calibration')
    if (review.get('schema')!='independent_bicycle_motion_runtime_review' or review.get('status')!='passed'
            or any(review.get(k) is not True for k in flags) or review['report_sha256']!=sha256(review['report'])
            or review.get('protocol_sha256')!=report.get('protocol_sha256')
            or report.get('schema')!=SCHEMA or report.get('status')!='passed'
            or report['dataset_manifest_sha256']!=sha256(Path(dataset)/'manifest.json')
            or report['dataset_index_sha256']!=sha256(Path(dataset)/'index.json')
            or report['models'][encoder]['bundle_manifest_sha256']!=sha256(Path(bundle)/'manifest.json')
            or report['models'][encoder]['weights_sha256']!=sha256(Path(bundle)/'weights.msgpack')
            or not report['models'][encoder]['parity_passed']
            or any(x.get(k) is not False for x in (review,report) for k in ('calibration_fitted','reserved_calibration_parents_used','benchmark_parents_used','model_promoted'))):
        raise ValueError('Changed or incomplete independent motion runtime qualification')
    return report

def observed_history(data,dataset):
    from .bicycle_policy import check_files as motion_calibration_check_files
    from .bicycle_data import source_map
    from .bicycle_observation import WINDOW_TICKS
    if not np.all(data['partition']=='development_calibration'):raise ValueError('Only original reserved calibration queries')
    mapping,bindings=source_map(dataset);motion_calibration_check_files(bindings)
    index=read(Path(dataset)/'index.json');lookup={}
    for row in index:
        record=mapping[str(Path(row['file']).resolve())]
        if (record['query_sha256']!=row['sha256'] or any(record[k]!=row[k] for k in ('group_id','query_origin','query_tick'))):
            raise ValueError('Changed acquisition identity')
        key=(row['group_id'],row['query_origin'],int(row['query_tick']))
        if key in lookup:raise ValueError('Duplicate reserved source identity')
        lookup[key]=record
    manifest=read(Path(dataset)/'manifest.json');dt=manifest['config']['robot']['dt'];n=len(data['group_id'])
    if not np.isfinite(dt) or dt<=0 or not n or np.any(data['query_tick']<0) or np.any(data['query_tick']!=data['query_tick'].astype(int)):
        raise ValueError('Invalid reserved history time coordinate')
    past=np.zeros((n,64,2),np.float32);elapsed=np.minimum(data['query_tick'],WINDOW_TICKS)*dt;requests={}
    for i,(g,o,t) in enumerate(zip(data['group_id'],data['query_origin'],data['query_tick'])):
        r=lookup[(g,o,int(t))];file=r['path']
        if file not in requests:requests[file]=dict(sha256=r['sha256'],rows=[])
        if requests[file]['sha256']!=r['sha256']:raise ValueError('Mixed history hashes')
        requests[file]['rows'].append(dict(index=i,group_id=g,origin=o,tick=int(t),past_tick=max(0,int(t)-WINDOW_TICKS)))
    def read_one(item):
        from .bicycle_policy import check_files as motion_calibration_check_files
        file,record=item;motion_calibration_check_files({file:record['sha256']})
        with np.load(file) as z:obs=z['observed_obstacles'];mask=z['mask'];noise=z['noise']
        for r in record['rows']:
            i,t,p=r['index'],r['tick'],r['past_tick']
            if not 0<=p<=t<len(obs):raise ValueError('Unavailable causal reserved history')
            np.testing.assert_array_equal(data['observed_obstacles'][i],obs[t]);np.testing.assert_array_equal(data['obstacle_mask'][i],mask);np.testing.assert_array_equal(data['noise'][i],noise)
            past[i]=obs[p,:,:2]
    with ThreadPoolExecutor(14) as pool:list(pool.map(read_one,requests.items()))
    motion_calibration_check_files(bindings)
    proof=dict(schema='bicycle_reserved_observed_motion_history',dataset_manifest_sha256=sha256(Path(dataset)/'manifest.json'),
        dataset_index_sha256=sha256(Path(dataset)/'index.json'),source_indices=bindings,sources=requests,queries=n,
        parents=len(set(data['group_id'])),dt=dt,window_ticks=WINDOW_TICKS,loaded_acquisition_fields=['observed_obstacles','mask','noise'],
        physical_truth_loaded=False,labels_used_for_history=False,all_current_observations_checked=True)
    return past,elapsed.astype(np.float64),proof


MOTION_QUALIFICATION_SCHEMA='bicycle_motion_history_runtime_qualification'


import copy


from scipy.optimize import minimize_scalar

from scipy.special import logsumexp


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
    from .bicycle_control import read
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
    from .bicycle_control import read
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
    from .bicycle_control import read
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
    for field in ('bundle','dataset','output'):p.add_argument('--'+field,required=True)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--variance-method',choices=['moment','mixture_likelihood'],default='moment')
    p.add_argument('--phase',choices=['full','fit','audit'],default='full')
    p.add_argument('--runtime-qualification');calibrate(**vars(p.parse_args()))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('protocol','output','directory'):p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))
