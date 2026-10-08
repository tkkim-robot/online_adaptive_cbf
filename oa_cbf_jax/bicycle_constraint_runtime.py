"""Qualified export and reserved calibration for observed-constraint encoders.

Unchanged genuine weights, a shared numerical inference port, measured backend
selection, and the existing parent-weighted mixture-variance fitting procedure.
No physical benchmark or trajectory-gate outcome selects these settings.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .bicycle_experiment import read
from .dataset import sha256
from .io import write_json

SCHEMA='bicycle_observed_constraint_runtime_qualification'
FIELDS=('observed_state','goal','observed_obstacles','obstacle_mask','points','route_mask','cursor','previous_control','previous_gain','noise')
ENCODERS=('gat','matched_fc')


def check_files(files):
    for path,digest in files.items():
        if sha256(path)!=digest:raise ValueError('Changed runtime input: '+path)


def validate_learning_review(spec):
    """Bind calibration to the exact reviewed study, source and checkpoints.

    A positive held-gain forecasting study permits calibration only. It does
    not supply the stronger runtime-integration approval used by feature fits.
    """
    review=read(spec['learning_review'])
    if (review.get('status')!='passed'
            or review.get('report_sha256')!=sha256(spec['learning_report'])):
        raise ValueError('Positive independently reviewed learning required')
    kind=spec.get('learning_kind','constraint_features')
    if kind=='constraint_features':
        if (review.get('warrants_runtime_integration') is not True
                or review.get('all_source_parent_feature_checkpoint_and_metric_bindings_verified') is not True):
            raise ValueError('Reviewed constraint feature integration required')
    elif kind=='long_forecast':
        required=('warrants_calibration_study',
            'all_original_source_observation_target_and_checkpoint_bindings_verified',
            'matched_encoder_training_verified')
        if (any(review.get(key) is not True for key in required)
                or review.get('calibration_fitted') is not False
                or review.get('model_promoted') is not False
                or review.get('protocol_sha256')!=sha256(spec['learning_protocol'])):
            raise ValueError('Reviewed long-forecast calibration study required')
        protocol=read(spec['learning_protocol']);report=read(spec['learning_report'])
        dataset=Path(spec['dataset']).resolve();training=Path(spec['training']).resolve()
        if (Path(protocol['dataset']).resolve()!=dataset
                or Path(protocol['versions']['long_current']).resolve()!=training
                or read(dataset/'manifest.json').get('horizon_steps')!=160
                or report.get('benchmark_parents_used') is not False
                or report.get('controller_changed') is not False):
            raise ValueError('Changed long-forecast source or scope')
        check_files(protocol['bound_files'])
        for encoder in ENCODERS:
            result=report['results']['long_current'][encoder]
            if result['forecast_horizon_steps']!=160 or result['encoder']!=encoder:
                raise ValueError('Changed reviewed forecasting horizon or encoder')
            check_files(result['bound_checkpoints'])
            for i in range(4):
                root=training/encoder/f'member_{i}'
                best=read(root/'best.json');settings=read(root/'settings.json')
                for path in (root/'settings.json',root/'best.json',root/'complete.json',
                             root/'checkpoints'/best['state_file']):
                    if result['bound_checkpoints'].get(str(path))!=sha256(path):
                        raise ValueError('Unreviewed long-forecast checkpoint')
                if (settings['dataset_manifest_sha256']!=sha256(dataset/'manifest.json')
                        or settings['bicycle_contract']['horizon_steps']!=160):
                    raise ValueError('Long-forecast training/source mismatch')
    else:
        raise ValueError('Unknown reviewed learning kind: '+kind)
    return review


def validate_qualification(path,bundle,dataset):
    if path is None:raise ValueError('Reviewed constraint runtime qualification is required')
    report=read(path);meta=read(Path(bundle)/'manifest.json');encoder=meta['architecture']['encoder']
    if (report.get('schema')!=SCHEMA or report.get('status')!='passed'
            or report['dataset_manifest_sha256']!=sha256(Path(dataset)/'manifest.json')
            or report['dataset_index_sha256']!=sha256(Path(dataset)/'index.json')
            or report['models'][encoder]['bundle_manifest_sha256']!=sha256(Path(bundle)/'manifest.json')
            or report['models'][encoder]['weights_sha256']!=sha256(Path(bundle)/'weights.msgpack')
            or not report['models'][encoder]['parity_passed']
            or report['benchmark_parents_used'] or report['reserved_calibration_parents_used']):
        raise ValueError('Changed or incomplete constraint runtime qualification')
    return report


def observed_graphs(data,config,dtype,batch=64):
    import jax
    import jax.numpy as jnp
    from .bicycle_features import bicycle_inference_graph
    fn=jax.jit(lambda *a:jax.vmap(lambda *v:bicycle_inference_graph(*v,config=config,compute_dtype=dtype))(*a))
    values=[];masks=[];exe=None;begin=time.monotonic()
    for start in range(0,len(data['group_id']),batch):
        n=min(batch,len(data['group_id'])-start)
        args=tuple(jnp.asarray(np.pad(data[k][start:start+n],((0,batch-n),)+((0,0),)*(data[k].ndim-1)),
            dtype=bool if k in ('obstacle_mask','route_mask') else jnp.float32) for k in FIELDS)
        if exe is None:exe=fn.lower(*args).compile()
        f,m=jax.device_get(exe(*args));values.append(f[:n]);masks.append(m[:n])
    if fn._cache_size():raise ValueError('Unexpected observed graph compilation')
    return np.concatenate(values),np.concatenate(masks),dict(signatures=1,implicit_jit_cache_entries=0,seconds=time.monotonic()-begin)


def raw_inference(predictor,features,mask,batch=64,repeats=3):
    import jax
    import jax.numpy as jnp
    from .models import predict_ensemble
    norm=predictor.metadata['normalization'];mean=jnp.asarray(norm['target_mean'],jnp.float32);scale=jnp.asarray(norm['target_scale'],jnp.float32)
    from .bicycle_gain_contract import model_bank
    bank=model_bank(predictor.metadata)
    def raw(p,f,m,g):
        result=predict_ensemble(predictor.model,p,f,m,g)
        return dict(mean=result['mean']*scale+mean,variance=jnp.exp(result['log_variance'])*scale**2,event_logits=result['event_logits'])
    fn=jax.jit(raw)
    def args(start):
        n=min(batch,len(features)-start)
        dtype=jnp.float64 if predictor.model.config.compute_dtype=='float64' else jnp.float32
        return (predictor.params,jnp.asarray(np.pad(features[start:start+n],((0,batch-n),(0,0),(0,0))),dtype=dtype),
            jnp.asarray(np.pad(mask[start:start+n],((0,batch-n),(0,0)))),jnp.asarray(np.broadcast_to(bank,(batch,len(bank),1))))
    begin=time.monotonic();exe=fn.lower(*args(0)).compile();jax.block_until_ready(exe(*args(0)));cold=time.monotonic()-begin
    arrays=[];timing=[]
    for repeat in range(repeats):
        begin=time.monotonic();parts=[]
        for start in range(0,len(features),batch):
            n=min(batch,len(features)-start);a=args(start);raw=exe(*a)
            # Include host transfer, the same rounded raw outputs and normalization as policy.
            p=jax.device_get(raw)
            parts.append({k:v[:,:n] for k,v in p.items()})
        timing.append(time.monotonic()-begin)
        if repeat==0:arrays={k:np.concatenate([v[k] for v in parts],axis=1) for k in parts[0]}
    if fn._cache_size() or any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite prediction or implicit compilation')
    return arrays,dict(compile_seconds=cold,p50_pass_seconds=float(np.median(timing)),pass_seconds=timing,signatures=1,implicit_jit_cache_entries=0,
        feature_input_dtype=str(args(0)[1].dtype))


def qualify(training,dataset,encoder,output):
    import jax
    from .bicycle_constraint_features import verify_features
    from .bicycle_experiment import control_config
    from .dataset import load_dataset
    from .inference import ResearchPredictor
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    raw=load_dataset(dataset,'validation')
    # Outcome-independent spread over every region of the ordered validation set.
    indices=np.unique(np.linspace(0,len(raw['group_id'])-1,256,dtype=int));data={k:v[indices] for k,v in raw.items()}
    config=control_config(read(Path(dataset)/'manifest.json')['config'])
    f64,mask,graph_proof=observed_graphs(data,config,'float64');np.testing.assert_array_equal(mask,data['node_mask'])
    proof64=verify_features(f64,mask,batch=64)
    proof32=verify_features(data['features'],mask,batch=64)
    preds={};timings={}
    for name,features in [('fp32',data['features']),('fp64',f64)]:
        bundle=Path(training)/encoder/('bundle' if name=='fp32' else 'bundle_fp64')
        predictor=ResearchPredictor(bundle,allow_uncalibrated=True)
        preds[name],timings[name]=raw_inference(predictor,features,mask)
    np.savez_compressed(out/'predictions.npz',indices=indices,group_id=data['group_id'],query_tick=data['query_tick'],features_fp64=f64,node_mask=mask,
        **{name+'_'+key:value for name,p in preds.items() for key,value in p.items()})
    result=dict(encoder=encoder,backend=jax.default_backend(),queries=len(indices),physical_parents=len(set(data['group_id'])),
        timing=timings,feature_proof_fp32=proof32,feature_proof_fp64=proof64,graph_proof=graph_proof,
        precision_change_max={k:float(np.max(abs(preds['fp32'][k]-preds['fp64'][k]))) for k in preds['fp32']},
        prediction_sha256=sha256(out/'predictions.npz'),benchmark_parents_used=False,reserved_calibration_parents_used=False)
    write_json(out/'report.json',result);print(result,flush=True)


def run(protocol,directory):
    spec=read(protocol);job=Path(directory);out=Path(spec['output']);training=Path(spec['training']);start=time.monotonic()
    check_files(spec['bound_files']);validate_learning_review(spec)
    out.mkdir(parents=True,exist_ok=False);write_json(job/'protocol.json',spec)
    def progress(stage,estimate):
        value=dict(stage=stage,elapsed_seconds=time.monotonic()-start,estimated_remaining_seconds=estimate)
        write_json(job/'progress.json',value);print(value,flush=True)
    def execute(name,module,args,slot=0,backend='cpu'):
        env=os.environ.copy();env.update(JAX_PLATFORMS='cpu' if backend=='cpu' else 'cuda',CUDA_VISIBLE_DEVICES='' if backend=='cpu' else str(slot),
            JAX_EXPLICIT_X64_DTYPES='allow',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
        env.pop('JAX_ENABLE_X64',None);exe=Path.cwd()/('.venv-cpu' if backend=='cpu' else '.venv')/'bin/python';begin=time.monotonic()
        with (job/(name+'.log')).open('w') as log:
            subprocess.run(['taskset','-c',f'{slot*14}-{slot*14+13}',str(exe),'-m','oa_cbf_jax.'+module,*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return time.monotonic()-begin
    progress('genuine_checkpoint_export_and_precision_derivation',1500)
    for slot,e in enumerate(ENCODERS):
        root=training/e
        if spec.get('reuse_exported_bundles',False):
            from .precision_bundle import validate_derivation
            for name in ('bundle','bundle_fp64'):
                for filename in ('manifest.json','weights.msgpack'):
                    path=(root/name/filename).resolve()
                    if spec['bound_files'].get(str(path))!=sha256(path):raise ValueError('Unbound reused export')
            validate_derivation(root/'bundle_fp64',read(root/'bundle_fp64/manifest.json'))
        else:
            execute(e+'_export','inference',['export-pilot','--members',*[root/f'member_{i}' for i in range(4)],'--output',root/'bundle'],slot)
            execute(e+'_derive','precision_bundle',['--source',root/'bundle','--output',root/'bundle_fp64'],slot)
    progress('observed_feature_and_cpu_gpu_inference_qualification',1300)
    def probe(item):
        slot,(e,b)=item
        return execute(e+'_'+b,'bicycle_constraint_runtime',['qualify','--training',training,'--dataset',spec['dataset'],'--encoder',e,'--output',out/e/b],slot,b)
    with ThreadPoolExecutor(4) as pool:list(pool.map(probe,enumerate((('gat','cpu'),('gat','gpu'),('matched_fc','cpu'),('matched_fc','gpu')))))
    qualification=dict(schema=SCHEMA,status='passed',dataset_manifest_sha256=sha256(Path(spec['dataset'])/'manifest.json'),
        dataset_index_sha256=sha256(Path(spec['dataset'])/'index.json'),models={},benchmark_parents_used=False,reserved_calibration_parents_used=False)
    for e in ENCODERS:
        reports={b:read(out/e/b/'report.json') for b in ('cpu','gpu')}
        with np.load(out/e/'cpu/predictions.npz') as a,np.load(out/e/'gpu/predictions.npz') as b:
            for key in ('indices','group_id','query_tick','node_mask'):np.testing.assert_array_equal(a[key],b[key])
            np.testing.assert_allclose(a['features_fp64'],b['features_fp64'],atol=1e-10,rtol=1e-10)
            errors={}
            for precision in ('fp32','fp64'):
                for key in ('mean','variance','event_logits'):
                    field=precision+'_'+key;np.testing.assert_allclose(a[field],b[field],atol=2e-5,rtol=2e-5)
                    errors[field]=float(np.max(abs(a[field]-b[field])))
        selected=min(reports,key=lambda b:reports[b]['timing']['fp64']['p50_pass_seconds'])
        qualification['models'][e]=dict(selected=selected,parity_passed=True,maximum_errors=errors,reports=reports,
            bundle_manifest_sha256=sha256(training/e/'bundle_fp64/manifest.json'),weights_sha256=sha256(training/e/'bundle_fp64/weights.msgpack'))
    write_json(out/'runtime_qualification.json',qualification)
    progress('new_reserved_parent_prediction_fits_and_audits',600)
    def calibrate(item,phase):
        slot,e=item
        return execute(e+'_calibration_'+phase,'bicycle_predictive_calibration',['--bundle',training/e/'bundle_fp64','--dataset',spec['dataset'],
            '--output',training/e/'prediction_calibration','--variance-method','mixture_likelihood','--phase',phase,'--runtime-qualification',out/'runtime_qualification.json'],
            slot,qualification['models'][e]['selected'])
    calibration_seconds={}
    for phase in ('fit','audit'):
        with ThreadPoolExecutor(2) as pool:calibration_seconds[phase]=list(pool.map(lambda item:calibrate(item,phase),enumerate(ENCODERS)))
    check_files(spec['bound_files'])
    report=dict(status='completed',output=str(training.resolve()),matched_training_settings_verified=True,
        learning_kind=spec.get('learning_kind','constraint_features'),
        forecast_horizon_steps=read(Path(spec['dataset'])/'manifest.json')['horizon_steps'],
        learning_review_sha256=sha256(spec['learning_review']),runtime_qualification=str((out/'runtime_qualification.json').resolve()),
        runtime_qualification_sha256=sha256(out/'runtime_qualification.json'),calibration_seconds=calibration_seconds,
        variance_method='mixture_likelihood',model_promoted=False,whole_goal_complete=False,elapsed_seconds=time.monotonic()-start)
    write_json(job/'complete.json',report)
    from .bicycle_bundles import models
    fitted=models(training,job/'complete.json')
    report['models']={e:{k:v for k,v in item.items() if k not in ('metadata','calibration')} for e,item in fitted.items()}
    write_json(out/'report.json',report);write_json(job/'complete.json',dict(report,report_sha256=sha256(out/'report.json')))
    progress('completed',0)


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    s=sub.add_parser('qualify')
    for k in ('training','dataset','encoder','output'):s.add_argument('--'+k,required=True)
    s=sub.add_parser('run')
    for k in ('protocol','directory'):s.add_argument('--'+k,required=True)
    args=vars(p.parse_args());action=args.pop('action');globals()[action](**args)
