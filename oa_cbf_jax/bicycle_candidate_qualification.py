"""Bound candidate-conditioned inference qualification before reserved calibration."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from .bicycle_experiment import read,control_config
from .bicycle_constraint_runtime import FIELDS,raw_inference,check_files
from .bicycle_motion_qualification import load_queries,diagnostic_fit
from .dataset import sha256
from .io import write_json

SCHEMA='bicycle_candidate_encoding_runtime_qualification'


def prepare(spec,out):
    data,query_sources=load_queries(spec)
    path=out/'inputs.npz';np.savez_compressed(path,**data)
    write_json(out/'observation_sources.json',dict(query_sources=query_sources,queries=len(data['indices']),
        physical_truth_loaded=False,labels_loaded=False,history_inputs_used=False))
    return path


def observed_graphs(data,config,precision):
    from .bicycle_constraint_runtime import observed_graphs as graphs
    from .bicycle_observed_audit import check_graph
    from .bicycle_constraint_features import verify_features
    from .bicycle_candidate_features import verify_conditioning
    f,mask,proof=graphs(data,config,precision)
    np.testing.assert_array_equal(mask,data['node_mask'])
    for i in range(len(f)):
        check_graph(f[i],mask[i],(data['observed_state'][i],data['goal'][i],data['observed_obstacles'][i],data['obstacle_mask'][i],
            data['points'][i],data['route_mask'][i],data['noise'][i],config,data['cursor'][i],data['previous_control'][i],data['previous_gain'][i]))
    proof.update(every_observed_graph_verified=True,constraint_features=verify_features(f,mask,batch=64),
        candidate_features=verify_conditioning(f,mask,data['gains'],batch=64))
    return f,mask,proof


def worker(protocol,encoder,backend):
    import jax
    from .inference import ResearchPredictor
    from .bicycle_candidate_features import validate_metadata
    from .bicycle_policy import BicycleSelector,BicyclePolicyConfig
    from .bicycle_policy_audit import check_selection
    spec=read(protocol);check_files(spec['bound_files']);out=Path(spec['output'])/encoder/backend;out.mkdir(parents=True,exist_ok=False)
    with np.load(spec['inputs']) as z:data={k:z[k] for k in z.files}
    config=control_config(read(Path(spec['dataset'])/'manifest.json')['config'])
    offline=data['features']
    f32,mask,proof32=observed_graphs(data,config,'float32');f64,_,proof64=observed_graphs(data,config,'float64')
    root=Path(spec['training'])/encoder;pred={};timing={}
    for name in ('bundle','bundle_fp64'):validate_metadata(read(root/name/'manifest.json'))
    for name,features,bundle in [('offline_fp32',offline,root/'bundle'),('runtime_fp32',f32,root/'bundle'),('runtime_fp64',f64,root/'bundle_fp64')]:
        pred[name],timing[name]=raw_inference(ResearchPredictor(bundle,allow_uncalibrated=True),features,mask)
    reviewed=read(read(spec['learning_report'])['reports']['candidate_encoding'][encoder]['path'])
    prediction_file=Path(spec['learning_report']).parent/'candidate_encoding'/encoder/'validation_predictions.npz'
    if sha256(prediction_file)!=reviewed['prediction_sha256']:raise ValueError('Changed reviewed checkpoint predictions')
    with np.load(prediction_file) as z:
        norm=read(root/'member_0/settings.json')['normalization'];mu=np.asarray(norm['target_mean'],np.float32);scale=np.asarray(norm['target_scale'],np.float32)
        expected=dict(mean=z['mean'][:,data['indices'],::2]*scale+mu,variance=z['variance'][:,data['indices'],::2]*scale**2,event_logits=z['event_logits'][:,data['indices'],::2])
    np.testing.assert_array_equal(data['gains'][:,::2],np.broadcast_to(np.geomspace(.5,8,8).astype(np.float32)[None,:,None],(len(mask),8,1)))
    for k in expected:np.testing.assert_allclose(pred['offline_fp32'][k],expected[k],atol=2e-5,rtol=2e-5)
    fit=diagnostic_fit(root/'bundle_fp64',out/'identity_test_fit.json')
    policy=BicycleSelector(root/'bundle_fp64',out/'identity_test_fit.json',reference_recording=True,numerical_test=True)
    cold=policy.warm(8,64);records=[];durations=[];n=len(mask)
    for repeat in range(3):
        for start in range(0,n,8):
            args=[data[k][start:start+8] for k in FIELDS];t=time.perf_counter()
            r=jax.tree.map(np.asarray,policy.predict(*args))
            durations.append(time.perf_counter()-t)
            if repeat==0:records.append(r)
    record={k:np.concatenate([r[k] for r in records]) for k in records[0]}
    for field,key in [('prediction_mean','mean'),('prediction_variance','variance'),('prediction_event_logits','event_logits')]:
        np.testing.assert_allclose(record[field],np.moveaxis(pred['runtime_fp64'][key],0,1),atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(record['features'],f64,atol=1e-10,rtol=1e-10)
    check_selection(dict(record,active=np.ones(n,bool),previous_gain=data['previous_gain']),dict(policy_config=asdict(BicyclePolicyConfig()),phase='gate_calibration'),fit)
    if policy._function._cache_size() or len(policy._compiled)!=1:raise ValueError('Unexpected live policy compilation')
    np.savez_compressed(out/'predictions.npz',indices=data['indices'],features_fp32=f32,features_fp64=f64,features_offline=offline,node_mask=mask,
        **{v+'_'+k:a for v,p in pred.items() for k,a in p.items()})
    np.savez_compressed(out/'selector.npz',**record)
    result=dict(encoder=encoder,backend=backend,queries=n,graph_proof_fp32=proof32,graph_proof_fp64=proof64,timing=timing,
        policy_timing=dict(compile_seconds=cold,p50_batch_seconds=float(np.median(durations)),p95_batch_seconds=float(np.percentile(durations,95)),batch=8,
            sample_seconds=durations,implicit_jit_cache_entries=0,compiled_signatures=1,mode='diagnostic fixed-reference selection, identity calibration; no physical controller solve'),
        exported_checkpoint_predictions_match_review=True,live_policy_matches_direct_inference=True,independent_selection_verified=True,
        predictions_sha256=sha256(out/'predictions.npz'),selector_sha256=sha256(out/'selector.npz'),identity_test_fit_sha256=sha256(out/'identity_test_fit.json'),
        calibration_fitted=False,benchmark_parents_used=False,reserved_calibration_parents_used=False,model_promoted=False)
    write_json(out/'report.json',result);print(result,flush=True)


def run(protocol,directory):
    spec=read(protocol);check_files(spec['bound_files']);job=Path(directory);out=Path(spec['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    import shutil
    if shutil.disk_usage(out).free/2**30<spec['minimum_free_gib']+1.:raise ValueError('Insufficient runtime artifact reserve')
    proof=read(spec['learning_review']);report=read(spec['learning_report'])
    if (proof['status']!='passed' or proof['report_sha256']!=sha256(spec['learning_report']) or not proof['warrants_runtime_integration']
            or not proof['all_source_parent_feature_checkpoint_and_metric_bindings_verified'] or not all(proof['development_checks'].values())
            or not proof['matched_encoder_only_training_verified'] or not proof['only_candidate_conditioned_encoding_changed']):
        raise ValueError('Positive independently reviewed frozen-model learning required')
    for encoder in ('gat','matched_fc'):
        entry=report['reports']['candidate_encoding'][encoder];check_files({entry['path']:entry['sha256']})
        r=read(entry['path']);check_files(r['bound_checkpoints'])
        if Path(r['training']).resolve()!=Path(spec['training']).resolve():raise ValueError('Unreviewed model source')
    def progress(stage):
        v=dict(stage=stage,elapsed_seconds=time.monotonic()-start,estimated_remaining_seconds=600);write_json(job/'progress.json',v);print(v,flush=True)
    progress('verify_original_observed_inputs')
    inputs=prepare(spec,out);spec.update(inputs=str(inputs));spec['bound_files'][str(inputs)]=sha256(inputs);write_json(job/'protocol.json',spec)
    def execute(name,module,args,slot=0,backend='cpu'):
        env=os.environ.copy();env.update(JAX_PLATFORMS='cpu' if backend=='cpu' else 'cuda,cpu',CUDA_VISIBLE_DEVICES=str(slot),
            JAX_EXPLICIT_X64_DTYPES='allow',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
        env.pop('JAX_ENABLE_X64',None);python=Path.cwd()/('.venv-cpu' if backend=='cpu' else '.venv')/'bin/python'
        with (job/(name+'.log')).open('w') as log:
            subprocess.run(['taskset','-c',f'{slot*14}-{slot*14+13}',str(python),'-m','oa_cbf_jax.'+module,*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    progress('export_genuine_weights_and_derive_fp64')
    def export(item):
        slot,e=item;root=Path(spec['training'])/e
        if spec.get('reuse_exported_bundles'):
            from .precision_bundle import validate_derivation
            for name in ('bundle','bundle_fp64'):
                for filename in ('manifest.json','weights.msgpack'):
                    path=str((root/name/filename).resolve())
                    if spec['bound_files'].get(path)!=sha256(path):raise ValueError('Unbound reused genuine bundle')
            validate_derivation(root/'bundle_fp64',read(root/'bundle_fp64/manifest.json'))
            return
        execute(e+'_export','inference',['export-pilot','--members',*[root/f'member_{i}' for i in range(4)],'--output',root/'bundle'],slot)
        execute(e+'_derive','precision_bundle',['--source',root/'bundle','--output',root/'bundle_fp64'],slot)
    with ThreadPoolExecutor(2) as pool:list(pool.map(export,enumerate(('gat','matched_fc'))))
    progress('actual_cpu_gpu_candidate_graph_and_live_selector_qualification')
    def probe(item):
        slot,(e,b)=item;execute(e+'_'+b,'bicycle_candidate_qualification',['worker','--protocol',job/'protocol.json','--encoder',e,'--backend',b],slot,b)
    with ThreadPoolExecutor(4) as pool:list(pool.map(probe,enumerate((('gat','cpu'),('gat','gpu'),('matched_fc','cpu'),('matched_fc','gpu')))))
    models={}
    for e in ('gat','matched_fc'):
        results={b:read(out/e/b/'report.json') for b in ('cpu','gpu')};errors={}
        for filename in ('predictions.npz','selector.npz'):
            with np.load(out/e/'cpu'/filename) as cpu,np.load(out/e/'gpu'/filename) as gpu:
                for k in cpu.files:
                    if cpu[k].dtype.kind in 'biu':np.testing.assert_array_equal(cpu[k],gpu[k])
                    else:
                        tol=1e-10 if k in ('features_fp64','features') else 2e-5
                        np.testing.assert_allclose(cpu[k],gpu[k],atol=tol,rtol=tol)
                        errors[filename+'/'+k]=float(np.max(np.abs(cpu[k]-gpu[k])))
        bundle=Path(spec['training'])/e/'bundle_fp64'
        models[e]=dict(bundle=str(bundle),bundle_manifest_sha256=sha256(bundle/'manifest.json'),weights_sha256=sha256(bundle/'weights.msgpack'),
            parity_passed=True,maximum_errors=errors,selected_backend=min(results,key=lambda b:results[b]['policy_timing']['p50_batch_seconds']),
            reports={b:dict(path=str(out/e/b/'report.json'),sha256=sha256(out/e/b/'report.json')) for b in results})
    result=dict(schema=SCHEMA,status='passed',models=models,inputs_sha256=sha256(inputs),observation_sources_sha256=sha256(out/'observation_sources.json'),
        protocol_sha256=sha256(job/'protocol.json'),dataset_manifest_sha256=sha256(Path(spec['dataset'])/'manifest.json'),
        dataset_index_sha256=sha256(Path(spec['dataset'])/'index.json'),learning_review_sha256=sha256(spec['learning_review']),
        calibration_fitted=False,benchmark_parents_used=False,reserved_calibration_parents_used=False,model_promoted=False,whole_goal_complete=False,
        limitation='Candidate-conditioned observed graph, raw prediction and fixed-reference numerical qualification only. Reserved prediction calibration, adaptive physical timing/trajectory gates and fresh navigation still required.')
    write_json(out/'report.json',result);check_files(spec['bound_files'])
    write_json(job/'complete.json',dict(status='completed',report=str(out/'report.json'),report_sha256=sha256(out/'report.json'),elapsed_seconds=time.monotonic()-start,whole_goal_complete=False));progress('completed')


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('run');q.add_argument('--protocol',required=True);q.add_argument('--directory',required=True)
    q=sub.add_parser('worker');q.add_argument('--protocol',required=True);q.add_argument('--encoder',required=True);q.add_argument('--backend',required=True)
    a=vars(p.parse_args());action=a.pop('action');globals()[action](**a)
