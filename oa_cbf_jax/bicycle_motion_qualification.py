"""Bound observed-history runtime qualification before any reserved calibration."""
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
from .dataset import sha256
from .io import write_json

SCHEMA='bicycle_motion_history_runtime_qualification'


def load_queries(spec,count=256):
    """Observed cache plus exact original graph arrays from bound query shards."""
    keys=(*FIELDS,'group_id','query_tick','query_origin','partition','gains')
    with np.load(spec['query_cache']) as z:
        if not np.all(z['partition']=='validation'):raise ValueError('Only development validation is authorized')
        indices=np.unique(np.linspace(0,len(z['group_id'])-1,count,dtype=int))
        data={k:z[k][indices] for k in keys}
    index=read(Path(spec['dataset'])/'index.json')
    lookup={(r['group_id'],r['query_origin'],int(r['query_tick'])):r for r in index}
    if len(lookup)!=len(index):raise ValueError('Ambiguous original query identity')
    features=[];masks=[];bindings={}
    for i,(g,o,t) in enumerate(zip(data['group_id'],data['query_origin'],data['query_tick'])):
        r=lookup[(g,o,int(t))];check_files({r['file']:r['sha256']})
        with np.load(r['file']) as z:
            for k in (*FIELDS,'gains','group_id','query_tick'):np.testing.assert_array_equal(data[k][i],z[k][0])
            features.append(z['features'][0]);masks.append(z['node_mask'][0])
        bindings[str(indices[i])]=dict(file=r['file'],sha256=r['sha256'])
    data.update(features=np.stack(features),node_mask=np.stack(masks),indices=indices)
    return data,bindings


def prepare(spec,out):
    from .bicycle_motion_runtime import ObservationHistory
    source=read(spec['history_source']);history=read(source['report'])
    check_files({source['report']:source['report_sha256'],source['review']:source['review_sha256']})
    entry=history['inputs']['validation'];check_files({entry['path']:entry['sha256']})
    data,query_sources=load_queries(spec);indices=data['indices']
    with np.load(entry['path']) as z:
        for key in ('group_id','query_tick','query_origin','observed_obstacles','noise','obstacle_mask'):
            np.testing.assert_array_equal(data[key],z[key][indices])
        past=z['past_positions'][indices];elapsed=z['elapsed'][indices]
    data.update(past_positions=past,history_elapsed=elapsed)
    wanted={int(i):j for j,i in enumerate(indices)};sources=read(spec['history_sources']);proof={};verified=set();ticks=0
    for file,record in sources.items():
        rows=[r for r in record['rows'] if r['role']=='validation' and r['index'] in wanted]
        if not rows:continue
        check_files({file:record['sha256']})
        # Whitelist excludes physical state, true obstacle velocities, errors and labels.
        with np.load(file) as z:observed=z['observed_obstacles'];mask=z['mask'];noise=z['noise']
        maximum=max(r['tick'] for r in rows);memory=ObservationHistory(1,64,spec['dt']);lookup={r['tick']:r for r in rows}
        for tick in range(maximum+1):
            p,e=memory.observe(observed[tick:tick+1],tick);ticks+=1
            if tick in lookup:
                r=lookup[tick];j=wanted[r['index']]
                np.testing.assert_array_equal(observed[tick],data['observed_obstacles'][j])
                np.testing.assert_array_equal(mask,data['obstacle_mask'][j]);np.testing.assert_array_equal(noise,data['noise'][j])
                np.testing.assert_array_equal(p[0],data['past_positions'][j]);np.testing.assert_array_equal(e[0],data['history_elapsed'][j])
                if j in verified:raise ValueError('Duplicate history source')
                verified.add(j)
        proof[file]=dict(sha256=record['sha256'],rows=rows,ticks=maximum+1)
    if verified!=set(range(len(indices))):raise ValueError('Missing runtime history coverage')
    path=out/'inputs.npz';np.savez_compressed(path,**data)
    write_json(out/'history_replay.json',dict(sources=proof,query_sources=query_sources,queries=len(indices),ticks=ticks,
        every_ring_query_equals_original_causal_history=True,physical_truth_loaded=False,labels_loaded=False))
    return path


def observed_graphs(data,config,precision,batch=64):
    import jax
    import jax.numpy as jnp
    from .bicycle_motion_runtime import graph
    from .bicycle_motion_features import numpy_features
    from .bicycle_observed_audit import check_graph
    names=(*FIELDS,'past_positions','history_elapsed')
    fn=jax.jit(jax.vmap(lambda *a:graph(*a,config=config,compute_dtype=precision)));exe=None;parts=[];masks=[]
    for start in range(0,len(data['group_id']),batch):
        ix=np.minimum(np.arange(start,start+batch),len(data['group_id'])-1);n=min(batch,len(data['group_id'])-start)
        args=tuple(jnp.asarray(data[k][ix],dtype=bool if k in ('obstacle_mask','route_mask') else jnp.float64 if k=='history_elapsed' else jnp.float32) for k in names)
        if exe is None:exe=fn.lower(*args).compile()
        f,m=jax.device_get(exe(*args));parts.append(f[:n]);masks.append(m[:n])
    f=np.concatenate(parts);mask=np.concatenate(masks);np.testing.assert_array_equal(mask,data['node_mask'])
    for i in range(len(f)):
        check_graph(f[i,:,:35],mask[i],(data['observed_state'][i],data['goal'][i],data['observed_obstacles'][i],data['obstacle_mask'][i],
            data['points'][i],data['route_mask'][i],data['noise'][i],config,data['cursor'][i],data['previous_control'][i],data['previous_gain'][i]))
        expected=numpy_features(f[i,:,:35],mask[i],data['observed_state'][i],data['observed_obstacles'][i],data['past_positions'][i],data['noise'][i],data['history_elapsed'][i])
        np.testing.assert_allclose(f[i],expected,atol=3e-6,rtol=2e-6)
    if fn._cache_size():raise ValueError('Implicit graph compilation')
    return f,mask,dict(signatures=1,implicit_jit_cache_entries=0,every_observed_graph_and_history_feature_verified=True)


def diagnostic_fit(bundle,path):
    """Identity transformation ONLY for numerical tests; never a calibrated fit."""
    from .bicycle_predictive_calibration import SCHEMA as FIT_SCHEMA
    meta=read(Path(bundle)/'manifest.json')
    result=dict(schema=FIT_SCHEMA,diagnostic_identity_only=True,calibration_fitted=False,
        weights_sha256=meta['weights_sha256'],bundle_manifest_sha256=sha256(Path(bundle)/'manifest.json'),
        bicycle_contract=meta['bicycle_contract'],controller=meta['controller'],variance_scale=[1.,1.],
        event_calibration=[dict(temperature=1.,bias=0.,usable_for_failure_budget=True) for _ in range(2)],
        candidates=np.geomspace(.5,8,8).astype(np.float32)[:,None].tolist())
    write_json(path,result);return result


def worker(protocol,encoder,backend):
    import jax
    from .inference import ResearchPredictor
    from .bicycle_motion_features import numpy_features
    from .bicycle_policy import BicycleSelector,BicyclePolicyConfig
    from .bicycle_policy_audit import check_selection
    spec=read(protocol);check_files(spec['bound_files']);out=Path(spec['output'])/encoder/backend;out.mkdir(parents=True,exist_ok=False)
    with np.load(spec['inputs']) as z:data={k:z[k] for k in z.files}
    config=control_config(read(Path(spec['dataset'])/'manifest.json')['config'])
    offline=np.stack([numpy_features(data['features'][i],data['node_mask'][i],data['observed_state'][i],data['observed_obstacles'][i],data['past_positions'][i],data['noise'][i],data['history_elapsed'][i]) for i in range(len(data['group_id']))])
    f32,mask,proof32=observed_graphs(data,config,'float32');f64,_,proof64=observed_graphs(data,config,'float64')
    root=Path(spec['training'])/encoder;pred={};timing={}
    for name,features,bundle in [('offline_fp32',offline,root/'bundle'),('runtime_fp32',f32,root/'bundle'),('runtime_fp64',f64,root/'bundle_fp64')]:
        pred[name],timing[name]=raw_inference(ResearchPredictor(bundle,allow_uncalibrated=True),features,mask)
    reviewed=read(read(spec['learning_report'])['reports']['motion_adapter'][encoder]['path'])
    prediction_file=Path(spec['learning_report']).parent/'motion_adapter'/encoder/'validation_predictions.npz'
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
            r=jax.tree.map(np.asarray,policy.predict(*args,past_positions=data['past_positions'][start:start+8],history_elapsed=data['history_elapsed'][start:start+8]))
            durations.append(time.perf_counter()-t)
            if repeat==0:records.append(r)
    record={k:np.concatenate([r[k] for r in records]) for k in records[0]}
    for field,key in [('prediction_mean','mean'),('prediction_variance','variance'),('prediction_event_logits','event_logits')]:
        np.testing.assert_allclose(record[field],np.moveaxis(pred['runtime_fp64'][key],0,1),atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(record['features'],f64,atol=1e-10,rtol=1e-10)
    np.testing.assert_array_equal(record['history_past_positions'],data['past_positions']);np.testing.assert_array_equal(record['history_elapsed'],data['history_elapsed'])
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
    proof=read(spec['learning_review']);report=read(spec['learning_report'])
    if (proof['status']!='passed' or proof['report_sha256']!=sha256(spec['learning_report']) or not proof['warrants_runtime_integration']
            or not proof['all_source_parent_feature_checkpoint_and_metric_bindings_verified'] or not all(proof['development_checks'].values())):
        raise ValueError('Positive independently reviewed frozen-model learning required')
    for encoder in ('gat','matched_fc'):
        r=read(report['reports']['motion_adapter'][encoder]['path']);check_files(r['bound_checkpoints'])
        if Path(r['training']).resolve()!=Path(spec['training']).resolve():raise ValueError('Unreviewed model source')
    def progress(stage):
        v=dict(stage=stage,elapsed_seconds=time.monotonic()-start,estimated_remaining_seconds=600);write_json(job/'progress.json',v);print(v,flush=True)
    progress('verify_causal_runtime_memory_on_original_observations')
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
    progress('actual_cpu_gpu_history_graph_and_live_selector_qualification')
    def probe(item):
        slot,(e,b)=item;execute(e+'_'+b,'bicycle_motion_qualification',['worker','--protocol',job/'protocol.json','--encoder',e,'--backend',b],slot,b)
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
    result=dict(schema=SCHEMA,status='passed',models=models,inputs_sha256=sha256(inputs),history_replay_sha256=sha256(out/'history_replay.json'),
        protocol_sha256=sha256(job/'protocol.json'),dataset_manifest_sha256=sha256(Path(spec['dataset'])/'manifest.json'),
        dataset_index_sha256=sha256(Path(spec['dataset'])/'index.json'),learning_review_sha256=sha256(spec['learning_review']),
        calibration_fitted=False,benchmark_parents_used=False,reserved_calibration_parents_used=False,model_promoted=False,whole_goal_complete=False,
        limitation='Observed-history, raw prediction and fixed-reference numerical qualification only. Reserved prediction calibration, adaptive physical timing/trajectory gates and fresh navigation still required.')
    write_json(out/'report.json',result);check_files(spec['bound_files'])
    write_json(job/'complete.json',dict(status='completed',report=str(out/'report.json'),report_sha256=sha256(out/'report.json'),elapsed_seconds=time.monotonic()-start,whole_goal_complete=False));progress('completed')


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('run');q.add_argument('--protocol',required=True);q.add_argument('--directory',required=True)
    q=sub.add_parser('worker');q.add_argument('--protocol',required=True);q.add_argument('--encoder',required=True);q.add_argument('--backend',required=True)
    a=vars(p.parse_args());action=a.pop('action');globals()[action](**a)
