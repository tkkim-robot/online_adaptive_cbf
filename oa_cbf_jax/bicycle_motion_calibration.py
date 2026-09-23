"""Reviewed history inference and observation-only reserved calibration inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from .bicycle_experiment import read
from .bicycle_constraint_runtime import check_files
from .dataset import sha256
from .io import write_json


def validate_qualification(review_path,bundle,dataset):
    if review_path is None:raise ValueError('Independent motion runtime qualification is required')
    review=read(review_path);report=read(review['report']);meta=read(Path(bundle)/'manifest.json')
    from .bicycle_motion_runtime import validate_metadata
    from .bicycle_motion_qualification import SCHEMA
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
    from .bicycle_history_sources import source_map
    from .bicycle_motion_observer import WINDOW_TICKS
    if not np.all(data['partition']=='development_calibration'):raise ValueError('Only original reserved calibration queries')
    mapping,bindings=source_map(dataset);check_files(bindings)
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
        file,record=item;check_files({file:record['sha256']})
        with np.load(file) as z:obs=z['observed_obstacles'];mask=z['mask'];noise=z['noise']
        for r in record['rows']:
            i,t,p=r['index'],r['tick'],r['past_tick']
            if not 0<=p<=t<len(obs):raise ValueError('Unavailable causal reserved history')
            np.testing.assert_array_equal(data['observed_obstacles'][i],obs[t]);np.testing.assert_array_equal(data['obstacle_mask'][i],mask);np.testing.assert_array_equal(data['noise'][i],noise)
            past[i]=obs[p,:,:2]
    with ThreadPoolExecutor(14) as pool:list(pool.map(read_one,requests.items()))
    check_files(bindings)
    proof=dict(schema='bicycle_reserved_observed_motion_history',dataset_manifest_sha256=sha256(Path(dataset)/'manifest.json'),
        dataset_index_sha256=sha256(Path(dataset)/'index.json'),source_indices=bindings,sources=requests,queries=n,
        parents=len(set(data['group_id'])),dt=dt,window_ticks=WINDOW_TICKS,loaded_acquisition_fields=['observed_obstacles','mask','noise'],
        physical_truth_loaded=False,labels_used_for_history=False,all_current_observations_checked=True)
    return past,elapsed.astype(np.float64),proof


def run(protocol,directory):
    spec=read(protocol);check_files(spec['bound_files']);job=Path(directory);start=time.monotonic();training=Path(spec['training'])
    report=read(read(spec['runtime_review'])['report'])
    for e in ('gat','matched_fc'):validate_qualification(spec['runtime_review'],training/e/'bundle_fp64',spec['dataset'])
    write_json(job/'protocol.json',spec);times={}
    def calibrate(item,phase):
        slot,e=item;backend=report['models'][e]['selected_backend']
        env=os.environ.copy();env.update(JAX_PLATFORMS='cpu' if backend=='cpu' else 'cuda,cpu',CUDA_VISIBLE_DEVICES=str(slot),JAX_EXPLICIT_X64_DTYPES='allow',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false');env.pop('JAX_ENABLE_X64',None)
        python=Path.cwd()/('.venv-cpu' if backend=='cpu' else '.venv')/'bin/python';begin=time.monotonic()
        with (job/(e+'_'+phase+'.log')).open('w') as log:
            subprocess.run(['taskset','-c',f'{slot*28}-{slot*28+27}',str(python),'-m','oa_cbf_jax.bicycle_predictive_calibration',
                '--bundle',str(training/e/'bundle_fp64'),'--dataset',spec['dataset'],'--output',str(training/e/'prediction_calibration'),
                '--variance-method','mixture_likelihood','--phase',phase,'--runtime-qualification',spec['runtime_review']],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return time.monotonic()-begin
    for phase in ('fit','audit'):
        progress=dict(stage='reserved_prediction_'+phase,elapsed_seconds=time.monotonic()-start,estimated_remaining_seconds=1200 if phase=='fit' else 600)
        write_json(job/'progress.json',progress);print(progress,flush=True)
        with ThreadPoolExecutor(2) as pool:times[phase]=list(pool.map(lambda item:calibrate(item,phase),enumerate(('gat','matched_fc'))))
        if phase=='fit':
            write_json(job/'both_fits_frozen.json',{e:sha256(training/e/'prediction_calibration/prediction_fit.json') for e in ('gat','matched_fc')})
    for e,digest in read(job/'both_fits_frozen.json').items():
        if digest!=sha256(training/e/'prediction_calibration/prediction_fit.json'):raise ValueError('Reserved audit changed frozen fit')
    check_files(spec['bound_files'])
    complete=dict(status='completed',output=str(training),matched_training_settings_verified=True,variance_method='mixture_likelihood',
        runtime_qualification=spec['runtime_review'],runtime_qualification_sha256=sha256(spec['runtime_review']),
        calibration_seconds=times,model_promoted=False,whole_goal_complete=False,elapsed_seconds=time.monotonic()-start)
    write_json(job/'complete.json',complete)
    from .bicycle_bundles import models
    fitted=models(training,job/'complete.json');complete['models']={e:{k:v for k,v in d.items() if k not in ('metadata','calibration')} for e,d in fitted.items()}
    out=Path(spec['output']);out.mkdir(parents=True,exist_ok=False);write_json(out/'report.json',complete)
    write_json(job/'complete.json',dict(complete,report=str(out/'report.json'),report_sha256=sha256(out/'report.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--directory',required=True);run(**vars(p.parse_args()))
