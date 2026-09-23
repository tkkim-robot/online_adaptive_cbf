"""Reviewed candidate encoding and separately reserved predictive calibration."""
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
    if review_path is None:raise ValueError('Independent candidate runtime qualification is required')
    review=read(review_path);report=read(review['report']);meta=read(Path(bundle)/'manifest.json')
    from .bicycle_candidate_features import validate_metadata
    from .bicycle_candidate_qualification import SCHEMA
    validate_metadata(meta);encoder=meta['architecture']['encoder']
    flags=('all_source_and_checkpoint_bindings_verified','all_candidate_and_observed_features_verified',
        'exported_weights_equal_selected_trained_members','cpu_gpu_and_live_selector_parity_verified',
        'independent_selection_statistics_verified','warrants_reserved_calibration')
    if (review.get('schema')!='independent_bicycle_candidate_runtime_review' or review.get('status')!='passed'
            or any(review.get(k) is not True for k in flags) or review['report_sha256']!=sha256(review['report'])
            or review.get('protocol_sha256')!=report.get('protocol_sha256')
            or report.get('schema')!=SCHEMA or report.get('status')!='passed'
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
    if (fit.get('dataset_manifest_sha256')!=report['dataset_manifest_sha256']
            or fit.get('dataset_index_sha256')!=report['dataset_index_sha256']):
        raise ValueError('Candidate calibration dataset differs from runtime qualification')
    return report


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
