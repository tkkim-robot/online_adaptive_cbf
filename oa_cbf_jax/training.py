"""Independent Flax GAT training with group bootstrap and resumable checkpoints.

One process per GPU avoids peer transfers. Canonical checkpoints use Flax
msgpack (no Python-object pickle); inference calibration is a separate artifact.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from flax.training import train_state

from .dataset import load_dataset,sha256,source_fingerprint
from .io import write_json
from .models import make_model,GATConfig

DATA_KEYS=['features','node_mask','gains','target','target_mask','events','event_mask']


def parent_bootstrap(group_ids,rng):
    ids,inverse,frequencies=np.unique(group_ids,return_inverse=True,return_counts=True)
    draws=rng.integers(0,len(ids),size=len(ids));multiplicity=np.bincount(draws,minlength=len(ids))
    weights=(multiplicity[inverse]/frequencies[inverse]).astype(np.float32)
    return weights,dict(ids=ids.tolist(),multiplicity=multiplicity.tolist(),row_weights=weights.tolist(),
        interpretation='Parent bootstrap with total weight shared across its acquired visits; validation parents equal total weight.')


def normalization(data,parent_weighted=False):
    means=[];scales=[]
    for i in range(data['target'].shape[-1]):
        values=data['target'][...,i][data['target_mask'][...,i]]
        if not len(values):raise ValueError('No observed target values')
        if parent_weighted:
            # Normalize each parent's observed branches/visits to total one.
            # Otherwise long surviving histories also set the target units.
            _,inverse=np.unique(data['group_id'],return_inverse=True)
            valid=data['target_mask'][...,i]
            counts=np.bincount(inverse,weights=valid.sum(axis=1))
            weights=np.broadcast_to((1/np.maximum(counts[inverse],1))[:,None],valid.shape)[valid]
            mean=np.average(values,weights=weights)
            scale=np.sqrt(np.average((values-mean)**2,weights=weights))
        else:mean=np.mean(values);scale=np.std(values)
        means.append(float(mean));scales.append(max(float(scale),.05))
    return dict(target_mean=means,target_scale=scales,features='fixed physical units from the dataset graph schema')


def prepare(data,norm):
    result={k:np.asarray(data[k]) for k in DATA_KEYS}
    result['target']=((result['target']-np.asarray(norm['target_mean']))/np.asarray(norm['target_scale'])).astype(np.float32)
    result['target']=np.where(result['target_mask'],result['target'],0.)
    return result


def progress_contrast_loss(prediction,target,valid,group_weight,replicas):
    """Learn within-query progress differences, averaging paired replicas first.

    Both arguments are normalized continuous progress. A query contributes
    only if all its retained-prefix progress labels are observed. Correlated
    replicas never receive independent scene weight. No variance target or
    privileged runtime input is introduced.
    """
    count=prediction.shape[1]
    if replicas<1 or count%replicas:raise ValueError('Invalid contrast replica layout')
    predicted=prediction.reshape(prediction.shape[0],count//replicas,replicas).mean(-1)
    actual=target.reshape(target.shape[0],count//replicas,replicas).mean(-1)
    difference=(predicted-predicted.mean(-1,keepdims=True))-(actual-actual.mean(-1,keepdims=True))
    weight=group_weight*jnp.all(valid,axis=1)
    return jnp.sum(jnp.mean(difference**2,axis=1)*weight)/jnp.maximum(jnp.sum(weight),1.)


def loss_and_metrics(model,params,batch,group_weight,progress_contrast_weight=0.,replicas=1):
    out=model.apply({'params':params},batch['features'],batch['node_mask'],batch['gains'])
    error=out['mean']-batch['target']
    nll=.5*(jnp.exp(-out['log_variance'])*error**2+out['log_variance']+jnp.log(2*jnp.pi))
    def grouped(value,mask):
        # Equal weight per independent observation group, not per valid branch.
        count=jnp.sum(mask,axis=1)
        per_group=jnp.sum(jnp.where(mask,value,0.),axis=1)/jnp.maximum(count,1)
        weight=(count>0)*group_weight[:,None]
        return jnp.sum(per_group*weight,axis=0)/jnp.maximum(jnp.sum(weight,axis=0),1)
    nll_heads=grouped(nll,batch['target_mask'])
    bce=optax.sigmoid_binary_cross_entropy(out['event_logits'],batch['events'])
    event_heads=grouped(bce,batch['event_mask'])
    loss=jnp.mean(nll_heads)+jnp.mean(event_heads)
    extra={}
    if progress_contrast_weight:
        contrast=progress_contrast_loss(out['mean'][...,1],batch['target'][...,1],batch['target_mask'][...,1],group_weight,replicas)
        extra=dict(base_loss=loss,progress_contrast_mse=contrast)
        loss=loss+progress_contrast_weight*contrast
    return loss,dict(loss=loss,nll=nll_heads,event_bce=event_heads,**extra,
                     normalized_mae=grouped(jnp.abs(error),batch['target_mask']),
                     brier=grouped((jax.nn.sigmoid(out['event_logits'])-batch['events'])**2,batch['event_mask']),
                     variance_mean=jnp.mean(jnp.exp(out['log_variance']),axis=(0,1)))


def make_updates(model,progress_contrast_weight=0.,replicas=1):
    def batch_update(state,batch,weight):
        (_,metrics),grad=jax.value_and_grad(lambda p:loss_and_metrics(model,p,batch,weight,progress_contrast_weight,replicas),has_aux=True)(state.params)
        metrics['gradient_norm']=optax.global_norm(grad)
        return state.apply_gradients(grads=grad),metrics
    @jax.jit
    def epoch_update(state,data,indices,weights):
        def step(s,inputs):
            idx,weight=inputs
            return batch_update(s,jax.tree.map(lambda a:a[idx],data),weight)
        state,metrics=jax.lax.scan(step,state,(indices,weights))
        return state,jax.tree.map(lambda a:jnp.mean(a,axis=0),metrics)
    @jax.jit
    def evaluate(params,data,weights=None):
        return loss_and_metrics(model,params,data,jnp.ones(data['features'].shape[0]) if weights is None else weights,progress_contrast_weight,replicas)[1]
    return epoch_update,evaluate


def initial_state(model,data,seed,learning_rate,total_steps):
    params=model.init(jax.random.key(seed),data['features'][:1],data['node_mask'][:1],data['gains'][:1])['params']
    schedule=optax.warmup_cosine_decay_schedule(learning_rate*.1,learning_rate,max(1,total_steps//20),total_steps,learning_rate*.05)
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(schedule,weight_decay=1e-4))
    # A Python integer step becomes an array after the first update and would
    # otherwise create a second runtime JIT signature immediately after warmup.
    return train_state.TrainState.create(apply_fn=model.apply,params=params,tx=optimizer).replace(step=jnp.int32(0))


def checkpoint(directory,state,metadata):
    """Atomic immutable generation; pointer changes only after both files exist."""
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    stem=f'epoch_{metadata["epoch"]:05d}'
    payload=serialization.to_bytes(state)
    path=root/(stem+'.msgpack');tmp=path.with_suffix('.tmp');tmp.write_bytes(payload);tmp.replace(path)
    info={**metadata,'state_file':path.name,'state_sha256':hashlib.sha256(payload).hexdigest()}
    write_json(root/(stem+'.json'),info);write_json(root/'latest.json',info)
    return info


def restore(directory,template):
    root=Path(directory);info=json.loads((root/'latest.json').read_text())
    path=root/info['state_file']
    if sha256(path)!=info['state_sha256']:raise ValueError('Checkpoint checksum mismatch')
    # Flax restores NumPy leaves. Convert before the warmed function so host
    # leaves do not create a second dispatch signature on checkpoint reload.
    restored=serialization.from_bytes(template,path.read_bytes())
    return jax.tree.map(jnp.asarray,restored),info


def train(dataset,output,seed=0,width=64,layers=2,heads=4,batch_groups=64,epochs=300,learning_rate=3e-4,
          validate_every=5,patience=12,benchmark_only=False,encoder='gat',flight_history_invariant=False,risk_log_variance_min=-10.,scalar_gain_quadratic=False,progress_contrast_weight=0.,quad3d_history_invariant=False,paired_gain_quadratic=False,continuous_log_variance_min=-10.,quad3d_obstacle_pooling=False):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    dataset_manifest=json.loads((Path(dataset)/'manifest.json').read_text())
    if dataset_manifest.get('weight_fit_authorized') is False:
        raise ValueError('Reserved calibration data cannot be used for weight fitting')
    bicycle=dataset_manifest['schema']=='oa_cbf_bicycle_acquired_history_hurdle_v67'
    quad3d=dataset_manifest['schema'] in ('oa_cbf_quad3d_acquired_hurdle_v94','oa_cbf_quad3d_observer_history_hurdle_v98','oa_cbf_quad3d_wide_gain_history_hurdle_v104')
    parent_weighted=bicycle or quad3d
    if quad3d:
        from .quad3d_learning_contract import validate_training_dataset
        validate_training_dataset(dataset)
        if encoder not in ('gat','full_fc'):raise ValueError('Quad3D requires complete observed graph inputs')
    if not math.isfinite(progress_contrast_weight) or not 0<=progress_contrast_weight<=10:
        raise ValueError('Invalid progress contrast weight')
    wide_quad3d=dataset_manifest['schema']=='oa_cbf_quad3d_wide_gain_history_hurdle_v104'
    if scalar_gain_quadratic and (not bicycle or encoder!='gat'):
        raise ValueError('Gain-response experiments require the audited bicycle OA GAT dataset')
    if progress_contrast_weight and (not (bicycle or wide_quad3d) or encoder!='gat'):
        raise ValueError('Gain-response contrast requires an audited bicycle or wide-gain Quad3D OA GAT dataset')
    if (quad3d_history_invariant or paired_gain_quadratic or continuous_log_variance_min!=-10. or quad3d_obstacle_pooling) and (not wide_quad3d or encoder!='gat'):
        raise ValueError('Quad3D variants require the audited wide-gain OA GAT dataset')
    if bicycle:
        from .bicycle_data import validate_training_dataset
        validate_training_dataset(dataset)
    if risk_log_variance_min!=-10. and (encoder!='gat' or dataset_manifest['schema']!='oa_cbf_quad2d_motion_history_hurdle_v1'):
        raise ValueError('Variance-floor pilot is restricted to audited OA flight history data')
    if dataset_manifest['schema']=='oa_cbf_quad2d_motion_history_hurdle_v1':
        from .quad2d_history_contract import validate_dataset
        validate_dataset(dataset)
    if flight_history_invariant and dataset_manifest['schema'] not in ('oa_cbf_quad2d_guided_hurdle_v1','oa_cbf_quad2d_motion_history_hurdle_v1'):
        raise ValueError('History-invariant variant requires the audited fixed-gain guided flight labels')
    if dataset_manifest['schema'] in ('oa_cbf_quad2d_initial_flight_v1','oa_cbf_quad2d_initial_hurdle_v2','oa_cbf_quad2d_guided_hurdle_v1'):
        audit=json.loads((Path(dataset)/'independent_replay.json').read_text())
        if not audit['audit_passed'] or not audit.get('all_collision_bound_branches_audited') or not (audit.get('all_initial_graph_features_independently_checked') or audit.get('all_observed_graph_features_independently_checked')) or audit['manifest_sha256']!=sha256(Path(dataset)/'manifest.json') or audit['index_sha256']!=sha256(Path(dataset)/'index.json'):
            raise ValueError('Complete independently audited flight dataset required')
        if dataset_manifest['schema']=='oa_cbf_quad2d_guided_hurdle_v1' and not (audit.get('all_observed_graph_features_independently_checked') and audit.get('all_guidance_approvals_checked')):
            raise ValueError('Guided observation and approval audit required')
        if 'terminal_transition_distance' in dataset_manifest.get('controller',{}).get('predictive_guidance',{}) and not audit.get('all_physical_task_target_inputs_checked'):
            raise ValueError('Physical terminal performance target audit required')
    raw=load_dataset(dataset,'train');validation=load_dataset(dataset,'validation')
    if set(raw['group_id'])&set(validation['group_id']):raise ValueError('Split leakage')
    norm=normalization(raw,parent_weighted=parent_weighted);data=jax.device_put(prepare(raw,norm));val=jax.device_put(prepare(validation,norm))
    count=len(raw['group_id']);batches=math.ceil(count/batch_groups)
    cfg=GATConfig(width=width,layers=layers,heads=heads,encoder=encoder,flight_history_invariant=flight_history_invariant,risk_log_variance_min=risk_log_variance_min,scalar_gain_quadratic=scalar_gain_quadratic,
        quad3d_history_invariant=quad3d_history_invariant,paired_gain_quadratic=paired_gain_quadratic,continuous_log_variance_min=continuous_log_variance_min,quad3d_obstacle_pooling=quad3d_obstacle_pooling)
    model=make_model(cfg);state=initial_state(model,data,seed,learning_rate,epochs*batches)
    update,evaluate=make_updates(model,progress_contrast_weight,dataset_manifest.get('replicas',1))
    rng=np.random.default_rng(seed+8191)
    bootstrap=np.arange(count,dtype=np.int32) if parent_weighted else rng.integers(0,count,size=count,dtype=np.int32)
    row_weights=np.ones(count,np.float32);validation_weights=None;parent_sampling=None
    if parent_weighted:
        # All visits of one scene share a bootstrap draw; visits are not
        # independent scenes and do not increase that scene's total weight.
        row_weights,parent_sampling=parent_bootstrap(raw['group_id'],rng)
        _,vi,vf=np.unique(validation['group_id'],return_inverse=True,return_counts=True);validation_weights=jnp.asarray((1/vf[vi]).astype(np.float32))
    settings=dict(dataset=str(Path(dataset).resolve()),dataset_manifest_sha256=sha256(Path(dataset)/'manifest.json'),
                  architecture=asdict(cfg),normalization=norm,seed=seed,batch_groups=batch_groups,epochs=epochs,
                  learning_rate=learning_rate,validate_every=validate_every,patience=patience,source_fingerprint=source_fingerprint(),
                  group_bootstrap=bootstrap.tolist(),selection='minimum validation mean NLL plus mean first-event BCE',
                  production_eligible=False,stage=dataset_manifest['stage'],device=str(jax.devices()[0]),
                  graph_features=int(raw['features'].shape[-1]),dataset_schema=dataset_manifest['schema'],events=dataset_manifest['events'],gain_dimension=int(raw['gains'].shape[-1]),
                  gain_domain=dataset_manifest.get('gain_domain',dict(lower=.3,upper=4.)),
                  controller=dataset_manifest.get('controller',dict(sensor_margin_scale=0.)),
                  jax_version=jax.__version__,precision='FP32 neural training; highest matmul precision',
                  targets=dataset_manifest['targets'])
    if parent_weighted:
        settings['parent_sampling']=parent_sampling
        settings['normalization_sampling']='Each physical parent has equal total weight over observed values; training partition only.'
    if bicycle:
        settings['bicycle_contract']={key:dataset_manifest[key] for key in ('config','sensor_schema','graph_schema','horizon_steps','capacity','acquisition_mode','snapshot_ticks','replicas') if key in dataset_manifest}
    if quad3d:
        from .quad3d_learning_contract import CONTRACT_FIELDS
        settings['quad3d_contract']={key:dataset_manifest[key] for key in CONTRACT_FIELDS}
    if progress_contrast_weight:
        settings['progress_contrast_weight']=progress_contrast_weight
        settings['selection']='minimum validation mean NLL plus mean first-event BCE plus declared within-query normalized progress contrast MSE'
    if 'scene_distribution' in dataset_manifest:
        counts,frequencies=np.unique(raw['obstacle_mask'].sum(axis=1),return_counts=True)
        settings.update(training_obstacle_capacity=dataset_manifest['capacity'],
            training_scene_distribution=dataset_manifest['scene_distribution'],
            training_obstacle_count_histogram={str(int(n)):int(v) for n,v in zip(counts,frequencies)})
    settings_path=root/'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text())!=settings:raise ValueError('Training contract changed; choose new run directory')
    write_json(settings_path,settings)
    epoch=0;best=float('inf');stale=0
    if (root/'checkpoints/latest.json').exists():
        state,info=restore(root/'checkpoints',state)
        if info['settings_sha256']!=sha256(settings_path):raise ValueError('Checkpoint/config mismatch')
        epoch=info['epoch'];best=info['best'];stale=info['stale'];rng.bit_generator.state=info['sampler_rng']
    weight=np.concatenate((row_weights,np.zeros(batches*batch_groups-count,np.float32))).reshape(batches,batch_groups)
    def sample_indices():
        ordered=bootstrap[rng.permutation(count)]
        return np.pad(ordered,(0,batches*batch_groups-count)).reshape(batches,batch_groups)
    def sample_weights(indices):
        if not parent_weighted:return weight
        return row_weights[indices]*np.concatenate((np.ones(count,np.float32),np.zeros(batches*batch_groups-count,np.float32))).reshape(batches,batch_groups)
    def validation_evaluate(params):
        return evaluate(params,val,validation_weights) if parent_weighted else evaluate(params,val)
    # Warm exactly the static training and validation signatures; restore state
    # and host RNG so compilation consumes no optimization/sample budget.
    warm_indices=np.pad(bootstrap,(0,batches*batch_groups-count)).reshape(batches,batch_groups)
    t=time.perf_counter();warm_state,warm_metrics=update(state,data,warm_indices,weight)
    jax.block_until_ready(warm_metrics);warm_validation=validation_evaluate(state.params);jax.block_until_ready(warm_validation)
    cold=time.perf_counter()-t
    if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves((warm_metrics,warm_validation))):raise ValueError('Nonfinite warmup')
    if benchmark_only:
        timings=[]
        for _ in range(10):
            t=time.perf_counter();_,metrics=update(state,data,warm_indices,weight);jax.block_until_ready(metrics);timings.append(time.perf_counter()-t)
        validation_timings=[]
        for _ in range(5):
            t=time.perf_counter();metrics=validation_evaluate(state.params);jax.block_until_ready(metrics);validation_timings.append(time.perf_counter()-t)
        write_json(root/'benchmark.json',dict(compile_seconds=cold,p50_epoch_seconds=float(np.median(timings)),
                     p95_epoch_seconds=float(np.percentile(timings,95)),updates_per_epoch=batches,groups_per_epoch=count,
                     warm_epoch_seconds=timings,validation_seconds=validation_timings,p50_validation_seconds=float(np.median(validation_timings)),
                     training_groups_per_second=count/float(np.median(timings)),device=str(jax.devices()[0]),
                     validation={k:np.asarray(v).tolist() for k,v in warm_validation.items()},training_signature_count=update._cache_size()))
        print((root/'benchmark.json').read_text(),flush=True);return
    clock=time.perf_counter()
    for epoch in range(epoch+1,epochs+1):
        tick=time.perf_counter();indices=sample_indices();state,metrics=update(state,data,indices,sample_weights(indices));jax.block_until_ready(metrics)
        if epoch%validate_every==0 or epoch==1 or epoch==epochs:
            validation_metrics=validation_evaluate(state.params);jax.block_until_ready(validation_metrics)
            measured=float(validation_metrics['loss'])
            record=dict(epoch=epoch,step=int(state.step),elapsed_seconds=time.perf_counter()-clock,
                        last_epoch_seconds=time.perf_counter()-tick,compile_seconds=cold,
                        train={k:np.asarray(v).tolist() for k,v in metrics.items()},
                        validation={k:np.asarray(v).tolist() for k,v in validation_metrics.items()},
                        training_signature_count=update._cache_size(),validation_signature_count=evaluate._cache_size())
            if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves((metrics,validation_metrics))):raise ValueError('Nonfinite training state')
            improved=measured<best-1e-4
            best=min(best,measured);stale=0 if improved else stale+1
            metadata=dict(epoch=epoch,best=best,stale=stale,sampler_rng=rng.bit_generator.state,settings_sha256=sha256(settings_path))
            info=checkpoint(root/'checkpoints',state,metadata)
            if improved:write_json(root/'best.json',info)
            with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            write_json(root/'progress.json',record);print(json.dumps(record),flush=True)
            if stale>=patience:break
    write_json(root/'complete.json',dict(status='completed',epoch=epoch,best_validation_loss=best,
                 elapsed_seconds=time.perf_counter()-clock,production_eligible=False,
                 training_signature_count=update._cache_size(),validation_signature_count=evaluate._cache_size()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--width',type=int,default=64)
    p.add_argument('--layers',type=int,default=2);p.add_argument('--heads',type=int,default=4)
    p.add_argument('--batch-groups',type=int,default=64);p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--validate-every',type=int,default=5)
    p.add_argument('--patience',type=int,default=12);p.add_argument('--benchmark-only',action='store_true')
    p.add_argument('--encoder',choices=['gat','legacy_fc','full_fc'],default='gat')
    p.add_argument('--flight-history-invariant',action='store_true')
    p.add_argument('--risk-log-variance-min',type=float,default=-10.)
    p.add_argument('--scalar-gain-quadratic',action='store_true');p.add_argument('--progress-contrast-weight',type=float,default=0.)
    p.add_argument('--quad3d-history-invariant',action='store_true');p.add_argument('--paired-gain-quadratic',action='store_true')
    p.add_argument('--continuous-log-variance-min',type=float,default=-10.)
    p.add_argument('--quad3d-obstacle-pooling',action='store_true')
    train(**vars(p.parse_args()))
