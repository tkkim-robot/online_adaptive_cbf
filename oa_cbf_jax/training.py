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

from .io import load_dataset, sha256, source_fingerprint

from .io import write_json

from .models import make_model, GATConfig

DATA_KEYS=['features','node_mask','gains','target','target_mask','events','event_mask']

def parent_bootstrap(group_ids,rng,eligible=None):
    ids,inverse,frequencies=np.unique(group_ids,return_inverse=True,return_counts=True)
    draws=rng.integers(0,len(ids),size=len(ids));multiplicity=np.bincount(draws,minlength=len(ids))
    weights=(multiplicity[inverse]/frequencies[inverse]).astype(np.float32)
    if eligible is not None:
        from .unicycle_training import parent_live_weights
        weights=parent_live_weights(group_ids,eligible,multiplicity)
    return weights,dict(ids=ids.tolist(),multiplicity=multiplicity.tolist(),row_weights=weights.tolist(),
        interpretation=('Parent bootstrap with total weight shared across LIVE acquired observations; absorbing observations have zero weight.'
            if eligible is not None else 'Parent bootstrap with total weight shared across its acquired visits; validation parents equal total weight.'))

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
    if 'reserve_mean' in norm and 'reserve_target' in data:
        result['reserve_mask']=np.asarray(data['reserve_mask'],bool)
        result['reserve_target']=np.where(result['reserve_mask'],
            (data['reserve_target']-norm['reserve_mean'])/norm['reserve_scale'],0.).astype(np.float32)
    return result

def progress_contrast_loss(prediction,target,valid,group_weight,replicas,normalization='none'):
    """Learn within-query progress differences, averaging paired replicas first.

    Both arguments are normalized continuous progress. A query contributes
    only if all its retained-prefix progress labels are observed. Correlated
    replicas never receive independent scene weight. The opt-in parent_spread
    mode scales by the within-parent label spread, floored at0.05 normalized
    target units. This prevents broad scene-to-scene variation from drowning
    out small gain effects, and penalizes false variation on flat gain targets.
    The scale is a training-loss weight only; no label enters runtime inference.
    """
    count=prediction.shape[1]
    if replicas<1 or count%replicas:raise ValueError('Invalid contrast replica layout')
    if normalization not in ('none','parent_spread'):raise ValueError('Unknown progress contrast normalization')
    predicted=prediction.reshape(prediction.shape[0],count//replicas,replicas).mean(-1)
    actual=target.reshape(target.shape[0],count//replicas,replicas).mean(-1)
    difference=(predicted-predicted.mean(-1,keepdims=True))-(actual-actual.mean(-1,keepdims=True))
    if normalization=='parent_spread':
        spread=jnp.maximum(jnp.std(actual,axis=-1,keepdims=True),.05)
        difference=difference/spread
    weight=group_weight*jnp.all(valid,axis=1)
    return jnp.sum(jnp.mean(difference**2,axis=1)*weight)/jnp.maximum(jnp.sum(weight),1.)

def progress_contrast_supported(manifest, encoder):
    if encoder not in ('gat', 'matched_fc', 'nearest_fc'):
        return False
    from .bicycle_gain_contract import TRAIN_SCHEMA, validate_manifest
    if manifest['schema']==TRAIN_SCHEMA:
        validate_manifest(manifest)
        return True
    if manifest['schema'] in ('oa_cbf_bicycle_acquired_history_hurdle_v67',
                              'oa_cbf_quad3d_wide_gain_history_hurdle_v104'):
        return True
    if manifest['schema'] == 'oa_cbf_quad2d_guided_hurdle_v1':
        from .quad2d_control import contract
        target = contract('terminal_task')
        return (manifest.get('weight_fit_authorized') is True
                and manifest.get('controller', {}).get('performance_target') == target
                and manifest.get('targets', [None, None])[1] == target['target'])
    return False

def validate_contrast_layout(data, replicas):
    """Consecutive replicas must represent the same queried gain, not neighbors."""
    gains = data['gains']
    if not isinstance(replicas, int) or replicas < 1 or gains.shape[1] % replicas:
        raise ValueError('Invalid progress-contrast replica layout')
    shaped = gains.reshape(gains.shape[0], gains.shape[1]//replicas, replicas, gains.shape[-1])
    if not np.array_equal(shaped, np.broadcast_to(shaped[:, :, :1], shaped.shape)):
        raise ValueError('Progress contrast requires consecutive same-gain replicas')

def loss_and_metrics(model,params,batch,group_weight,progress_contrast_weight=0.,replicas=1,progress_contrast_normalization='none',reserve_auxiliary_weight=0.,variance_beta=0.,prediction=None,risk_censor_floor=None,risk_tail_objective=False,local_stop_contrast_weight=0.,flight_viable_progress_weight=0.):
    out=model.apply({'params':params},batch['features'],batch['node_mask'],batch['gains']) if prediction is None else prediction
    error=out['mean']-batch['target']
    nll=.5*(jnp.exp(-out['log_variance'])*error**2+out['log_variance']+jnp.log(2*jnp.pi))
    if risk_censor_floor is not None:
        from .clipped_inference import nll as clipped_nll, mixture_nll
        if 'risk_component_mean' in out:
            risk_nll=mixture_nll(batch['target'][...,0],out['risk_component_mean'],
                out['risk_component_log_variance'],out['risk_component_logits'],risk_censor_floor)
        else:
            risk_nll=clipped_nll(batch['target'][...,0],out['mean'][...,0],out['log_variance'][...,0],risk_censor_floor)
        nll=nll.at[...,0].set(risk_nll)
    elif 'risk_component_mean' in out:
        raise ValueError('Mixture risk outputs require their explicit clipped likelihood')
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
        contrast=progress_contrast_loss(out['mean'][...,1],batch['target'][...,1],batch['target_mask'][...,1],group_weight,replicas,progress_contrast_normalization)
        extra=dict(base_loss=loss,progress_contrast_mse=contrast)
        loss=loss+progress_contrast_weight*contrast
    if local_stop_contrast_weight:
        from .unicycle_training import loss as stop_contrast_loss
        stop_contrast=stop_contrast_loss(jax.nn.sigmoid(out['event_logits'][...,1]),
            batch['events'][...,1],batch['event_mask'][...,1],group_weight,replicas)
        extra.update(base_loss=loss,stop_contrast_mse=stop_contrast)
        loss=loss+local_stop_contrast_weight*stop_contrast
    if reserve_auxiliary_weight:
        # A separate head supplies gradients without diluting either original
        # continuous/event loss. Checkpoint selection still uses primary loss.
        auxiliary=grouped((out['prefix_reserve']-batch['reserve_target'])**2,batch['reserve_mask']).mean()
        extra.update(primary_selection_loss=loss,reserve_auxiliary_mse=auxiliary)
        loss=loss+reserve_auxiliary_weight*auxiliary
    objective=loss
    if risk_tail_objective:
        if risk_censor_floor is None or variance_beta or reserve_auxiliary_weight:
            raise ValueError('Tail objective requires clipped-risk likelihood without other auxiliary objectives')
        from .clipped_inference import tail_loss
        tail=tail_loss(batch['target'][...,0],out['mean'][...,0],out['log_variance'][...,0],risk_censor_floor)
        # Use exactly the existing parent/mask reduction, not branch weighting.
        tail_mean=grouped(tail[...,None],batch['target_mask'][...,:1])[0]
        objective=objective+tail_mean
        extra.update(primary_selection_loss=loss,optimization_loss=objective,upper_tail_pinball=tail_mean)
    if variance_beta:
        from .uncertainty import variance_weighted_nll
        weighted=grouped(variance_weighted_nll(nll,out['log_variance'],variance_beta),batch['target_mask'])
        objective=jnp.mean(weighted)+jnp.mean(event_heads)
        if progress_contrast_weight:objective+=progress_contrast_weight*contrast
        if reserve_auxiliary_weight:raise ValueError('Variance weighting cannot be combined with a reserve auxiliary')
        extra.update(primary_selection_loss=loss,optimization_loss=objective,variance_weighted_nll=weighted)
    if flight_viable_progress_weight:
        from .training_extensions import loss as viable_progress_loss
        viable=viable_progress_loss(out['mean'][...,1],batch['target'][...,1],
            batch['target_mask'][...,1],batch['events'],batch['event_mask'],group_weight,replicas)
        objective=objective+flight_viable_progress_weight*viable
        extra.update(viable_progress_contrast_mse=viable,optimization_loss=objective,primary_selection_loss=loss)
    return objective,dict(loss=loss,nll=nll_heads,event_bce=event_heads,**extra,
                     normalized_mae=grouped(jnp.abs(error),batch['target_mask']),
                     brier=grouped((jax.nn.sigmoid(out['event_logits'])-batch['events'])**2,batch['event_mask']),
                     variance_mean=jnp.mean(jnp.exp(out['log_variance']),axis=(0,1)))

def make_updates(model,progress_contrast_weight=0.,replicas=1,quad3d_augmentation_seed=None,progress_contrast_normalization='none',reserve_auxiliary_weight=0.,reserve_gradient_guard=False,variance_beta=0.,flight_reflection_seed=None,risk_censor_floor=None,risk_tail_objective=False,local_stop_contrast_weight=0.,local_reflection_seed=None,flight_viable_progress_weight=0.):
    if flight_viable_progress_weight and (reserve_auxiliary_weight or reserve_gradient_guard or variance_beta or risk_tail_objective or local_stop_contrast_weight):
        raise ValueError('Viable progress must be an isolated flight optimization change')
    if local_reflection_seed is not None and (flight_reflection_seed is not None or quad3d_augmentation_seed is not None):
        raise ValueError('Local paired reflection cannot be combined with another orientation treatment')
    if local_stop_contrast_weight and (progress_contrast_weight or reserve_auxiliary_weight or variance_beta or risk_tail_objective):
        raise ValueError('Stop contrast must be an isolated objective experiment')
    if reserve_gradient_guard and not reserve_auxiliary_weight:raise ValueError('Gradient guard requires the reserve auxiliary')
    def batch_update(state,batch,weight):
        if local_reflection_seed is not None:
            from .unicycle_training import select_orientation
            batch=select_orientation(batch,local_reflection_seed,state.step)
        if flight_reflection_seed is not None:
            from .quad2d_training import select_orientation
            batch=select_orientation(batch,flight_reflection_seed,state.step)
        if quad3d_augmentation_seed is not None:
            from .quad3d_features import augment_batch
            batch=augment_batch(batch,quad3d_augmentation_seed,state.step)
        if reserve_gradient_guard:
            from .training_extensions import combine
            def objectives(params):
                _,m=loss_and_metrics(model,params,batch,weight,progress_contrast_weight,replicas,progress_contrast_normalization,reserve_auxiliary_weight,variance_beta,risk_censor_floor=risk_censor_floor,risk_tail_objective=risk_tail_objective,local_stop_contrast_weight=local_stop_contrast_weight,flight_viable_progress_weight=flight_viable_progress_weight)
                return jnp.stack((m['primary_selection_loss'],m['event_bce'][1],m['reserve_auxiliary_mse'])),m
            values,pullback,metrics=jax.vjp(objectives,state.params,has_aux=True)
            basis=jnp.eye(3,dtype=values.dtype)
            gradients=[pullback(basis[i])[0] for i in range(3)]
            grad,guard_metrics=combine(*gradients,weight=reserve_auxiliary_weight)
            metrics.update(guard_metrics)
        else:
            (_,metrics),grad=jax.value_and_grad(lambda p:loss_and_metrics(model,p,batch,weight,progress_contrast_weight,replicas,progress_contrast_normalization,reserve_auxiliary_weight,variance_beta,risk_censor_floor=risk_censor_floor,risk_tail_objective=risk_tail_objective,local_stop_contrast_weight=local_stop_contrast_weight,flight_viable_progress_weight=flight_viable_progress_weight),has_aux=True)(state.params)
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
        prediction=None
        if model.config.bicycle_candidate_encoding:
            # Bound attention memory while retaining the exact complete-query
            # loss reduction and censoring weights. This is not mean-of-batches.
            size=len(data['features']);batch=64;pad=(-size)%batch
            blocks=tuple(jnp.pad(data[key],[(0,pad)]+[(0,0)]*(data[key].ndim-1)).reshape(-1,batch,*data[key].shape[1:])
                         for key in ('features','node_mask','gains'))
            outputs=jax.lax.map(lambda args:model.apply({'params':params},*args),blocks)
            prediction=jax.tree.map(lambda value:value.reshape(-1,*value.shape[2:])[:size],outputs)
        return loss_and_metrics(model,params,data,jnp.ones(data['features'].shape[0]) if weights is None else weights,progress_contrast_weight,replicas,progress_contrast_normalization,reserve_auxiliary_weight,variance_beta,prediction,risk_censor_floor,risk_tail_objective,local_stop_contrast_weight,flight_viable_progress_weight)[1]
    return epoch_update,evaluate

def initial_state(model,data,seed,learning_rate,total_steps,params=None,trainable_mask=None):
    template=model.init(jax.random.key(seed),data['features'][:1],data['node_mask'][:1],data['gains'][:1])['params']
    if params is None:params=template
    elif jax.tree.structure(params)!=jax.tree.structure(template) or any(a.shape!=b.shape for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(template))):
        raise ValueError('Initial parameters do not match model shape')
    params=jax.tree.map(jnp.asarray,params)
    schedule=optax.warmup_cosine_decay_schedule(learning_rate*.1,learning_rate,max(1,total_steps//20),total_steps,learning_rate*.05)
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(schedule,weight_decay=1e-4))
    if trainable_mask is not None:
        from .bicycle_training import mask_transform
        optimizer=optax.chain(mask_transform(trainable_mask),optimizer,mask_transform(trainable_mask))
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
          validate_every=5,patience=12,benchmark_only=False,encoder='gat',flight_history_invariant=False,risk_log_variance_min=-10.,scalar_gain_quadratic=False,progress_contrast_weight=0.,quad3d_history_invariant=False,paired_gain_quadratic=False,continuous_log_variance_min=-10.,quad3d_obstacle_pooling=False,quad3d_quarter_turns=False,progress_contrast_normalization='none',flight_obstacle_pooling=False,flight_gain_basis=False,flight_gain_balanced_sampling=False,bicycle_history_invariant=False,bicycle_safety_balanced_sampling=False,bicycle_obstacle_pooling=False,bicycle_route_context=False,bicycle_reserve_labels=None,bicycle_reserve_gradient_guard=False,bicycle_constraint_features=False,variance_beta=0.,bicycle_affine_gain=False,bicycle_motion_history=None,bicycle_motion_adapter_source=None,bicycle_candidate_encoding=False,benchmark_epochs=10,benchmark_validations=5,flight_local_residual_source=None,flight_reflection_labels=None,variance_refit_source=None,variance_refit_mode='variance_only',clipped_risk_likelihood=False,risk_tail_objective=False,clipped_risk_components=1,flight_additional_dataset=None,flight_additional_reflection=None,local_additional_dataset=None,unicycle_ego_frame=False,unicycle_constraint_features=False,local_stop_contrast_weight=0.,local_prediction_selection=False,local_reflection_base_job=None,local_reflection_shared_job=None,flight_warmstart_source=None,flight_constraint_features=False,flight_gain_attention=False,flight_freeze_original=False,flight_full_parent_ensemble=False,flight_action_dataset=None,flight_action_reflection=None,flight_viable_progress_weight=0.):
    if flight_freeze_original and not (flight_gain_attention and flight_warmstart_source):
        raise ValueError('Frozen flight predictor requires added gain attention and original warm start')
    if flight_gain_attention and not flight_warmstart_source:
        raise ValueError('Gain attention requires its retained flight warm start')
    if flight_constraint_features and not flight_warmstart_source:
        raise ValueError('Flight coefficient treatment requires its retained warm start')
    if flight_warmstart_source and (encoder!='gat' or not flight_history_invariant or not flight_obstacle_pooling
            or variance_refit_source or flight_local_residual_source or bicycle_motion_adapter_source
            or flight_reflection_labels or flight_additional_dataset or variance_beta or clipped_risk_likelihood
            or risk_tail_objective or bicycle_reserve_labels):
        raise ValueError('Flight warm start must retain the original Gaussian training objective and data')
    if flight_local_residual_source and (bicycle_motion_adapter_source or encoder!='gat' or not flight_history_invariant):
        raise ValueError('Local scene residual requires flight GAT only')
    if bicycle_motion_adapter_source and not bicycle_motion_history:
        raise ValueError('Frozen motion adapter requires the observed-history treatment')
    if variance_refit_mode not in ('variance_only','all_parameters','clipped_risk_tail') or (variance_refit_mode != 'variance_only' and not variance_refit_source):
        raise ValueError('Proper-likelihood mode requires an explicit refit source')
    if clipped_risk_components not in (1,2) or (clipped_risk_components==2 and
            (not clipped_risk_likelihood or risk_tail_objective or variance_refit_source)):
        raise ValueError('Two-component clipped risk requires its own likelihood-only trial')
    if (bool(flight_additional_dataset) != bool(flight_additional_reflection)
            or flight_additional_dataset and (not flight_reflection_labels or variance_refit_source
                or not flight_gain_balanced_sampling or encoder not in ('gat','nearest_fc'))):
        raise ValueError('Additional flight TRAIN data requires its audited reflection and shared sampling')
    if (bool(flight_action_dataset) != bool(flight_action_reflection)
            or flight_action_dataset and (not flight_additional_dataset or not flight_full_parent_ensemble)):
        raise ValueError('Action-context coverage requires paired labels and the retained full-parent TRAIN union')
    if (flight_viable_progress_weight not in (0.,1.)
            or flight_viable_progress_weight and (not flight_action_dataset or not flight_full_parent_ensemble
                or variance_beta or variance_refit_source or risk_tail_objective or clipped_risk_likelihood
                or bicycle_reserve_labels or local_stop_contrast_weight)):
        raise ValueError('Viable progress requires the matched action-context Gaussian recipe and fixed weight1')
    clipped_tail_refit = variance_refit_mode == 'clipped_risk_tail'
    if clipped_tail_refit and not (clipped_risk_likelihood and risk_tail_objective):
        raise ValueError('Clipped tail refit requires explicit clipped likelihood and tail objective')
    if variance_refit_source and (variance_beta or bicycle_motion_adapter_source or flight_local_residual_source
            or not flight_reflection_labels or encoder not in ('gat','nearest_fc')):
        raise ValueError('Proper-likelihood refit requires original paired flight heads and unweighted likelihood')
    if not math.isfinite(variance_beta) or not 0 <= variance_beta <= 1:
        raise ValueError('Variance beta must be finite and in [0, 1]')
    if risk_tail_objective and not clipped_risk_likelihood:
        raise ValueError('Upper-tail objective requires the explicit clipped-risk distribution')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    dataset_manifest=json.loads((Path(dataset)/'manifest.json').read_text())
    if dataset_manifest.get('weight_fit_authorized') is False:
        raise ValueError('Reserved calibration data cannot be used for weight fitting')
    if flight_full_parent_ensemble and (dataset_manifest['schema']!='oa_cbf_quad2d_guided_hurdle_v1'
            or encoder not in ('gat','nearest_fc') or not flight_additional_dataset
            or not flight_reflection_labels or variance_refit_source or flight_warmstart_source
            or variance_beta or clipped_risk_likelihood or risk_tail_objective):
        raise ValueError('Full-parent ensemble requires the original expanded Gaussian flight treatment')
    local_unicycle=dataset_manifest['schema']=='unicycle_local_observation_learning_v1'
    if local_unicycle:
        from .unicycle_data import dataset_validate as validate_local_unicycle
        validate_local_unicycle(dataset)
        from .unicycle_policy import validate_bank
        validate_bank(dataset,dataset_manifest)
        if encoder not in ('gat','nearest_fc'):
            raise ValueError('Local unicycle comparison requires GAT or paper nearest-only FC')
    if unicycle_constraint_features and (not local_unicycle or dataset_manifest['config']['clearance_buffer']!=.05):
        raise ValueError('Coefficient study requires the unchanged local unicycle .05 clearance buffer')
    if unicycle_ego_frame and (not local_unicycle or encoder!='gat'):
        raise ValueError('Robot-heading graph requires the local unicycle GAT dataset')
    if clipped_risk_likelihood and (dataset_manifest.get('schema')!='oa_cbf_quad2d_guided_hurdle_v1'
            or dataset_manifest.get('weight_fit_authorized') is not True or dataset_manifest.get('graph_features')!=40
            or encoder not in ('gat','nearest_fc') or not flight_reflection_labels
            or variance_beta or (variance_refit_source and not clipped_tail_refit) or bicycle_reserve_labels
            or bicycle_motion_adapter_source or flight_local_residual_source):
        raise ValueError('Clipped risk pilot requires original paired flight data/encoders and its own likelihood')
    if variance_beta:
        from .uncertainty import validate_variance_treatment
        validate_variance_treatment(dataset_manifest,encoder,bicycle_constraint_features,
                                    bicycle_reserve_labels,flight_reflection_labels)
    from .bicycle_gain_contract import TRAIN_SCHEMA, LEGACY_SCHEMA
    wide_bicycle=dataset_manifest['schema']==TRAIN_SCHEMA
    bicycle=dataset_manifest['schema'] in (LEGACY_SCHEMA,TRAIN_SCHEMA)
    if wide_bicycle and encoder != 'nearest_fc' and (encoder not in ('gat','matched_fc') or not bicycle_candidate_encoding):
        raise ValueError('Wide-gain study requires the matched candidate-conditioned encoders')
    if (type(bicycle_safety_balanced_sampling) is not bool or bicycle_safety_balanced_sampling
            and (not bicycle or encoder not in ('gat','matched_fc','nearest_fc'))):
        raise ValueError('Safety-balanced sampling requires audited matched bicycle training')
    quad3d=dataset_manifest['schema'] in ('oa_cbf_quad3d_acquired_hurdle_v94','oa_cbf_quad3d_observer_history_hurdle_v98','oa_cbf_quad3d_wide_gain_history_hurdle_v104')
    parent_weighted=bicycle or quad3d or local_unicycle
    if quad3d:
        from .quad3d_learning_contract import validate_training_dataset
        validate_training_dataset(dataset)
        if encoder not in ('gat','full_fc','matched_fc','nearest_fc'):raise ValueError('Unregistered Quad3D encoder')
    if not math.isfinite(progress_contrast_weight) or not 0<=progress_contrast_weight<=10:
        raise ValueError('Invalid progress contrast weight')
    if (progress_contrast_normalization not in ('none','parent_spread')
            or (progress_contrast_normalization!='none' and not progress_contrast_weight)):
        raise ValueError('Progress contrast normalization requires an active contrast loss')
    if local_stop_contrast_weight:
        from .unicycle_training import stop_contrast_contract
        stop_contrast_contract(local_stop_contrast_weight)
        if (not local_unicycle or encoder not in ('gat','nearest_fc') or unicycle_constraint_features
                or progress_contrast_weight or bicycle_reserve_labels or variance_beta or risk_tail_objective):
            raise ValueError('Stop contrast requires the original local-unicycle encoders and primary losses')
    if local_prediction_selection:
        from .unicycle_training import contract as prediction_selection_contract, score as prediction_score
        if (not local_unicycle or encoder not in ('gat','nearest_fc') or unicycle_constraint_features
                or local_stop_contrast_weight or progress_contrast_weight or bicycle_reserve_labels
                or variance_beta or risk_tail_objective or variance_refit_source or bicycle_motion_adapter_source):
            raise ValueError('Prediction checkpoint selection requires original local encoders and optimization losses')
    wide_quad3d=dataset_manifest['schema']=='oa_cbf_quad3d_wide_gain_history_hurdle_v104'
    if quad3d_quarter_turns:
        if not wide_quad3d or encoder not in ('gat','matched_fc','nearest_fc'):raise ValueError('Quarter turns require registered wide-gain Quad3D models')
        from .quad3d_features import validate as validate_symmetry
        from .quad3d_control import control_config
        validate_symmetry(control_config(dataset_manifest['config']))
    if scalar_gain_quadratic and (not bicycle or encoder not in ('gat','matched_fc')):
        raise ValueError('Gain-response experiments require the audited bicycle OA GAT dataset')
    if progress_contrast_weight and not progress_contrast_supported(dataset_manifest, encoder):
        raise ValueError('Gain-response contrast requires audited bicycle, wide-gain Quad3D, or authorized terminal-task Quad2D data')
    if (quad3d_history_invariant or paired_gain_quadratic or continuous_log_variance_min!=-10. or quad3d_obstacle_pooling) and (not wide_quad3d or encoder not in ('gat','matched_fc')):
        raise ValueError('Quad3D variants require the audited wide-gain OA GAT dataset')
    if bicycle:
        from .bicycle_data import validate_training_dataset
        validate_training_dataset(dataset)
    if bicycle_candidate_encoding and (not bicycle or not bicycle_constraint_features or bicycle_motion_history or bicycle_motion_adapter_source or variance_beta):
        raise ValueError('Candidate encoding requires original matched constraint features and objective')
    if bicycle_affine_gain and (not bicycle or not bicycle_constraint_features or variance_beta):
        raise ValueError('Affine gain study requires original matched constraint features and objective')
    if bicycle_constraint_features and (not bicycle or encoder not in ('gat','matched_fc')):
        raise ValueError('Current-constraint features require matched bicycle graph35 data')
    if bicycle_history_invariant and (not bicycle or encoder not in ('gat','matched_fc')):
        raise ValueError('Bicycle history invariance requires audited acquired bicycle labels')
    if bicycle_obstacle_pooling and (not bicycle or encoder not in ('gat','matched_fc') or dataset_manifest['graph_features']!=35):
        raise ValueError('Bicycle obstacle pooling requires audited acquired graph35 labels')
    if risk_log_variance_min!=-10. and (encoder not in ('gat','matched_fc') or dataset_manifest['schema']!='oa_cbf_quad2d_motion_history_hurdle_v1'):
        raise ValueError('Variance-floor pilot is restricted to audited OA flight history data')
    if dataset_manifest['schema']=='oa_cbf_quad2d_motion_history_hurdle_v1':
        from .quad2d_history_data import validate_dataset
        validate_dataset(dataset)
    if flight_history_invariant and dataset_manifest['schema'] not in ('oa_cbf_quad2d_guided_hurdle_v1','oa_cbf_quad2d_motion_history_hurdle_v1'):
        raise ValueError('History-invariant variant requires the audited fixed-gain guided flight labels')
    if (flight_obstacle_pooling or flight_gain_basis) and (encoder not in ('gat','matched_fc')
            or dataset_manifest['schema']!='oa_cbf_quad2d_guided_hurdle_v1'
            or dataset_manifest['graph_features']!=40):
        raise ValueError('Planar head variants require audited graph40 guided flight labels')
    if dataset_manifest['schema'] in ('oa_cbf_quad2d_initial_flight_v1','oa_cbf_quad2d_initial_hurdle_v2','oa_cbf_quad2d_guided_hurdle_v1'):
        audit=json.loads((Path(dataset)/'independent_replay.json').read_text())
        if not audit['audit_passed'] or not audit.get('all_collision_bound_branches_audited') or not (audit.get('all_initial_graph_features_independently_checked') or audit.get('all_observed_graph_features_independently_checked')) or audit['manifest_sha256']!=sha256(Path(dataset)/'manifest.json') or audit['index_sha256']!=sha256(Path(dataset)/'index.json'):
            raise ValueError('Complete independently audited flight dataset required')
        if dataset_manifest['schema']=='oa_cbf_quad2d_guided_hurdle_v1' and not (audit.get('all_observed_graph_features_independently_checked') and audit.get('all_guidance_approvals_checked')):
            raise ValueError('Guided observation and approval audit required')
        if 'terminal_transition_distance' in dataset_manifest.get('controller',{}).get('predictive_guidance',{}) and not audit.get('all_physical_task_target_inputs_checked'):
            raise ValueError('Physical terminal performance target audit required')
    raw=load_dataset(dataset,'train');validation=load_dataset(dataset,'validation')
    local_expansion_proof=None
    if local_additional_dataset:
        if not local_unicycle or encoder not in ('gat','nearest_fc'):
            raise ValueError('Shared local coverage requires the registered GAT or nearest-FC unicycle models')
        from .unicycle_data import extend
        raw,validation,local_expansion_proof=extend(dataset,local_additional_dataset,raw,validation)
        write_json(root/'local_training_expansion.json',local_expansion_proof)
    local_reflection_proof=None
    if local_reflection_base_job or local_reflection_shared_job:
        if (not local_reflection_base_job or not local_reflection_shared_job or not local_additional_dataset
                or not local_unicycle or encoder not in ('gat','nearest_fc')
                or flight_reflection_labels or unicycle_constraint_features or local_stop_contrast_weight
                or local_prediction_selection or progress_contrast_weight or variance_beta
                or (encoder=='gat' and not unicycle_ego_frame)):
            raise ValueError('Local reflection requires both complete paired corpora and original ego-frame/nearest-FC treatment')
        from .unicycle_training import load_pairs
        local_reflected,local_reflected_validation,local_reflection_proof=load_pairs(
            dataset,local_additional_dataset,local_reflection_base_job,local_reflection_shared_job,raw,validation)
        # Only original validation selects checkpoints. The paired validation
        # loader above checks its role/order without adding observations to it.
        del local_reflected_validation
    live_query_proof=None
    if local_unicycle:
        from .unicycle_training import live_query_view, parent_live_weights, CONTRACT as LIVE_QUERY_CONTRACT
        raw,train_query_proof=live_query_view(raw)
        validation,validation_query_proof=live_query_view(validation)
        live_query_proof=dict(train=train_query_proof,validation=validation_query_proof)
        if local_reflection_proof is not None:
            local_reflected,reflected_query_proof=live_query_view(local_reflected)
            if reflected_query_proof!=train_query_proof:
                raise ValueError('Reflection changed live-parent supervision')
        write_json(root/'live_query_supervision.json',live_query_proof)
        if local_stop_contrast_weight:
            from .unicycle_training import validate_population
            for population in (raw,validation):validate_population(population,dataset_manifest['replicas'])
    if flight_gain_balanced_sampling and (dataset_manifest['schema']!='oa_cbf_quad2d_guided_hurdle_v1'
            or not progress_contrast_supported(dataset_manifest,encoder)):
        raise ValueError('Gain-opportunity weights require authorized terminal-task Quad2D training data')
    if set(raw['group_id'])&set(validation['group_id']):raise ValueError('Split leakage')
    route_proofs = None
    if bicycle_route_context:
        if not bicycle or encoder not in ('gat', 'matched_fc'):
            raise ValueError('Route-context pilot requires matched bicycle development data')
        from .bicycle_features import route_context_augment_development as augment_development
        raw, train_proof = augment_development(raw, 'train')
        validation, val_proof = augment_development(validation, 'validation')
        route_proofs = dict(train=train_proof, validation=val_proof)
        write_json(root/'route_features.json', route_proofs)
    motion_proofs=None
    if bicycle_motion_history:
        if not bicycle or not bicycle_constraint_features or bicycle_affine_gain or variance_beta or bicycle_route_context or bicycle_reserve_labels:
            raise ValueError('Motion features require the original matched constraint-model treatment')
        from .bicycle_features import augment_development
        raw,train_motion=augment_development(raw,'train',bicycle_motion_history)
        validation,val_motion=augment_development(validation,'validation',bicycle_motion_history)
        motion_proofs=dict(train=train_motion,validation=val_motion)
        write_json(root/'motion_features.json',motion_proofs)
    if progress_contrast_weight:
        for split in (raw, validation):
            validate_contrast_layout(split, dataset_manifest.get('replicas', 1))
    if bicycle_reserve_gradient_guard and not bicycle_reserve_labels:raise ValueError("Guarded auxiliary requires verified labels")
    reserve_proof = None
    if bicycle_reserve_labels:
        if not bicycle or encoder not in ('gat','matched_fc') or bicycle_route_context:
            raise ValueError('Reserve auxiliary requires original matched graph35 bicycle development data')
        from .bicycle_training import attach_labels, reserve_normalization, contract as reserve_contract
        raw, reserve_proof = attach_labels(raw, bicycle_reserve_labels, dataset, 'train')
        validation, val_reserve_proof = attach_labels(validation, bicycle_reserve_labels, dataset, 'validation')
        if reserve_proof != val_reserve_proof: raise ValueError('Different auxiliary label sources')
    norm=normalization(raw,parent_weighted=parent_weighted)
    if reserve_proof is not None: norm.update(reserve_normalization(raw))
    reflection_proof=None;expansion_proof=None
    prepared=prepare(raw,norm)
    if local_reflection_proof is not None:
        from .unicycle_training import pack
        prepared=pack(prepared,prepare(local_reflected,norm))
    if flight_reflection_labels:
        if (encoder not in ('gat','nearest_fc') or bicycle or quad3d or flight_local_residual_source
                or reserve_proof is not None or quad3d_quarter_turns):
            raise ValueError('Paired flight orientations require original GAT or nearest FC heads')
        from .quad2d_training import load_pair, pack
        reflected,reflection_proof=load_pair(dataset,flight_reflection_labels,raw)
        if flight_additional_dataset:
            from .quad2d_training import extend
            raw,reflected,expansion_proof=extend(dataset,flight_additional_dataset,
                flight_additional_reflection,raw,reflected)
            if flight_action_dataset:
                from .quad2d_training import append_contexts
                raw,reflected,expansion_proof=append_contexts(flight_action_dataset,
                    flight_action_reflection,raw,reflected,expansion_proof)
            # Keep the normalization computed from the PRIMARY TRAIN above.
            prepared=prepare(raw,norm)
        prepared=pack(prepared,prepare(reflected,norm))
    data=jax.device_put(prepared);val=jax.device_put(prepare(validation,norm))
    count=len(raw['group_id']);batches=math.ceil(count/batch_groups)
    nearest_options = {}
    if encoder == 'nearest_fc':
        from .nearest_fc import from_manifest, contract as nearest_contract
        dynamics, yaw_scale = from_manifest(dataset_manifest)
        nearest_options = dict(nearest_dynamics=dynamics, nearest_yaw_scale=yaw_scale)
    cfg=GATConfig(**nearest_options,width=width,layers=layers,heads=heads,encoder=encoder,flight_history_invariant=flight_history_invariant,risk_log_variance_min=risk_log_variance_min,scalar_gain_quadratic=scalar_gain_quadratic,
        quad3d_history_invariant=quad3d_history_invariant,paired_gain_quadratic=paired_gain_quadratic,continuous_log_variance_min=continuous_log_variance_min,quad3d_obstacle_pooling=quad3d_obstacle_pooling,flight_obstacle_pooling=flight_obstacle_pooling,flight_gain_basis=flight_gain_basis,bicycle_history_invariant=bicycle_history_invariant,bicycle_obstacle_pooling=bicycle_obstacle_pooling,bicycle_route_context=bicycle_route_context,bicycle_reserve_auxiliary=reserve_proof is not None,bicycle_constraint_features=bicycle_constraint_features,bicycle_affine_gain=bicycle_affine_gain,bicycle_motion_history=bool(bicycle_motion_history),bicycle_candidate_encoding=bicycle_candidate_encoding,flight_local_residual=bool(flight_local_residual_source),unicycle_ego_frame=unicycle_ego_frame,unicycle_constraint_features=unicycle_constraint_features,flight_constraint_features=flight_constraint_features,flight_gain_attention=flight_gain_attention)
    model=make_model(cfg,risk_components=clipped_risk_components);adapter_proof=None;adapter_mask=None;adapter_params=None
    if bicycle_motion_adapter_source:
        from .bicycle_training import initialize as initialize_adapter
        adapter_params,adapter_mask,adapter_proof=initialize_adapter(model,seed,norm,bicycle_motion_adapter_source)
        write_json(root/'adapter_initialization.json',adapter_proof)
    if flight_local_residual_source:
        from .training_extensions import initialize as initialize_residual
        adapter_params,adapter_mask,adapter_proof=initialize_residual(model,data,seed,norm,flight_local_residual_source,sha256(Path(dataset)/'manifest.json'))
        write_json(root/'adapter_initialization.json',adapter_proof)
    if variance_refit_source:
        from .training_extensions import variance_refit_initialize as initialize_variance
        adapter_params,adapter_mask,adapter_proof=initialize_variance(model,seed,norm,variance_refit_source,sha256(Path(dataset)/'manifest.json'),variance_refit_mode)
        write_json(root/'adapter_initialization.json',adapter_proof)
    if flight_warmstart_source:
        from .quad2d_features import initialize as initialize_flight
        adapter_params,adapter_mask,adapter_proof=initialize_flight(model,data,seed,norm,flight_warmstart_source,sha256(Path(dataset)/'manifest.json'),flight_freeze_original)
        write_json(root/'adapter_initialization.json',adapter_proof)
    state=initial_state(model,data,seed,learning_rate,epochs*batches,adapter_params,adapter_mask)
    censor_floor=None
    if clipped_risk_likelihood:
        from .clipped_inference import FLOOR
        censor_floor=float(np.float32((FLOOR-norm['target_mean'][0])/norm['target_scale'][0]))
    update,evaluate=make_updates(model,progress_contrast_weight,dataset_manifest.get('replicas',1),seed+27001 if quad3d_quarter_turns else None,progress_contrast_normalization,.25 if reserve_proof is not None else 0.,bicycle_reserve_gradient_guard,variance_beta,seed+37001 if reflection_proof is not None else None,censor_floor,risk_tail_objective,local_stop_contrast_weight,seed+47001 if local_reflection_proof is not None else None,flight_viable_progress_weight)
    rng=np.random.default_rng(seed+8191)
    from .training_extensions import parent_indices, full_parent_contract
    bootstrap=np.arange(count,dtype=np.int32) if parent_weighted else parent_indices(count,rng,flight_full_parent_ensemble)
    row_weights=np.ones(count,np.float32);validation_weights=None;parent_sampling=None
    opportunity_sampling=None
    if flight_gain_balanced_sampling:
        from .quad2d_training import gain_opportunity_weights
        row_weights,opportunity_sampling=gain_opportunity_weights(raw,dataset_manifest['replicas'])
    if parent_weighted:
        # All visits of one scene share a bootstrap draw; visits are not
        # independent scenes and do not increase that scene's total weight.
        row_weights,parent_sampling=parent_bootstrap(raw['group_id'],rng,raw['prior_status']==0 if local_unicycle else None)
        _,vi,vf=np.unique(validation['group_id'],return_inverse=True,return_counts=True);validation_weights=jnp.asarray((1/vf[vi]).astype(np.float32))
        if local_unicycle:
            validation_weights=jnp.asarray(parent_live_weights(validation['group_id'],validation['prior_status']==0))
    safety_sampling=None
    if bicycle_safety_balanced_sampling:
        from .bicycle_training import safety_opportunity_weights
        multiplier,safety_sampling=safety_opportunity_weights(raw,dataset_manifest['replicas'])
        row_weights*=multiplier
        safety_sampling['effective_row_weights_sha256']=hashlib.sha256(row_weights.tobytes()).hexdigest()
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
    if flight_full_parent_ensemble:
        settings['ensemble_sampling']=full_parent_contract()
    if local_stop_contrast_weight:
        settings['local_unicycle_stop_contrast']=stop_contrast_contract(local_stop_contrast_weight)
        settings['selection']='minimum validation original Gaussian NLL and event BCE plus gain-centered stop-probability MSE'
    if local_prediction_selection:
        settings['local_unicycle_prediction_selection']=prediction_selection_contract()
        settings['selection']='minimum validation mean normalized absolute prediction error plus mean event Brier'
    if parent_weighted:
        settings['parent_sampling']=parent_sampling
        settings['normalization_sampling']='Each physical parent has equal total weight over observed values; training partition only.'
    if flight_gain_attention:
        from .quad2d_features import gain_attention_contract
        settings['flight_gain_attention_contract']=gain_attention_contract()
    if flight_freeze_original:
        from .quad2d_features import frozen_predictor_contract
        settings['flight_frozen_predictor_contract']=frozen_predictor_contract()
    if encoder == 'nearest_fc':
        settings['nearest_fc_contract'] = nearest_contract(dynamics, yaw_scale,unicycle_constraint_features)
    if unicycle_constraint_features:
        from .unicycle_features import contract as coefficient_contract
        settings['unicycle_constraint_features_contract']=coefficient_contract()
    if local_unicycle:
        from .unicycle_calibration import training_contract
        from .unicycle_policy import validate_bank
        validate_bank(dataset,dataset_manifest)
        settings['local_unicycle_contract']=training_contract(dataset_manifest)
        settings['local_unicycle_supervision']=LIVE_QUERY_CONTRACT
        settings['live_query_supervision_sha256']=sha256(root/'live_query_supervision.json')
        if local_expansion_proof is not None:
            settings['local_unicycle_expansion']=local_expansion_proof
    if flight_gain_balanced_sampling:
        settings['flight_gain_balanced_sampling']=True
        settings['gain_opportunity_sampling']=opportunity_sampling
    if bicycle_safety_balanced_sampling:
        settings['bicycle_safety_balanced_sampling']=True
        settings['safety_opportunity_sampling']=safety_sampling
    if bicycle:
        settings['bicycle_contract']={key:dataset_manifest[key] for key in ('config','sensor_schema','graph_schema','horizon_steps','capacity','acquisition_mode','snapshot_ticks','replicas') if key in dataset_manifest}
    if wide_bicycle:
        settings['bicycle_gain_contract']=dataset_manifest['bicycle_gain_contract']
        if 'bicycle_task_progress_contract' in dataset_manifest:
            settings['bicycle_task_progress_contract']=dataset_manifest['bicycle_task_progress_contract']
        settings['offline_wide_gain_pilot']=True
    if bicycle_constraint_features:
        from .bicycle_features import constraint_features_contract as constraint_contract
        settings['bicycle_constraint_features_contract']=constraint_contract()
        settings['offline_constraint_features_pilot']=True
    if bicycle_candidate_encoding:
        from .bicycle_features import contract as candidate_contract
        settings['bicycle_candidate_encoding_contract']=candidate_contract(settings['gain_domain'])
        settings['offline_candidate_encoding_pilot']=True
    if motion_proofs is not None:
        from .bicycle_features import MOTION_FEATURES_SCHEMA as MOTION_SCHEMA, motion_features_contract as motion_contract
        settings['bicycle_contract']['source_graph_schema']=settings['bicycle_contract']['graph_schema']
        settings['bicycle_contract']['graph_schema']=MOTION_SCHEMA
        settings['bicycle_motion_history_contract']=motion_contract()
        settings['bicycle_motion_history_source']=str(Path(bicycle_motion_history).resolve())
        settings['bicycle_motion_history_source_sha256']=sha256(bicycle_motion_history)
        settings['motion_features_sha256']=sha256(root/'motion_features.json')
        settings['offline_motion_history_pilot']=True
    if bicycle_motion_adapter_source:
        from .bicycle_training import motion_adapter_contract as adapter_contract
        settings.update(bicycle_motion_adapter_contract=adapter_contract(),
            bicycle_motion_adapter_source=str(Path(bicycle_motion_adapter_source).resolve()),
            bicycle_motion_adapter_source_sha256=sha256(bicycle_motion_adapter_source),initial_checkpoint_eligible=True)
    if flight_local_residual_source:
        from .training_extensions import flight_local_residual_contract as residual_contract
        settings.update(flight_local_residual_contract=residual_contract(),
            flight_local_residual_source=str(Path(flight_local_residual_source).resolve()),
            flight_local_residual_source_manifest_sha256=sha256(Path(flight_local_residual_source)/'manifest.json'),
            initial_checkpoint_eligible=True)
    if bicycle_affine_gain:
        from .bicycle_control import contract as gain_contract
        settings['bicycle_affine_gain_contract']=gain_contract()
        settings['offline_affine_gain_pilot']=True
    if route_proofs is not None:
        from .bicycle_features import ROUTE_CONTEXT_SCHEMA as ROUTE_SCHEMA, route_context_contract as route_contract
        settings['bicycle_contract']['source_graph_schema'] = settings['bicycle_contract']['graph_schema']
        settings['bicycle_contract']['graph_schema'] = ROUTE_SCHEMA
        settings['bicycle_route_context'] = route_contract()
        settings['route_features_sha256'] = sha256(root/'route_features.json')
        settings['offline_route_context_pilot'] = True
    if reserve_proof is not None:
        settings['bicycle_reserve_auxiliary']=dict(contract=reserve_contract(),labels=reserve_proof)
        settings['offline_reserve_auxiliary_pilot']=True
        if bicycle_reserve_gradient_guard:
            from .training_extensions import contract as gradient_contract
            settings['bicycle_reserve_gradient_guard']=gradient_contract()
    if quad3d:
        from .quad3d_learning_contract import CONTRACT_FIELDS
        settings['quad3d_contract']={key:dataset_manifest[key] for key in CONTRACT_FIELDS}
    if progress_contrast_weight:
        settings['progress_contrast_weight']=progress_contrast_weight
        settings['selection']='minimum validation mean NLL plus mean first-event BCE plus declared within-query normalized progress contrast MSE'
        if progress_contrast_normalization!='none':
            settings['progress_contrast_normalization']=progress_contrast_normalization
            settings['progress_contrast_scale_floor']=.05
    if reflection_proof is not None:
        settings['flight_reflection']=reflection_proof
        settings['flight_reflection_seed']=seed+37001
    if local_reflection_proof is not None:
        settings['local_unicycle_reflection']=local_reflection_proof
        settings['local_unicycle_reflection_seed']=seed+47001
    if expansion_proof is not None:
        settings['flight_training_expansion']=expansion_proof
    if flight_viable_progress_weight:
        from .training_extensions import viable_progress_contract
        settings['flight_viable_progress_contract']=viable_progress_contract()
    if quad3d_quarter_turns:
        settings['quad3d_quarter_turns']=True
        settings['augmentation']=dict(schema='linearized_quad3d_quarter_turn_v1',seed=seed+27001,
            sampling='Independent uniform quarter turn per query/optimizer step, paired across encoders and all gain/replica branches. Masks, targets and group weights unchanged; no validation augmentation.')
    if 'scene_distribution' in dataset_manifest:
        counts,frequencies=np.unique(raw['obstacle_mask'].sum(axis=1),return_counts=True)
        settings.update(training_obstacle_capacity=dataset_manifest['capacity'],
            training_scene_distribution=dataset_manifest['scene_distribution'],
            training_obstacle_count_histogram={str(int(n)):int(v) for n,v in zip(counts,frequencies)})
    if variance_beta:
        from .uncertainty import variance_objective_contract
        settings['variance_weighted_objective']=variance_objective_contract(variance_beta)
        settings['offline_variance_objective_pilot']=True
    if clipped_risk_likelihood:
        from .clipped_inference import contract as clipped_contract
        settings['risk_distribution_contract']=clipped_contract(clipped_risk_components)
        if clipped_risk_components!=1:settings['clipped_risk_components']=clipped_risk_components
        settings['normalized_risk_censor_floor']=censor_floor
        settings['selection']='minimum validation clipped-risk likelihood plus progress Gaussian NLL, event BCE and original progress contrast'
        if clipped_risk_components==2:
            settings['selection']=settings['selection'].replace('clipped-risk likelihood','clipped-risk mixture likelihood')
        if risk_tail_objective:
            from .clipped_inference import tail_objective_contract
            settings['risk_tail_objective']=tail_objective_contract(clipped_tail_refit)
        if clipped_tail_refit:
            settings['selection']='minimum validation optimization_loss: clipped likelihood plus fixed upper99 pinball, progress NLL, event BCE and original progress contrast'
    if variance_refit_source:
        from .training_extensions import variance_refit_contract
        settings.update(variance_refit_contract=variance_refit_contract(variance_refit_mode),
            variance_refit_source=str(Path(variance_refit_source).resolve()),
            variance_refit_source_manifest_sha256=sha256(Path(variance_refit_source)/'manifest.json'),
            initial_checkpoint_eligible=True)
        if variance_refit_mode != 'variance_only':settings['variance_refit_mode']=variance_refit_mode
    if flight_warmstart_source:
        from .quad2d_features import contract as coefficient_contract
        settings.update(flight_warmstart_source=str(Path(flight_warmstart_source).resolve()),
            flight_warmstart_source_manifest_sha256=sha256(Path(flight_warmstart_source)/'manifest.json'),
            initial_checkpoint_eligible=True,flight_constraint_features_contract=coefficient_contract() if flight_constraint_features else None)
    settings_path=root/'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text())!=settings:raise ValueError('Training contract changed; choose new run directory')
    write_json(settings_path,settings)
    epoch=0;best=float('inf');stale=0
    restored_checkpoint=(root/'checkpoints/latest.json').exists()
    if restored_checkpoint:
        state,info=restore(root/'checkpoints',state)
        if info['settings_sha256']!=sha256(settings_path):raise ValueError('Checkpoint/config mismatch')
        epoch=info['epoch'];best=info['best'];stale=info['stale'];rng.bit_generator.state=info['sampler_rng']
    weight=np.concatenate((row_weights,np.zeros(batches*batch_groups-count,np.float32))).reshape(batches,batch_groups)
    def sample_indices():
        ordered=bootstrap[rng.permutation(count)]
        return np.pad(ordered,(0,batches*batch_groups-count)).reshape(batches,batch_groups)
    def sample_weights(indices):
        if not (parent_weighted or flight_gain_balanced_sampling):return weight
        return row_weights[indices]*np.concatenate((np.ones(count,np.float32),np.zeros(batches*batch_groups-count,np.float32))).reshape(batches,batch_groups)
    def validation_evaluate(params):
        return evaluate(params,val,validation_weights) if parent_weighted else evaluate(params,val)
    # Warm exactly the static training and validation signatures; restore state
    # and host RNG so compilation consumes no optimization/sample budget.
    warm_indices=np.pad(bootstrap,(0,batches*batch_groups-count)).reshape(batches,batch_groups)
    warm_weights=sample_weights(warm_indices)
    t=time.perf_counter();warm_state,warm_metrics=update(state,data,warm_indices,warm_weights)
    jax.block_until_ready(warm_metrics);warm_validation=validation_evaluate(state.params);jax.block_until_ready(warm_validation)
    cold=time.perf_counter()-t
    if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves((warm_metrics,warm_validation))):raise ValueError('Nonfinite warmup')
    if benchmark_only:
        if benchmark_epochs<1 or benchmark_validations<1:
            raise ValueError('Benchmark repeats must be positive')
        timings=[]
        for _ in range(benchmark_epochs):
            t=time.perf_counter();_,metrics=update(state,data,warm_indices,warm_weights);jax.block_until_ready(metrics);timings.append(time.perf_counter()-t)
        validation_timings=[]
        for _ in range(benchmark_validations):
            t=time.perf_counter();metrics=validation_evaluate(state.params);jax.block_until_ready(metrics);validation_timings.append(time.perf_counter()-t)
        write_json(root/'benchmark.json',dict(compile_seconds=cold,p50_epoch_seconds=float(np.median(timings)),
                     p95_epoch_seconds=float(np.percentile(timings,95)),updates_per_epoch=batches,groups_per_epoch=count,
                     warm_epoch_seconds=timings,validation_seconds=validation_timings,p50_validation_seconds=float(np.median(validation_timings)),
                     training_groups_per_second=count/float(np.median(timings)),device=str(jax.devices()[0]),
                     validation={k:np.asarray(v).tolist() for k,v in warm_validation.items()},training_signature_count=update._cache_size(),
                     validation_signature_count=evaluate._cache_size()))
        print((root/'benchmark.json').read_text(),flush=True);return
    if adapter_proof is not None and not restored_checkpoint:
        best=float(warm_validation['optimization_loss' if clipped_tail_refit else 'loss'])
        metadata=dict(epoch=0,best=best,stale=0,sampler_rng=rng.bit_generator.state,settings_sha256=sha256(settings_path))
        write_json(root/'best.json',checkpoint(root/'checkpoints',state,metadata))
        with (root/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(dict(epoch=0,step=0,initial_checkpoint=True,
                validation={k:np.asarray(v).tolist() for k,v in warm_validation.items()}))+'\n')
    clock=time.perf_counter()
    for epoch in range(epoch+1,epochs+1):
        tick=time.perf_counter();indices=sample_indices();state,metrics=update(state,data,indices,sample_weights(indices));jax.block_until_ready(metrics)
        if epoch%validate_every==0 or epoch==1 or epoch==epochs:
            if adapter_proof is not None:
                from .bicycle_training import frozen_digest
                if frozen_digest(state.params,adapter_mask)!=adapter_proof['frozen_parameters_sha256']:
                    raise ValueError('Frozen predictor parameters changed during adapter training')
            validation_metrics=validation_evaluate(state.params);jax.block_until_ready(validation_metrics)
            selection_key = ('optimization_loss' if clipped_tail_refit else
                'primary_selection_loss' if reserve_proof is not None else 'loss')
            measured=float(validation_metrics[selection_key])
            if local_prediction_selection:
                measured=prediction_score(validation_metrics)
                validation_metrics=dict(validation_metrics,prediction_selection_score=measured)
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
    p.add_argument('--benchmark-epochs',type=int,default=10);p.add_argument('--benchmark-validations',type=int,default=5)
    p.add_argument('--encoder',choices=['gat','legacy_fc','full_fc','matched_fc','nearest_fc'],default='gat')
    p.add_argument('--flight-history-invariant',action='store_true')
    p.add_argument('--risk-log-variance-min',type=float,default=-10.)
    p.add_argument('--scalar-gain-quadratic',action='store_true');p.add_argument('--progress-contrast-weight',type=float,default=0.)
    p.add_argument('--progress-contrast-normalization',choices=['none','parent_spread'],default='none')
    p.add_argument('--flight-gain-balanced-sampling',action='store_true')
    p.add_argument('--bicycle-history-invariant',action='store_true')
    p.add_argument('--bicycle-obstacle-pooling',action='store_true')
    p.add_argument('--bicycle-route-context',action='store_true')
    p.add_argument('--bicycle-constraint-features',action='store_true')
    p.add_argument('--bicycle-affine-gain',action='store_true')
    p.add_argument('--bicycle-candidate-encoding',action='store_true')
    p.add_argument('--bicycle-motion-history')
    p.add_argument('--bicycle-motion-adapter-source')
    p.add_argument('--flight-local-residual-source')
    p.add_argument('--flight-reflection-labels')
    p.add_argument('--flight-additional-dataset')
    p.add_argument('--local-additional-dataset')
    p.add_argument('--local-reflection-base-job')
    p.add_argument('--local-reflection-shared-job')
    p.add_argument('--flight-additional-reflection')
    p.add_argument('--flight-action-dataset')
    p.add_argument('--flight-action-reflection')
    p.add_argument('--flight-viable-progress-weight',type=float,default=0.,choices=[0.,1.])
    p.add_argument('--variance-beta',type=float,default=0.)
    p.add_argument('--variance-refit-source')
    p.add_argument('--variance-refit-mode',choices=['variance_only','all_parameters','clipped_risk_tail'],default='variance_only')
    p.add_argument('--clipped-risk-likelihood',action='store_true')
    p.add_argument('--risk-tail-objective',action='store_true')
    p.add_argument('--clipped-risk-components',type=int,choices=[1,2],default=1)
    p.add_argument('--bicycle-reserve-labels')
    p.add_argument('--bicycle-reserve-gradient-guard',action='store_true')
    p.add_argument('--bicycle-safety-balanced-sampling',action='store_true')
    p.add_argument('--quad3d-history-invariant',action='store_true');p.add_argument('--paired-gain-quadratic',action='store_true')
    p.add_argument('--continuous-log-variance-min',type=float,default=-10.)
    p.add_argument('--quad3d-obstacle-pooling',action='store_true')
    p.add_argument('--flight-obstacle-pooling',action='store_true')
    p.add_argument('--flight-gain-basis',action='store_true')
    p.add_argument('--flight-warmstart-source')
    p.add_argument('--flight-constraint-features',action='store_true')
    p.add_argument('--unicycle-ego-frame',action='store_true')
    p.add_argument('--unicycle-constraint-features',action='store_true')
    p.add_argument('--local-stop-contrast-weight',type=float,default=0.)
    p.add_argument('--local-prediction-selection',action='store_true')
    p.add_argument('--flight-gain-attention',action='store_true')
    p.add_argument('--flight-freeze-original',action='store_true')
    p.add_argument('--flight-full-parent-ensemble',action='store_true')
    p.add_argument('--quad3d-quarter-turns',action='store_true')
    train(**vars(p.parse_args()))
