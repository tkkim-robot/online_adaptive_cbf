"""Training extensions functions and shared contracts."""

import jax

import jax.numpy as jnp

def contract():
    return dict(schema='bicycle_reserve_shared_gradient_guard',
        scope='Shared encoder and candidate hidden layers only; separate reserve output keeps its own gradient.',
        rule='Add auxiliary shared gradient only when its cosine with adverse-BCE gradient exceeds1e-4; otherwise zero that contribution.',
        agreement_floor=1e-4,weight=.25,primary_gradient_changed=False,
        limitation='Local raw-gradient heuristic before the unchanged AdamW transform; no Adam-step, validation or physical-safety guarantee.')

def combine(primary, adverse, auxiliary, weight=.25):
    shared=[k for k in primary if k not in ('output','prefix_reserve')]
    dot=sum(jnp.vdot(a,b) for k in shared for a,b in zip(jax.tree.leaves(adverse[k]),jax.tree.leaves(auxiliary[k])))
    aa=sum(jnp.vdot(a,a) for k in shared for a in jax.tree.leaves(adverse[k]))
    bb=sum(jnp.vdot(a,a) for k in shared for a in jax.tree.leaves(auxiliary[k]))
    cosine=dot/jnp.sqrt(jnp.maximum(aa*bb,1e-24))
    # A near-zero event gradient provides no reliably agreeing direction.
    allow=(aa>1e-12)&(bb>1e-12)&(cosine>1e-4)
    guarded={k:jax.tree.map(lambda a:a if k=='prefix_reserve' else jnp.where(allow,a,jnp.zeros_like(a)),v) for k,v in auxiliary.items()}
    result=jax.tree.map(lambda p,a:p+weight*a,primary,guarded)
    return result,dict(auxiliary_shared_gradient_allowed=allow.astype(jnp.float32),auxiliary_adverse_cosine=cosine)


import numpy as np

def full_parent_contract():
    return dict(schema='all_parent_deep_ensemble_v1',
        sampling='Every independent TRAIN parent once per epoch, shuffled separately by member.',
        diversity='All four fixed independent initialization and shuffle seeds retained; no member removal.',
        weighting='Original gain-opportunity weights and actual paired orientation sampling unchanged.',
        random_stream='Consume the original bootstrap draw before replacing its indices, preserving the subsequent host shuffle stream.',
        uncertainty='Original variance heads and maximum-member risk retained; fresh predictive and trajectory calibration required.')

def parent_indices(count, rng, full_parent=False):
    if type(full_parent) is not bool or count < 1:
        raise ValueError('Positive parent count and explicit boolean sampling required')
    sampled = rng.integers(0, count, size=count, dtype=np.int32)
    return np.arange(count, dtype=np.int32) if full_parent else sampled

def validate_contract(settings):
    if settings.get('ensemble_sampling') != full_parent_contract():
        raise ValueError('Unknown full-parent ensemble sampling contract')
    values = np.asarray(settings['group_bootstrap'])
    if not np.array_equal(values, np.arange(len(values))) or not len(values):
        raise ValueError('Full-parent ensemble must include every TRAIN parent exactly once')
    if settings.get('parent_sampling'):
        raise ValueError('Repeated-visit bootstrap cannot be combined with full-parent sampling')


from pathlib import Path

from flax import serialization


from .io import sha256

from .nearest_fc import read

from .bicycle_training import frozen_digest

from .models import GATConfig, make_model

def flight_local_residual_contract():
    return dict(schema='flight_local_scene_residual_v1',
        local_input='Robot plus the one nearest observed obstacle; same explicit paper FC contract.',
        scene_input='Full observed graph40 with existing history invariance and obstacle pooling.',
        predictions='Local Gaussian means/log variances and event logits plus zero-initialized GAT corrections.',
        frozen='All original local predictor parameters, including optimizer weight decay.',
        trained='Scene attention and residual prediction head, on existing training parents only.',
        checkpoint='Epoch zero is eligible; checkpoint selection uses the existing validation objective.',
        comparison='A new hybrid OA variant; graph contribution requires comparison to its frozen local-only predictor.',
        controller_changed=False,labels_changed=False,physical_truth_used=False)

def initialize(model,data,seed,normalization,bundle,dataset_sha256):
    root=Path(bundle).resolve();meta=read(root/'manifest.json')
    from .nearest_fc import validate_metadata
    validate_metadata(meta)
    if (not model.config.flight_local_residual or meta['architecture'].get('nearest_dynamics')!='quad2d'
            or meta['normalization']!=normalization or meta['dataset_manifest_sha256']!=dataset_sha256):
        raise ValueError('Mismatched frozen local model, training data or normalization')
    weights=root/'weights.msgpack'
    if sha256(weights)!=meta['weights_sha256']:raise ValueError('Changed frozen local weights')
    seeds=[r['seed'] for r in meta['members']]
    if seeds.count(seed)!=1:raise ValueError('Use the corresponding independent local member seed')
    member=seeds.index(seed);original=serialization.msgpack_restore(weights.read_bytes())
    local=jax.tree.map(lambda v:np.array(v[member],copy=True),original)
    params=model.init(jax.random.key(seed),data['features'][:1],data['node_mask'][:1],data['gains'][:1])['params']
    params=jax.tree.map(lambda v:np.array(v,copy=True),params)
    if jax.tree.structure(local)!=jax.tree.structure(params['local_model']) or any(
            a.shape!=b.shape for a,b in zip(jax.tree.leaves(local),jax.tree.leaves(params['local_model']))):
        raise ValueError('Frozen local parameter shape mismatch')
    params['local_model']=local
    mask=jax.tree.map(lambda v:np.ones(v.shape,bool),params)
    mask['local_model']=jax.tree.map(lambda v:np.zeros(v.shape,bool),local)
    local_model=make_model(GATConfig(**meta['architecture']))
    args=(data['features'][:2],data['node_mask'][:2],data['gains'][:2])
    a=model.apply({'params':jax.tree.map(jnp.asarray,params)},*args)
    b=local_model.apply({'params':jax.tree.map(jnp.asarray,local)},*args)
    for k in b:np.testing.assert_array_equal(a[k],b[k],err_msg='Epoch-zero local identity')
    proof=dict(contract=flight_local_residual_contract(),source_bundle=str(root),source_manifest_sha256=sha256(root/'manifest.json'),
        source_weights_sha256=sha256(weights),source_member=member,source_seed=seed,
        frozen_parameters_sha256=frozen_digest(params,mask),epoch_zero_exact_identity=True,
        trainable_parameters=int(sum(np.count_nonzero(x) for x in jax.tree.leaves(mask))))
    return params,mask,proof


from dataclasses import asdict


from .quad2d_static_inputs import read as variance_refit_read

from .uncertainty import variance_objective_contract

def variance_refit_contract(mode='variance_only'):
    if mode == 'clipped_risk_tail':
        return dict(schema='flight_clipped_risk_tail_refit_v1',
            source='Corresponding selected member of the paired clipped-likelihood ensemble, without tail training.',
            trained='Only risk log-variance column 2 of the final output kernel and bias.',
            frozen='Both latent means, progress variance, events, representation and all other parameters, including AdamW decay.',
            objective='Mean of clipped-risk and progress NLL, original event/contrast losses, plus fixed normalized upper99 pinball divided by .01.',
            selection='Minimum validation optimization_loss (proper likelihood plus fixed proper quantile score); epoch zero eligible.',
            training_data='Same original/reflected TRAIN pairs, normalization, bootstrap, weights and seeds.',
            calibration='Not calibration; reserved prediction and forward parents untouched.',
            controller_changed=False,safety_thresholds_changed=False)
    if mode == 'all_parameters':
        return dict(schema='flight_proper_likelihood_finetune_v1',
            source='Corresponding selected member of paired beta=.5 development ensemble.',
            trained='All existing parameters; architecture and information boundary unchanged.',
            frozen='No network parameters; original source checkpoints remain immutable.',
            objective='Original unweighted Gaussian NLL plus unchanged event BCE and progress contrast.',
            selection='Original proper validation loss; epoch zero eligible.',
            training_data='Same original/reflected TRAIN parent pairs, normalization, bootstrap, weights and seeds.',
            calibration='Not calibration; original624 reserved calibration parents untouched.',
            controller_changed=False, safety_thresholds_changed=False)
    if mode != 'variance_only':
        raise ValueError('Unknown proper-likelihood treatment')
    return dict(schema='flight_frozen_mean_variance_refit_v1',
        source='Corresponding selected member of paired beta=.5 development ensemble.',
        trained='Only two log-variance columns of final output kernel and bias.',
        frozen='All means, events, attention/dense representations and remaining output columns, including AdamW decay.',
        objective='Original unweighted Gaussian NLL; unchanged event/contrast terms have zero allowed gradients.',
        selection='Original proper validation loss; epoch zero eligible.',
        training_data='Same original/reflected TRAIN parent pairs, normalization, bootstrap, weights and seeds.',
        calibration='Not calibration; original624 reserved calibration parents untouched.',
        controller_changed=False, safety_thresholds_changed=False)

def mask(params, mode='variance_only'):
    variance_refit_contract(mode)
    if mode == 'all_parameters':
        return jax.tree.map(lambda x:np.ones(x.shape, bool), params)
    allowed = jax.tree.map(lambda x:np.zeros(x.shape, bool), params)
    if set(params['output']) != {'kernel','bias'} or params['output']['bias'].shape != (6,):
        raise ValueError('Expected two means, two variances and two events')
    end = 3 if mode == 'clipped_risk_tail' else 4
    allowed['output']['kernel'][:,2:end] = True
    allowed['output']['bias'][2:end] = True
    return allowed

def variance_refit_initialize(model, seed, normalization, bundle, dataset_sha256, mode='variance_only'):
    root = Path(bundle).resolve()
    meta = variance_refit_read(root/'manifest.json')
    variance_refit_contract(mode)
    if mode == 'clipped_risk_tail':
        from .clipped_inference import contract as clipped_contract
        source_valid = (meta.get('risk_distribution_contract') == clipped_contract()
            and not meta.get('risk_tail_objective') and not meta.get('variance_refit_contract'))
    else:
        source_valid = meta.get('variance_weighted_objective') == variance_objective_contract(.5)
    if (model.config.encoder not in ('gat','nearest_fc') or meta['architecture'] != asdict(model.config)
            or meta['normalization'] != normalization or meta['dataset_manifest_sha256'] != dataset_sha256
            or not source_valid
            or not meta.get('flight_reflection')):
        raise ValueError('Expected matching paired flight source for '+mode)
    if sha256(root/'weights.msgpack') != meta['weights_sha256']:
        raise ValueError('Changed source weights')
    seeds = [row['seed'] for row in meta['members']]
    if seeds.count(seed) != 1:
        raise ValueError('Missing corresponding independent member seed')
    slot = seeds.index(seed)
    all_params = serialization.msgpack_restore((root/'weights.msgpack').read_bytes())
    params = jax.tree.map(lambda x:np.array(x[slot], copy=True), all_params)
    allowed = mask(params, mode)
    proof = dict(contract=variance_refit_contract(mode), source_bundle=str(root), source_manifest_sha256=sha256(root/'manifest.json'),
        source_weights_sha256=meta['weights_sha256'], source_seed=seed, source_member=slot,
        frozen_parameters_sha256=frozen_digest(params, allowed),
        trainable_parameters=int(sum(x.sum() for x in jax.tree.leaves(allowed))))
    return params, allowed, proof


def viable_progress_contract():
    return dict(schema='observed_nonadverse_progress_contrast_v1',weight=1.0,
        eligibility='All paired replicas have observed progress and both observed event flags false; at least two eligible gains per parent.',
        objective='Parent-weighted mean squared centered progress error over eligible gains, after averaging common-noise replicas. Normalized original progress units; no spread division.',
        effect='Add only to the optimization objective, alongside every original continuous/event/all-gain-contrast loss. Never remove failed trials from original losses.',
        selection='Original primary validation loss; auxiliary is logged but not a checkpoint selection criterion.',
        inference='No label-conditioned mask, new feature, safety threshold, gain eligibility or architecture change at inference.',
        limitations='Empirical nonadversity of four offline replicas is not a safety certificate.')

def loss(prediction,target,valid,events,event_mask,group_weight,replicas):
    if (replicas<1 or prediction.ndim!=2 or prediction.shape!=target.shape
            or valid.shape!=target.shape or prediction.shape[1]%replicas
            or events.shape!=(*target.shape,2) or event_mask.shape!=events.shape
            or group_weight.shape!=(target.shape[0],)):
        raise ValueError('Complete paired progress/event layout required')
    n,k=prediction.shape; q=k//replicas
    eligible=(valid.reshape(n,q,replicas).all(-1)
        & event_mask.reshape(n,q,replicas,2).all((-1,-2))
        & ~events.astype(bool).reshape(n,q,replicas,2).any((-1,-2)))
    counts=eligible.sum(-1)
    predicted=prediction.reshape(n,q,replicas).mean(-1)
    actual=target.reshape(n,q,replicas).mean(-1)
    error=predicted-actual
    center=jnp.where(eligible,error,0.).sum(-1)/jnp.maximum(counts,1)
    per_parent=jnp.where(eligible,(error-center[:,None])**2,0.).sum(-1)/jnp.maximum(counts,1)
    weight=group_weight*(counts>=2)
    return jnp.sum(per_parent*weight)/jnp.where(weight.sum()>0,weight.sum(),1.)
