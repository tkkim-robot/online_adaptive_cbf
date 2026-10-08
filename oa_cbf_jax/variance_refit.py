"""Explicit proper-likelihood continuations of a reviewed development ensemble."""


from dataclasses import asdict
from pathlib import Path

from flax import serialization
import jax
import numpy as np

from .bicycle_motion_adapter import frozen_digest
from .dataset import sha256
from .quad2d_static_inputs import read
from .probability_losses import variance_objective_contract


def contract(mode='variance_only'):
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
    contract(mode)
    if mode == 'all_parameters':
        return jax.tree.map(lambda x:np.ones(x.shape, bool), params)
    allowed = jax.tree.map(lambda x:np.zeros(x.shape, bool), params)
    if set(params['output']) != {'kernel','bias'} or params['output']['bias'].shape != (6,):
        raise ValueError('Expected two means, two variances and two events')
    end = 3 if mode == 'clipped_risk_tail' else 4
    allowed['output']['kernel'][:,2:end] = True
    allowed['output']['bias'][2:end] = True
    return allowed


def initialize(model, seed, normalization, bundle, dataset_sha256, mode='variance_only'):
    root = Path(bundle).resolve()
    meta = read(root/'manifest.json')
    contract(mode)
    if mode == 'clipped_risk_tail':
        from .censored_risk import contract as clipped_contract
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
    proof = dict(contract=contract(mode), source_bundle=str(root), source_manifest_sha256=sha256(root/'manifest.json'),
        source_weights_sha256=meta['weights_sha256'], source_seed=seed, source_member=slot,
        frozen_parameters_sha256=frozen_digest(params, allowed),
        trainable_parameters=int(sum(x.sum() for x in jax.tree.leaves(allowed))))
    return params, allowed, proof
