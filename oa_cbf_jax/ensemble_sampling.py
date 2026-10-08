"""Explicit all-parent ensemble sampling without selecting ensemble members."""
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


def bootstrap_control_settings(settings, count):
    """Strip only the independently checked sampler treatment for matched review."""
    validate_contract(settings)
    if len(settings['group_bootstrap']) != count:
        raise ValueError('Full-parent sampler count differs from the reserved TRAIN union')
    result = dict(settings)
    result.pop('ensemble_sampling')
    result['group_bootstrap'] = parent_indices(count,
        np.random.default_rng(settings['seed'] + 8191)).tolist()
    return result
