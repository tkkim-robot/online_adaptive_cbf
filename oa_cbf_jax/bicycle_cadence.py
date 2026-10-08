"""Explicit development-only neural cadence with a physical QP every tick.

Skipped neural calls are recorded as missing predictions, never as new model
decisions. The original every-tick deployed/research policy remains unchanged.
"""
import numpy as np


CONTRACT = dict(intervals=[1, 10], physical_every_ticks=1, event_trigger=False,
                diagnostic_only=True, refit_required_before_promotion=True)


def validate_cadence(phase, interval, source):
    if type(interval) is not int or interval < 1:
        raise ValueError('Cadence must be a positive integer')
    if phase == 'cadence_diagnosis':
        if source.get('cadence_development') != CONTRACT or interval not in CONTRACT['intervals']:
            raise ValueError('Explicit frozen cadence development contract required')
    elif interval != 1:
        raise ValueError('Changed cadence is only authorized for cadence_diagnosis')


def held_selection(template, previous):
    """Zero placeholders denote an absent query; -4 denotes a timed gain hold."""
    result = {k: np.zeros_like(v) for k, v in template.items()}
    result['controller_gain'] = np.array(previous, copy=True)
    result['selected_index'].fill(-4)
    return result


def audit_schedule(data, manifest):
    """Reject gain changes or invented neural scores between scheduled calls."""
    n = len(data['active'])
    interval = manifest.get('query_every_ticks', 1)
    validate_cadence(manifest['phase'], interval, manifest)
    if manifest['phase'] != 'cadence_diagnosis':
        if 'neural_query' in data:
            raise ValueError('Unrecognized query-mask schema')
        return np.ones(n, bool)
    query = np.asarray(data['neural_query'])
    if query.dtype != np.dtype(bool):
        raise ValueError('Query marker must be boolean')
    np.testing.assert_array_equal(query, np.arange(n) % interval == 0)
    held = ~query
    np.testing.assert_array_equal(data['controller_gain'][held], data['previous_gain'][held])
    np.testing.assert_array_equal(data['selected_index'][held], np.full(held.sum(), -4))
    absent = ('uncertainty_fallback', 'accepted', 'cs_score', 'finite_member_cvar',
              'adverse_probability', 'predicted_progress', 'ranking_score',
              'features', 'node_mask', 'prediction_mean', 'prediction_variance',
              'prediction_event_logits')
    absent += tuple(k for k in data if k.startswith('incumbent_') or k == 'selection_eligible')
    for key in absent:
        if np.any(data[key][held] != 0):
            raise ValueError('A held tick claimed a neural prediction: '+key)
    return query
