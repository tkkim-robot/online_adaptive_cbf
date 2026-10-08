"""Prospective targets apply only to queries made while acquisition is live.

Keep terminated acquisitions and their factual outcomes in the evidence. They
are not future queries, so their absorbing stop flag must not teach that every
counterfactual gain would fail. Genuine failures after a live query stay targets.
"""
import numpy as np


CONTRACT = dict(schema='local_unicycle_live_query_supervision_v1',
    population='Acquisition is live before the prospective gain query (prior_status == 0).',
    terminal_observations='Retained, with zero supervised target weight; never revived.',
    live_failure_targets='All observed collision, QP, admissibility and bound failures retained.',
    horizon_seconds=8., adaptation_seconds=.2,
    weighting='Equal physical-parent weight shared over live observations; parent bootstrap for training.')


def live_query_view(data):
    prior = np.asarray(data['prior_status'])
    if prior.shape != (len(data['group_id']),) or prior.dtype.kind not in 'iu':
        raise ValueError('One integer prior status per observation is required')
    if not np.isin(prior, (0, 1, 2, 3, 4, 5, 8)).all():
        raise ValueError('Unknown acquisition status')
    live = prior == 0
    result = dict(data)
    for key, values in (('target_mask', 'target'), ('event_mask', 'events')):
        mask = np.asarray(data[key])
        if mask.dtype != bool or mask.ndim != 3 or mask.shape != data[values].shape:
            raise ValueError('Invalid prospective target mask')
        result[key] = mask & live[:, None, None]
    proof = dict(contract=CONTRACT, observations=len(prior), live_observations=int(live.sum()),
        already_terminal_observations=int((~live).sum()),
        excluded_absorbing_stop_labels=int(data['event_mask'][~live, :, 1].sum()),
        live_stop_targets_before=int((data['events'][live, :, 1] * data['event_mask'][live, :, 1]).sum()),
        live_stop_targets_after=int((result['events'][live, :, 1] * result['event_mask'][live, :, 1]).sum()),
        physical_parent_count=len(set(data['group_id'])), outcome_rows_removed=0)
    if proof['live_stop_targets_before'] != proof['live_stop_targets_after']:
        raise ValueError('A genuine live-query failure was removed')
    # Validate that no parent silently disappears from the supervised population.
    parent_live_weights(data['group_id'], live)
    return result, proof


def parent_live_weights(groups, live, multiplicity=None):
    """Parent weight does not depend on how many absorbing visits it produced."""
    groups = np.asarray(groups); live = np.asarray(live, bool)
    if groups.ndim != 1 or live.shape != groups.shape:
        raise ValueError('Invalid live-parent mask')
    ids, inverse = np.unique(groups, return_inverse=True)
    count = np.bincount(inverse, weights=live, minlength=len(ids))
    if np.any(count == 0):
        raise ValueError('Every reserved parent must retain at least one live observation')
    multiplier = np.ones(len(ids)) if multiplicity is None else np.asarray(multiplicity)
    if multiplier.shape != ids.shape or not np.isfinite(multiplier).all() or np.any(multiplier < 0):
        raise ValueError('Invalid parent bootstrap multiplicities')
    return np.where(live, multiplier[inverse] / count[inverse], 0.).astype(np.float32)
