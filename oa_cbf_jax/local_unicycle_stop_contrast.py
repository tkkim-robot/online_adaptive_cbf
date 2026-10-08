"""Optional supervision of within-scene differences in held-gain stop risk.

This changes only the training objective. Labels remain eight-second held-gain
outcomes; no label, rollout search, or new decision rule enters deployment.
"""
import math
import numpy as np
import jax.numpy as jnp


def contract(weight=1.):
    if not math.isfinite(weight) or weight != 1.:
        raise ValueError('The registered stop-contrast experiment uses weight one')
    return dict(schema='local_unicycle_stop_probability_contrast_v1', weight=weight,
        target='controller_stop', event_index=1, replicas=2,
        objective='MSE of gain-centered predicted probability and paired-replica event frequency.',
        query_weight='Original live-observation parent-bootstrap weights; all flat and nonflat queries retained.',
        masks='Every gain/replica event must be observed; already absorbing queries remain masked.',
        primary_losses='Original Gaussian NLL and collision/controller-stop BCE unchanged.',
        checkpoint_selection='Original validation loss plus declared contrast MSE.',
        horizon_seconds=8., adaptation_seconds=.2, runtime_changed=False)


def validate_population(data, replicas):
    from .training import validate_contrast_layout
    validate_contrast_layout(data, replicas)
    if replicas != 2 or data['events'].shape[-1] != 2:
        raise ValueError('Expected two paired futures and original two event heads')
    live = np.asarray(data['prior_status']) == 0
    mask = np.asarray(data['event_mask'])[..., 1]
    if not np.array_equal(mask, np.broadcast_to(live[:, None], mask.shape)):
        raise ValueError('Every live gain/replica stop outcome must remain supervised')
    values = np.asarray(data['events'])[live, :, 1]
    if not np.isin(values, (0., 1.)).all():
        raise ValueError('Stop outcomes must be factual binary events')


def loss(probability, events, observed, parent_weight, replicas=2):
    """Equivalent to half the mean squared error of all pairwise contrasts.

    Average replicas before centering so they do not become independent scene
    weight. All-gain masks are justified by validate_population, never by which
    gain succeeds. False variation on truly flat targets is also penalized.
    """
    if (probability.ndim != 2 or probability.shape != events.shape
            or probability.shape != observed.shape
            or parent_weight.shape != probability.shape[:1]
            or replicas < 1 or probability.shape[1] % replicas):
        raise ValueError('Invalid stop-contrast tensor layout')
    valid = jnp.all(observed, axis=1)
    # Mask before arithmetic, including for wholly unobserved absorbing rows.
    p = jnp.where(observed, probability, 0.).reshape(len(probability), -1, replicas).mean(-1)
    y = jnp.where(observed, events, 0.).reshape(len(probability), -1, replicas).mean(-1)
    residual = (p-p.mean(-1, keepdims=True))-(y-y.mean(-1, keepdims=True))
    weight = parent_weight*valid
    return jnp.sum(jnp.mean(residual**2, axis=1)*weight)/jnp.maximum(jnp.sum(weight), 1.)
