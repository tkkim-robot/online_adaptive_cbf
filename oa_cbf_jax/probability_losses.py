"""Explicit training objectives; predictive likelihood reporting stays separate."""
import math

import jax
import jax.numpy as jnp


def variance_objective_contract(beta):
    if not math.isfinite(beta) or not 0 < beta <= 1:
        raise ValueError('Variance weighting requires a finite beta in (0, 1]')
    return dict(schema='detached_variance_weighted_gaussian_nll', beta=float(beta),
        formula='stop_gradient(exp(beta * log_variance)) * Gaussian_NLL',
        normalized_targets=True, unchanged_target_and_event_masks=True,
        unchanged_parent_weights=True, unchanged_event_and_contrast_losses=True,
        checkpoint_selection='Original unweighted Gaussian NLL + event BCE + progress contrast',
        source='https://arxiv.org/abs/2203.09168',
        implementation_reference='https://github.com/martius-lab/beta-nll',
        limitation='Training objective only; uncertainty must be independently recalibrated before deployment.')


def variance_weighted_nll(nll, log_variance, beta):
    """Detach the entire variance multiplier, including its log argument.

    The beta=0 path returns the original value exactly. Both continuous heads
    use the same beta; weighting is applied before original masked reductions.
    """
    if not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError('Variance beta must be finite and in [0, 1]')
    if beta == 0:
        return nll
    return nll * jax.lax.stop_gradient(jnp.exp(beta * log_variance))
