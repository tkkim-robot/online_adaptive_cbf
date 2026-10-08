"""Explicit training objectives; predictive likelihood reporting stays separate."""
import math

import jax
import jax.numpy as jnp


def validate_variance_treatment(manifest, encoder, bicycle_constraint_features=False,
                                bicycle_reserve_labels=None, flight_reflection_labels=None):
    """Limit the opt-in objective to the two registered development treatments."""
    bicycle = (encoder in ('gat', 'matched_fc') and bicycle_constraint_features
               and bicycle_reserve_labels is None)
    flight = (encoder in ('gat', 'nearest_fc') and bool(flight_reflection_labels)
              and manifest.get('schema') == 'oa_cbf_quad2d_guided_hurdle_v1'
              and manifest.get('graph_features') == 40
              and manifest.get('weight_fit_authorized') is True
              and not bicycle_constraint_features and bicycle_reserve_labels is None)
    if not (bicycle or flight):
        raise ValueError('Variance objective requires registered bicycle constraints or paired flight development data')


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
