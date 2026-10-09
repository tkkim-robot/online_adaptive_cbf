"""Uncertainty functions and shared contracts."""

import jax

import jax.numpy as jnp

from jax.scipy.special import ndtri

def gaussian_log_overlap(mean_a, variance_a, mean_b, variance_b):
    v = variance_a + variance_b
    return -.5 * jnp.sum(jnp.log(2 * jnp.pi * v) + (mean_a - mean_b)**2 / v, axis=-1)

@jax.jit
def cs_disagreement(means, variances):
    """Mean pairwise Cauchy–Schwarz divergence, ensemble on penultimate axis.

    Means/variances [..., E, D]. Positive finite variances are a caller contract;
    invalid predictions remain nonfinite for rejection, never silently accepted.
    """
    pair = gaussian_log_overlap(means[..., :, None, :], variances[..., :, None, :],
                                means[..., None, :, :], variances[..., None, :, :])
    own = gaussian_log_overlap(means, variances, means, variances)
    distances = .5 * (own[..., :, None] + own[..., None, :]) - pair
    # Only remove roundoff after using a mathematically nonnegative statistic.
    return jnp.maximum(jnp.mean(distances, axis=(-2, -1)), 0.)

@jax.jit
def worst_member_cvar(means, variances, epsilon=.01):
    """Finite-member ambiguity CVaR, not an upper bound on GMM CVaR."""
    z = ndtri(1 - epsilon)
    coefficient = jnp.exp(-z*z/2) / jnp.sqrt(2*jnp.pi) / epsilon
    return jnp.max(means + jnp.sqrt(variances) * coefficient, axis=-1)

def conformal_threshold(scores, coverage=.95):
    import math
    import numpy as np
    values = np.asarray(scores, float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Calibration requires finite independent-group scores")
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie strictly between zero and one")
    rank = math.ceil((len(values) + 1) * coverage)
    return {"threshold": float(np.sort(values)[rank-1]) if rank <= len(values) else float("inf"),
            "status": "calibrated" if rank <= len(values) else "insufficient_calibration",
            "n_groups": len(values), "rank": rank, "coverage": coverage}


import math

from statistics import NormalDist


from jax.scipy.special import ndtr

def gaussian_mixture_cvar(mean, variance, tail_mass=.01):
    """Return upper-tail VaR/CVaR; ensemble members are on the LAST axis.

    Fixed 40-step bracketed solve, no data-dependent compilation or iteration
    count. The mixture quantile lies between the component quantiles. Evaluate
    q + E[(X-q)+]/tail_mass, which is also an upper objective bound at any q.
    Nonfinite or nonpositive inputs return NaN and must fail admission.
    """
    if not 0 < tail_mass < 1:
        raise ValueError('Tail mass must lie strictly between zero and one')
    if mean.shape != variance.shape or mean.ndim < 1 or mean.shape[-1] < 1:
        raise ValueError('Matching arrays with a nonempty final member axis required')
    sigma = jnp.sqrt(variance)
    component_q = mean + sigma * NormalDist().inv_cdf(1-tail_mass)
    lower, upper = component_q.min(-1), component_q.max(-1)

    def step(_, bounds):
        lo, hi = bounds
        mid = lo + (hi-lo)*.5
        survival = ndtr((mean-mid[..., None])/sigma).mean(-1)
        return jnp.where(survival > tail_mass, mid, lo), jnp.where(survival > tail_mass, hi, mid)

    lower, upper = jax.lax.fori_loop(0, 40, step, (lower, upper))
    q = lower + (upper-lower)*.5
    z = (q[..., None]-mean)/sigma
    excess = (mean-q[..., None])*ndtr(-z) + sigma*jnp.exp(-.5*z*z)/math.sqrt(2*math.pi)
    cvar = q + excess.mean(-1)/tail_mass
    valid = jnp.all(jnp.isfinite(mean) & jnp.isfinite(variance) & (variance > 0), axis=-1)
    return dict(quantile=jnp.where(valid,q,jnp.nan), cvar=jnp.where(valid,cvar,jnp.nan),
                tail_probability=jnp.where(valid,ndtr(-z).mean(-1),jnp.nan))

def numpy_reference_batch(mean, variance, tail_mass=.01):
    """Independent FP64 vectorized CDF inversion for recorded-query audits.

    Unlike the runtime kernel this uses SciPy's CDF and a fixed 64-iteration
    NumPy solve. The scalar Brent/integral tests remain a separate reference.
    """
    import numpy as np
    from scipy.special import ndtr as cdf
    mean,variance=np.asarray(mean,float),np.asarray(variance,float)
    if (not 0<tail_mass<1 or mean.ndim<1 or mean.shape!=variance.shape or not mean.shape[-1]
            or not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance<=0)):
        raise ValueError('Finite matching means, positive variances and valid tail mass required')
    sigma=np.sqrt(variance);bounds=mean+sigma*NormalDist().inv_cdf(1-tail_mass)
    lower,upper=bounds.min(-1),bounds.max(-1)
    for _ in range(64):
        q=(lower+upper)/2;right=cdf((mean-q[...,None])/sigma).mean(-1)>tail_mass
        lower,upper=np.where(right,q,lower),np.where(right,upper,q)
    q=(lower+upper)/2;z=(q[...,None]-mean)/sigma
    excess=(mean-q[...,None])*cdf(-z)+sigma*np.exp(-z*z/2)/np.sqrt(2*np.pi)
    return q,q+excess.mean(-1)/tail_mass


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
