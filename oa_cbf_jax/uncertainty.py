"""Analytic Gaussian disagreement and explicitly defined tail-risk screens."""

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
