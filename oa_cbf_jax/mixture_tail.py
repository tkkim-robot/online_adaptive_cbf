"""Tail expectation of an equally weighted Gaussian predictive ensemble.

This is CVaR of the mixture distribution, not the maximum of the individual
component CVaRs. Neither statistic universally bounds the other. No physical
safety guarantee follows from an unverified predictive distribution.

Positive-part CVaR representation: Rockafellar and Uryasev (2002),
https://doi.org/10.1016/S0378-4266(02)00271-6 .
"""
import math
from statistics import NormalDist

import jax
import jax.numpy as jnp
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
