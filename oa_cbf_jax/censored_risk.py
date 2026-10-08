"""Likelihood and tail arithmetic for the recorded clipped clearance cost.

The observation is Y=max(-2,Z), where Z is modeled as Gaussian. There is a
probability atom at -2, not a Gaussian density at that point. Earlier controller
failures remain separately masked; they are not observations of this atom.
"""
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, ndtr, ndtri, logsumexp
import jax

FLOOR = -2.


def contract(components=1):
    if components not in (1,2):raise ValueError('Only one or two risk components are registered')
    result = dict(schema='quad2d_clipped_gaussian_risk_v1',floor=FLOOR,
        recorded_cost='-min(minimum_physical_clearance,0.6)/0.3 = max(-2, latent_cost)',
        distribution='Gaussian latent cost Z; observed cost Y=max(-2,Z).',
        likelihood='At floor: -log Phi((floor-mu)/sigma). Above floor: original Gaussian negative log density.',
        model_heads='Risk mean and variance describe latent Z; progress and event heads retain original meanings.',
        validation='Score observed-cost distribution, including its atom; use clipped mean for risk MSE.',
        normalization='Original TRAIN-only target normalization; floor transformed using the same units.',
        controller_failure_mask='Unchanged; no risk likelihood for an earlier censored failure.',
        safety_threshold_changed=False,development_only=True,
        deployment='Blocked until this distribution has explicit predictive calibration and policy integration.')
    if components == 2:
        result.update(schema='quad2d_clipped_gaussian_mixture_risk_v1',components=2,
            distribution='Two learned Gaussian latent-cost components with softmax weights; observed cost Y=max(-2,Z).',
            likelihood='At floor: negative log of weighted component CDFs. Above floor: negative log of weighted component densities.',
            model_heads='Explicit risk component means, variances and probabilities; ordinary risk mean/variance are latent mixture moments, never a Gaussian tail approximation. Progress/event heads retain original meanings.',
            validation='Proper mixture likelihood including atom, mixture clipped mean, and root-solved mixture quantile; no moment-matched tail substitution.')
    return result


def nll(target, mean, log_variance, floor=FLOOR):
    density = .5*(jnp.exp(-log_variance)*(target-mean)**2+log_variance+jnp.log(2*jnp.pi))
    atom = -log_ndtr((floor-mean)*jnp.exp(-.5*log_variance))
    return jnp.where(target<=floor,atom,density)


def tail_objective_contract(joint_selection=False):
    result = dict(schema='clipped_risk_upper_tail_pinball_v1',mass=.01,weight=1.,normalized_targets=True,
        formula='pinball_0.99(target - max(floor, mu + sigma*Phi^-1(0.99))) / 0.01',
        equivalence='The CVaR variational objective at q minus the observed target; proper minimizer is the 99th percentile.',
        masking='Original observable-risk mask and original equal-parent/opportunity weights.',
        selection='Unchanged clipped Gaussian likelihood + progress NLL + event BCE + progress contrast; tail term is optimizer-only.',
        both_encoders=True,safety_thresholds_changed=False)
    if joint_selection:
        result['selection']='Minimum validation optimization_loss: unchanged clipped likelihood, progress/event/contrast plus this proper quantile score; epoch zero eligible.'
    return result


def tail_loss(target,mean,log_variance,floor=FLOOR):
    q=upper_quantile(mean,jnp.exp(log_variance),.01,floor)
    error=target-q
    return jnp.maximum(.99*error,-.01*error)/.01


def clipped_mean(mean, variance, floor=FLOOR):
    sigma = jnp.sqrt(variance)
    z = (mean-floor)/sigma
    density = jnp.exp(-.5*z*z)/jnp.sqrt(2*jnp.pi)
    return jnp.maximum(floor,floor+(mean-floor)*ndtr(z)+sigma*density)


def upper_quantile(mean, variance, mass=.01, floor=FLOOR):
    return jnp.maximum(floor,mean+jnp.sqrt(variance)*ndtri(1-mass))


def mixture_nll(target, mean, log_variance, logits, floor=FLOOR):
    """Component axis is last; sum atom probabilities or interior densities."""
    terms=-nll(target[...,None],mean,log_variance,floor)
    return -logsumexp(jax.nn.log_softmax(logits,axis=-1)+terms,axis=-1)


def mixture_clipped_mean(mean, variance, weights, floor=FLOOR):
    return jnp.sum(weights*clipped_mean(mean,variance,floor),axis=-1)


def mixture_tail(mean, variance, weights, mass=.01, floor=FLOOR):
    """Exact clipped-mixture quantile/CVaR, with a fixed 48-step solve.

    Even when the upper quantile is at the atom, the positive-part identity
    includes precisely the necessary fraction of its probability mass.
    Invalid distributions fail closed with NaN, not a finite risk approval.
    """
    if not 0<mass<1:raise ValueError('Tail mass must be between zero and one')
    if mean.shape!=variance.shape or mean.shape!=weights.shape or mean.ndim<1:
        raise ValueError('Matching component arrays required')
    sigma=jnp.sqrt(variance)
    component_q=mean+sigma*ndtri(1-mass)
    def step(_,bounds):
        lo,hi=bounds;mid=lo+(hi-lo)*.5
        probability=jnp.sum(weights*ndtr((mean-mid[...,None])/sigma),axis=-1)
        return jnp.where(probability>mass,mid,lo),jnp.where(probability>mass,hi,mid)
    lo,hi=jax.lax.fori_loop(0,48,step,(component_q.min(-1),component_q.max(-1)))
    quantile=jnp.maximum(floor,lo+(hi-lo)*.5)
    z=(mean-quantile[...,None])/sigma
    excess=jnp.maximum(0.,(mean-quantile[...,None])*ndtr(z)+sigma*jnp.exp(-.5*z*z)/jnp.sqrt(2*jnp.pi))
    cvar=quantile+jnp.sum(weights*excess,axis=-1)/mass
    valid=(jnp.all(jnp.isfinite(mean)&jnp.isfinite(variance)&(variance>0)&jnp.isfinite(weights)&(weights>=0),axis=-1)
        & (jnp.abs(weights.sum(-1)-1)<1e-5))
    return dict(quantile=jnp.where(valid,quantile,jnp.nan),cvar=jnp.where(valid,cvar,jnp.nan))
