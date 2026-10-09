"""Clipped inference functions and shared contracts."""

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


import json

from pathlib import Path

from flax import serialization


import numpy as np

from .io import sha256

from .inference import ResearchPredictor

from .models import GATConfig, make_model, predict_ensemble

def disagreement(mean, variance, weights):
    """Exact mixed-measure CS divergence; arrays [..., members, components]."""
    a, b = mean[..., :, None, :, None], mean[..., None, :, None, :]
    va, vb = variance[..., :, None, :, None], variance[..., None, :, None, :]
    logw = jnp.log(weights)
    atom = logsumexp(logw + log_ndtr((FLOOR-mean)/jnp.sqrt(variance)), axis=-1)
    product_variance = va*vb/(va+vb)
    product_mean = (a*vb+b*va)/(va+vb)
    overlap = -.5*(jnp.log(2*jnp.pi*(va+vb))+(a-b)**2/(va+vb))
    overlap += log_ndtr((product_mean-FLOOR)/jnp.sqrt(product_variance))
    overlap += logw[..., :, None, :, None]+logw[..., None, :, None, :]
    continuous = logsumexp(overlap, axis=(-2,-1))
    pair = jnp.logaddexp(atom[..., :, None]+atom[..., None, :], continuous)
    own = jnp.diagonal(pair, axis1=-2, axis2=-1)
    distance = .5*(own[..., :, None]+own[..., None, :])-pair
    valid = (jnp.all(jnp.isfinite(mean)&jnp.isfinite(variance)&(variance>0)
                    & jnp.isfinite(weights)&(weights>=0), axis=(-2,-1))
             & jnp.all(jnp.abs(weights.sum(-1)-1)<1e-5, axis=-1))
    return jnp.where(valid, jnp.maximum(distance.mean((-2,-1)), 0.), jnp.nan)

def statistics(raw, normalization, variance_scale, temperature, bias, tail_mass=.01):
    """Member axis first, component axis last; never approximate mixture tails."""
    scale=jnp.asarray(normalization['target_scale'],jnp.float32)
    center=jnp.asarray(normalization['target_mean'],jnp.float32)
    cm=raw['risk_component_mean']*scale[0]+center[0]
    cv=jnp.exp(raw['risk_component_log_variance'])*scale[0]**2*variance_scale[0]
    weights=jax.nn.softmax(raw['risk_component_logits'],axis=-1)
    latent_mean=jnp.sum(weights*cm,axis=-1)
    latent_var=jnp.sum(weights*(cv+(cm-latent_mean[...,None])**2),axis=-1)
    mean=jnp.stack((latent_mean,raw['mean'][...,1]*scale[1]+center[1]),axis=-1)
    variance=jnp.stack((latent_var,jnp.exp(raw['log_variance'][...,1])*scale[1]**2*variance_scale[1]),axis=-1)
    tail=mixture_tail(cm,cv,weights,tail_mass)
    cs=disagreement(*(jnp.moveaxis(a,0,-2) for a in (cm,cv,weights)))
    return dict(mean=mean,variance=variance,
        risk_component_mean=cm,risk_component_variance=cv,risk_component_probability=weights,
        observed_risk_mean=mixture_clipped_mean(cm,cv,weights),
        member_upper_quantile=tail['quantile'],member_cvar=tail['cvar'],
        finite_member_cvar=jnp.max(tail['cvar'],axis=0),disagreement=cs,
        event_logits=raw['event_logits'],
        event_probability=jax.nn.sigmoid(raw['event_logits']/temperature+bias))

class ClippedRiskPredictor:
    """Research-only AOT feature inference with an explicit distribution API.

    This class alone does not enable a physical FlightPolicy or confer calibrated
    safety. Ordinary ResearchPredictor intentionally still refuses these bundles.
    """
    warm_features=ResearchPredictor.warm_features
    predict_features=ResearchPredictor.predict_features

    def __init__(self,bundle,allow_uncalibrated=False,device=None):
        root=Path(bundle)
        self.metadata=json.loads((root/'manifest.json').read_text())
        m=self.metadata
        if not allow_uncalibrated:
            raise ValueError('Explicit research-only clipped-mixture inference required')
        if ((root/'INVALIDATED.json').exists() or m.get('schema')!='oa_cbf_jax_research_bundle_v1'
                or m.get('risk_distribution_contract')!=contract(2)
                or m.get('clipped_risk_components')!=2 or len(m.get('members',[]))!=4
                or m.get('dataset_schema')!='oa_cbf_quad2d_guided_hurdle_v1'
                or m.get('graph_features')!=40 or m.get('gain_dimension')!=2
                or m['weights_sha256']!=sha256(root/'weights.msgpack')):
            raise ValueError('Exact valid guided Quad2D clipped-mixture bundle required')
        config=GATConfig(**m['architecture'])
        if config.encoder not in ('gat','nearest_fc') or config.compute_dtype!='float32':
            raise ValueError('Qualified FP32 GAT or nearest-FC architecture required')
        if config.encoder=='nearest_fc':
            from .nearest_fc import validate_metadata
            validate_metadata(m)
        norm=m['normalization']
        if (np.shape(norm['target_mean'])!=(2,) or np.shape(norm['target_scale'])!=(2,)
                or not np.isfinite(norm['target_mean']).all()
                or not np.all(np.isfinite(norm['target_scale']) & (np.asarray(norm['target_scale'])>0))):
            raise ValueError('Invalid physical target normalization')
        self.gain_dimension=2
        self.device=device or jax.devices()[0]
        self.model=make_model(config,risk_components=2)
        raw=serialization.msgpack_restore((root/'weights.msgpack').read_bytes())
        if not all(np.isfinite(a).all() and a.shape[0]==4 for a in jax.tree.leaves(raw)):
            raise ValueError('Nonfinite or incomplete ensemble')
        self.params=jax.device_put(raw,self.device)
        def predict(params,features,mask,gains):
            out=predict_ensemble(self.model,params,features,mask,gains)
            return statistics(out,norm,jnp.ones(2),jnp.ones(2),jnp.zeros(2))
        self._feature_function=jax.jit(predict)
        self._compiled_features={}


SCHEMA='quad2d_clipped_mixture_prediction_calibration_v1'

DISAGREEMENT='CS with respect to unit atom at physical risk -2 plus Lebesgue above -2; exact component overlaps.'

def validate_calibration(info,metadata,bundle):
    """Require the actual matched distribution; never inherit Gaussian scales."""
    if (info.get('schema') not in (SCHEMA,'oa_cbf_frozen_trajectory_gate_v1')
            or info.get('risk_distribution_contract')!=contract(2)
            or metadata.get('risk_distribution_contract')!=contract(2)
            or info.get('disagreement_contract')!=DISAGREEMENT
            or info.get('event_statistic')!='maximum_member_probability'
            or info.get('bundle_manifest_sha256')!=sha256(Path(bundle)/'manifest.json')
            or info.get('weights_sha256')!=metadata['weights_sha256']):
        raise ValueError('Explicit matched clipped-mixture calibration required')
    scale=np.asarray(info['variance_scale'],float)
    if scale.shape!=(2,) or not np.isfinite(scale).all() or np.any(scale<1):
        raise ValueError('Invalid non-shrinking component variance calibration')
    for path,digest in info['bindings'].items():
        if sha256(path)!=digest:raise ValueError('Changed clipped calibration input')
    if info.get('schema')=='oa_cbf_frozen_trajectory_gate_v1' and info.get('predictive_calibration_schema')!=SCHEMA:
        raise ValueError('Different trajectory-gate predictive distribution')


from scipy.special import log_ndtr as clipped_risk_reference_log_ndtr, logsumexp as clipped_risk_reference_logsumexp, ndtr as clipped_risk_reference_ndtr

from scipy.stats import norm

CLIPPED_RISK_REFERENCE_FLOOR=-2.

def parameters(prediction,scale=1.):
    m,v,w=(np.asarray(prediction[k],float) for k in
           ('risk_component_mean','risk_component_variance','risk_component_probability'))
    if (m.shape!=v.shape or m.shape!=w.shape or m.shape[-1]!=2
            or not all(np.isfinite(a).all() for a in (m,v,w)) or np.any(v<=0) or np.any(w<0)):
        raise ValueError('Invalid mixture arrays')
    np.testing.assert_allclose(w.sum(-1),1.,atol=2e-6,rtol=0)
    return m,v*scale,w/w.sum(-1,keepdims=True)

def tail(mean,variance,weights,mass=.01):
    sigma=np.sqrt(variance)
    bounds=mean+sigma*norm.isf(mass)
    lo,hi=bounds.min(-1),bounds.max(-1)
    for _ in range(64):
        mid=lo+(hi-lo)*.5
        above=(weights*clipped_risk_reference_ndtr((mean-mid[...,None])/sigma)).sum(-1)>mass
        lo,hi=np.where(above,mid,lo),np.where(above,hi,mid)
    q=np.maximum(CLIPPED_RISK_REFERENCE_FLOOR,lo+(hi-lo)*.5)
    z=(mean-q[...,None])/sigma
    excess=np.maximum(0.,(mean-q[...,None])*clipped_risk_reference_ndtr(z)+sigma*norm.pdf(z))
    return q,q+(weights*excess).sum(-1)/mass

def clipped_risk_reference_disagreement(mean,variance,weights):
    """Independent explicit member/component sums; member axis first."""
    e=mean.shape[0]
    overlap=np.empty((e,e,*mean.shape[1:-1]),float)
    # Work in log space so even well-separated, narrow distributions are valid.
    with np.errstate(divide='ignore'):
        atom_log=clipped_risk_reference_logsumexp(np.log(weights)+clipped_risk_reference_log_ndtr((CLIPPED_RISK_REFERENCE_FLOOR-mean)/np.sqrt(variance)),axis=-1)
        for i in range(e):
            for j in range(e):
                terms=[atom_log[i]+atom_log[j]]
                for a in range(mean.shape[-1]):
                    for b in range(mean.shape[-1]):
                        m,n=mean[i,...,a],mean[j,...,b]
                        v,w=variance[i,...,a],variance[j,...,b]
                        precision=1/v+1/w
                        product_mean=(m/v+n/w)/precision
                        terms.append(np.log(weights[i,...,a])+np.log(weights[j,...,b])
                            +norm.logpdf(m,n,np.sqrt(v+w))
                            +clipped_risk_reference_log_ndtr((product_mean-CLIPPED_RISK_REFERENCE_FLOOR)*np.sqrt(precision)))
                overlap[i,j]=clipped_risk_reference_logsumexp(np.stack(terms),axis=0)
    own=np.stack([overlap[i,i] for i in range(e)])
    return np.maximum((.5*(own[:,None]+own[None,:])-overlap).mean((0,1)),0.)
