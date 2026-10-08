"""Explicit clipped-mixture predictions; ordinary Gaussian loading stays closed.

The disagreement uses densities with respect to delta_floor + Lebesgue above
the floor in physical risk units. It includes both the atom and the continuous
density. It is a calibrated development statistic, not an OOD guarantee.
"""
import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, logsumexp
import numpy as np

from .censored_risk import FLOOR, contract, mixture_tail, mixture_clipped_mean
from .dataset import sha256
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
