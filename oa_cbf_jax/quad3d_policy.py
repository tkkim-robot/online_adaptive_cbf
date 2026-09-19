"""Observed Quad3D GAT/FC candidate scoring; no physical rollout gain search."""
from dataclasses import dataclass,asdict
from pathlib import Path
from statistics import NormalDist
import math
import numpy as np
import jax
import jax.numpy as jnp
from .quad3d_features import history_graph
from .quad3d_candidate_data import candidate_bank,WIDE_SCHEMA
from .quad3d_learning_contract import validate_model,read
from .quad3d_control import control_config,envelope_rows
from .quad3d import cylinder_hocbf
from .quad3d_observation import controller_obstacles
from .quad3d_predictive_calibration import SCHEMA as FIT_SCHEMA
from .inference import ResearchPredictor
from .models import predict_ensemble
from .uncertainty import cs_disagreement
from .dataset import sha256

GATE_SCHEMA='oa_cbf_quad3d_trajectory_gate_v96'


@dataclass(frozen=True)
class Quad3DPolicyConfig:
    query_every_ticks:int=4
    tail_mass:float=.01
    conditional_risk_limit:float=0.
    adverse_probability_limit:float=.05
    gain_switch_penalty:float=.01
    initial_gain:float=2.

    def __post_init__(self):
        if type(self.query_every_ticks) is not int or self.query_every_ticks<1:raise ValueError('Positive integer query cadence required')
        if not all(math.isfinite(v) for v in asdict(self).values()) or not 0<self.tail_mass<1 or not 0<self.adverse_probability_limit<1 or self.gain_switch_penalty<0 or self.initial_gain not in (2.,4.,6.,8.):raise ValueError('Invalid Quad3D policy configuration')


def admissibility(x,obstacles,mask,noise,bank,c):
    o=controller_obstacles(obstacles,mask,noise)
    def value(g):
        _,_,psi=cylinder_hocbf(x,o,mask,g,c.robot,c.clearance_buffer)
        return jnp.min(jnp.where(mask[:,None],psi,jnp.inf))
    psi=jax.vmap(value)(bank.astype(x.dtype));_,_,envelope=envelope_rows(x,c)
    return psi,jnp.min(envelope)


def select(means,variances,logits,previous,bank,temperature,bias,threshold,psi,domain,c,p):
    # E,K,D -> K,E,1. Predictions are FP32; physical admission remains FP64.
    cs=cs_disagreement(jnp.moveaxis(means[...,:1],0,1),jnp.moveaxis(variances[...,:1],0,1))
    z=NormalDist().inv_cdf(1-p.tail_mass);coefficient=math.exp(-z*z/2)/math.sqrt(2*math.pi)/p.tail_mass
    risk=jnp.max(means[...,0]+jnp.sqrt(variances[...,0])*coefficient,axis=0)
    adverse=jax.nn.sigmoid(logits/temperature+bias)[...,1].max(axis=0)
    progress=means[...,1].mean(axis=0)
    score=progress-p.gain_switch_penalty*jnp.mean(jnp.abs(jnp.log(bank/previous.astype(jnp.float32))),axis=-1)
    finite=jnp.all(jnp.isfinite(means)&jnp.isfinite(variances)&(variances>0)&jnp.isfinite(logits),axis=(0,2))&jnp.isfinite(cs)&jnp.isfinite(risk)&jnp.isfinite(adverse)
    screened=finite&(cs<=threshold)&(risk<=p.conditional_risk_limit)&(adverse<=p.adverse_probability_limit)
    admissible=(psi>=-c.qp_tolerance)&(domain>=-c.qp_tolerance)
    accepted=screened&admissible
    index=jnp.argmax(jnp.where(accepted,score,-jnp.inf));any_=jnp.any(accepted)
    proposed=jnp.where(any_,bank[index].astype(previous.dtype),previous)
    return dict(network_gain=proposed,selected_index=jnp.where(any_,index,-1),uncertainty_fallback=~jnp.any(screened),
        admission_fallback=jnp.any(screened)&~any_,accepted=accepted,screened=screened,admissible=admissible,
        cs_score=cs,finite_member_cvar=risk,adverse_probability=adverse,predicted_progress=progress,ranking_score=score,
        candidate_psi=psi,candidate_domain=domain)


class Quad3DSelector:
    def __init__(self,bundle,prediction_fit,gate=None,reference=False,config=Quad3DPolicyConfig()):
        self.predictor=ResearchPredictor(bundle,allow_uncalibrated=True);self.metadata=self.predictor.metadata;validate_model(self.metadata)
        self.fit=read(prediction_fit);self.config=config;self.reference=reference
        fit=self.fit
        if fit['schema']!=FIT_SCHEMA or fit['weights_sha256']!=self.metadata['weights_sha256'] or fit['bundle_manifest_sha256']!=sha256(Path(bundle)/'manifest.json'):raise ValueError('Quad3D prediction lineage mismatch')
        for field in ('quad3d_contract','controller','targets','events','gain_domain'):
            if fit[field]!=self.metadata[field]:raise ValueError('Quad3D prediction semantics mismatch: '+field)
        if not fit['event_calibration'][1]['usable_for_failure_budget']:raise ValueError('Adverse-event calibration support required')
        self.bank=candidate_bank(self.metadata['dataset_schema'])
        if fit['candidates']!=self.bank.tolist():raise ValueError('Wrong four-gain bank')
        if self.metadata['dataset_schema']!=WIDE_SCHEMA and config.initial_gain!=2.:raise ValueError('Legacy Quad3D initialization is frozen at2')
        self.robot=control_config(fit['quad3d_contract']['config'])
        if gate is None:
            if not reference:raise ValueError('Adaptive Quad3D requires a frozen trajectory gate')
            self.threshold=np.float32(np.inf)
        else:
            g=read(gate)
            if g['schema']!=GATE_SCHEMA or g['weights_sha256']!=self.metadata['weights_sha256'] or g['prediction_fit_sha256']!=sha256(prediction_fit) or g['policy_config']!=asdict(config) or not math.isfinite(g['threshold']) or g['threshold']<0:raise ValueError('Quad3D trajectory gate mismatch')
            self.threshold=np.float32(g['threshold'])
        norm=self.metadata['normalization'];mean=jnp.asarray(norm['target_mean'],jnp.float32);scale=jnp.asarray(norm['target_scale'],jnp.float32)
        variance_scale=jnp.asarray(fit['variance_scale'],jnp.float32);temperature=jnp.asarray([e['temperature'] for e in fit['event_calibration']],jnp.float32);bias=jnp.asarray([e['bias'] for e in fit['event_calibration']],jnp.float32)
        bank=jnp.asarray(self.bank,jnp.float32);c=self.robot;p=config
        def predict(params,x,goal,o,mask,points,rm,cursor,previous_u,previous_gain,noise,nominal_bias=None):
            f,m=history_graph(x,goal,o,mask,points,rm,cursor,previous_u,previous_gain,noise,config=c,nominal_bias=nominal_bias)
            f=f.astype(jnp.float32)
            raw=predict_ensemble(self.predictor.model,params,f[None],m[None],bank[None])
            means=raw['mean'][:,0]*scale+mean;variances=jnp.exp(raw['log_variance'][:,0])*scale**2*variance_scale;logits=raw['event_logits'][:,0]
            psi,domain=admissibility(x,o,mask,noise,bank,c)
            result=select(means,variances,logits,previous_gain,bank,temperature,bias,jnp.asarray(self.threshold),psi,domain,c,p)
            if reference:
                result.update(network_gain=jnp.full(4,p.initial_gain,x.dtype),selected_index=jnp.int32(-2),uncertainty_fallback=jnp.bool_(False),admission_fallback=jnp.bool_(False))
            result.update(features=f,node_mask=m,prediction_mean=means,prediction_variance=variances,prediction_event_logits=logits)
            return result
        self.predict=predict
