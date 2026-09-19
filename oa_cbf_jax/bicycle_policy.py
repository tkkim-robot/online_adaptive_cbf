"""Observation-only learned scalar-gain selection with explicit research gates."""
from dataclasses import dataclass,asdict
from pathlib import Path
import math
import time
import jax
import jax.numpy as jnp
import numpy as np
from .bicycle_experiment import read,control_config
from .bicycle_features import bicycle_graph,bicycle_inference_graph
from .bicycle_predictive_calibration import SCHEMA as FIT_SCHEMA
from .inference import ResearchPredictor
from .models import predict_ensemble
from .uncertainty import cs_disagreement
from .dataset import sha256


@dataclass(frozen=True)
class BicyclePolicyConfig:
    tail_mass:float=.01
    conditional_risk_limit:float=0.
    adverse_probability_limit:float=.05
    gain_switch_penalty:float=.01
    initial_gain:float=float(np.geomspace(.5,8,8).astype(np.float32)[3])

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()) or not 0<self.tail_mass<1 or not 0<self.adverse_probability_limit<1 or self.gain_switch_penalty<0 or not .5<=self.initial_gain<=8:
            raise ValueError('Invalid bicycle policy thresholds')


def select_gain(means,variances,logits,previous,bank,temperature,bias,cs_limit,config=BicyclePolicyConfig()):
    # E,B,K,D -> B,K,E,1 for risk disagreement. Variances already calibrated.
    risk_means=jnp.transpose(means[...,:1],(1,2,0,3));risk_variances=jnp.transpose(variances[...,:1],(1,2,0,3))
    cs=cs_disagreement(risk_means,risk_variances)
    from statistics import NormalDist
    z=NormalDist().inv_cdf(1-config.tail_mass);coefficient=math.exp(-z*z/2)/math.sqrt(2*math.pi)/config.tail_mass
    risk=jnp.max(means[...,0]+jnp.sqrt(variances[...,0])*coefficient,axis=0)
    adverse=jax.nn.sigmoid(logits/temperature+bias)[...,1].max(axis=0)
    progress=means[...,1].mean(axis=0)
    finite=jnp.all(jnp.isfinite(means)&jnp.isfinite(variances)&(variances>0)&jnp.isfinite(logits),axis=(0,3))&jnp.isfinite(cs)&jnp.isfinite(risk)&jnp.isfinite(adverse)
    accepted=finite&(cs<=cs_limit)&(risk<=config.conditional_risk_limit)&(adverse<=config.adverse_probability_limit)
    score=progress-config.gain_switch_penalty*jnp.abs(jnp.log(bank[None,:,0]/previous[:,None]))
    index=jnp.argmax(jnp.where(accepted,score,-jnp.inf),axis=1);any_=jnp.any(accepted,axis=1)
    gain=jnp.where(any_,bank[index,0],previous)
    return dict(controller_gain=gain,selected_index=jnp.where(any_,index,-1),uncertainty_fallback=~any_,accepted=accepted,
        cs_score=cs,finite_member_cvar=risk,adverse_probability=adverse,predicted_progress=progress,ranking_score=score)


class BicycleSelector:
    def __init__(self,bundle,prediction_fit,trajectory_gate=None,reference_recording=False,config=BicyclePolicyConfig()):
        self.predictor=ResearchPredictor(bundle,allow_uncalibrated=True);self.metadata=self.predictor.metadata
        self.fit_path=Path(prediction_fit);self.fit=read(self.fit_path);self.config=config;self.reference_recording=reference_recording
        if self.fit['schema']!=FIT_SCHEMA or self.fit['weights_sha256']!=self.metadata['weights_sha256'] or self.fit['bundle_manifest_sha256']!=sha256(Path(bundle)/'manifest.json'):
            raise ValueError('Bicycle model/prediction fit lineage mismatch')
        if self.metadata.get('gain_dimension')!=1 or self.metadata.get('graph_features')!=35 or self.fit['bicycle_contract']!=self.metadata.get('bicycle_contract') or not self.fit['event_calibration'][1]['usable_for_failure_budget']:
            raise ValueError('Bicycle predictive contract/support mismatch')
        self.robot=control_config(self.fit['bicycle_contract']['config'])
        from .bicycle_guidance import guidance_from_controller
        if self.fit.get('controller')!=self.metadata.get('controller'):raise ValueError('Bicycle fitted controller semantics mismatch')
        self.guidance=guidance_from_controller(self.metadata['controller'])
        self.bank=np.geomspace(.5,8,8).astype(np.float32)[:,None]
        np.testing.assert_array_equal(self.bank,np.asarray(self.fit['candidates'],np.float32))
        if trajectory_gate is None:
            if not reference_recording:raise ValueError('Adaptive bicycle selection requires a frozen trajectory gate')
            self.threshold=np.float32(np.inf)
        else:
            gate=read(trajectory_gate)
            if (gate['schema']!='oa_cbf_bicycle_trajectory_gate_v70' or gate['prediction_fit_sha256']!=sha256(self.fit_path)
                    or gate['weights_sha256']!=self.metadata['weights_sha256'] or gate['policy_config']!=asdict(config)
                    or not math.isfinite(gate['threshold']) or gate['threshold']<0):raise ValueError('Bicycle trajectory gate mismatch')
            self.threshold=np.float32(gate['threshold'])
        norm=self.metadata['normalization'];mean=jnp.asarray(norm['target_mean'],jnp.float32);scale=jnp.asarray(norm['target_scale'],jnp.float32)
        variance_scale=jnp.asarray(self.fit['variance_scale'],jnp.float32);temperature=jnp.asarray([e['temperature'] for e in self.fit['event_calibration']],jnp.float32);bias=jnp.asarray([e['bias'] for e in self.fit['event_calibration']],jnp.float32)
        bank=jnp.asarray(self.bank)
        compute_dtype=self.predictor.model.config.compute_dtype
        def predict(params,x,goal,obstacles,mask,points,route_mask,cursor,previous_control,previous_gain,noise,threshold):
            graph=bicycle_graph if compute_dtype=='float32' else lambda *a,config:bicycle_inference_graph(*a,config=config,compute_dtype=compute_dtype)
            features,node_mask=jax.vmap(lambda *a:graph(*a,config=self.robot))(x,goal,obstacles,mask,points,route_mask,cursor,previous_control,previous_gain,noise)
            gains=jnp.broadcast_to(bank,(len(x),8,1));raw=predict_ensemble(self.predictor.model,params,features,node_mask,gains)
            means=raw['mean']*scale+mean;variances=jnp.exp(raw['log_variance'])*scale**2*variance_scale
            result=select_gain(means,variances,raw['event_logits'],previous_gain,bank,temperature,bias,threshold,config)
            if reference_recording:
                # This mode records statistics on a specified fixed controller;
                # it never applies an ungated neural proposal.
                result.update(controller_gain=jnp.full_like(previous_gain,config.initial_gain),selected_index=jnp.full_like(result['selected_index'],-2),uncertainty_fallback=jnp.zeros_like(result['uncertainty_fallback']))
            result.update(features=features,node_mask=node_mask,prediction_mean=jnp.moveaxis(means,0,1),prediction_variance=jnp.moveaxis(variances,0,1),prediction_event_logits=jnp.moveaxis(raw['event_logits'],0,1))
            return result
        self._function=jax.jit(predict);self._compiled={}

    def warm(self,batch=8,route_capacity=64):
        args=(jnp.zeros((batch,4),jnp.float32),jnp.ones((batch,2),jnp.float32),jnp.zeros((batch,64,5),jnp.float32),jnp.zeros((batch,64),bool),
            jnp.zeros((batch,route_capacity,2),jnp.float32),jnp.zeros((batch,route_capacity),bool),jnp.zeros(batch,jnp.float32),jnp.zeros((batch,2),jnp.float32),jnp.ones(batch,jnp.float32),jnp.zeros((batch,6),jnp.float32),jnp.asarray(self.threshold))
        start=time.monotonic();exe=self._function.lower(self.predictor.params,*args).compile();jax.block_until_ready(exe(self.predictor.params,*args));self._compiled[(batch,route_capacity)]=exe
        return time.monotonic()-start

    def predict(self,x,goal,obstacles,mask,points,route_mask,cursor,previous_control,previous_gain,noise):
        key=(x.shape[0],points.shape[1])
        if key not in self._compiled:raise ValueError('Unwarmed bicycle selector signature')
        if x.shape!=(key[0],4) or goal.shape!=(key[0],2) or obstacles.shape!=(key[0],64,5) or mask.shape!=(key[0],64) or route_mask.shape!=points.shape[:2] or noise.shape!=(key[0],6):raise ValueError('Unsupported bicycle selector input shape')
        raw=(x,goal,obstacles,mask,points,route_mask,cursor,previous_control,previous_gain,noise)
        args=tuple(jax.device_put(np.asarray(a,dtype=bool if i in (3,5) else np.float32),self.predictor.device) for i,a in enumerate(raw))
        return self._compiled[key](self.predictor.params,*args,jnp.asarray(self.threshold))
