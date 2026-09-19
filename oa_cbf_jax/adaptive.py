"""Learned online gain selection, with explicit development calibration limits.

All neural proposals pass an actual fixed-budget CBF-QP branch check. A common
nonlearned backup has its own source code in the decision log. Nothing here
certifies trajectory-level calibration or converts a rejected solve to success.
"""

from dataclasses import dataclass
from typing import NamedTuple
import json
import math
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np

from .config import UnicycleConfig
from .dataset import sha256
from .inference import ResearchPredictor
from .models import route_graph,predict_ensemble
from .route_control import route_control,rollout_route
from .routing import physical_route_coordinate
from .simulation import GOAL,TIMEOUT
from .uncertainty import cs_disagreement,worst_member_cvar
from .sensor_margin import clearance_inflation,controller_contract,require_matching_controller

LEARNED,BACKUP,SEARCH,FIXED,REJECTED,UNGATED=range(6)
SOURCE_NAMES={LEARNED:'learned',BACKUP:'common_backup',SEARCH:'nonlearned_search',FIXED:'fixed',REJECTED:'rejected',UNGATED:'ungated_learned'}


@dataclass(frozen=True)
class PolicyConfig:
    mode:str='learned'
    validation_horizon:int=80
    interval:int=4
    shortlist:int=4
    tail_mass:float=.01
    risk_threshold:float=0.
    collision_probability_limit:float=.01
    failure_probability_limit:float=.25
    gain_change_penalty:float=.01
    fixed_gain:tuple=(2.,2.)
    reactive_reselection:bool=False
    backup_gains:tuple=((1.,1.),(2.,2.),(3.,3.))
    sensor_margin_scale:float=0.
    margin_guidance:bool=False
    shared_clearance_budget:bool=False
    motion_observer_window:int=0
    filter_obstacle_position:bool=False

    def __post_init__(self):
        controller_contract(self.sensor_margin_scale,self.margin_guidance,self.shared_clearance_budget,self.motion_observer_window,self.filter_obstacle_position)
        object.__setattr__(self,'fixed_gain',tuple(self.fixed_gain))
        object.__setattr__(self,'backup_gains',tuple(tuple(g) for g in self.backup_gains))
        if self.mode not in ('learned','ungated','search','fixed','fixed_backup'):raise ValueError('Unknown policy mode')
        if self.validation_horizon<1 or not 1<=self.interval<=self.validation_horizon or self.shortlist<1:raise ValueError('Invalid policy budget')
        if not 0<self.tail_mass<1:raise ValueError('Invalid risk configuration')
        if any(len(g)!=2 or any(not math.isfinite(v) or v<=0 for v in g) for g in (self.fixed_gain,*self.backup_gains)):
            raise ValueError('Backup and fixed gains must be positive finite pairs')


class Decision(NamedTuple):
    gains:jax.Array
    accepted:jax.Array
    source:jax.Array
    stages:jax.Array
    primary_valid:jax.Array
    backup_valid:jax.Array
    risk:jax.Array
    disagreement:jax.Array
    event_probability:jax.Array


def make_selector(model,normalization,robot,config):
    if config.mode in ('learned','ungated'):
        mean=jnp.asarray(normalization['target_mean'],jnp.float32)
        scale=jnp.asarray(normalization['target_scale'],jnp.float32)
    def validate(x,goal,obs,mask,gains,points,route_mask,cursor,previous_gain,noise):
        result,trace=jax.vmap(lambda g:rollout_route(x,goal,obs,mask,g,points,route_mask,cursor,config=robot,
                         steps=config.validation_horizon,speed_uncertainty=1.15*noise[2],
                         clearance_uncertainty=clearance_inflation(noise,config.sensor_margin_scale),margin_guidance=config.margin_guidance,shared_clearance_budget=config.shared_clearance_budget))(gains)
        first=physical_route_coordinate(x[:2],points,route_mask,cursor)
        last=jax.vmap(lambda state,c:physical_route_coordinate(state[:2],points,route_mask,c))(
            result.final_state,trace['route_progress'][:,-1])
        score=(last-first)/(config.validation_horizon*robot.dt*robot.v_max)
        score-=config.gain_change_penalty*jnp.sum((jnp.log(gains)-jnp.log(previous_gain))**2,axis=-1)
        score+=.05*jnp.minimum(result.min_clearance,.5)/.5
        valid=((result.status==GOAL)|(result.status==TIMEOUT))&(result.min_clearance>0)
        return valid,score

    def select(params,calibration,x,goal,obs,mask,candidates,points,route_mask,cursor,previous_gain,previous_control,noise):
        pool=jnp.concatenate((candidates,previous_gain[None]))
        same=jnp.max(jnp.abs(jnp.log(pool[:,None])-jnp.log(pool[None,:])),axis=-1)<1e-6
        unique=~jnp.any(same&(jnp.arange(len(pool))[None,:]<jnp.arange(len(pool))[:,None]),axis=1)
        zeros=jnp.zeros(7,jnp.int32)
        nan=jnp.asarray(jnp.nan,x.dtype)
        if config.mode=='fixed':
            return Decision(jnp.asarray(config.fixed_gain,x.dtype),jnp.asarray(True),jnp.int32(FIXED),zeros,
                            jnp.int32(0),jnp.int32(0),nan,nan,jnp.full(2,jnp.nan))
        if config.mode in ('learned','ungated'):
            features,node_mask=route_graph(x,goal,obs,mask,points,route_mask,cursor,previous_control,previous_gain,noise,robot)
            output=predict_ensemble(model,params,features[None],node_mask[None],pool[None])
            mu=output['mean'][:,0]*scale+mean
            variance=jnp.exp(output['log_variance'][:,0])*scale**2*calibration['variance_scale']
            means=mu[:,:,0].T;variances=variance[:,:,0].T
            cs=cs_disagreement(means[...,None],variances[...,None])
            risk=worst_member_cvar(means,variances,config.tail_mass)
            event=jnp.max(jax.nn.sigmoid(output['event_logits'][:,0]/calibration['temperature']+calibration['bias']),axis=0)
            finite=jnp.all(jnp.isfinite(mu)&jnp.isfinite(variance),axis=(0,2))&jnp.all(jnp.isfinite(event),axis=-1)&jnp.isfinite(cs)&jnp.isfinite(risk)
            qp,h,psi,_,_,_=jax.vmap(lambda g:route_control(x,goal,obs,mask,g,points,route_mask,cursor,robot,
                1.15*noise[2],clearance_inflation(noise,config.sensor_margin_scale),config.margin_guidance,config.shared_clearance_budget))(pool)
            current=finite&unique&qp.feasible&(h>=-robot.qp_tolerance)&(psi>=-robot.qp_tolerance)
            epistemic=current&(cs<=calibration['cs_threshold'])
            risk_ok=epistemic&(risk<=config.risk_threshold)
            event_ok=risk_ok&(event[:,0]<=config.collision_probability_limit)&(event[:,1]<=config.failure_probability_limit)
            admitted=event_ok if config.mode=='learned' else current
            stages=jnp.stack((jnp.int32(len(pool)),jnp.sum(finite),jnp.sum(current),jnp.sum(epistemic),jnp.sum(risk_ok),jnp.sum(event_ok),jnp.sum(admitted))).astype(jnp.int32)
            score=jnp.mean(mu[:,:,1],axis=0)-config.gain_change_penalty*jnp.sum((jnp.log(pool)-jnp.log(previous_gain))**2,axis=-1)
            _,indices=jax.lax.top_k(jnp.where(admitted,score,-jnp.inf),config.shortlist)
            primary=pool[indices];screen=admitted[indices];ranking=score[indices]
            source=LEARNED if config.mode=='learned' else UNGATED
        elif config.mode=='fixed_backup':
            primary=jnp.asarray(config.fixed_gain,x.dtype)[None];screen=jnp.ones(1,bool)
            ranking=jnp.zeros(1);stages=zeros;source=FIXED
            indices=jnp.zeros(1,jnp.int32)
        else:
            primary=pool;screen=unique;ranking=jnp.zeros(len(pool));stages=zeros;source=SEARCH
            indices=jnp.arange(len(pool))
        valid,actual_score=validate(x,goal,obs,mask,primary,points,route_mask,cursor,previous_gain,noise)
        valid &= screen
        if config.mode in ('search','fixed_backup'):ranking=actual_score
        chosen=jnp.argmax(jnp.where(valid,ranking,-jnp.inf));primary_ok=jnp.any(valid)
        # A documented common backup, evaluated from the same copied state.
        # Under batched vmap this conditional may execute both sides; single-
        # observation online timing must be measured separately from throughput.
        backup_pool=jnp.concatenate((previous_gain[None],jnp.asarray(config.backup_gains,x.dtype).reshape(-1,2)))
        def backup(_):
            ok,value=validate(x,goal,obs,mask,backup_pool,points,route_mask,cursor,previous_gain,noise)
            index=jnp.argmax(jnp.where(ok,value,-jnp.inf))
            return backup_pool[index],jnp.any(ok),jnp.sum(ok).astype(jnp.int32)
        backup_gain,backup_ok,backup_count=jax.lax.cond(primary_ok,
                    lambda _:(previous_gain,jnp.asarray(False),jnp.int32(0)),backup,operand=None)
        accepted=primary_ok|backup_ok
        gain=jnp.where(primary_ok,primary[chosen],jnp.where(backup_ok,backup_gain,previous_gain))
        selected_source=jnp.where(primary_ok,source,jnp.where(backup_ok,BACKUP,REJECTED)).astype(jnp.int32)
        if config.mode in ('learned','ungated'):
            selected=indices[chosen]
            selected_risk=jnp.where(primary_ok,risk[selected],nan);selected_cs=jnp.where(primary_ok,cs[selected],nan)
            selected_event=jnp.where(primary_ok,event[selected],jnp.full(2,jnp.nan))
        else:selected_risk=nan;selected_cs=nan;selected_event=jnp.full(2,jnp.nan)
        return Decision(gain,accepted,selected_source,stages,jnp.sum(valid).astype(jnp.int32),backup_count,
                        selected_risk,selected_cs,selected_event)
    return select


class NonlearnedPolicy:
    """Same selection/backup interface without a neural bundle dependency."""
    def __init__(self,config,robot):
        if config.mode not in ('search','fixed','fixed_backup'):
            raise ValueError('Nonlearned policy requires an analytic mode')
        self.config=config;self.robot=robot;self.params=None;self.calibration=None
        self.selector=make_selector(None,None,robot,config)
        self.metadata={'interpretation':'Nonlearned policy; no model or neural calibration used.'}


class DevelopmentPolicy:
    """Strict warmed single-observation interface for the development policy."""
    def __init__(self,bundle,calibration,config=PolicyConfig(),robot=None,allow_development=False,device=None):
        if not allow_development:raise ValueError('Observation calibration is development-only; explicit opt-in required')
        predictor=ResearchPredictor(bundle,allow_uncalibrated=True,device=device)
        info=json.loads(Path(calibration).read_text())
        require_matching_controller(predictor.metadata,info,config.sensor_margin_scale,config.margin_guidance,config.shared_clearance_budget,config.motion_observer_window,config.filter_obstacle_position)
        if info['schema']!='oa_cbf_development_calibration_v1' or info['weights_sha256']!=predictor.metadata['weights_sha256']:
            raise ValueError('Calibration/weights mismatch')
        if info['targets']!=predictor.metadata['targets'] or predictor.metadata.get('graph_features')!=31:
            raise ValueError('Incompatible feature/target contract')
        if info['dataset_manifest_sha256']!=predictor.metadata['dataset_manifest_sha256'] or info['events']!=predictor.metadata['events']:
            raise ValueError('Incompatible calibration data/event contract')
        calibrated_robot=UnicycleConfig(**info['robot'])
        robot=calibrated_robot if robot is None else robot
        if calibrated_robot!=robot or info['horizon_steps']!=80:raise ValueError('Unsupported robot or prediction horizon')
        # Historical 31-feature datasets record .3..4 queries. New domains
        # require explicit matching training and calibration provenance.
        domain=predictor.metadata.get('gain_domain',dict(lower=.3,upper=4.))
        if info.get('gain_domain',dict(lower=.3,upper=4.))!=domain:
            raise ValueError('Calibration/model gain-domain mismatch')
        if not all(math.isfinite(domain[k]) for k in ('lower','upper')) or not 0<domain['lower']<domain['upper']:
            raise ValueError('Invalid calibrated gain domain')
        if any(v<domain['lower'] or v>domain['upper'] for pair in (config.fixed_gain,*config.backup_gains) for v in pair):
            raise ValueError('Fixed/backup gains lie outside the calibrated gain domain')
        self.gain_domain=domain
        if info['cs_gate']['status']!='calibrated':raise ValueError('Insufficient epistemic calibration')
        self.predictor=predictor;self.params=predictor.params;self.device=predictor.device;self.config=config;self.robot=robot
        values=dict(variance_scale=np.asarray(info['variance_scale'],np.float32),
                    temperature=np.asarray([e['temperature'] for e in info['event_calibration']],np.float32),
                    bias=np.asarray([e['bias'] for e in info['event_calibration']],np.float32),
                    cs_threshold=np.float32(info['cs_gate']['threshold']))
        if not all(np.isfinite(a).all() for a in values.values()):raise ValueError('Nonfinite calibration')
        if np.any(values['variance_scale']<=0) or np.any(values['temperature']<=0) or values['cs_threshold']<0:
            raise ValueError('Invalid calibration scale/threshold')
        self.calibration=jax.device_put(values,self.device);self.metadata=info
        with jax.default_device(self.device):
            self.selector=make_selector(predictor.model,predictor.metadata['normalization'],robot,config)
        self.function=jax.jit(self.selector);self.compiled={}

    def warm(self,capacity=16,candidates=64,route_capacity=32):
        if candidates+1<self.config.shortlist:raise ValueError('Shortlist exceeds candidate pool')
        key=(capacity,candidates,route_capacity)
        with jax.default_device(self.device):
            route=jnp.zeros((route_capacity,2),jnp.float32).at[1:,0].set(5.)
            args=(self.params,self.calibration,jnp.zeros(4,jnp.float32),jnp.array([5.,0.],jnp.float32),
                  jnp.zeros((capacity,5),jnp.float32),jnp.zeros(capacity,bool),jnp.ones((candidates,2),jnp.float32),
                  route,jnp.arange(route_capacity)<2,jnp.float32(0),jnp.array([2.,2.],jnp.float32),
                  jnp.zeros(2,jnp.float32),jnp.zeros(6,jnp.float32))
            start=time.perf_counter();exe=self.function.lower(*args).compile();jax.block_until_ready(exe(*args))
        self.compiled[key]=exe;return time.perf_counter()-start

    def propose(self,x,goal,obstacles,mask,candidates,route,route_mask,cursor,previous_gain,previous_control,noise):
        values=[x,goal,obstacles,mask,candidates,route,route_mask,cursor,previous_gain,previous_control,noise]
        values=[np.asarray(a,dtype=bool if i in (3,6) else np.float32) for i,a in enumerate(values)]
        if values[2].ndim!=2 or values[4].ndim!=2 or values[5].ndim!=2:raise ValueError('Invalid policy array ranks')
        key=(values[2].shape[0],values[4].shape[0],values[5].shape[0])
        if key not in self.compiled:raise ValueError(f'Unwarmed policy signature {key}; runtime compilation is forbidden')
        expected=[(4,),(2,),(key[0],5),(key[0],),(key[1],2),(key[2],2),(key[2],),(),(2,),(2,),(6,)]
        if any(a.shape!=shape for a,shape in zip(values,expected)):raise ValueError('Invalid policy observation shapes')
        if not all(np.isfinite(a).all() for a in values):raise ValueError('Nonfinite policy observation')
        if np.any(values[4]<=0) or np.any(values[8]<=0) or np.any(values[10]<0):raise ValueError('Invalid gain/noise input')
        if any(np.any(a<np.float32(self.gain_domain['lower'])) or np.any(a>np.float32(self.gain_domain['upper'])) for a in (values[4],values[8])):
            raise ValueError('Candidate/previous gain lies outside the calibrated gain domain')
        if np.any(values[2][values[3],2]<0) or values[6].sum()<2 or np.any(np.diff(values[6].astype(int))>0):
            raise ValueError('Invalid obstacle radius or padded route mask')
        return self.compiled[key](self.params,self.calibration,*(jax.device_put(a,self.device) for a in values))
