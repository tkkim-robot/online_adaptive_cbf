"""Local GAT/nearest-FC gain selection and shared physical CBF-QP execution.

No prospective physical rollout searches gains at runtime. Predictions rank
candidates; at most four ranked proposals undergo instantaneous QP checks.
When none is admitted, retain the previous gain and check its actual QP.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jnp
from .local_unicycle_collection import ROBOT,NOMINAL,K
from .controllers import nominal_unicycle,unicycle_cbf_qp,solve_qp2
from .dynamics import integrate_unicycle,signed_clearance,swept_disk_clearance
from .models import unicycle_graph,predict_ensemble
from .obstacle_selection import nearest_obstacles
from .uncertainty import cs_disagreement,worst_member_cvar
from .inference import ResearchPredictor
from .dataset import sha256
from .local_unicycle_calibration import validate_bundle,validate_calibration_data


@dataclass(frozen=True)
class LocalPolicyConfig:
    interval:int=4
    shortlist:int=4
    tail_mass:float=.01
    risk_threshold:float=0.
    collision_limit:float=.01
    stop_limit:float=.25
    gain_change_penalty:float=.01
    initial_gain:tuple=(4.,1.)
    ranking_objective:str='conditional_progress'

    def __post_init__(self):
        object.__setattr__(self,'initial_gain',tuple(self.initial_gain))
        if self.interval!=4 or self.shortlist!=4:raise ValueError('Fixed0.2second/four-proposal contract required')
        if self.ranking_objective not in ('conditional_progress','viability_weighted_progress','nonincreasing_stop'):
            raise ValueError('Unknown local ranking objective')
        if (not 0<self.tail_mass<1 or not 0<=self.collision_limit<=1 or not 0<=self.stop_limit<=1
            or not np.isfinite([self.risk_threshold,self.gain_change_penalty,*self.initial_gain]).all()
            or self.gain_change_penalty<0 or len(self.initial_gain)!=2 or min(self.initial_gain)<=0):raise ValueError('Invalid local policy')


def policy_config_record(config):
    result=asdict(config)
    # Preserve old frozen metadata for the unchanged original rule.
    if config.ranking_objective=='conditional_progress':result.pop('ranking_objective')
    return result


def candidate_score(progress,event,pool,previous,config):
    value=progress
    if config.ranking_objective=='viability_weighted_progress':
        value=jnp.maximum(progress,0.)*jnp.clip(1.-jnp.sum(event,axis=-1),0.,1.)
    return value-config.gain_change_penalty*jnp.sum((jnp.log(pool)-jnp.log(previous))**2,axis=-1)


def limit_stop_increase(admitted,event,previous_valid):
    """An optional relative stop screen; the last prediction is the incumbent.

    A currently invalid incumbent must not block a feasible replacement.
    Existing absolute risk/event/disagreement checks still apply to proposals.
    These empirical probabilities are not a safety certificate.
    """
    comparable=previous_valid&jnp.isfinite(event[-1,1])
    return admitted&(~comparable|(event[:,1]<=event[-1,1]))


def qp_control(observed,goal,rows,mask,gain):
    _,a,b,h,psi=unicycle_cbf_qp(observed,goal,rows,mask,gain,ROBOT)
    result=solve_qp2(nominal_unicycle(observed,goal,NOMINAL),a,b,jnp.ones(2,jnp.float32),ROBOT.qp_tolerance)
    admissible=jnp.all(jnp.where(mask,(h>=-ROBOT.qp_tolerance)&(psi>=-ROBOT.qp_tolerance),True))
    return result,admissible


def make_statistics(model,normalization):
    center=jnp.asarray(normalization['target_mean'],jnp.float32)
    scale=jnp.asarray(normalization['target_scale'],jnp.float32)
    def statistics(params,calibration,features,mask,candidates):
        out=predict_ensemble(model,params,features[None],mask[None],candidates[None])
        mean=out['mean'][:,0]*scale+center
        variance=jnp.exp(out['log_variance'][:,0])*scale**2*calibration['variance_scale']
        risk_mean=mean[:,:,0].T;risk_variance=variance[:,:,0].T
        return dict(progress=mean[:,:,1].mean(0),
            risk=worst_member_cvar(risk_mean,risk_variance,calibration['tail_mass']),
            cs=cs_disagreement(risk_mean[...,None],risk_variance[...,None]),
            event=jax.nn.sigmoid(out['event_logits'][:,0]/calibration['temperature']+calibration['bias']).mean(0),
            finite=jnp.all(jnp.isfinite(mean)&jnp.isfinite(variance)&(variance>0),axis=(0,2)))
    return statistics


def make_selector(model,normalization,config=LocalPolicyConfig()):
    statistics=make_statistics(model,normalization)
    def select(params,calibration,observed,goal,rows,mask,bank,previous):
        pool=jnp.concatenate((bank,previous[None]));features,nodes=unicycle_graph(observed,goal,rows,mask,ROBOT)
        pred=statistics(params,calibration,features,nodes,pool)
        unique=~jnp.any((jnp.max(jnp.abs(pool[:,None]-pool[None,:]),axis=-1)<1e-6)&
            (jnp.arange(len(pool))[None,:]<jnp.arange(len(pool))[:,None]),axis=1)
        finite=pred['finite']&jnp.isfinite(pred['risk'])&jnp.isfinite(pred['cs'])&jnp.isfinite(pred['event']).all(-1)
        epistemic=finite&unique&(pred['cs']<=calibration['cs_threshold'])
        safe=epistemic&(pred['risk']<=config.risk_threshold)
        admitted=safe&(pred['event'][:,0]<=config.collision_limit)&(pred['event'][:,1]<=config.stop_limit)
        if config.ranking_objective=='nonincreasing_stop':
            previous_qp,previous_domain=qp_control(observed,goal,rows,mask,previous)
            admitted=limit_stop_increase(admitted,pred['event'],previous_qp.feasible&previous_domain)
        score=candidate_score(pred['progress'],pred['event'],pool,previous,config)
        _,order=jax.lax.top_k(jnp.where(admitted,score,-jnp.inf),config.shortlist)
        qp,domain=jax.vmap(lambda g:qp_control(observed,goal,rows,mask,g))(pool[order])
        valid=admitted[order]&qp.feasible&domain
        chosen=jnp.argmax(valid);accepted=jnp.any(valid);index=order[chosen]
        gain=jnp.where(accepted,pool[index],previous)
        stages=jnp.stack((jnp.sum(finite),jnp.sum(epistemic),jnp.sum(safe),jnp.sum(admitted),jnp.sum(valid))).astype(jnp.int32)
        return gain,dict(source=jnp.where(accepted,0,1).astype(jnp.int32),accepted=accepted,stages=stages,
            selected_index=jnp.where(accepted,index,-1).astype(jnp.int32),
            selected_risk=jnp.where(accepted,pred['risk'][index],0.),selected_cs=jnp.where(accepted,pred['cs'][index],0.))
    return select


def empty_decision(source=1):
    return dict(source=jnp.int32(source),accepted=jnp.bool_(False),stages=jnp.zeros(5,jnp.int32),
        selected_index=jnp.int32(-1),selected_risk=jnp.float32(0.),selected_cs=jnp.float32(0.))


def make_rollout(model=None,normalization=None,config=LocalPolicyConfig(),position_observer=False):
    selector=make_selector(model,normalization,config) if model is not None else None
    if position_observer:
        from .local_unicycle_observed_policy import make_rollout as observed_rollout
        return observed_rollout(selector,config)
    def rollout(params,calibration,initial,world,goal,errors,bank):
        def tick(carry,inputs):
            x,status,done,minimum,gain=carry;k,error=inputs
            observed=x.at[:2].add(error[0]);seen=world.at[:,:2].add(error[1:])
            rows,mask,ids=nearest_obstacles(observed[:2],seen,jnp.ones(len(world),bool),K)
            query=(status==0)&(k%config.interval==0)
            if selector is not None:
                gain,decision=jax.lax.cond(query,
                    lambda _:selector(params,calibration,observed,goal,rows,mask,bank,gain),
                    lambda _:(gain,empty_decision()),operand=None)
            else:decision=empty_decision(2)
            qp,admissible=qp_control(observed,goal,rows,mask,gain)
            active=(status==0)&qp.feasible&admissible
            u=jnp.where(active,qp.control,jnp.zeros(2,jnp.float32))
            y,sub=integrate_unicycle(x,u,ROBOT.dt,ROBOT.integration_substeps)
            starts=jnp.concatenate((x[None],sub[:-1]))
            clear=jnp.min(jax.vmap(lambda a,b:swept_disk_clearance(a,b,world,jnp.ones(len(world),bool),ROBOT.radius,0.,0.))(starts,sub))
            bounds=jnp.max(jnp.maximum(-sub[:,3],sub[:,3]-ROBOT.v_max))
            reached=(jnp.linalg.norm(y[:2]-goal)<=ROBOT.goal_tolerance)&(jnp.abs(y[3])<=.2)
            ns=jnp.where((status==0)&~qp.feasible,3,status);ns=jnp.where((status==0)&~admissible,5,ns)
            ns=jnp.where(active&reached,1,ns);ns=jnp.where(active&(bounds>ROBOT.qp_tolerance),8,ns)
            ns=jnp.where(active&(clear<=0.),2,ns)
            state=jnp.where(active,y,x);minimum=jnp.minimum(minimum,jnp.where(active,clear,jnp.inf))
            trace=dict(before=x,state=state,observed=observed,control=u,active=active,status=ns,selected_ids=ids,
                feasible=qp.feasible,admissible=admissible,clearance=jnp.where(active,clear,0.),gain=gain,query=query,**decision)
            return (state,ns,done+active.astype(jnp.int32),minimum,gain),trace
        start=(initial,jnp.int32(0),jnp.int32(0),
            jnp.min(signed_clearance(initial[:2],world,jnp.ones(len(world),bool),ROBOT.radius)),jnp.asarray(config.initial_gain,jnp.float32))
        final,trace=jax.lax.scan(tick,start,(jnp.arange(len(errors)),errors))
        state,status,steps,clearance,gain=final
        return dict(final_state=state,status=jnp.where(status==0,4,status),steps=steps,min_clearance=clearance,
            progress=jnp.linalg.norm(goal-initial[:2])-jnp.linalg.norm(goal-state[:2])),trace
    return rollout


class LocalPolicy:
    def __init__(self,bundle,prediction_calibration,gate=None,config=LocalPolicyConfig(),numerical_pilot=False):
        info=json.loads(Path(prediction_calibration).read_text());meta=validate_bundle(bundle,info['dataset'])
        self.position_observer=meta['controller'].get('position_observer') is not None
        if self.position_observer:
            from .local_unicycle_observer import contract
            if meta['controller']['position_observer']!=contract():raise ValueError('Unknown learned observer contract')
        validate_calibration_data(info['calibration_dataset'])
        if (info['schema']!='unicycle_local_prediction_calibration_v1' or info['local_unicycle_contract']!=meta['local_unicycle_contract']
            or info['controller']!=meta['controller']
            or info['weights_sha256']!=meta['weights_sha256'] or info['bundle_manifest_sha256']!=sha256(Path(bundle)/'manifest.json')
            or info['calibration_manifest_sha256']!=sha256(Path(info['calibration_dataset'])/'manifest.json')
            or info['calibration_audit_sha256']!=sha256(Path(info['calibration_dataset'])/'audit.json')):raise ValueError('Changed local prediction calibration')
        if sha256(info['qualification'])!=info['qualification_sha256']:
            raise ValueError('Changed predictive qualification')
        if 'policy_visited_stop_calibration' in info:
            from .local_unicycle_policy_calibration import validate_supplement
            validate_supplement(info)
        coefficients=[*info['variance_scale'],*[v['temperature'] for v in info['event_calibration']],
            *[v['bias'] for v in info['event_calibration']]]
        if not np.isfinite(coefficients).all() or min(coefficients[:4])<=0:
            raise ValueError('Invalid empirical prediction calibration')
        if meta['architecture']['encoder']=='nearest_fc':
            from .nearest_fc_qualification import validate_fit
            validate_fit(info,bundle)
        if gate is None:
            if not numerical_pilot:raise ValueError('Trajectory screen required; only explicit numerical pilots may disable it')
            threshold=np.inf
        else:
            g=json.loads(Path(gate).read_text())
            if (g['schema']!='local_unicycle_reference_trajectory_screen_v1'
                or g['prediction_calibration_sha256']!=sha256(prediction_calibration)
                or g['bundle_manifest_sha256']!=sha256(Path(bundle)/'manifest.json') or g['cs_gate']['status']!='calibrated'):
                raise ValueError('Changed trajectory screen')
            if g.get('position_observer')!=meta['controller'].get('position_observer'):
                raise ValueError('Trajectory screen uses a different observer')
            for path,digest in g['bound_files'].items():
                if sha256(path)!=digest:raise ValueError('Changed trajectory calibration evidence')
            if sha256(Path(gate).parent/'scores.npz')!=g['score_sha256']:
                raise ValueError('Changed trajectory disagreement scores')
            threshold=g['cs_gate']['threshold']
        self.predictor=ResearchPredictor(bundle,allow_uncalibrated=True)
        self.params=self.predictor.params;self.config=config
        self.calibration={k:jnp.asarray(v,jnp.float32) for k,v in dict(variance_scale=info['variance_scale'],
            temperature=[v['temperature'] for v in info['event_calibration']],bias=[v['bias'] for v in info['event_calibration']],
            cs_threshold=threshold,tail_mass=config.tail_mass).items()}
        self.rollout=make_rollout(self.predictor.model,meta['normalization'],config,position_observer=self.position_observer)
        self.selector=make_selector(self.predictor.model,meta['normalization'],config)
        self.metadata=dict(bundle=str(Path(bundle).resolve()),bundle_manifest_sha256=sha256(Path(bundle)/'manifest.json'),
            prediction_calibration=str(Path(prediction_calibration).resolve()),prediction_calibration_sha256=sha256(prediction_calibration),
            trajectory_screen=str(Path(gate).resolve()) if gate else None,numerical_pilot=numerical_pilot,policy=policy_config_record(config),
            trajectory_screen_sha256=sha256(gate) if gate else None,
            interpretation='Learned prediction-ranked gains, instantaneous CBF-QP, previous-gain-only fallback. Reference screen is not adaptive trajectory coverage.')
        if self.position_observer:self.metadata['position_observer']=meta['controller']['position_observer']
