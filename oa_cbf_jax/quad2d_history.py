"""History-conditioned fixed-gain counterfactuals with preserved physical state.

Simulator state belongs only to label generation. Graph construction uses raw
observations and causal observer memory; no physical state/bias is an input.
Future replicas resample innovations, never latent states against copied history.
"""
from dataclasses import asdict
import numpy as np
import jax
import jax.numpy as jnp
from .motion_observer import MotionState
from .quad2d_motion import update
from .quad2d_features import flight_graph
from .quad2d_control import FlightConfig,flight_arrived,physical_envelope_violation
from .quad2d_guidance import ObservedMotionGuidanceConfig,predictive_flight_control
from .quad2d import integrate_quad2d
from .routing import physical_route_coordinate
from .dynamics import signed_clearance,swept_disk_clearance
from .quad2d_rollout import GOAL,COLLISION,TIMEOUT,PLANNER_FAILURE,STATE_BOUND

SCHEMA='oa_cbf_quad2d_motion_history_hurdle_v1'
GRAPH_SCHEMA='quad2d_observed_motion_history_50_v1'


def snapshot(data,query,window=32):
    """Capture BEFORE processing query's raw reading or applying its command."""
    if type(query) is not int or not 0<=query<len(data['active']):raise ValueError('Invalid query index')
    if not data['active'][:query].all():raise ValueError('Cannot branch beyond an unobserved physical stop')
    anchor=max(0,(query-1)//window)*window
    memory=MotionState(data['observed_obstacles'][anchor,:,:2].copy(),
        data['observed_obstacles'][max(0,anchor-window),:,:2].copy(),np.int32(query))
    truth=data['true_obstacles'].copy()
    # Keep these simulator origins for the entire branch. Global time is explicit
    # so shared prefixes/branches do not change the physical obstacle trajectory.
    return dict(state=(data['state'][query-1] if query else data['true_initial_state']).copy(),
        truth_obstacles=truth,ego_bias=data['true_initial_state']-data['initial_observation'],
        obstacle_bias=truth-data['observed_obstacles_initial'],time_tick=np.int32(query),
        observed=data['observed_state'][query].copy(),obstacles=data['observed_obstacles'][query].copy(),
        cursor=np.float32(data['route_progress'][query-1] if query else 0.),
        previous_control=(data['control'][query-1] if query else np.full(2,4.905,np.float32)).copy(),
        previous_gain=(data['gain'][query-1] if query else np.array([4.,4.],np.float32)).copy(),memory=memory)


def graph(observed,goal,obstacles,mask,points,route_mask,cursor,previous_u,previous_gain,noise,memory,config=FlightConfig(),guidance=ObservedMotionGuidanceConfig()):
    """50 observed features: legacy40 plus causal motion support, no truth."""
    _,estimate,bound,_,_=update(memory,obstacles,mask,noise,config.robot.dt,guidance.motion_window)
    base,node_mask=flight_graph(observed,goal,estimate,mask,points,route_mask,cursor,previous_u,previous_gain,noise,config)
    track=jnp.concatenate(((obstacles[:,3:5]-estimate[:,3:5])/config.velocity_limit,bound/config.velocity_limit,
        (memory.current_anchor-obstacles[:,:2])/5,(memory.previous_anchor-obstacles[:,:2])/5),axis=-1)
    extra=jnp.concatenate((jnp.zeros((2,8),observed.dtype),track))
    clock=jnp.stack((jnp.minimum(memory.ticks/guidance.motion_window,2.),(memory.ticks%guidance.motion_window)/guidance.motion_window))
    extra=jnp.concatenate((extra,jnp.broadcast_to(clock,(len(base),2))),axis=-1)
    return jnp.where(node_mask[:,None],jnp.concatenate((base,extra),axis=-1),0.),node_mask


def branch(context,goal,mask,points,route_mask,noise,gains,key,ready=True,config=FlightConfig(),guidance=ObservedMotionGuidanceConfig(),steps=160,innovations=None):
    """A physical continuation with held gains and future online observer updates."""
    if type(guidance) is not ObservedMotionGuidanceConfig:raise ValueError('Matched current/future observer required')
    c=config.robot;observed=context['observed'];obstacles=context['obstacles'];truth=context['truth_obstacles'];initial=context['state']
    if innovations is None:innovations=jax.random.uniform(key,(steps,6+obstacles.size),dtype=observed.dtype,minval=-1.,maxval=1.)
    xs=noise[jnp.array([0,0,1,2,2,3])];os=noise[jnp.array([4,4,6,5,5])]
    t0=context['time_tick']*c.dt;initial_obs=truth.at[:,:2].set(truth[:,:2]+t0*truth[:,3:5])
    minimum=jnp.min(signed_clearance(initial[:2],initial_obs,mask,c.radius));status=jnp.where(flight_arrived(initial,goal,config),GOAL,0)
    status=jnp.where(physical_envelope_violation(initial,config)>c.qp_tolerance,STATE_BOUND,status)
    status=jnp.where(minimum<=0,COLLISION,status);status=jnp.where(ready,status,PLANNER_FAILURE)
    def tick(carry,inputs):
        x,status,count,minimum,cursor,residual,memory=carry;k,innovation=inputs;active=status==0
        global_tick=context['time_tick']+k
        sensed=x-context['ego_bias']+.15*xs*innovation[:6]
        seen=truth.at[:,:2].set(truth[:,:2]+global_tick*c.dt*truth[:,3:5])-context['obstacle_bias']+.15*os*innovation[6:].reshape(obstacles.shape)
        sensed=jnp.where(k==0,observed,sensed);seen=jnp.where(k==0,obstacles,seen)
        memory,estimate,bound,lag,bad=update(memory,seen,mask,noise,c.dt,guidance.motion_window)
        (qp,h,psi,domain,proposed,remaining,target),info=predictive_flight_control(sensed,goal,seen,mask,gains,points,route_mask,cursor,config,guidance,noise,estimate)
        admissible=(h>=-c.qp_tolerance)&(psi>=-c.qp_tolerance)&(domain>=-c.qp_tolerance)
        accepted=active&info['approved']&qp.feasible&admissible
        u=jnp.where(accepted,qp.control,jnp.zeros(2,observed.dtype));y,sub=integrate_quad2d(x,u,c)
        starts=jnp.concatenate((x[None],sub[:-1]));times=global_tick*c.dt+jnp.arange(c.integration_substeps)*c.dt/c.integration_substeps
        clearance=jnp.min(jax.vmap(lambda a,b,t:swept_disk_clearance(a,b,truth,mask,c.radius,t,t+c.dt/c.integration_substeps))(starts,sub,times))
        envelope=jnp.max(jax.vmap(lambda s:physical_envelope_violation(s,config))(jnp.concatenate((x[None],sub))))
        status=jnp.where(active&~qp.feasible,3,status);status=jnp.where(active&~admissible,5,status);status=jnp.where(active&~info['approved'],6,status)
        status=jnp.where(accepted&flight_arrived(y,goal,config),GOAL,status);status=jnp.where(accepted&(envelope>c.qp_tolerance),STATE_BOUND,status)
        status=jnp.where(accepted&(clearance<=0),COLLISION,status)
        x=jnp.where(accepted,y,x);cursor=jnp.where(accepted,proposed,cursor);count+=accepted.astype(jnp.int32)
        minimum=jnp.minimum(minimum,jnp.where(accepted,clearance,jnp.inf));residual=jnp.maximum(residual,jnp.where(accepted,qp.max_violation,-jnp.inf))
        trace=dict(state=x,control=u,active=accepted,status=status,observed_state=sensed,observed_obstacles=seen,
            clearance=jnp.where(accepted,clearance,jnp.nan),state_bound_violation=jnp.where(accepted,envelope,jnp.nan),qp_violation=jnp.where(accepted,qp.max_violation,jnp.nan),
            h=h,psi1=psi,envelope_domain=domain,route_progress=cursor,route_target=target,route_remaining=remaining,gain=gains,
            forecast_obstacles=estimate,forecast_velocity_bound=bound,forecast_lag=lag,forecast_inconsistent=bad,
            **{'guidance_'+name:value for name,value in info.items()})
        return (x,status,count,minimum,cursor,residual,memory),trace
    carry=(initial,status,jnp.int32(0),minimum,context['cursor'],jnp.float32(-jnp.inf),context['memory'])
    template=jax.eval_shape(tick,carry,(jnp.int32(0),innovations[0]))[1]
    storage=jax.tree.map(lambda a:jnp.zeros((steps,*a.shape),a.dtype),template)
    def condition(s):return (s[0]<steps)&((s[0]==0)|(s[1][1]==0))
    def body(s):
        k,carry,trace=s;carry,record=tick(carry,(k,innovations[k]))
        return k+1,carry,jax.tree.map(lambda a,b:jax.lax.dynamic_update_index_in_dim(a,b,k,0),trace,record)
    _,last,trace=jax.lax.while_loop(condition,body,(jnp.int32(0),carry,storage))
    final,status,count,minimum,cursor,residual,_=last;status=jnp.where(status==0,TIMEOUT,status)
    progress=physical_route_coordinate(final[:2],points,route_mask,cursor)-physical_route_coordinate(initial[:2],points,route_mask,context['cursor'])
    summary=dict(final_state=final,physical_initial_state=initial,status=status,steps=count,min_clearance=minimum,worst_qp_violation=residual,
        route_progress=progress,goal_progress=jnp.linalg.norm(initial[:2]-goal)-jnp.linalg.norm(final[:2]-goal),final_cursor=cursor)
    return summary,trace


def join_trace(acquisition,continuation,query):
    """Audit from the original physical origin; no relaxed visited-state prior."""
    result={k:np.concatenate((acquisition[k][:query],v),axis=0) for k,v in continuation.items()}
    for k in ['true_initial_state','true_obstacles','initial_observation','observed_obstacles_initial','noise','obstacle_mask','goal']:
        result[k]=acquisition[k]
    return result
