"""Actual six-state flight branches with explicit observation-conditioned noise."""
from functools import partial
import jax
import jax.numpy as jnp
from .quad2d import integrate_quad2d
from .quad2d_control import FlightConfig,flight_control,flight_arrived,physical_envelope_violation
from .dynamics import signed_clearance,swept_disk_clearance
from .routing import physical_route_coordinate
from .simulation import RUNNING,GOAL,COLLISION,INFEASIBLE,TIMEOUT,STATUS_NAMES

INADMISSIBLE=5;PLANNER_FAILURE=7;STATE_BOUND=8
NAMES={**STATUS_NAMES,INADMISSIBLE:'hocbf_inadmissible',6:'policy_rejected',PLANNER_FAILURE:'planner_failure',STATE_BOUND:'state_bound_violation'}


def flight_sensor_model(observed,obstacles,mask,noise,key,steps):
    """Known ranges[xy,pitch,vxy,pitch_rate,obs_xy,obs_velocity,radius].

    Independent uniform latent offsets and15% subsequent innovations; first
    innovation is zero so every branch starts from exactly the same observation.
    Positive physical obstacle radius is part of the generative prior. Physical
    flight states are never clipped to their operational envelope.
    """
    xkey,okey,nkey=jax.random.split(key,3)
    xs=jnp.stack((noise[0],noise[0],noise[1],noise[2],noise[2],noise[3]))
    os=jnp.stack((noise[4],noise[4],noise[6],noise[5],noise[5]))
    xb=jax.random.uniform(xkey,(6,),dtype=observed.dtype,minval=-1.,maxval=1.)*xs
    ob=jax.random.uniform(okey,obstacles.shape,dtype=obstacles.dtype,minval=-1.,maxval=1.)*os
    ob=jnp.where(mask[:,None],ob,0.);truth=obstacles+ob
    truth=truth.at[:,2].set(jnp.maximum(truth[:,2],1e-4));ob=truth-obstacles
    innovations=jax.random.uniform(nkey,(steps,6+obstacles.size),dtype=observed.dtype,minval=-1.,maxval=1.).at[0].set(0.)
    return observed+xb,truth,xb,ob,xs,os,innovations


@partial(jax.jit,static_argnames=('config','steps','guidance'))
def flight_branch(observed,goal,obstacles,mask,gains,points,route_mask,cursor,noise,key,ready=True,config=FlightConfig(),steps=160,guidance=None):
    c=config.robot
    initial,truth_obs,xb,ob,xs,os,innovations=flight_sensor_model(observed,obstacles,mask,noise,key,steps)
    minimum=jnp.min(signed_clearance(initial[:2],truth_obs,mask,c.radius))
    status=jnp.where(flight_arrived(initial,goal,config),GOAL,RUNNING)
    status=jnp.where(physical_envelope_violation(initial,config)>c.qp_tolerance,STATE_BOUND,status)
    status=jnp.where(minimum<=0,COLLISION,status);status=jnp.where(ready,status,PLANNER_FAILURE)
    def tick(carry,inputs):
        x,status,count,minimum,cursor,max_residual=carry;k,innovation=inputs;active=status==RUNNING
        sensed=x-xb+.15*xs*innovation[:6]
        seen=truth_obs.at[:,:2].set(truth_obs[:,:2]+k*c.dt*truth_obs[:,3:5])-ob+.15*os*innovation[6:].reshape(obstacles.shape)
        guidance_info={};approved=jnp.asarray(True)
        if guidance is None:
            qp,h,psi,domain,proposed,remaining,target=flight_control(sensed,goal,seen,mask,gains,points,route_mask,cursor,config)
        else:
            from .quad2d_guidance import predictive_flight_control
            (qp,h,psi,domain,proposed,remaining,target),info=predictive_flight_control(sensed,goal,seen,mask,gains,points,route_mask,cursor,config,guidance,noise)
            approved=info['approved'];guidance_info={'guidance_'+key:value for key,value in info.items()}
        admissible=(h>=-c.qp_tolerance)&(psi>=-c.qp_tolerance)&(domain>=-c.qp_tolerance)
        accepted=active&approved&qp.feasible&admissible
        control=jnp.where(accepted,qp.control,jnp.zeros(2,observed.dtype))
        y,sub=integrate_quad2d(x,control,c);starts=jnp.concatenate((x[None],sub[:-1]))
        times=k*c.dt+jnp.arange(c.integration_substeps)*c.dt/c.integration_substeps
        clear=jnp.min(jax.vmap(lambda a,b,t:swept_disk_clearance(a,b,truth_obs,mask,c.radius,t,t+c.dt/c.integration_substeps))(starts,sub,times))
        bound=jnp.max(jax.vmap(lambda s:physical_envelope_violation(s,config))(jnp.concatenate((x[None],sub))))
        status=jnp.where(active&~qp.feasible,INFEASIBLE,status)
        status=jnp.where(active&~admissible,INADMISSIBLE,status)
        status=jnp.where(active&~approved,6,status)
        status=jnp.where(accepted&flight_arrived(y,goal,config),GOAL,status)
        status=jnp.where(accepted&(bound>c.qp_tolerance),STATE_BOUND,status)
        status=jnp.where(accepted&(clear<=0),COLLISION,status)
        x=jnp.where(accepted,y,x);cursor=jnp.where(accepted,proposed,cursor);count+=accepted.astype(jnp.int32)
        minimum=jnp.minimum(minimum,jnp.where(accepted,clear,jnp.inf))
        max_residual=jnp.maximum(max_residual,jnp.where(accepted,qp.max_violation,-jnp.inf))
        trace=dict(state=x,control=control,active=accepted,status=status,observed_state=sensed,observed_obstacles=seen,
            clearance=jnp.where(accepted,clear,jnp.nan),state_bound_violation=jnp.where(accepted,bound,jnp.nan),
            qp_violation=jnp.where(accepted,qp.max_violation,jnp.nan),h=h,psi1=psi,envelope_domain=domain,
            route_progress=cursor,route_target=target,route_remaining=remaining,**guidance_info)
        return (x,status,count,minimum,cursor,max_residual),trace
    carry=(initial,status,jnp.int32(0),minimum,jnp.asarray(cursor,observed.dtype),jnp.asarray(-jnp.inf,observed.dtype))
    if guidance is None:
        (final,status,count,minimum,final_cursor,residual),trace=jax.lax.scan(tick,carry,(jnp.arange(steps),innovations))
    else:
        # The expensive predictive nominal stops after the last active branch.
        # AOT callers discard history for summary batches, eliminating buffers.
        template=jax.eval_shape(tick,carry,(jnp.int32(0),innovations[0]))[1]
        storage=jax.tree.map(lambda a:jnp.zeros((steps,*a.shape),a.dtype),template)
        def condition(state):
            k,c,_=state
            return (k<steps)&((k==0)|(c[1]==RUNNING))
        def body(state):
            k,c,history=state;c,record=tick(c,(k,innovations[k]))
            history=jax.tree.map(lambda a,v:jax.lax.dynamic_update_index_in_dim(a,v,k,0),history,record)
            return k+1,c,history
        _,(final,status,count,minimum,final_cursor,residual),trace=jax.lax.while_loop(condition,body,(jnp.int32(0),carry,storage))
    status=jnp.where(status==RUNNING,TIMEOUT,status)
    route_progress=physical_route_coordinate(final[:2],points,route_mask,final_cursor)-physical_route_coordinate(initial[:2],points,route_mask,cursor)
    summary=dict(final_state=final,status=status,steps=count,min_clearance=minimum,worst_qp_violation=residual,
        route_progress=route_progress,goal_progress=jnp.linalg.norm(initial[:2]-goal)-jnp.linalg.norm(final[:2]-goal),final_cursor=final_cursor)
    from .quad2d_guidance import TerminalGuidanceConfig
    if isinstance(guidance,TerminalGuidanceConfig):summary['physical_initial_state']=initial
    return summary,trace,dict(initial_state=initial,obstacles=truth_obs)
