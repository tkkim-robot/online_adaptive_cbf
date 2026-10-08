"""AOT learned-policy scan: observed inputs, checked gain switches, actual physics."""
import numpy as np
import jax
import jax.numpy as jnp
from .quad3d import integrate_quad3d
from .quad3d_observation import observe,observed_arrived
from .quad3d_observed_rollout import observed_control


def checked_control(x,goal,o,mask,proposed,previous,points,rm,cursor,noise,c,nominal_bias=None,previous_result=None):
    primary=observed_control(x,goal,o,mask,proposed,points,rm,cursor,noise,c,True,nominal_bias)
    valid=primary[1]&(primary[2]>=-c.qp_tolerance)&(primary[3]>=-c.qp_tolerance)
    retry=~valid&jnp.any(proposed!=previous)
    result=jax.lax.cond(retry,lambda _:observed_control(x,goal,o,mask,previous,points,rm,cursor,noise,c,True,nominal_bias) if previous_result is None else previous_result,lambda _:primary,operand=None)
    proof=dict(attempted_control=primary[0],attempted_feasible=primary[1],attempted_psi=primary[2],attempted_domain=primary[3],
        attempted_residual=primary[4],attempted_iterations=primary[5])
    return result,jnp.where(retry,previous,proposed),retry,proof


def failure_query_trigger(running,scheduled,arrived,held_feasible):
    """One extra learned query only when the held QP has actually failed."""
    return running&~scheduled&~arrived&~held_feasible


def make_policy_rollout(selector,steps=1600,ordered=False,failure_requery=False,return_stepper=False):
    c=selector.robot;p=selector.config
    estimating=c.nominal_bias_observer=='innovation_ema_v97'
    def rollout(params,x,goal,obs,mask,initial_gain,points,rm,noise,bx,bo,ix,io,waypoint_count=None):
        if ordered and waypoint_count is None:raise ValueError('Ordered mission needs a waypoint count')
        goals=goal;routes=points;route_masks=rm
        def mission(leg):
            return (goals[leg],routes[leg],route_masks[leg]) if ordered else (goals,routes,route_masks)
        dt=jnp.asarray(np.asarray(c.robot.dt,np.float64),x.dtype)
        def current(state,k):
            true_o=obs.at[:,:2].add(k.astype(x.dtype)*dt*obs[:,3:5])
            return observe(state,true_o,mask,bx,bo,noise,ix[k],io[k])
        # Shape-only tracing supplies a typed zero record for non-query ticks.
        def predict(seen,so,goal,points,rm,cursor,previous_u,previous_gain,bias_estimate):
            args=(params,seen,goal,so,mask,points,rm,cursor,previous_u,previous_gain,noise)
            return selector.predict(*args,nominal_bias=bias_estimate) if estimating else selector.predict(*args)
        template=jax.eval_shape(predict,x,obs,*mission(0),jnp.zeros((),x.dtype),jnp.zeros(4,x.dtype),initial_gain,jnp.zeros(12,x.dtype))
        zero=jax.tree.map(lambda a:jnp.zeros(a.shape,a.dtype),template)
        def tick(carry,k):
            state,status,count,cursor,previous_u,previous_gain,bias_estimate,previous_seen,leg=carry
            seen,so=current(state,k);handoff=jnp.bool_(False)
            if ordered:
                handoff=(status==0)&(leg<waypoint_count-1)&observed_arrived(seen,goals[leg],noise,c)
                leg=leg+handoff.astype(jnp.int32);cursor=jnp.where(handoff,jnp.zeros_like(cursor),cursor)
            goal,points,rm=mission(leg)
            requery=(status==0)&((k%p.query_every_ticks==0)|handoff)
            if estimating:
                from .quad3d_observer import update_bias
                updated=update_bias(bias_estimate,previous_seen,seen,previous_u,noise,c)
                bias_estimate=jnp.where(k>0,updated,bias_estimate)
            if failure_requery:
                # Same observation, same past observer memory, original QP.
                # The cached held solve also supplies the previous-gain retry.
                held=observed_control(seen,goal,so,mask,previous_gain,points,rm,cursor,noise,c,True,
                    nominal_bias=bias_estimate if estimating else None)
                extra_query=failure_query_trigger(status==0,requery,held[-1],held[1])
                requery=requery|extra_query
            prediction=jax.lax.cond(requery,lambda _:predict(seen,so,goal,points,rm,cursor,previous_u,previous_gain,bias_estimate),lambda _:zero,None)
            proposed_gain=jnp.where(requery,prediction['network_gain'],previous_gain)
            if failure_requery:
                held_attempt=dict(attempted_control=held[0],attempted_feasible=held[1],attempted_psi=held[2],
                    attempted_domain=held[3],attempted_residual=held[4],attempted_iterations=held[5])
                result,gain,retry,attempt=jax.lax.cond(requery,
                    lambda _:checked_control(seen,goal,so,mask,proposed_gain,previous_gain,points,rm,cursor,noise,c,
                        nominal_bias=bias_estimate if estimating else None,previous_result=held),
                    lambda _:(held,previous_gain,jnp.bool_(False),held_attempt),None)
            else:
                result,gain,retry,attempt=checked_control(seen,goal,so,mask,proposed_gain,previous_gain,points,rm,cursor,noise,c,
                    nominal_bias=bias_estimate if estimating else None)
            u,feasible,psi,domain,residual,iterations,target,next_progress,remaining,visible,done=result
            final_leg=(leg==waypoint_count-1) if ordered else jnp.bool_(True)
            status=jnp.where((status==0)&done&final_leg,1,status)
            status=jnp.where((status==0)&((psi<-c.qp_tolerance)|(domain<-c.qp_tolerance)),2,status)
            status=jnp.where((status==0)&~feasible,3,status);active=status==0
            applied=jnp.where(active,u,jnp.zeros(4,state.dtype));yy,sub=integrate_quad3d(state,applied,c.robot);next_state=jnp.where(active,yy,state)
            times=(k+jnp.asarray(np.arange(1,c.robot.integration_substeps+1)/c.robot.integration_substeps,state.dtype))*dt
            centers=obs[None,:,:2]+times[:,None,None]*obs[None,:,3:5]
            distance=jnp.linalg.norm(sub[:,None,:2]-centers,axis=-1)-c.robot.radius-obs[None,:,2]
            clear=jnp.min(jnp.where(mask[None,:],distance,jnp.inf))
            limits=jnp.asarray(np.array([c.tilt_limit,c.tilt_limit,c.yaw_limit]+[c.velocity_limit]*3+[c.rate_limit]*3),state.dtype)
            violation=jnp.maximum(jnp.max(jnp.abs(sub[:,3:])-limits),jnp.maximum(jnp.max(sub[:,2]-c.altitude_max),jnp.max(c.altitude_min-sub[:,2])))
            status=jnp.where(active&(clear<=0),4,status);status=jnp.where(active&(status==0)&(violation>c.qp_tolerance),5,status)
            next_cursor=jnp.where(active,next_progress,cursor)
            data=dict(state=state,next_state=next_state,observed=seen,control=applied,proposed=u,active=active,status=status,feasible=feasible,
                psi=psi,domain=domain,residual=residual,iterations=iterations,clearance=clear,envelope=violation,route_target=target,
                route_cursor_before=cursor,route_progress=next_cursor,route_remaining=remaining,route_visible=visible,
                requery=requery,controller_gain=gain,network_gain=proposed_gain,previous_gain=previous_gain,previous_control=previous_u,qp_switch_fallback=retry)
            data.update(attempt)
            if failure_requery:
                data.update(failure_requery=extra_query,held_proposed=held[0],held_feasible=held[1],
                    held_psi=held[2],held_domain=held[3],held_residual=held[4],held_iterations=held[5])
            if estimating:data.update(nominal_bias_estimate=bias_estimate,nominal_observation=seen-bias_estimate)
            if ordered:data.update(waypoint_index=leg,waypoint_handoff=handoff,mission_goal=goal)
            return (next_state,status,count+active.astype(jnp.int32),next_cursor,jnp.where(active,applied,previous_u),jnp.where(active,gain,previous_gain),bias_estimate,seen,leg),(data,prediction)
        initial=(x,jnp.int32(0),jnp.int32(0),jnp.zeros((),x.dtype),jnp.zeros(4,x.dtype),initial_gain,jnp.zeros(12,x.dtype),jnp.zeros(12,x.dtype),jnp.int32(0))
        def final_status(carry,elapsed):
            state,status=carry[:2];leg=carry[-1]
            seen,_=current(state,jnp.asarray(elapsed,jnp.int32))
            final_goal,_,_=mission(leg)
            final_leg=(leg==waypoint_count-1) if ordered else jnp.bool_(True)
            status=jnp.where((status==0)&final_leg&observed_arrived(seen,final_goal,noise,c),1,status)
            return jnp.where(status==0,6,status)
        if return_stepper:
            return tick,initial,final_status,dict(initial_state=x,obstacles=obs)
        result,trace=jax.lax.scan(tick,initial,jnp.arange(steps,dtype=jnp.int32))
        state,_,count,*_=result;leg=result[-1];status=final_status(result,steps)
        summary=dict(final_state=state,status=status,steps=count)
        if ordered:summary.update(waypoint_index=leg,waypoints_visited=leg+(status==1).astype(jnp.int32))
        return summary,trace
    return rollout
