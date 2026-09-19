"""Actual acquired-history bicycle branches with censored physical futures."""
import jax
import jax.numpy as jnp
from .bicycle import integrate_bicycle,bicycle_state_violation
from .bicycle_control import BicycleControlConfig,bicycle_control,bicycle_arrived,constant,observed32
from .bicycle_guidance import guided_bicycle_control,BicycleGuidanceConfig
from .bicycle_observation import unit_errors,observe,speed_error_bound
from .bicycle_rollout import RUNNING,GOAL,COLLISION,INFEASIBLE,TIMEOUT,INADMISSIBLE,PLANNER_FAILURE,STATE_BOUND
from .dynamics import signed_clearance,swept_disk_clearance
from .routing import physical_route_coordinate


def make_observed_episode(config=BicycleControlConfig(),steps=80,guidance=BicycleGuidanceConfig(),stop_at_goal=True):
    c=config.robot
    def episode(initial,goal,obstacles,mask,alpha,points,route_mask,ready,cursor,first_x,first_o,bias_x,bias_o,noise,key):
        initial=initial.astype(jnp.float64);obstacles=obstacles.astype(jnp.float64)
        dt=constant(c.dt,jnp.float64);radius=constant(c.radius,jnp.float64);error=speed_error_bound(noise)
        minimum=jnp.min(signed_clearance(initial[:2],obstacles,mask,radius))
        status=jnp.where(ready,RUNNING,PLANNER_FAILURE)
        if stop_at_goal:status=jnp.where(bicycle_arrived(initial,goal,config),GOAL,status)
        status=jnp.where(bicycle_state_violation(initial,c)>config.qp_tolerance,STATE_BOUND,status)
        status=jnp.where(minimum<=0,COLLISION,status)
        initial_cursor=cursor
        def tick(carry,k):
            x,status,count,minimum,cursor,worst=carry;active=status==RUNNING
            current=obstacles.at[:,:2].set(obstacles[:,:2]+k.astype(jnp.float64)*dt*obstacles[:,3:5])
            ix,io=unit_errors(jax.random.fold_in(key,k),len(mask))
            # The first observation was actually acquired before this branch.
            # Future replica innovations never alter it or the sampled history.
            ix=jnp.where(k==0,0.,ix);io=jnp.where(k==0,0.,io)
            sensed,seen=observe(x,current,mask,bias_x,bias_o,noise,ix,io)
            sensed=jnp.where(k==0,first_x,sensed);seen=jnp.where(k==0,first_o,seen)
            guide={}
            if guidance is None:
                qp,h,domain,proposed,remaining,target=bicycle_control(sensed,goal,seen,mask,alpha,points,route_mask,cursor,config,error)
            else:
                qp,h,domain,proposed,remaining,target,info=guided_bicycle_control(sensed,goal,seen,mask,alpha,points,route_mask,cursor,config,guidance,error,noise=noise)
                guide={'guidance_'+k:v for k,v in info.items()}
            admissible=(h>=-config.qp_tolerance)&(domain>0);accepted=active&qp.feasible&admissible
            u=jnp.where(accepted,qp.control,jnp.zeros(2,jnp.float32))
            y,sub=integrate_bicycle(x,u,c);starts=jnp.concatenate((x[None],sub[:-1]))
            times=k.astype(jnp.float64)*dt+jnp.arange(c.integration_substeps,dtype=jnp.float64)*dt/c.integration_substeps
            clearance=jnp.min(jax.vmap(lambda a,b,t:swept_disk_clearance(a,b,obstacles,mask,radius,t,t+dt/c.integration_substeps))(starts,sub,times))
            violation=jnp.max(jax.vmap(lambda s:bicycle_state_violation(s,c))(jnp.concatenate((x[None],sub))))
            status=jnp.where(active&~qp.feasible,INFEASIBLE,status);status=jnp.where(active&~admissible,INADMISSIBLE,status)
            if stop_at_goal:status=jnp.where(accepted&bicycle_arrived(y,goal,config),GOAL,status)
            status=jnp.where(accepted&(violation>config.qp_tolerance),STATE_BOUND,status);status=jnp.where(accepted&(clearance<=0),COLLISION,status)
            before=x;before_cursor=cursor
            x=jnp.where(accepted,y,x);cursor=jnp.where(accepted,proposed,cursor);count+=accepted.astype(jnp.int32)
            minimum=jnp.minimum(minimum,jnp.where(accepted,clearance,jnp.inf));worst=jnp.maximum(worst,jnp.where(accepted,qp.max_violation,-jnp.inf))
            trace=dict(state=x,state_before=before,observed_state=sensed,observed_obstacles=seen,control=u,active=accepted,status=status,h=h,domain=domain,
                qp_violation=qp.max_violation,route_progress=cursor,cursor_before=before_cursor,route_remaining=remaining,route_target=target,
                clearance=jnp.where(accepted,clearance,jnp.nan),state_violation=jnp.where(accepted,violation,jnp.nan),innovation_x=ix,innovation_o=io,**guide)
            return (x,status,count,minimum,cursor,worst),trace
        carry=(initial,status,jnp.int32(0),minimum,cursor,jnp.float32(-jnp.inf))
        template=jax.eval_shape(tick,carry,jnp.int32(0))[1];storage=jax.tree.map(lambda a:jnp.zeros((steps,*a.shape),a.dtype),template)
        def body(data):
            k,carry,history=data;carry,record=tick(carry,k)
            history=jax.tree.map(lambda a,v:jax.lax.dynamic_update_index_in_dim(a,v,k,0),history,record)
            return k+1,carry,history
        _,(x,status,count,minimum,cursor,worst),trace=jax.lax.while_loop(lambda data:(data[0]<steps)&((data[0]==0)|(data[1][1]==RUNNING)),body,(jnp.int32(0),carry,storage))
        status=jnp.where(status==RUNNING,TIMEOUT,status)
        progress=physical_route_coordinate(observed32(x)[:2],points,route_mask,cursor)-physical_route_coordinate(observed32(initial)[:2],points,route_mask,initial_cursor)
        summary=dict(final_state=x,status=status,steps=count,min_clearance=minimum,worst_qp_violation=worst,final_cursor=cursor,route_progress=progress,
            goal_progress=jnp.linalg.norm(initial[:2]-goal)-jnp.linalg.norm(x[:2]-goal))
        return summary,trace
    return episode
