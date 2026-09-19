"""Actual affine-bicycle fixed-class-K pilot; perfect observations declared.

No learned policy or noisy-observation guarantee is implied. Every attempted
command, terminal rejection and moving obstacle is retained for replay.
"""
import jax
import jax.numpy as jnp
from .bicycle import integrate_bicycle,bicycle_state_violation
from .bicycle_control import BicycleControlConfig,bicycle_control,bicycle_arrived,constant,observed32
from .dynamics import signed_clearance,swept_disk_clearance

RUNNING,GOAL,COLLISION,INFEASIBLE,TIMEOUT,INADMISSIBLE,PLANNER_FAILURE,STATE_BOUND=range(8)
NAMES={0:'running',1:'goal_reached',2:'collision',3:'qp_rejected',4:'timeout',5:'barrier_inadmissible',6:'planner_failure',7:'state_bound_violation'}


def make_bicycle_episode(config=BicycleControlConfig(),steps=1600,guidance=None):
    c=config.robot
    def episode(initial,goal,obstacles,mask,alpha,points,route_mask,ready):
        # Accumulate physical motion in64; expose only rounded current32
        # observations to control. Save those exact observations for row audit.
        initial=initial.astype(jnp.float64);obstacles=obstacles.astype(jnp.float64)
        dt=constant(c.dt,jnp.float64);radius=constant(c.radius,jnp.float64)
        minimum=jnp.min(signed_clearance(initial[:2],obstacles,mask,radius))
        status=jnp.where(ready,RUNNING,PLANNER_FAILURE)
        status=jnp.where(bicycle_arrived(initial,goal,config),GOAL,status)
        status=jnp.where(bicycle_state_violation(initial,c)>config.qp_tolerance,STATE_BOUND,status)
        status=jnp.where(minimum<=0,COLLISION,status)
        def tick(carry,k):
            x,status,count,minimum,cursor,worst=carry;active=status==RUNNING
            seen=observed32(obstacles.at[:,:2].set(obstacles[:,:2]+k.astype(jnp.float64)*dt*obstacles[:,3:5]))
            observed=observed32(x)
            if guidance is None:
                qp,h,domain,proposed,remaining,target=bicycle_control(observed,goal,seen,mask,alpha,points,route_mask,cursor,config)
            else:
                from .bicycle_guidance import guided_bicycle_control
                qp,h,domain,proposed,remaining,target,guide=guided_bicycle_control(observed,goal,seen,mask,alpha,points,route_mask,cursor,config,guidance)
            admissible=(h>=-config.qp_tolerance)&(domain>0)
            accepted=active&qp.feasible&admissible
            u=jnp.where(accepted,qp.control,jnp.zeros(2,jnp.float32))
            y,sub=integrate_bicycle(x,u,c);starts=jnp.concatenate((x[None],sub[:-1]))
            times=k.astype(jnp.float64)*dt+jnp.arange(c.integration_substeps,dtype=jnp.float64)*dt/c.integration_substeps
            clearance=jnp.min(jax.vmap(lambda a,b,t:swept_disk_clearance(a,b,obstacles,mask,radius,t,t+dt/c.integration_substeps))(starts,sub,times))
            violation=jnp.max(jax.vmap(lambda s:bicycle_state_violation(s,c))(jnp.concatenate((x[None],sub))))
            status=jnp.where(active&~qp.feasible,INFEASIBLE,status)
            status=jnp.where(active&~admissible,INADMISSIBLE,status)
            status=jnp.where(accepted&bicycle_arrived(y,goal,config),GOAL,status)
            status=jnp.where(accepted&(violation>config.qp_tolerance),STATE_BOUND,status)
            status=jnp.where(accepted&(clearance<=0),COLLISION,status)
            x=jnp.where(accepted,y,x);cursor=jnp.where(accepted,proposed,cursor);count+=accepted.astype(jnp.int32)
            minimum=jnp.minimum(minimum,jnp.where(accepted,clearance,jnp.inf))
            worst=jnp.maximum(worst,jnp.where(accepted,qp.max_violation,-jnp.inf))
            record=dict(state=x,observed_state=observed,observed_centers=seen[:,:2],control=u,active=accepted,status=status,h=h,domain=domain,
                qp_violation=qp.max_violation,route_progress=cursor,route_remaining=remaining,route_target=target,
                clearance=jnp.where(accepted,clearance,jnp.nan),state_violation=jnp.where(accepted,violation,jnp.nan))
            if guidance is not None:record.update({'guidance_'+key:value for key,value in guide.items()})
            return (x,status,count,minimum,cursor,worst),record
        carry=(initial,status,jnp.int32(0),minimum,jnp.float32(0),jnp.float32(-jnp.inf))
        template=jax.eval_shape(tick,carry,jnp.int32(0))[1]
        storage=jax.tree.map(lambda a:jnp.zeros((steps,*a.shape),a.dtype),template)
        def condition(data):
            k,c,_=data
            return (k<steps)&((k==0)|(c[1]==RUNNING))
        def body(data):
            k,c,history=data;c,record=tick(c,k)
            history=jax.tree.map(lambda a,v:jax.lax.dynamic_update_index_in_dim(a,v,k,0),history,record)
            return k+1,c,history
        _,(x,status,count,minimum,cursor,worst),trace=jax.lax.while_loop(condition,body,(jnp.int32(0),carry,storage))
        status=jnp.where(status==RUNNING,TIMEOUT,status)
        summary=dict(final_state=x,status=status,steps=count,min_clearance=minimum,worst_qp_violation=worst,
            goal_progress=jnp.linalg.norm(initial[:2]-goal)-jnp.linalg.norm(x[:2]-goal))
        return summary,trace
    return episode
