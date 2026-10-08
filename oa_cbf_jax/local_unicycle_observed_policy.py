"""Online gain adaptation with the same causal observer used for held labels."""
import jax
import jax.numpy as jnp
from .local_unicycle_collection import ROBOT,K
from .local_unicycle_observer import initialize,observe,advance
from .local_unicycle_policy import qp_control,empty_decision
from .dynamics import integrate_unicycle,signed_clearance,swept_disk_clearance
from .obstacle_selection import nearest_obstacles


def make_rollout(selector,config):
    def rollout(params,calibration,initial,world,goal,errors,bank):
        def tick(carry,inputs):
            x,status,done,minimum,gain,memory=carry;k,error=inputs
            raw=x.at[:2].add(error[0]);raw_world=world.at[:,:2].add(error[1:])
            observed,seen=observe(memory,raw,raw_world)
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
            trace=dict(before=x,state=state,observed=observed,observed_world=seen,control=u,active=active,status=ns,
                selected_ids=ids,feasible=qp.feasible,admissible=admissible,clearance=jnp.where(active,clear,0.),gain=gain,
                query=query,observer_prediction=memory.predicted_position,observer_centers=memory.obstacle_centers,
                observer_ready=memory.ready,**decision)
            return (state,ns,done+active.astype(jnp.int32),minimum,gain,advance(observed,seen,u,active)),trace
        start=(initial,jnp.int32(0),jnp.int32(0),
            jnp.min(signed_clearance(initial[:2],world,jnp.ones(len(world),bool),ROBOT.radius)),
            jnp.asarray(config.initial_gain,jnp.float32),initialize(initial,world))
        final,trace=jax.lax.scan(tick,start,(jnp.arange(len(errors)),errors))
        state,status,steps,clearance,_,_=final
        return dict(final_state=state,status=jnp.where(status==0,4,status),steps=steps,min_clearance=clearance,
            progress=jnp.linalg.norm(goal-initial[:2])-jnp.linalg.norm(goal-state[:2])),trace
    return rollout
