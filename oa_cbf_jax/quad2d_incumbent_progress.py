"""Audit the current-gain comparison from observed queries and live scores.

The predictive witness is recomputed with the frozen controller. Independent
physical integration/CBF auditing remains separate and unchanged.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .quad2d_guidance import predictive_flight_control
from .quad2d_guidance import terminal_guidance_from_contract


def make_auditor(config,guidance,batch,candidates):
    if batch not in (8,32):raise ValueError('Declared incumbent-audit batch required')
    guidance=terminal_guidance_from_contract(guidance)
    candidates=np.asarray(candidates,np.float32)
    def witness(x,goal,obs,mask,gain,points,rm,cursor,noise):
        result,info=predictive_flight_control(x,goal,obs,mask,gain,points,rm,cursor,config,guidance,noise)
        qp,h,psi,domain,*_=result;t=config.robot.qp_tolerance
        return info['approved']&qp.feasible&(h>=-t)&(psi>=-t)&(domain>=-t)
    evaluate=jax.jit(jax.vmap(witness));executable=None

    def audit(data):
        nonlocal executable
        query=np.asarray(data['requery'],bool);indices=np.flatnonzero(query)
        before=np.vstack((np.full((1,2),4.,np.float32),data['gain'][:-1]))
        cursor=np.r_[np.float32(0),data['route_progress'][:-1]].astype(np.float32)
        feasible=np.asarray(data['guidance_incumbent_previous_feasible'],bool)
        if feasible.shape!=query.shape:raise ValueError('Missing incumbent witness record')
        for start in range(0,len(indices),batch):
            chosen=indices[start:start+batch];padded=np.pad(chosen,(0,batch-len(chosen)),mode='edge')
            repeat=lambda a:np.broadcast_to(a,(batch,*a.shape))
            args=tuple(map(jnp.asarray,(data['observed_state'][padded],repeat(data['goal']),data['observed_obstacles'][padded],
                (data['controller_obstacle_mask'][padded] if 'controller_obstacle_mask' in data else repeat(data['obstacle_mask'])),before[padded],repeat(data['points']),repeat(data['route_mask']),cursor[padded],repeat(data['noise']))))
            if executable is None:executable=evaluate.lower(*args).compile()
            actual=np.asarray(executable(*args))[:len(chosen)]
            if not np.array_equal(actual,feasible[chosen]):raise ValueError('Changed previous-gain witness')
        improvements=recoveries=0
        for k in indices:
            pool=np.vstack((candidates,before[k]))
            mean=np.asarray(data['guidance_query_mean'][k],float)
            if mean.ndim!=3 or mean.shape[1:]!=(len(pool),2) or not np.isfinite(mean).all():
                raise ValueError('Missing finite live candidate predictions')
            ranking=mean[...,1].mean(0)-.01*np.sum(np.log(pool.astype(float)/before[k])**2,-1)
            previous=float(data['guidance_incumbent_previous_score'][k]);selected=float(data['guidance_incumbent_selected_score'][k])
            np.testing.assert_allclose(previous,ranking[-1],atol=2e-6,rtol=2e-6)
            matched=np.all(np.abs(pool-data['gain'][k])<1e-6,-1)
            if not matched.any() or not np.any(np.isclose(selected,ranking[matched],atol=2e-6,rtol=2e-6)):
                raise ValueError('Selected score does not match the recorded gain')
            if data['active'][k] and int(data['source'][k])==0:
                if feasible[k]:
                    if not selected>previous:raise ValueError('Learned switch failed the incumbent progress comparison')
                    improvements+=1
                else:recoveries+=1
            elif data['active'][k]:
                if not feasible[k] or not np.array_equal(data['gain'][k],before[k]):
                    raise ValueError('Fallback bypassed the incumbent physical witness')
        return dict(incumbent_query_witnesses_recomputed=len(indices),incumbent_improving_queries=improvements,
            incumbent_recovery_queries=recoveries,incumbent_audit_passed=True)
    return audit
