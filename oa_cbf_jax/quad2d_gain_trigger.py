"""Verify gain-query triggers from recorded observations and the previous gain.

Physical integration and applied CBF commands retain their independent auditor.
This separate audit recomputes the trigger with the frozen observed controller;
it is not an independent proof of the predictive controller algorithm itself.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .quad2d_guidance import TerminalGuidanceConfig, predictive_flight_control


def make_trigger_auditor(config, guidance, batch):
    guidance=TerminalGuidanceConfig(**guidance)
    if batch not in (8,32):raise ValueError('Declared trigger audit batch required')

    def predicate(x,goal,obs,mask,gain,points,rm,cursor,noise):
        result,info=predictive_flight_control(x,goal,obs,mask,gain,points,rm,cursor,config,guidance,noise)
        qp,h,psi,domain,*_=result
        tolerance=config.robot.qp_tolerance
        return info['approved']&qp.feasible&(h>=-tolerance)&(psi>=-tolerance)&(domain>=-tolerance)

    evaluate=jax.jit(jax.vmap(predicate))

    def audit(data):
        active=np.asarray(data['active'],bool); query=np.asarray(data['requery'],bool)
        held=active&~query
        recorded=np.asarray(data['guidance_trigger_previous_feasible'],bool)
        if recorded.shape!=active.shape or np.any(recorded[query]) or not recorded[held].all():
            raise ValueError('Query appeared without a failed previous-gain check')
        if np.any(data['source'][held]!=4):raise ValueError('Non-query action did not retain the previous gain')
        previous=np.vstack((np.array([[4.,4.]],np.float32),data['gain'][:-1]))
        np.testing.assert_array_equal(data['gain'][held],previous[held])
        cursor=np.r_[np.float32(0),data['route_progress'][:-1]].astype(np.float32)
        indices=np.flatnonzero(query)
        for start in range(0,len(indices),batch):
            selected=indices[start:start+batch]
            padded=np.pad(selected,(0,batch-len(selected)),mode='edge')
            repeat=lambda value:np.broadcast_to(value,(batch,*value.shape))
            args=(data['observed_state'][padded],repeat(data['goal']),data['observed_obstacles'][padded],
                  repeat(data['obstacle_mask']),previous[padded],repeat(data['points']),
                  repeat(data['route_mask']),cursor[padded],repeat(data['noise']))
            actual=np.asarray(evaluate(*map(jnp.asarray,args)))[:len(selected)]
            if actual.any():raise ValueError('Recorded neural query had a feasible previous gain on replay')
        return dict(trigger_queries_recomputed=len(indices),trigger_held_actions_checked=int(held.sum()),
                    trigger_audit_passed=True)
    return audit
