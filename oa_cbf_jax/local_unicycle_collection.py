"""Noisy, acquired-state pilot for shared local graph/QP observations.

The pilot authorizes no fitting. It qualifies the new eight-second label path
before reserving independent training, calibration and benchmark parents.
"""


from dataclasses import asdict, replace


import numpy as np
import jax
import jax.numpy as jnp
from .config import UnicycleConfig
from .controllers import nominal_unicycle, unicycle_cbf_qp, solve_qp2
from .dynamics import integrate_unicycle,signed_clearance,swept_disk_clearance
from .obstacle_selection import nearest_obstacles,nearest_numpy,contract


from .models import unicycle_graph


ROBOT=UnicycleConfig(a_max=.5,w_max=.5,v_max=2.,stationary_obstacles=True)
NOMINAL=replace(ROBOT,v_max=1.5)
K=10
HORIZON=160


def rollout(initial,world,goal,gain,position_errors,prior_status=0):
    """One branch; supplied current/future sensor innovations are gain-independent."""
    def tick(carry,error):
        x,status,done,minimum=carry
        observed=x.at[:2].add(error[0]);all_seen=world.at[:,:2].add(error[1:])
        rows,mask,ids=nearest_obstacles(observed[:2],all_seen,jnp.ones(len(world),bool),K)
        _,a,b,h,psi=unicycle_cbf_qp(observed,goal,rows,mask,gain,ROBOT)
        reference=nominal_unicycle(observed,goal,NOMINAL)
        qp=solve_qp2(reference,a,b,jnp.ones(2,jnp.float32),ROBOT.qp_tolerance)
        admissible=jnp.all((h>=-ROBOT.qp_tolerance)&(psi>=-ROBOT.qp_tolerance))
        active=(status==0)&qp.feasible&admissible
        control=jnp.where(active,qp.control,jnp.zeros(2,jnp.float32))
        y,sub=integrate_unicycle(x,control,ROBOT.dt,ROBOT.integration_substeps)
        starts=jnp.concatenate((x[None],sub[:-1]))
        clearance=jnp.min(jax.vmap(lambda a,b:swept_disk_clearance(a,b,world,jnp.ones(len(world),bool),ROBOT.radius,0.,0.))(starts,sub))
        bounds=jnp.max(jnp.maximum(-sub[:,3],sub[:,3]-ROBOT.v_max))
        reached=(jnp.linalg.norm(y[:2]-goal)<=ROBOT.goal_tolerance)&(jnp.abs(y[3])<=.2)
        ns=jnp.where((status==0)&~qp.feasible,3,status)
        ns=jnp.where((status==0)&~admissible,5,ns)
        ns=jnp.where(active&reached,1,ns);ns=jnp.where(active&(bounds>ROBOT.qp_tolerance),8,ns)
        ns=jnp.where(active&(clearance<=0.),2,ns)
        state=jnp.where(active,y,x);minimum=jnp.minimum(minimum,jnp.where(active,clearance,jnp.inf))
        record=dict(before=x,state=state,observed=observed,control=control,active=active,status=ns,selected_ids=ids,
            feasible=qp.feasible,admissible=admissible,clearance=jnp.where(active,clearance,jnp.float32(0.)))
        return (state,ns,done+active.astype(jnp.int32),minimum),record
    start=(initial,jnp.asarray(prior_status,jnp.int32),jnp.int32(0),
        jnp.min(signed_clearance(initial[:2],world,jnp.ones(len(world),bool),ROBOT.radius)))
    final,trace=jax.lax.scan(tick,start,position_errors)
    state,status,done,clearance=final
    return dict(final_state=state,status=jnp.where(status==0,4,status),steps=done,min_clearance=clearance,
        progress=jnp.linalg.norm(goal-initial[:2])-jnp.linalg.norm(goal-state[:2])),trace


def graph_inputs(initial,world,goal,error):
    observed=initial.at[:2].add(error[0]);seen=world.at[:,:2].add(error[1:])
    selected,valid,ids=nearest_obstacles(observed[:2],seen,jnp.ones(len(world),bool),K)
    features,mask=unicycle_graph(observed,goal,selected,valid,ROBOT)
    return features,mask,observed,selected,ids


def audit_branch(initial,world,goal,gain,errors,result,trace,prior_status=0):
    """NumPy geometry and independent exact-held-input quadrature at every tick."""
    x=np.asarray(trace['before'],float);y=np.asarray(trace['state'],float);u=np.asarray(trace['control'],float)
    active=np.asarray(trace['active'],bool);o=np.asarray(world,float);obs=x.copy();obs[:,:2]+=errors[:,0]
    np.testing.assert_allclose(obs,trace['observed'],atol=2e-6,rtol=0)
    # Selection must use the rounded observed input that the controller received.
    obs=np.asarray(trace['observed'],float);seen=(np.asarray(world,np.float32)[None,:,:2]+errors[:,1:]).astype(float)
    # Independent observed-center distance and geometric tie ordering.
    distance=np.sum((seen-obs[:,None,:2])**2,axis=-1)
    ids=np.lexsort((np.zeros_like(distance),np.zeros_like(distance),np.broadcast_to(o[:,2],distance.shape),seen[:,:,1],seen[:,:,0],distance),axis=-1)[:,:K]
    np.testing.assert_array_equal(ids,trace['selected_ids'])
    selected=np.take_along_axis(seen,ids[:,:,None],axis=1)
    radius=o[ids,2];delta=obs[:,None,:2]-selected
    d=np.column_stack((np.cos(obs[:,2]),np.sin(obs[:,2])));normal=np.column_stack((-d[:,1],d[:,0]))
    velocity=obs[:,None,3:4]*d[:,None,:]
    h=np.sum(delta**2,-1)-(radius+ROBOT.radius+ROBOT.clearance_buffer)**2
    hd=2*np.sum(delta*velocity,-1)
    authority=2*np.stack((np.sum(delta*d[:,None,:],-1),obs[:,None,3]*np.sum(delta*normal[:,None,:],-1)),axis=-1)
    rhs=2*np.sum(velocity**2,-1)+gain.sum()*hd+gain.prod()*h
    violation=-np.sum(authority*u[:,None,:],-1)-rhs
    if active.any() and violation[active].max()>5e-5:raise ValueError('Observed CBF violation')
    if np.any(np.abs(u[active])>np.array([.5,.5])+2e-5):raise ValueError('Input violation')
    if not np.array_equal(x[0],initial) or int(active.sum())!=int(result['steps']):raise ValueError('Changed initial state/step count')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(u).all():raise ValueError('Nonfinite physical trace')
    np.testing.assert_array_equal(x[1:],y[:-1]);np.testing.assert_array_equal(y[-1],result['final_state'])
    if np.any(u[~active]!=0) or np.any(y[~active]!=x[~active]):raise ValueError('Stopped branch advanced')
    nodes,weights=np.polynomial.legendre.leggauss(16);tt=(nodes+1)*ROBOT.dt/2
    err=0.;min_clear=np.inf
    for fraction in (.25,.5,.75,1.):
        v=x[:,3,None]+u[:,0,None]*tt*fraction;theta=x[:,2,None]+u[:,1,None]*tt*fraction
        pos=x[:,:2]+ROBOT.dt*fraction/2*np.column_stack((np.sum(weights*v*np.cos(theta),-1),np.sum(weights*v*np.sin(theta),-1)))
        if fraction==1.:
            expected=np.column_stack((pos,np.angle(np.exp(1j*(x[:,2]+u[:,1]*ROBOT.dt))),x[:,3]+u[:,0]*ROBOT.dt))
            if active.any():
                diff=np.abs(expected[active]-y[active]);diff[:,2]=np.abs(np.angle(np.exp(1j*(expected[active,2]-y[active,2]))))
                err=float(diff.max())
                if err>8e-6:raise ValueError('Independent integration failed')
        clear=np.min(np.linalg.norm(pos[:,None,:]-o[None,:,:2],axis=-1)-ROBOT.radius-o[None,:,2],axis=-1)
        if active.any():
            min_clear=min(min_clear,float(clear[active].min()))
            if clear[active].min()<-1e-5 and int(result['status'])!=2:raise ValueError('Unreported all-world collision')
    if np.any((y[active,3]<-1e-5)|(y[active,3]>ROBOT.v_max+1e-5)) and int(result['status'])!=8:
        raise ValueError('Unreported state-bound violation')
    if prior_status and active.any():raise ValueError('Failed acquisition revived')
    return dict(applied_steps=int(active.sum()),maximum_motion_error=err,
        maximum_cbf_violation=float(violation[active].max()) if active.any() else None,
        minimum_sampled_clearance=min_clear if np.isfinite(min_clear) else None,audit_passed=True)
