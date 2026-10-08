"""Independent NumPy audit of every applied local adaptive-control step."""
import numpy as np
from .local_unicycle_collection import ROBOT,K

def audit_trajectory(initial,world,goal,errors,result,trace,filtered_observation=False,observer_memory=None):
    """NumPy geometry and independent exact-held-input quadrature at every tick."""
    x=np.asarray(trace['before'],float);y=np.asarray(trace['state'],float);u=np.asarray(trace['control'],float)
    active=np.asarray(trace['active'],bool);o=np.asarray(world,float);obs=x.copy();obs[:,:2]+=errors[:,0]
    observer_audit={}
    if filtered_observation:
        from .local_unicycle_observer import audit_observations
        observer_audit=audit_observations(world,errors,trace,observer_memory)
    else:
        np.testing.assert_allclose(obs,trace['observed'],atol=2e-6,rtol=0)
    # Selection must use the rounded observed input that the controller received.
    obs=np.asarray(trace['observed'],float);seen=(np.asarray(world,np.float32)[None,:,:2]+errors[:,1:]).astype(float)
    if filtered_observation:seen=np.asarray(trace['observed_world'][:,:,:2],float)
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
    gains=np.asarray(trace['gain'],float)
    if gains.shape!=(len(x),2) or not np.isfinite(gains).all() or (gains<=0).any():raise ValueError('Invalid applied gains')
    rhs=2*np.sum(velocity**2,-1)+gains.sum(-1)[:,None]*hd+gains.prod(-1)[:,None]*h
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
    was_live=np.concatenate(([True],np.asarray(trace['status'][:-1])==0))
    if not np.array_equal(active,was_live&trace['feasible']&trace['admissible']):raise ValueError('False active decision')
    if not np.array_equal(trace['query'],was_live&(np.arange(len(x))%4==0)):raise ValueError('Changed adaptation cadence')
    psi=hd+gains[:,0,None]*h
    domain=np.minimum(h.min(-1),psi.min(-1))
    if np.any(active&(domain<-5e-5)):raise ValueError('Applied inadmissible gain')
    changed=np.any(gains[1:]!=gains[:-1],axis=-1)
    if np.any(changed&~np.asarray(trace['query'][1:])):raise ValueError('Gain changed outside query')
    if np.any(np.asarray(trace['accepted'])&~np.asarray(trace['query'])):raise ValueError('Learned proposal outside query')
    reached=(np.linalg.norm(y[:,:2]-goal,axis=-1)<=ROBOT.goal_tolerance)&(np.abs(y[:,3])<=.2)
    if np.any((np.asarray(trace['status'])==1)&~reached):raise ValueError('False goal status')
    final_status=int(trace['status'][-1]) or 4
    if int(result['status'])!=final_status:raise ValueError('Wrong final status')
    return_gain_changes=int(changed.sum())
    return dict(applied_steps=int(active.sum()),gain_changes=return_gain_changes,maximum_motion_error=err,
        maximum_cbf_violation=float(violation[active].max()) if active.any() else None,
        minimum_sampled_clearance=min_clear if np.isfinite(min_clear) else None,audit_passed=True,**observer_audit)
