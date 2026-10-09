"""Shared bicycle margin audit implementation."""

import numpy as np

from fractions import Fraction

from .bicycle_control import BicycleControlConfig

from .bicycle_audit import reference_barrier

def fused_linear_observation(position,velocity,duration):
    """Independently reproduce FP64 fused motion followed by FP32 sensing.

    Near an FP32 midpoint, separate NumPy multiply/add can round to the other
    neighbor from the compiled fused operation. A conservative FP64 roundoff
    interval identifies ambiguous conversions; exact rational arithmetic then
    supplies the correctly rounded FP64 fused value before FP32 conversion.
    Normal values use vectorized NumPy. No JAX/controller code is called.
    """
    position,velocity=np.broadcast_arrays(np.asarray(position,float),np.asarray(velocity,float))
    duration=float(duration);motion=duration*velocity;approx=position+motion
    error=4*np.finfo(float).eps*(abs(position)+abs(motion))+4*np.finfo(float).tiny
    result=approx.astype(np.float32)
    ambiguous=(approx-error).astype(np.float32)!=(approx+error).astype(np.float32)
    for index in zip(*np.nonzero(ambiguous)):
        exact=Fraction(float(position[index]))+Fraction(duration)*Fraction(float(velocity[index]))
        result[index]=np.float32(float(exact))
    return result.astype(float)

def predicted_observation(state, control, obstacles, config=BicycleControlConfig(), *, linear_fused=True):
    """Independent complex-plane held-input displacement, then sensor rounding."""
    x=np.asarray(state,float);u=np.asarray(control,float);o=np.asarray(obstacles,float);dt=config.robot.dt
    distance=dt*(x[...,3]+.5*dt*u[...,0]);angle=u[...,1]*distance/config.robot.rear_axle_distance
    shift=distance*(1+1j*u[...,1])*np.exp(1j*(x[...,2]+angle/2))*np.sinc(angle/(2*np.pi))
    y=x.copy();y[...,0]+=shift.real;y[...,1]+=shift.imag;y[...,2]+=angle;y[...,3]+=dt*u[...,0]
    future=o.copy()
    # CPU fusion and GPU separate multiply/add are both legal FP64 evaluation
    # paths. Their FP32 sensor conversions can differ at an exact midpoint.
    future[...,:2]=(fused_linear_observation(o[...,:2],o[...,3:5],dt) if linear_fused
                    else (o[...,:2]+dt*o[...,3:5]).astype(np.float32).astype(float))
    return y.astype(np.float32).astype(float),future.astype(np.float32).astype(float)

def reference_margin(state,control,obstacles,mask,noise,config=BicycleControlConfig(), *, linear_fused=True):
    x=np.asarray(state,float);u=np.asarray(control,float);o=np.asarray(obstacles,float);n=np.asarray(noise,float)
    y,future=predicted_observation(x,u,o,config,linear_fused=linear_fused);dt=config.robot.dt;lr=config.robot.rear_axle_distance
    rounding=8*np.finfo(np.float32).eps*(1+np.maximum.reduce([np.max(abs(x),axis=-1),np.max(abs(y),axis=-1),np.max(abs(o),axis=(-2,-1)),np.max(abs(future),axis=(-2,-1))]))
    ev=.15*n[...,2]+n[...,2]+rounding;et=1.15*n[...,1]+rounding
    turn=abs(u[...,1])*dt/lr
    ep_ego=.3*n[...,0]+dt*np.hypot(1,u[...,1])*(ev+(abs(x[...,3])+abs(u[...,0])*dt)*np.minimum(et+turn*ev,2))+rounding
    et_ego=.3*n[...,1]+turn*ev+rounding;ev_ego=.3*n[...,2]+rounding
    ep_obs=.3*n[...,3]+dt*(1.15*n[...,4]+rounding)+rounding
    ev_obs=.3*n[...,4]+rounding;er=.3*n[...,5]+rounding
    ep=ep_ego+ep_obs;relative_ev=ev_obs+ev_ego+abs(y[...,3])*np.minimum(et_ego,2)
    p=future[...,:2]-y[...,None,:2];v=future[...,3:5]-y[...,None,3,None]*np.stack((np.cos(y[...,2]),np.sin(y[...,2])),axis=-1)[...,None,:]
    d=np.linalg.norm(p,axis=-1);speed=np.linalg.norm(v,axis=-1);unit=p/np.maximum(d[...,None],1e-12)
    radial=np.sum(unit*v,axis=-1);lateral=unit[...,0]*v[...,1]-unit[...,1]*v[...,0]
    error=speed*2*np.sin(np.arcsin(np.minimum(ep[...,None]/np.maximum(d,1e-12),1))/2)+relative_ev[...,None]
    radius=config.barrier_inflation*(future[...,2]+config.robot.radius+config.clearance_buffer+er[...,None])
    supported=(d>ep[...,None]+radius)&(radius>0)
    root=np.sqrt(np.maximum((d-ep[...,None])**2-radius**2,0));factor=np.sqrt(config.barrier_inflation**2-1)/np.maximum(radius,1e-12)*root
    lower=radial-error+factor*(1+.5*np.maximum(abs(lateral)-error,0)**2/np.sqrt((speed+relative_ev[...,None])**2+config.relative_speed_epsilon**2))
    lower=np.where(mask,np.where(supported,lower,-np.inf),np.inf)
    return dict(lower=np.min(lower,axis=-1),per_obstacle=lower,predicted_state=y,predicted_obstacles=future,
        ego_position_error=ep_ego,ego_heading_error=et_ego,ego_speed_error=ev_ego,obstacle_position_error=ep_obs,
        obstacle_velocity_error=ev_obs,position_error=ep,velocity_error=relative_ev,radius_error=er,rounding_allowance=rounding)

def audit_margin_trace(d,config=BicycleControlConfig()):
    indices=np.flatnonzero(d['active']);mask=np.asarray(d['mask'],bool)
    if not len(indices):return dict(applied=0,checked_next_observations=0,positive_bounds=0)
    r=reference_margin(d['observed_state'][indices],d['control'][indices],d['observed_obstacles'][indices],mask,d['noise'],config)
    saved=d['guidance_next_observation_lower'][indices]
    alternate_count=0
    different=~np.isclose(saved,r['lower'],atol=2e-10,rtol=2e-10)
    if np.any(different):
        alternate=reference_margin(d['observed_state'][indices[different]],d['control'][indices[different]],
            d['observed_obstacles'][indices[different]],mask,d['noise'],config,linear_fused=False)
        # Recompute the entire bound from the separately rounded observation;
        # never widen its tolerance or choose arbitrary neighboring values.
        np.testing.assert_allclose(saved[different],alternate['lower'],atol=2e-10,rtol=2e-10)
        alternate_count=int(np.sum(different))
        for key in r:r[key][different]=alternate[key]
    np.testing.assert_allclose(saved,r['lower'],atol=2e-10,rtol=2e-10)
    assert np.all(~d['guidance_margin_preferred'][indices] | (saved>=0))
    if np.all(d['noise']==0):assert not np.any(d['guidance_margin_preferred'])
    present=indices+1<len(d['observed_state']);k=indices[present];next_x=d['observed_state'][k+1].astype(float);next_o=d['observed_obstacles'][k+1].astype(float)
    pred_x=r['predicted_state'][present];pred_o=r['predicted_obstacles'][present]
    # No latent error realization enters the bound. Actual subsequent sensor
    # outputs are used only to audit retained transitions, including rejection.
    for actual,limit in [(np.linalg.norm(next_x[:,:2]-pred_x[:,:2],axis=-1),r['ego_position_error'][present]),
                         (abs(next_x[:,2]-pred_x[:,2]),r['ego_heading_error'][present]),
                         (abs(next_x[:,3]-pred_x[:,3]),r['ego_speed_error'][present]),
                         (np.max(np.linalg.norm(next_o[:,mask,:2]-pred_o[:,mask,:2],axis=-1),axis=-1),r['obstacle_position_error'][present]),
                         (np.max(np.linalg.norm(next_o[:,mask,3:5]-pred_o[:,mask,3:5],axis=-1),axis=-1),r['obstacle_velocity_error'][present]),
                         (np.max(abs(next_o[:,mask,2]-pred_o[:,mask,2]),axis=-1),r['radius_error'][present])]:
        if np.any(actual>limit+1e-10):raise ValueError('Next observation escaped declared model error envelope')
    valid=np.isfinite(saved[present])
    if np.any(valid):
        h=reference_barrier(next_x[valid,None,:],next_o[valid][:,mask],config)
        if not np.all(np.isfinite(h)) or np.any(np.min(h,axis=-1)<saved[present][valid]-1e-10):raise ValueError('Observed barrier below recorded lower bound')
    return dict(applied=len(indices),checked_next_observations=len(k),positive_bounds=int(np.sum(saved>=0)),
        preferred=int(np.sum(d['guidance_margin_preferred'][indices])),no_positive_profile=int(np.sum(d['guidance_margin_profiles'][indices]==0)),
        separate_linear_rounding_checks=alternate_count)
