"""Bicycle observation functions and shared contracts."""

import jax

import jax.numpy as jnp

import numpy as np

from .bicycle_control import observed32, constant

SCHEMA='bicycle_acquired_isotropic_bias_innovation_v67'

BASE_NOISE=np.array([.015,.01,.015,.02,.02,.008],np.float32)

INNOVATION_FRACTION=.15

def unit_errors(key,capacity=64):
    """Uniform disks for vectors, uniform[-1,1] for scalar components."""
    a,b=jax.random.split(key)
    uv=jax.random.uniform(a,(1+2*capacity,2),dtype=jnp.float32)
    angle=2*jnp.pi*uv[:,1];disks=jnp.sqrt(uv[:,0])[:,None]*jnp.stack((jnp.cos(angle),jnp.sin(angle)),axis=1)
    scalar=jax.random.uniform(b,(2+capacity,),minval=-1.,maxval=1.,dtype=jnp.float32)
    x=jnp.concatenate((disks[0],scalar[:2]))
    obs=jnp.concatenate((disks[1:1+capacity],scalar[2:,None],disks[1+capacity:]),axis=1)
    return x,obs

def scales(noise):
    return noise[jnp.array([0,0,1,2])],noise[jnp.array([3,3,5,4,4])]

def sample_bias(key,noise,mask):
    x,o=unit_errors(key,len(mask));xs,os=scales(noise)
    return x*xs,jnp.where(mask[:,None],o*os,0.)

def observe(physical,obstacles,mask,bias_x,bias_o,noise,innovation_x,innovation_o):
    """Current physical obstacles already include elapsed motion; no future input."""
    xs,os=scales(noise);fraction=constant(INNOVATION_FRACTION,jnp.float64)
    x=physical.astype(jnp.float64)+bias_x.astype(jnp.float64)+fraction*xs.astype(jnp.float64)*innovation_x.astype(jnp.float64)
    o=obstacles.astype(jnp.float64)+bias_o.astype(jnp.float64)+fraction*os.astype(jnp.float64)*innovation_o.astype(jnp.float64)
    return observed32(x),jnp.where(mask[:,None],observed32(o),jnp.zeros_like(obstacles,dtype=jnp.float32))

def speed_error_bound(noise):
    # FP32 multiply used to sample the latent bias and observation rounding both
    # contribute tiny errors. Include conservative rounding terms, without a
    # nonzero artificial margin in the exact zero-noise regression.
    base=constant(1+INNOVATION_FRACTION,jnp.float64)*noise[2].astype(jnp.float64)
    return base+jnp.where(noise[2]>0,constant(5e-7,jnp.float64),constant(0.,jnp.float64))


EPS = float(np.finfo(np.float32).eps)

def numpy_estimate(current, past, mask, noise, elapsed):
    """Independent obstacle-loop implementation, including the output rounding."""
    current = np.asarray(current); past = np.asarray(past); mask = np.asarray(mask, bool)
    output = current.copy(); radius = np.zeros(len(mask)); raw_bounds = np.zeros(len(mask))
    used = np.zeros(len(mask), bool); contradiction = np.zeros(len(mask), bool)
    if not np.isfinite(elapsed) or elapsed < 0 or np.any(np.asarray(noise) < 0):
        raise ValueError('Nonnegative finite history and noise required')
    for i in np.flatnonzero(mask):
        raw = current[i, 3:5].astype(float)
        rb = 1.15*float(noise[4])+8*EPS*(1+float(np.hypot(*raw)))
        raw_bounds[i] = radius[i] = rb
        if elapsed <= 0: continue
        now = current[i, :2].astype(float); before = past[i].astype(float)
        secant = (now-before)/float(elapsed)
        rounded_endpoints = 8*EPS*(1+float(np.hypot(*now))+float(np.hypot(*before)))
        sb = (.3*float(noise[3])+rounded_endpoints)/float(elapsed)+8*EPS*(1+float(np.hypot(*secant)))
        contradiction[i] = np.hypot(*(secant-raw)) > rb+sb
        if noise[4] > 0 and sb <= rb and not contradiction[i]:
            used[i] = True; output[i, 3:5] = secant.astype(current.dtype); radius[i] = sb
    return dict(obstacles=output, radius=radius, raw_radius=raw_bounds, used=used, contradiction=contradiction)

def held_offset(current, past, mask, noise, elapsed):
    """Development branch only: hold a causally estimated bias correction.

    Unlike a sliding secant, a fixed correction preserves the original velocity
    innovation differences used by the next-observation margin. Reserve another
    0.3*velocity noise for the difference from the acquisition innovation to any
    future innovation. Reject corrections that cannot retain the original full
    absolute sensor error allowance. No physical value enters this calculation.
    """
    r = numpy_estimate(current, past, mask, noise, elapsed)
    offset = (r['obstacles'][:, 3:5].astype(float)-np.asarray(current)[:, 3:5]).astype(np.float32)
    rounding = 16*EPS*(1+np.linalg.norm(np.asarray(current)[:, 3:5], axis=-1)+np.linalg.norm(offset, axis=-1))
    future_bound = r['radius']+.3*float(noise[4])+rounding
    used = r['used'] & (future_bound <= 1.15*float(noise[4]))
    offset[~used] = 0.
    return dict(offset=offset, used=used, future_radius=np.where(used, future_bound, r['raw_radius']))


from .bicycle_control import BicycleControlConfig

def next_observation_errors(state,control,predicted,obstacles,predicted_obstacles,noise,config=BicycleControlConfig()):
    """Disk/scalar envelopes under persistent bias + independent .15 innovations.

    Integrating two observed/true affine-slip states with the SAME input gives
    the position error by bounding the velocity integral and rotation chord.
    Persistent position/heading/speed biases cancel at the next observation;
    their effect on the predicted dynamics remains explicitly bounded.
    """
    dtype=jnp.float64;x=state.astype(dtype);u=control.astype(dtype);y=predicted.astype(dtype);n=noise.astype(dtype);c=config.robot
    dt=constant(c.dt,dtype);lr=constant(c.rear_axle_distance,dtype)
    magnitude=jnp.max(jnp.concatenate((jnp.abs(x),jnp.abs(y),jnp.abs(obstacles).reshape(-1),jnp.abs(predicted_obstacles).reshape(-1))))
    rounding=constant(8*np.finfo(np.float32).eps,dtype)*(1+magnitude)
    current_speed=constant(1.15,dtype)*n[2]+rounding
    current_heading=constant(1.15,dtype)*n[1]+rounding
    angle=current_heading+jnp.abs(u[1])*dt/lr*current_speed
    motion=dt*jnp.sqrt(1+u[1]**2)*(current_speed+(jnp.abs(x[3])+jnp.abs(u[0])*dt)*jnp.minimum(angle,2.))
    ego_position=constant(.3,dtype)*n[0]+motion+rounding
    ego_heading=constant(.3,dtype)*n[1]+jnp.abs(u[1])*dt/lr*current_speed+rounding
    ego_speed=constant(.3,dtype)*n[2]+rounding
    obstacle_position=constant(.3,dtype)*n[3]+dt*(constant(1.15,dtype)*n[4]+rounding)+rounding
    obstacle_velocity=constant(.3,dtype)*n[4]+rounding
    obstacle_radius=constant(.3,dtype)*n[5]+rounding
    return dict(position_error=ego_position+obstacle_position,velocity_error=obstacle_velocity+ego_speed+jnp.abs(y[3])*jnp.minimum(ego_heading,2.),radius_error=obstacle_radius,
        ego_position_error=ego_position,ego_heading_error=ego_heading,ego_speed_error=ego_speed,obstacle_position_error=obstacle_position,obstacle_velocity_error=obstacle_velocity,rounding_allowance=rounding)

def lower_barrier(state,obstacles,mask,position_error,velocity_error,radius_error,config=BicycleControlConfig()):
    """Lower h over relative-position/velocity disks and radius interval.

    Each positive parabolic term is bounded from below. Unsupported geometric
    domains return -inf; square-root guards never turn them into valid margins.
    """
    dtype=jnp.float64;x=state.astype(dtype);o=obstacles.astype(dtype)
    p=o[:,:2]-x[:2];d=jnp.sqrt(jnp.maximum(jnp.sum(p*p,axis=1),constant(1e-24,dtype)));unit=p/d[:,None]
    v=o[:,3:5]-x[3]*jnp.stack((jnp.cos(x[2]),jnp.sin(x[2])))
    speed=jnp.sqrt(jnp.sum(v*v,axis=1));radial=jnp.sum(unit*v,axis=1);lateral=unit[:,0]*v[:,1]-unit[:,1]*v[:,0]
    eta=2*jnp.sin(jnp.arcsin(jnp.clip(position_error/d,0.,1.))/2)
    component_error=speed*eta+velocity_error
    radius=(constant(config.robot.radius+config.clearance_buffer,dtype)+o[:,2]+radius_error)*constant(config.barrier_inflation,dtype)
    domain=(d-position_error)**2-radius**2;supported=(d>position_error)&(radius>0)&(d-position_error>radius)
    root=jnp.sqrt(jnp.maximum(domain,constant(1e-24,dtype)));shape=jnp.sqrt(constant(config.barrier_inflation**2-1,dtype))/jnp.maximum(radius,constant(1e-12,dtype))
    q=jnp.sqrt((speed+velocity_error)**2+constant(config.relative_speed_epsilon**2,dtype))
    low=radial-component_error+.5*shape*root*jnp.maximum(jnp.abs(lateral)-component_error,0.)**2/q+shape*root
    finite=jnp.isfinite(low)&jnp.isfinite(domain)&(position_error>=0)&(velocity_error>=0)&(radius_error>=0)
    per_obstacle=jnp.where(mask,jnp.where(supported&finite,low,-jnp.inf),jnp.inf)
    return dict(lower=jnp.min(per_obstacle),domain_supported=jnp.all(jnp.where(mask,supported&finite,True)),per_obstacle=per_obstacle)

def next_margin(state,control,predicted,obstacles,predicted_obstacles,mask,noise,config=BicycleControlConfig()):
    errors=next_observation_errors(state,control,predicted,obstacles,predicted_obstacles,noise,config)
    bound=lower_barrier(predicted,predicted_obstacles,mask,errors['position_error'],errors['velocity_error'],errors['radius_error'],config)
    return {**bound,**errors}


WINDOW_TICKS=20

def contract():
    return dict(schema='bicycle_observed_position_secant_velocity',window_ticks=WINDOW_TICKS,
        input='Current observed obstacles, observed positions min(20,tick) ticks ago, mask, declared noise, elapsed time.',
        rule='Inverse marginal variance fusion: persistent position bias cancels; independent bounded disk innovations remain.',
        innovation_fraction=.15,raw_velocity_component_variance='(1+.15^2)*noise_velocity_radius^2/4',
        secant_component_variance='2*.15^2*noise_position_radius^2/(4*elapsed^2)',
        initial='No history or zero declared velocity noise: retain the exact raw velocity.',
        identity='Persistent obstacle indices; no association with latent state.',
        physical_truth_used=False,labels_used=False,controller_solve=False,
        limitation='Assumes constant obstacle velocity and persistent position bias across the one-second window. Approximate moments ignore FP32 rounding. Not a barrier certificate or a changed controller.')

def velocity(current,past_positions,mask,noise,elapsed):
    """Single observation with fixed padded obstacle capacity; AOT/vmap friendly."""
    current=jnp.where(mask[:,None],current,0.).astype(jnp.float64)
    past=jnp.where(mask[:,None],past_positions,0.).astype(jnp.float64)
    noise=noise.astype(jnp.float64);elapsed=jnp.asarray(elapsed,dtype=jnp.float64)
    available=elapsed>0
    duration=jnp.maximum(elapsed,1e-12)
    secant=(current[:,:2]-past)/duration
    raw_variance=(1.+.15**2)*noise[4]**2/4.
    secant_variance=2.*.15**2*noise[3]**2/(4.*duration**2)
    weight=jnp.where(available & (noise[4]>0),raw_variance/jnp.maximum(raw_variance+secant_variance,1e-30),0.)
    estimate=current[:,3:5]+weight*(secant-current[:,3:5])
    return dict(velocity=jnp.where(mask[:,None],estimate,0.),weight=weight,
        history_available=available,elapsed=jnp.where(available,elapsed,0.))

def numpy_velocity(current,past_positions,mask,noise,elapsed):
    """Independent scalar-obstacle reference; no JAX calls."""
    current=np.asarray(current);mask=np.asarray(mask,bool)
    output=np.zeros((len(mask),2),float);weight=0.
    if elapsed>0 and noise[4]>0:
        raw=(float(noise[4])**2)*(1.+.15**2)
        sec=2.*(float(noise[3])*.15/float(elapsed))**2
        weight=1./(1.+sec/raw)
    for i in np.flatnonzero(mask):
        raw=np.asarray(current[i,3:5],float)
        if weight:
            slope=(current[i,:2].astype(float)-np.asarray(past_positions[i],float))/float(elapsed)
            output[i]=(1.-weight)*raw+weight*slope
        else:output[i]=raw
    return dict(velocity=output,weight=weight,history_available=bool(elapsed>0),elapsed=max(0.,float(elapsed)))
