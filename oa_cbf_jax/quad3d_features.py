"""Quad3d features functions and shared contracts."""

import numpy as np

import jax.numpy as jnp

from .quad3d_routing import flight_target, numpy_flight_target

from .quad3d_observation import guidance_obstacles, numpy_obstacles, BASE_NOISE

SCHEMA='quad3d_observed_flight_50_v94'

FEATURES=50

OBSERVER_SCHEMA='quad3d_observed_bias_history_58_v98'

OBSERVER_FEATURES=58

BIAS_INDICES=np.array([3,4,6,7,8,9,10,11])

BIAS_SCALE=np.array([BASE_NOISE[1]]*2+[BASE_NOISE[2]]*3+[BASE_NOISE[3]]*3)

def graph(x,goal,obstacles,mask,points,route_mask,cursor,previous_u,previous_gain,noise,config):
    c=config;r=c.robot;n=len(obstacles)+2;dtype=x.dtype
    positions=jnp.concatenate((x[None,:2],goal[None,:2],obstacles[:,:2]))-x[:2]
    velocities=jnp.concatenate((x[None,6:8],jnp.zeros((1,2),dtype),obstacles[:,3:5]))-x[6:8]
    radii=jnp.concatenate((jnp.asarray(np.array([r.radius,0.]),dtype),obstacles[:,2]))
    clear=jnp.sqrt(jnp.sum(positions**2,axis=-1)+1e-12)-r.radius-radii
    types=jnp.concatenate((jnp.eye(3,dtype=dtype)[jnp.array([0,2])],jnp.tile(jnp.asarray(np.array([0.,1.,0.]),dtype),(n-2,1))))
    target,_,remaining,_=flight_target(x,goal,guidance_obstacles(obstacles,mask,noise),mask,points,route_mask,cursor,c)
    constant=lambda value:jnp.asarray(np.asarray(value,np.float64),dtype)
    context=jnp.concatenate(((goal-x[:3])/5,(target-x[:3])/5,jnp.stack((cursor/10,remaining/10)),
        jnp.reshape((x[2]-c.altitude_min)/(c.altitude_max-c.altitude_min),(1,)),
        x[3:6]/constant([c.tilt_limit,c.tilt_limit,c.yaw_limit]),x[6:9]/c.velocity_limit,x[9:12]/c.rate_limit,
        (2*previous_u-r.input_max-r.input_min)/(r.input_max-r.input_min),jnp.log(previous_gain),noise/constant(BASE_NOISE),
        constant([r.mass/3,r.inertia_x/.5,r.inertia_y/.5,r.inertia_z/.5,r.arm/.3,r.yaw_coefficient/.1,r.gravity/9.8,r.dt/.05])))
    result=jnp.concatenate((types,positions/5,velocities/c.velocity_limit,radii[:,None],clear[:,None]/3,jnp.broadcast_to(context,(n,len(context)))),axis=-1)
    node_mask=jnp.concatenate((jnp.ones(2,bool),mask))
    assert result.shape[-1]==FEATURES
    return jnp.where(node_mask[:,None],result,0.),node_mask

def numpy_graph(x,goal,o,mask,points,rm,cursor,previous_u,previous_gain,noise,c):
    """Independent explicit NumPy schema and routing reconstruction."""
    x,goal,o,previous_u,previous_gain,noise=map(np.asarray,(x,goal,o,previous_u,previous_gain,noise));r=c.robot
    target,_,remaining,_=numpy_flight_target(x,goal,numpy_obstacles(o,mask,noise,True),mask,points,rm,cursor,c)
    context=np.r_[(goal-x[:3])/5,(target-x[:3])/5,cursor/10,remaining/10,(x[2]-c.altitude_min)/(c.altitude_max-c.altitude_min),
        x[3:6]/np.array([c.tilt_limit,c.tilt_limit,c.yaw_limit]),x[6:9]/c.velocity_limit,x[9:12]/c.rate_limit,
        (2*previous_u-r.input_max-r.input_min)/(r.input_max-r.input_min),np.log(previous_gain),noise/BASE_NOISE,
        r.mass/3,r.inertia_x/.5,r.inertia_y/.5,r.inertia_z/.5,r.arm/.3,r.yaw_coefficient/.1,r.gravity/9.8,r.dt/.05]
    result=np.zeros((len(o)+2,FEATURES));node_mask=np.r_[True,True,mask]
    for i in np.flatnonzero(node_mask):
        kind=0 if i==0 else 2 if i==1 else 1
        pos=(x[:2] if i==0 else goal[:2] if i==1 else o[i-2,:2])-x[:2]
        vel=(x[6:8] if i==0 else np.zeros(2) if i==1 else o[i-2,3:5])-x[6:8]
        radius=r.radius if i==0 else 0. if i==1 else o[i-2,2]
        result[i]=np.r_[np.eye(3)[kind],pos/5,vel/c.velocity_limit,radius,(np.sqrt(np.dot(pos,pos)+1e-12)-r.radius-radius)/3,context]
    return result,node_mask

def history_graph(*args,config,nominal_bias=None):
    """Versioned observer features, preserving the legacy 50-column graph."""
    features,mask=graph(*args,config)
    if config.nominal_bias_observer=='none':
        if nominal_bias is not None:raise ValueError('Observer history supplied to legacy graph')
        return features,mask
    if nominal_bias is None:raise ValueError('Causal observer history required')
    bias=nominal_bias[jnp.asarray(BIAS_INDICES)]/jnp.asarray(BIAS_SCALE,features.dtype)
    result=jnp.concatenate((features,jnp.broadcast_to(bias,(len(mask),8))),axis=-1)
    return jnp.where(mask[:,None],result,0.),mask

def numpy_history_graph(*args,config,nominal_bias=None):
    features,mask=numpy_graph(*args,config)
    if config.nominal_bias_observer=='none':
        if nominal_bias is not None:raise ValueError('Observer history supplied to legacy graph')
        return features,mask
    if nominal_bias is None:raise ValueError('Causal observer history required')
    bias=np.asarray(nominal_bias)[[3,4,6,7,8,9,10,11]]/np.array([.004,.004,.015,.015,.015,.008,.008,.008])
    result=np.concatenate((features,np.broadcast_to(bias,(len(mask),8))),axis=-1)
    return np.where(mask[:,None],result,0.),mask


import jax


def validate(config):
    if config.robot.inertia_x!=config.robot.inertia_y:
        raise ValueError('Quarter-turn augmentation requires equal horizontal inertias')
    if config.nominal_bias_observer!='innovation_ema_v97':
        raise ValueError('Quarter-turn features require the58-column history contract')

def rotate_xy(values,turns):
    matrices=jnp.asarray([[[1,0],[0,1]],[[0,-1],[1,0]],
                          [[-1,0],[0,-1]],[[0,1],[-1,0]]],values.dtype)
    return values@matrices[turns%4].T

def rotate_control(control,turns):
    return jnp.take(control,(jnp.arange(4)-turns)%4,axis=-1)

def rotate_features(features,turns):
    """Transform one stored66x58 observation graph; masks and targets unchanged."""
    if features.shape[-1]!=58:raise ValueError('Expected the58-column observed Quad3D graph')
    result=features
    # Relative positions/velocities, goal/route target, and current velocity.
    for index in (3,5,9,12,21,52):
        result=result.at[...,index:index+2].set(rotate_xy(features[...,index:index+2],turns))
    # Pitch/roll, rates, and matching normalized observer-bias pairs.
    for index in (18,24,50,55):
        result=result.at[...,index:index+2].set(rotate_xy(features[...,index:index+2],-turns))
    sign=jnp.where(turns%2,-1,1)
    for index in (20,26,57):result=result.at[...,index].set(sign*features[...,index])
    return result.at[...,27:31].set(rotate_control(features[...,27:31],turns))

def augment_batch(batch,seed,step):
    """Same seeded transforms for both encoders; all branches share a rotation."""
    key=jax.random.fold_in(jax.random.PRNGKey(seed),step)
    turns=jax.random.randint(key,(len(batch['features']),),0,4)
    return dict(batch,features=jax.vmap(rotate_features)(batch['features'],turns))
