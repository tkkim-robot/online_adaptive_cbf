"""Guidance functions and shared contracts."""

import jax

import jax.numpy as jnp

def _preview_guidance(x, reference, obstacles, mask, config, horizon, vectorized, speed_limit=None,consider_static=False,clearance_credit=0.,return_diagnostics=False):
    offsets=jnp.linspace(-1.4,1.4,9,dtype=x.dtype)
    if config.guidance_wide_turns:
        # Retain every original target and add headings behind the robot's
        # route bearing. The same bounded acceleration/yaw recurrence applies;
        # these are candidate nominal motions, not instantaneous heading jumps.
        offsets=jnp.concatenate((jnp.asarray([-jnp.pi,-2.4,-1.9],x.dtype),offsets,
                                 jnp.asarray([1.9,2.4,jnp.pi],x.dtype)))
    speed_fractions=jnp.asarray([1.,.6,.25,0.],x.dtype)
    heading_offsets=jnp.tile(offsets,len(speed_fractions))
    fractions=jnp.repeat(speed_fractions,len(offsets))
    nominal_speed=jnp.clip(x[3]+reference[0]/2,0.,config.v_max)
    nominal_bearing=x[2]+reference[1]/2
    headings=nominal_bearing+heading_offsets
    cruise=jnp.maximum(nominal_speed,config.guidance_min_speed)
    if speed_limit is not None:
        cruise=jnp.minimum(cruise,speed_limit)
    speeds=cruise*fractions
    directions=jnp.stack((jnp.cos(nominal_bearing),jnp.sin(nominal_bearing)))
    lateral=jnp.stack((-directions[1],directions[0]))
    dt=horizon/24
    radius=obstacles[:,2]+config.radius+config.clearance_buffer
    if vectorized:
        # Exact solution of the *discrete* saturated proportional recurrence
        # used below. This removes the nested 24-step sequential loop from each
        # controller tick; it does not substitute a continuous-time trajectory.
        k=jnp.arange(24,dtype=x.dtype)[:,None]
        heading_error=jnp.arctan2(jnp.sin(headings-x[2]),jnp.cos(headings-x[2]))
        angle_errors=_discrete_errors(heading_error,config.w_max,dt,k)
        speed_errors=_discrete_errors(speeds-x[3],config.a_max,dt,k)
        angles=x[2]+heading_error-angle_errors
        velocities=speeds-speed_errors
        mid_angles=angles+.5*dt*jnp.clip(2*angle_errors,-config.w_max,config.w_max)
        mid_velocities=velocities+.5*dt*jnp.clip(2*speed_errors,-config.a_max,config.a_max)
        increments=dt*mid_velocities[...,None]*jnp.stack((jnp.cos(mid_angles),jnp.sin(mid_angles)),axis=-1)
        positions=x[:2]+jnp.cumsum(increments,axis=0)
        obs_positions=obstacles[None,:,:2]+(k+1)[:,:,None]*dt*obstacles[None,:,3:5]
        distances=jnp.linalg.norm(positions[:,:,None,:]-obs_positions[:,None,:,:],axis=-1)-radius
        clearances=jnp.where(mask,distances,jnp.inf)
        final_positions=positions[-1]
    else:
        initial=jnp.broadcast_to(x,(len(headings),4))
        def step(state,k):
            error=jnp.arctan2(jnp.sin(headings-state[:,2]),jnp.cos(headings-state[:,2]))
            turn=jnp.clip(2*error,-config.w_max,config.w_max)
            acceleration=jnp.clip(2*(speeds-state[:,3]),-config.a_max,config.a_max)
            mid_heading=state[:,2]+.5*dt*turn
            mid_speed=state[:,3]+.5*dt*acceleration
            new=state.at[:,:2].add(dt*mid_speed[:,None]*jnp.stack((jnp.cos(mid_heading),jnp.sin(mid_heading)),axis=-1))
            new=new.at[:,2].add(dt*turn).at[:,3].add(dt*acceleration)
            obs_position=obstacles[:,:2]+(k+1)*dt*obstacles[:,3:5]
            clearance=jnp.linalg.norm(new[:,:2,None]-obs_position.T[None],axis=1)-radius
            return new,jnp.where(mask,clearance,jnp.inf)
        final,clearances=jax.lax.scan(step,initial,jnp.arange(24))
        final_positions=final[:,:2]
    nominal_index=len(offsets)//2
    relevant=mask if consider_static else mask&(jnp.linalg.norm(obstacles[:,3:5],axis=-1)>1e-5)
    dynamic_clearance=jnp.min(jnp.where(relevant,clearances[:,nominal_index],jnp.inf))
    if config.guidance_turn_return:
        from .guidance import preview_turn_return
        extra_positions,extra_headings,extra_speeds,extra_offsets,_=preview_turn_return(x,nominal_bearing,cruise,config,horizon)
        obs_positions=obstacles[None,:,:2]+jnp.arange(1,25,dtype=x.dtype)[:,None,None]*dt*obstacles[None,:,3:5]
        extra_clearances=jnp.linalg.norm(extra_positions[:,:,None,:]-obs_positions[:,None,:,:],axis=-1)-radius
        clearances=jnp.concatenate((clearances,jnp.where(mask,extra_clearances,jnp.inf)),axis=1)
        final_positions=jnp.concatenate((final_positions,extra_positions[-1]),axis=0)
        headings=jnp.concatenate((headings,extra_headings))
        speeds=jnp.concatenate((speeds,extra_speeds))
        heading_offsets=jnp.concatenate((heading_offsets,extra_offsets))
    minimum=jnp.min(clearances,axis=(0,2))
    moving=mask&(jnp.linalg.norm(obstacles[:,3:5],axis=-1)>1e-5)
    # Clearance padding is only a preference in the approximate nominal preview.
    # Exact CBF constraints and physical radii are unchanged.
    endpoint_slack=jnp.min(jnp.where(mask,jnp.linalg.norm(x[:2]-obstacles[:,:2],axis=-1)-radius,jnp.inf))
    # The caller may already include an uncertainty allowance in the preview
    # radii. Credit that allowance against the preferred extra .2m gap so the
    # two preferences share a budget. Inflated radii and hard CBF rows remain
    # unchanged; this is solely nominal trajectory ranking.
    desired_slack=jnp.minimum(jnp.maximum(.2-clearance_credit,0.),jnp.maximum(endpoint_slack-.02,0.))
    # A soft variant reserves eligibility for nonnegative inflated clearance;
    # additional clearance remains in the score below. The original preference
    # could exclude every forward option while accepting a stationary preview.
    # Neither approximate threshold is a certificate for the applied CBF action.
    safe=minimum>=(0. if config.guidance_soft_clearance else desired_slack)
    displacement=final_positions-x[:2]
    progress=jnp.sum(displacement*directions,axis=-1)/horizon
    cross_track=jnp.abs(jnp.sum(displacement*lateral,axis=-1))/horizon
    score=progress-.2*cross_track-.025*heading_offsets**2+.05*jnp.minimum(minimum,.5)
    # If no preview is clear, prefer the largest approximate margin, while the
    # downstream QP retains authority to reject an infeasible actual action.
    score=jnp.where(jnp.any(safe),jnp.where(safe,score,-jnp.inf),minimum+.02*score)
    chosen=jnp.argmax(score)
    angle=jnp.arctan2(jnp.sin(headings[chosen]-x[2]),jnp.cos(headings[chosen]-x[2]))
    suggested=jnp.stack((2*(speeds[chosen]-x[3]),2*angle))
    active=jnp.any(relevant)&(dynamic_clearance<.4)&(cruise>.02)
    result=jnp.where(active,suggested,reference)
    if return_diagnostics:
        return result,dict(active=active,chosen=chosen,final_positions=final_positions,
            minimum_clearance=minimum,safe=safe,score=score,desired_slack=desired_slack,
            headings=headings,speeds=speeds,nominal_bearing=nominal_bearing)
    return result

def _discrete_errors(initial,limit,dt,k):
    magnitude=jnp.abs(initial)
    saturated_steps=jnp.ceil(jnp.maximum(magnitude-limit/2,0)/(limit*dt))
    residual=magnitude-saturated_steps*limit*dt
    decay=jnp.power(jnp.maximum(1-2*dt,0),jnp.maximum(k-saturated_steps,0))
    values=jnp.where(k<saturated_steps,magnitude-k*limit*dt,residual*decay)
    return jnp.sign(initial)*values

def preview_guidance(x, reference, obstacles, mask, config, horizon, speed_limit=None,consider_static=False,clearance_credit=0.):
    # Rounding at discrete selection boundaries can change the chosen command.
    # The implementation is therefore part of the data/controller contract;
    # historical trained/calibrated bundles retain their original scan default.
    return _preview_guidance(x,reference,obstacles,mask,config,horizon,
                             config.guidance_kernel=='vectorized',speed_limit,consider_static,clearance_credit)


from .routing import route_nominal

def preview_route(x,goal,obstacles,mask,points,route_mask,progress,config,horizon):
    dt=horizon/24
    progress=jnp.asarray(progress,x.dtype)
    radius=obstacles[:,2]+config.radius+config.clearance_buffer
    initial_gap=jnp.min(jnp.where(mask,jnp.linalg.norm(x[:2]-obstacles[:,:2],axis=-1)-radius,jnp.inf))

    def step(carry,k):
        state,cursor,minimum=carry
        reference,updated,_,_=route_nominal(state,goal,points,route_mask,cursor,config)
        # Bound the approximate action, not the resulting physical plant state.
        acceleration=jnp.clip(reference[0],jnp.maximum(-config.a_max,-state[3]/dt),
                              jnp.minimum(config.a_max,(config.v_max-state[3])/dt))
        turn=jnp.clip(reference[1],-config.w_max,config.w_max)
        angle=state[2]+.5*dt*turn;speed=state[3]+.5*dt*acceleration
        new=state.at[:2].add(dt*speed*jnp.stack((jnp.cos(angle),jnp.sin(angle))))
        new=new.at[2].add(dt*turn).at[3].add(dt*acceleration)
        # Exact relative line-segment clearance for this approximate midpoint
        # trajectory. Checking only sample endpoints misses fast crossing disks.
        start=state[:2]-obstacles[:,:2]-k*dt*obstacles[:,3:5]
        end=new[:2]-obstacles[:,:2]-(k+1)*dt*obstacles[:,3:5]
        displacement=end-start
        fraction=jnp.clip(-jnp.sum(start*displacement,axis=-1)/jnp.maximum(jnp.sum(displacement**2,axis=-1),1e-12),0.,1.)
        gap=jnp.linalg.norm(start+fraction[:,None]*displacement,axis=-1)-radius
        minimum=jnp.minimum(minimum,jnp.min(jnp.where(mask,gap,jnp.inf)))
        return (new,updated,minimum),(new,jnp.stack((acceleration,turn)))

    (final,cursor,minimum),trace=jax.lax.scan(step,(x,progress,initial_gap),jnp.arange(24))
    return final,cursor,minimum,trace

def route_preview_reference(x,goal,obstacles,mask,points,route_mask,progress,config,
                            reference,bank_reference,clearance_credit=0.):
    final,cursor,minimum,_=preview_route(x,goal,obstacles,mask,points,route_mask,progress,config,config.guidance_horizon)
    radius=obstacles[:,2]+config.radius+config.clearance_buffer
    gap=jnp.min(jnp.where(mask,jnp.linalg.norm(x[:2]-obstacles[:,:2],axis=-1)-radius,jnp.inf))
    preferred=jnp.minimum(jnp.maximum(.2-clearance_credit,0.),jnp.maximum(gap-.02,0.))
    reached=(jnp.linalg.norm(final[:2]-goal)<=config.goal_tolerance)&(jnp.abs(final[3])<=.2)
    usable=(minimum>=preferred)&((cursor>progress+.01)|reached)
    return jnp.where(usable,reference,bank_reference)


import hashlib

import json

from pathlib import Path

from .io import sha256

def inventory(source):
    source=Path(source)
    manifest=json.loads((source/'manifest.json').read_text())
    expected=len(manifest['groups'])//manifest['shard_groups'];records=[]
    for i in range(expected):
        path=source/f'shard_{i:05d}.json'
        if not path.exists():continue
        record=json.loads(path.read_text())
        if record['file']!=f'shard_{i:05d}.npz' or record['groups']!=manifest['shard_groups']:
            raise ValueError('Invalid source shard identity')
        records.append(dict(record=record,metadata_sha256=sha256(path)))
    return records

def origin_for(source,target_manifest):
    source=Path(source).resolve();original=json.loads((source/'manifest.json').read_text())
    if 'recovery_origin' in original:raise ValueError('Nested recovery needs a separately audited migration')
    old_capacity=original.get('route_capacity',32);new_capacity=target_manifest.get('route_capacity',32)
    if new_capacity<=old_capacity:raise ValueError('Recovery must increase route capacity')
    excluded={'source_fingerprint','route_capacity','recovery_origin'}
    if {k:v for k,v in original.items() if k not in excluded}!={k:v for k,v in target_manifest.items() if k not in excluded}:
        raise ValueError('Recovery may only change source implementation and route capacity')
    records=inventory(source)
    if not records:raise ValueError('No completed source shards to recover')
    digest=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    return dict(dataset=str(source),manifest_sha256=sha256(source/'manifest.json'),
        source_fingerprint=original['source_fingerprint'],inventory_sha256=digest,completed_shards=len(records),
        original_route_capacity=old_capacity,target_route_capacity=new_capacity,
        interpretation='Keep every existing parent, feature, physical/sensor/context/query/key and recorded outcome. Only append inactive route storage. Recorded trajectories retain their original source and numerical route shape; all previously uncollected parents are simulated with the new capacity. No physical/noise reset or outcome resampling.')


import math


def controller_contract(sensor_margin_scale=0.,margin_guidance=False,shared_clearance_budget=False,motion_observer_window=0,filter_obstacle_position=False):
    value=float(sensor_margin_scale)
    if not math.isfinite(value) or value<0:raise ValueError('Sensor margin scale must be finite and nonnegative')
    if not isinstance(margin_guidance,bool):raise ValueError('Margin guidance must be boolean')
    if not isinstance(shared_clearance_budget,bool):raise ValueError('Shared clearance budget must be boolean')
    if shared_clearance_budget and not margin_guidance:raise ValueError('Shared clearance budget requires margin guidance')
    if isinstance(motion_observer_window,bool) or not isinstance(motion_observer_window,int) or not 0<=motion_observer_window<=512:
        raise ValueError('Motion observer window must be an integer between zero and512')
    if not isinstance(filter_obstacle_position,bool) or (filter_obstacle_position and not motion_observer_window):
        raise ValueError('Position filtering requires a motion observer and boolean opt-in')
    result=dict(sensor_margin_scale=value,margin_guidance=margin_guidance,shared_clearance_budget=shared_clearance_budget,motion_observer_window=motion_observer_window)
    # Preserve legacy manifest identities when this optional feature is absent.
    if filter_obstacle_position:result['filter_obstacle_position']=True
    return result

def clearance_inflation(noise,scale=1.):
    return scale*1.15*(jnp.sqrt(jnp.asarray(2.,noise.dtype))*(noise[0]+noise[3])+noise[5])

def require_matching_controller(model,calibration,scale,margin_guidance=False,shared_clearance_budget=False,motion_observer_window=0,filter_obstacle_position=False):
    expected=controller_contract(scale,margin_guidance,shared_clearance_budget,motion_observer_window,filter_obstacle_position)
    if controller_contract(**model.get('controller',{}))!=expected or controller_contract(**calibration.get('controller',{}))!=expected:
        raise ValueError('Controller/model/calibration mismatch; collect and calibrate matching targets')


def preview_turn_return(x,bearing,cruise,config,horizon):
    # No straight candidate is needed: the original bank already includes it.
    offsets=jnp.asarray([-2.4,-1.4,-.7,.7,1.4,2.4],x.dtype)
    offsets=jnp.tile(offsets,4)
    speeds=cruise*jnp.repeat(jnp.asarray([1.,.6,.25,0.],x.dtype),6)
    initial_headings=bearing+offsets
    from .guidance import _discrete_errors
    dt=horizon/24
    k=jnp.arange(24,dtype=x.dtype)[:,None]
    first_error=jnp.arctan2(jnp.sin(initial_headings-x[2]),jnp.cos(initial_headings-x[2]))
    first_heading=x[2]+first_error-_discrete_errors(first_error,config.w_max,dt,jnp.minimum(k,12))
    switch_heading=x[2]+first_error-_discrete_errors(first_error,config.w_max,dt,jnp.asarray(12,x.dtype))
    second_error=jnp.arctan2(jnp.sin(bearing-switch_heading),jnp.cos(bearing-switch_heading))
    second_residual=_discrete_errors(second_error,config.w_max,dt,jnp.maximum(k-12,0))
    second_heading=switch_heading+second_error-second_residual
    headings=jnp.where(k<12,first_heading,second_heading)
    errors=jnp.where(k<12,_discrete_errors(first_error,config.w_max,dt,k),second_residual)
    turn=jnp.clip(2*errors,-config.w_max,config.w_max)
    speed_error=_discrete_errors(speeds-x[3],config.a_max,dt,k)
    velocity=speeds-speed_error
    acceleration=jnp.clip(2*speed_error,-config.a_max,config.a_max)
    midpoint_heading=headings+.5*dt*turn
    midpoint_speed=velocity+.5*dt*acceleration
    increments=dt*midpoint_speed[...,None]*jnp.stack((jnp.cos(midpoint_heading),jnp.sin(midpoint_heading)),axis=-1)
    positions=x[:2]+jnp.cumsum(increments,axis=0)
    return positions,initial_headings,speeds,offsets,jnp.stack((acceleration,turn),axis=-1)
