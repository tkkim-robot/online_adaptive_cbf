"""Shared bicycle guidance implementation."""

from dataclasses import dataclass

import jax

import jax.numpy as jnp

import numpy as np

from .bicycle import integrate_bicycle, steering_to_slip, bicycle_state_violation

from .bicycle_control import BicycleControlConfig, bicycle_rows, nominal_bicycle, project_bicycle_reference, observed32, constant, parabolic_terms

from .controllers import QPResult

from .dynamics import swept_disk_clearance

from .routing import route_target_from_position, physical_route_coordinate, route_geometry

@dataclass(frozen=True)
class BicycleGuidanceConfig:
    horizon:int=40
    observation_margin:bool=False
    terminal_heading:bool=False
    turn_primitives:bool=False
    speed_primitives:bool=False
    recover_margin:bool=False
    recovery_preserve_preview:bool=False
    deterministic_witness:bool=False

    def __post_init__(self):
        if type(self.horizon) is not int or not 2<=self.horizon<=80:raise ValueError('Preview horizon must be2..80fixed ticks')
        if type(self.observation_margin) is not bool:raise ValueError('Observation margin must be a boolean')
        if type(self.terminal_heading) is not bool:raise ValueError('Terminal heading must be a boolean')
        if type(self.turn_primitives) is not bool:raise ValueError('Turn primitives must be a boolean')
        if type(self.speed_primitives) is not bool:raise ValueError('Speed primitives must be a boolean')
        if type(self.recover_margin) is not bool or (self.recover_margin and not self.observation_margin):
            raise ValueError('Margin recovery requires explicit observation-margin guidance')
        if type(self.recovery_preserve_preview) is not bool or (self.recovery_preserve_preview and not self.recover_margin):
            raise ValueError('Preview preservation requires explicit margin recovery')
        if type(self.deterministic_witness) is not bool or (self.deterministic_witness and not self.recovery_preserve_preview):
            raise ValueError('Deterministic witness requires preview-preserving recovery')

def guidance_from_controller(controller):
    """Require explicit label semantics; legacy missing margin means False."""
    if not isinstance(controller,dict) or not isinstance(controller.get('predictive_guidance'),dict):
        raise ValueError('Missing bicycle predictive guidance contract')
    value=controller['predictive_guidance']
    if 'horizon' not in value or set(value)-{'horizon','observation_margin','terminal_heading','turn_primitives','speed_primitives','recover_margin','recovery_preserve_preview','deterministic_witness'}:
        raise ValueError('Unsupported bicycle predictive guidance contract')
    return BicycleGuidanceConfig(**value)

def margin_recovery_index(original, lower, noisy, eligible=None):
    """Use only present-observation bounds at the externally supplied gain.

    A negative bound is not a certificate. If every candidate lacks a witness,
    try the largest finite bound instead of reverting to progress alone.
    Preserve the original index for zero noise, witnesses, unsupported domains
    and ties. No alternative gain or future measurement enters this rule.
    """
    # Explicit FP64 with default scalar types32 needs a boolean argmax: JAX's
    # nested floating argmax otherwise creates an FP32 reduction initializer.
    supported=lower if eligible is None else jnp.where(eligible,lower,-jnp.inf)
    best=jnp.argmax(supported==jnp.max(supported))
    recover=(noisy & (lower[original]<0) & ~jnp.any(lower>=0)
             & jnp.isfinite(supported[best])
             & (lower[best]>lower[original]+constant(1e-10,lower.dtype)))
    return jnp.where(recover,best,original),recover

def deterministic_witness_index(original, lower, noisy, eligible):
    """At zero declared noise, retain a nonnegative bound of equal preview length.

    Only existing nominal profiles at the supplied gain are considered. A
    negative or missing bound is never accepted as a witness; unchanged/noisy
    cases retain the original action. This is not recursive feasibility.
    """
    supported=jnp.where(eligible & jnp.isfinite(lower) & (lower>=0),lower,-jnp.inf)
    best=jnp.argmax(supported==jnp.max(supported))
    changed=(~noisy & (lower[original]<0) & jnp.isfinite(supported[best]))
    return jnp.where(changed,best,original),changed

def profile_bank(speed_primitives=False):
    # Candidate0 is the complete unchanged route nominal. Remaining candidates
    # include both turn directions, including the wrap boundary behind the ego.
    angles=np.deg2rad(np.array([0,-30,30,-60,60,-90,90,-120,120,-180,180],np.float32))
    fractions=[1.,.6,.3]+([1.5,2.] if speed_primitives else [])
    return (jnp.asarray(np.r_[0.,np.tile(angles,len(fractions))].astype(np.float32)),
        jnp.asarray(np.r_[1.,np.repeat(fractions,11)].astype(np.float32)))

def profile_reference(state,goal,target,offset,fraction,is_original,config):
    delta=target-state[:2];bearing=jnp.arctan2(delta[1],delta[0])+offset-state[2]
    error=jnp.arctan2(jnp.sin(bearing),jnp.cos(bearing))
    steering=jnp.clip(config.steering_feedback*error,-config.robot.steering_max,config.robot.steering_max)
    distance=jnp.maximum(jnp.linalg.norm(goal-state[:2])-.1,0.)
    base=jnp.minimum(config.cruise_speed,1.2*distance)*fraction
    speed=jnp.maximum(config.robot.speed_min,base*jnp.maximum(jnp.cos(error),.35))
    reference=jnp.stack((config.speed_feedback*(speed-state[3]),steering_to_slip(steering,config.robot)))
    return jnp.where(is_original,nominal_bicycle(state,goal,target,config),reference)

def turn_profile_bank(horizon):
    """Twelve fixed steering arcs: both turns, three speeds, two durations.

    This changes nominal inputs only. Each input is still projected through the
    same CBF-QP at the externally supplied gain. No gain is searched here.
    """
    rows=np.array([(direction,fraction,duration*horizon)
        for direction in (-1.,1.) for fraction in (1.,.6,.3) for duration in (.5,1.)],np.float32)
    return tuple(jnp.asarray(rows[:,i]) for i in range(3))

def turn_reference(state,goal,target,direction,fraction,turning,config):
    speed=jnp.maximum(config.robot.speed_min,
        jnp.minimum(config.cruise_speed,1.2*jnp.maximum(jnp.linalg.norm(goal-state[:2])-.1,0.))*fraction)
    reference=jnp.stack((config.speed_feedback*(speed-state[3]),
        direction*constant(config.robot.slip_max,state.dtype)))
    return jnp.where(turning,reference,nominal_bicycle(state,goal,target,config))

def terminal_heading_cost(state,target,config=BicycleControlConfig()):
    """Observable, continuous turn-distance surrogate for terminal ranking.

    Scale the cosine heading error by the bicycle's minimum geometric turning
    radius. This rewards progress through a turn before route arclength grows.
    It is a value heuristic, not a reachability or safety certificate; no future
    physical state or obstacle identity is used by the controller.
    """
    delta=target-state[:2]
    distance=jnp.linalg.norm(delta)
    direction=jnp.stack((jnp.cos(state[2]),jnp.sin(state[2])))
    alignment=jnp.sum(delta*direction)/jnp.maximum(distance,1e-6)
    c=config.robot
    radius=c.rear_axle_distance*np.sqrt(1+c.slip_max**2)/c.slip_max
    return jnp.where(distance>1e-6,constant(radius,state.dtype)*(1-jnp.clip(alignment,-1.,1.)),0.)

def preview_profiles(state,goal,obstacles,mask,alpha,points,route_mask,cursor,config=BicycleControlConfig(),guidance=BicycleGuidanceConfig(),record=False,speed_error=0.,noise=None):
    if guidance.observation_margin and (noise is None or noise.shape!=(6,)):
        raise ValueError('Observation-margin guidance requires six declared noise ranges')
    # Optional faster references add nominal choices, never change the supplied
    # gain, existing profile indices, actuator limits, speed rows or CBF rows.
    offsets,fractions=profile_bank(guidance.speed_primitives);legacy_size=len(offsets)
    if guidance.turn_primitives:
        directions,turn_fractions,durations=turn_profile_bank(guidance.horizon)
        offsets=jnp.concatenate((offsets,jnp.zeros_like(directions)))
        fractions=jnp.concatenate((fractions,turn_fractions))
    size=len(offsets);c=config.robot;dt=constant(c.dt,jnp.float64);radius=constant(c.radius,jnp.float64)
    original=jnp.arange(size)==0
    x=jnp.broadcast_to(state.astype(jnp.float64),(size,4));cursors=jnp.full((size,),cursor,jnp.float32)
    obs0=obstacles.astype(jnp.float64)
    initial=(x,cursors,jnp.ones(size,bool),jnp.zeros(size,jnp.int32),jnp.full(size,jnp.inf,jnp.float64),
        jnp.full(size,jnp.inf,jnp.float64),jnp.zeros(size,jnp.float32),jnp.full(size,-jnp.inf,jnp.float64))
    def tick(carry,k):
        x,cursors,alive,steps,minimum,min_h,energy,first_margin=carry
        was_alive=alive
        observed=observed32(x)
        seen=observed32(obs0.at[:,:2].set(obs0[:,:2]+k.astype(jnp.float64)*dt*obs0[:,3:5]))
        target,updated,remaining=jax.vmap(lambda s,p:route_target_from_position(s[:2],s[3],points,route_mask,p))(observed,cursors)
        # Return toward the route over the second half of the finite preview.
        decay=jnp.clip(2.-2.*k/guidance.horizon,0.,1.)
        references=jax.vmap(lambda s,t,o,f,z:profile_reference(s,goal,t,o*decay,f,z,config))(observed,target,offsets,fractions,original)
        if guidance.turn_primitives:
            arcs=jax.vmap(lambda s,t,d,f,h:turn_reference(s,goal,t,d,f,k<h,config))(
                observed[legacy_size:],target[legacy_size:],directions,turn_fractions,durations)
            references=references.at[legacy_size:].set(arcs)
        a,b,h,domain=jax.vmap(lambda s:bicycle_rows(s,seen,mask,alpha,config,speed_error))(observed)
        qp=jax.vmap(lambda r,a,b:project_bicycle_reference(r,a,b,jnp.float32,config))(references,a,b)
        hmin=jnp.min(jnp.where(mask,h,jnp.inf),axis=1);dmin=jnp.min(jnp.where(mask,domain,jnp.inf),axis=1)
        accepted=alive&qp.feasible&(hmin>=-config.qp_tolerance)&(dmin>0)
        u=jnp.where(accepted[:,None],qp.control,jnp.zeros((size,2),jnp.float32))
        y,sub=jax.vmap(lambda s,v:integrate_bicycle(s,v,c))(x,u)
        starts=jnp.concatenate((x[:,None],sub[:,:-1]),axis=1)
        times=k.astype(jnp.float64)*dt+jnp.arange(c.integration_substeps,dtype=jnp.float64)*dt/c.integration_substeps
        clearance=jax.vmap(lambda starts,ends:jnp.min(jax.vmap(lambda a,b,t:swept_disk_clearance(a,b,obs0,mask,radius,t,t+dt/c.integration_substeps))(starts,ends,times)))(starts,sub)
        violation=jnp.max(jax.vmap(jax.vmap(lambda s:bicycle_state_violation(s,c)))(sub),axis=1)
        after_seen=observed32(obs0.at[:,:2].set(obs0[:,:2]+(k.astype(jnp.float64)+1)*dt*obs0[:,3:5])).astype(jnp.float64)
        after_h,after_domain,_,_=jax.vmap(lambda s:parabolic_terms(observed32(s).astype(jnp.float64),after_seen,config))(y)
        next_h=jnp.min(jnp.where(mask,after_h,jnp.inf),axis=1);next_domain=jnp.min(jnp.where(mask,after_domain,jnp.inf),axis=1)
        if guidance.observation_margin:
            from .bicycle_observation import next_margin
            def compute_margin(_):
                bound=jax.vmap(lambda s,u,y:next_margin(s,u,observed32(y),seen,after_seen,mask,noise,config)['lower'])(observed,u,y)
                return jnp.where(accepted,bound,-jnp.inf)
            first_margin=jax.lax.cond(k==0,compute_margin,lambda _:first_margin,operand=None)
        alive=accepted&(clearance>0)&(violation<=config.qp_tolerance)&(next_h>=-config.qp_tolerance)&(next_domain>0)
        x=jnp.where(accepted[:,None],y,x);cursors=jnp.where(accepted,updated,cursors);steps+=accepted.astype(jnp.int32)
        minimum=jnp.minimum(minimum,jnp.where(accepted,clearance,jnp.inf));min_h=jnp.minimum(min_h,jnp.where(was_alive,jnp.where(accepted,jnp.minimum(hmin,next_h),hmin),jnp.inf))
        energy+=jnp.where(accepted,jnp.sum((u/constant([c.acceleration_max,c.slip_max],jnp.float32))**2,axis=1),0.)
        detail=dict(control=u,state=x,observed_state=observed,active=accepted,clearance=clearance,barrier=hmin,after_barrier=next_h)
        if not record:detail=dict(control=qp.control,feasible=qp.feasible,violation=qp.max_violation,objective=qp.objective)
        return (x,cursors,alive,steps,minimum,min_h,energy,first_margin),detail
    (final,cursors,alive,steps,minimum,min_h,energy,first_margin),history=jax.lax.scan(tick,initial,jnp.arange(guidance.horizon))
    coordinate=jax.vmap(lambda x,p:physical_route_coordinate(observed32(x)[:2],points,route_mask,p))(final,cursors)
    start_coordinate=physical_route_coordinate(state[:2],points,route_mask,cursor)
    vectors,valid,lengths,cumulative=route_geometry(points,route_mask)
    segment=jax.vmap(lambda s:jnp.argmax(valid&(cumulative[1:]>=s-1e-6)))(coordinate)
    position=points[segment]+((coordinate-cumulative[segment])/jnp.maximum(lengths[segment],1e-12))[:,None]*vectors[segment]
    cross_track=jnp.linalg.norm(final[:,:2]-position,axis=1)
    progress=coordinate-start_coordinate
    score=progress-.4*cross_track+.15*jnp.clip(minimum,0.,.5)+.05*jnp.clip(min_h,0.,.5)-.02*energy/guidance.horizon
    if guidance.terminal_heading:
        # Preview dynamics, gain, feasible set and observation witness stay the
        # same. Only the terminal value changes in this explicit label contract.
        observed_final=observed32(final)
        targets=jax.vmap(lambda x,p:route_target_from_position(x[:2],x[3],points,route_mask,p)[0])(observed_final,cursors)
        heading_cost=jax.vmap(lambda x,t:terminal_heading_cost(x,t,config))(observed_final,targets)
        score-=heading_cost
    rank=jnp.where(alive,score,-1e6+steps*100.+score)
    if guidance.observation_margin:
        # A missing witness preserves the original ranking; it does not certify
        # the chosen input. Zero-noise parents retain the legacy selection.
        witness=first_margin>=0.
        prefer=jnp.any(noise>0)&jnp.any(witness)
        rank=jnp.where(prefer&~witness,-jnp.inf,rank)
    index=jnp.argmax(rank==jnp.max(rank))
    if guidance.recover_margin:
        original_index=index
        eligible=None
        if guidance.recovery_preserve_preview:
            eligible=(steps>=steps[index]) & (~alive[index]|alive)
        index,recovered=margin_recovery_index(index,first_margin,jnp.any(noise>0),eligible)
        if guidance.deterministic_witness:
            index,witness_recovered=deterministic_witness_index(index,first_margin,jnp.any(noise>0),eligible)
            recovered=recovered|witness_recovered
    metrics=dict(selected=index,complete=alive[index],complete_profiles=jnp.sum(alive),steps=steps[index],clearance=minimum[index],
        barrier=min_h[index],route_progress=progress[index],cross_track=cross_track[index],score=score[index])
    if guidance.observation_margin:
        metrics.update(next_observation_lower=first_margin[index],margin_profiles=jnp.sum(witness),margin_preferred=prefer)
    if guidance.recover_margin:
        metrics.update(recovery_applied=recovered,recovery_original_index=original_index,
            recovery_profile_lower=first_margin,recovery_first_controls=history['control'][0])
        if guidance.recovery_preserve_preview:
            metrics.update(recovery_profile_steps=steps,recovery_profile_complete=alive)
    if guidance.terminal_heading:
        metrics['terminal_heading_cost']=heading_cost[index]
    return metrics,history,dict(final_state=final,steps=steps,complete=alive,clearance=minimum,barrier=min_h,progress=progress,score=score)

def guided_bicycle_control(state,goal,obstacles,mask,alpha,points,route_mask,cursor,config=BicycleControlConfig(),guidance=BicycleGuidanceConfig(),speed_error=0.,noise=None):
    metrics,history,_=preview_profiles(state,goal,obstacles,mask,alpha,points,route_mask,cursor,config,guidance,speed_error=speed_error,noise=noise)
    index=metrics['selected']
    # Use the projected first input that was actually previewed; do not return
    # an unprojected nominal and claim the preview covered another command.
    result=QPResult(history['control'][0,index],history['feasible'][0,index],history['violation'][0,index],history['objective'][0,index])
    _,_,h,domain=bicycle_rows(state,obstacles,mask,alpha,config,speed_error)
    target,updated,remaining=route_target_from_position(state[:2],state[3],points,route_mask,cursor)
    return result,jnp.min(jnp.where(mask,h,jnp.inf)).astype(state.dtype),jnp.min(jnp.where(mask,domain,jnp.inf)).astype(state.dtype),updated,remaining,target,metrics
