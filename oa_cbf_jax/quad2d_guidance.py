"""Quad2d guidance functions and shared contracts."""

from dataclasses import dataclass

import math

import jax

import jax.numpy as jnp

from .quad2d_control import FlightConfig, nominal_flight_velocity, flight_rows, reduced_flight_qp, physical_envelope_violation

from .quad2d import integrate_quad2d

from .routing import route_target_from_position, physical_route_coordinate

from .dynamics import swept_disk_clearance

def guidance_branch(x,goal,obs,mask,gains,points,rm,cursor,angle,speed_scale,config=FlightConfig(),steps=40,capture_first_control=False,clearance_uncertainty=0.,current_obstacles=None):
    c=config.robot
    def tick(carry,k):
        state,progress,valid,minimum,count=carry[:5]
        target,next_progress,_=route_target_from_position(state[:2],jnp.linalg.norm(state[3:5]),points,rm,progress)
        delta=target-state[:2];direction=delta/jnp.maximum(jnp.linalg.norm(delta),1e-8)
        rotated=jnp.array([jnp.cos(angle)*direction[0]-jnp.sin(angle)*direction[1],jnp.sin(angle)*direction[0]+jnp.cos(angle)*direction[1]])
        remaining=jnp.maximum(jnp.linalg.norm(goal-state[:2])-.08,0.)
        speed=jnp.minimum(config.cruise_speed,jnp.minimum(1.5*remaining,jnp.sqrt(1.2*remaining)))*speed_scale
        nominal=nominal_flight_velocity(state,speed*rotated,config)
        seen=obs.at[:,:2].set(obs[:,:2]+k*c.dt*obs[:,3:5])
        if current_obstacles is not None:
            seen=jnp.where(k==0,current_obstacles,seen)
        A,b,h,psi,domain=flight_rows(state,seen,mask,gains,config,clearance_uncertainty)
        qp=reduced_flight_qp(nominal,A,b,len(obs),config)
        accepted=valid&qp.feasible&(h>=-c.qp_tolerance)&(psi>=-c.qp_tolerance)&(domain>=-c.qp_tolerance)
        u=jnp.where(accepted,qp.control,jnp.zeros(2,x.dtype));y,sub=integrate_quad2d(state,u,c)
        starts=jnp.concatenate((state[None],sub[:-1]));times=k*c.dt+jnp.arange(c.integration_substeps)*c.dt/c.integration_substeps
        clear=jnp.min(jax.vmap(lambda a,b,t:swept_disk_clearance(a,b,obs,mask,c.radius,t,t+c.dt/c.integration_substeps))(starts,sub,times))
        bounds=jnp.max(jax.vmap(lambda s:physical_envelope_violation(s,config))(jnp.concatenate((state[None],sub))))
        valid=accepted&(clear>0)&(bounds<=c.qp_tolerance)
        next_carry=(jnp.where(accepted,y,state),jnp.where(accepted,next_progress,progress),valid,jnp.minimum(minimum,jnp.where(accepted,clear,jnp.inf)),count+accepted.astype(jnp.int32))
        if capture_first_control:next_carry+=(jnp.where(k==0,qp.control,carry[5]),)
        return next_carry,None
    carry=(x,cursor,jnp.asarray(True),jnp.float32(jnp.inf),jnp.int32(0))
    if capture_first_control:carry+=(jnp.zeros(2,x.dtype),)
    result,_=jax.lax.scan(tick,carry,jnp.arange(steps))
    final,last,valid,clear,count=result[:5]
    progress=physical_route_coordinate(final[:2],points,rm,last)-physical_route_coordinate(x[:2],points,rm,cursor)
    prediction=dict(valid=valid,steps=count,clearance=clear,progress=progress,final_state=final)
    if capture_first_control:prediction['first_control']=result[5]
    return prediction

@dataclass(frozen=True)
class GuidanceConfig:
    horizon: int = 40
    angles_degrees: tuple = (0.,-30.,30.,-60.,60.,-90.,90.,-120.,120.)
    speed_scales: tuple = (1.,.5,.25)
    clearance_reward: float = .10
    deviation_penalty: float = .02
    def __post_init__(self):
        if not isinstance(self.horizon,int) or isinstance(self.horizon,bool) or self.horizon<1:raise ValueError('Invalid prediction horizon')
        if not self.angles_degrees or not self.speed_scales or not all(math.isfinite(v) for v in (*self.angles_degrees,*self.speed_scales,self.clearance_reward,self.deviation_penalty)):raise ValueError('Invalid guidance bank')
        if min(self.speed_scales)<0 or max(self.speed_scales)>1 or self.clearance_reward<0 or self.deviation_penalty<0:raise ValueError('Invalid guidance score')

@dataclass(frozen=True)
class NoiseClearanceGuidanceConfig(GuidanceConfig):
    """OA design pilot: soft preference for clearance at the declared noise.

    A distinct contract forces new labels/calibration before learned use. The
    default GuidanceConfig and existing bundles retain their exact semantics.
    """
    noise_clearance_weight: float = .5
    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.noise_clearance_weight) or self.noise_clearance_weight<=0:raise ValueError('Positive finite noise-clearance weight required')

@dataclass(frozen=True)
class TerminalGuidanceConfig(NoiseClearanceGuidanceConfig):
    """Blend arclength progress into destination-distance progress near arrival.

    One nominal prediction travel distance (2m by default) sets the transition.
    The same observation-derived blend applies to every profile in a query.
    It gives returning from an overshoot a positive reward even after the route
    coordinate saturates. This changes labels and requires a new learned bundle.
    """
    terminal_transition_distance: float = 2.
    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.terminal_transition_distance) or self.terminal_transition_distance<=0:raise ValueError('Positive finite terminal transition required')

@dataclass(frozen=True)
class GuardedGuidanceConfig(TerminalGuidanceConfig):
    """OA design pilot; requires new matched labels before learned deployment."""
    clearance_guard: str = 'one_step_v1'
    hard_prediction_clearance: bool = False
    def __post_init__(self):
        super().__post_init__()
        if self.clearance_guard!='one_step_v1' or type(self.hard_prediction_clearance) is not bool:
            raise ValueError('Invalid clearance-guard contract')

@dataclass(frozen=True)
class ForecastMotionGuidanceConfig(TerminalGuidanceConfig):
    """Forecast-only motion observer; raw current QP and physical plant retained.

    Pilot requires matched history-aware labels before learned deployment.
    """
    motion_window: int = 32
    motion_contract: str = 'identified_constant_velocity_two_anchor_v1'
    def __post_init__(self):
        super().__post_init__()
        if type(self.motion_window) is not int or self.motion_window < 1:
            raise ValueError('Positive integer motion window required')
        if self.motion_contract != 'identified_constant_velocity_two_anchor_v1':
            raise ValueError('Unknown flight motion-observation contract')

@dataclass(frozen=True)
class ObservedMotionGuidanceConfig(ForecastMotionGuidanceConfig):
    """Use the same causal velocity estimate in current and predicted CBF rows.

    This is a distinct measurement contract, not a relaxation of raw-row audit
    tolerances. Geometric/ego uncertainty remains; no robustness claim follows.
    """
    motion_application: str = 'current_and_future_v1'
    def __post_init__(self):
        super().__post_init__()
        if self.motion_application != 'current_and_future_v1':
            raise ValueError('Unknown motion application')

@dataclass(frozen=True)
class InflatedGuidanceConfig(GuardedGuidanceConfig):
    """Match current/predicted CBF geometry to declared position/radius error.

    The first-command guard still accounts for velocity/pitch/rate uncertainty.
    Constant inflation through the nominal horizon is not a feedback tube.
    """
    cbf_clearance_inflation: str = 'current_position_radius_v1'
    def __post_init__(self):
        super().__post_init__()
        if self.cbf_clearance_inflation!='current_position_radius_v1':
            raise ValueError('Invalid CBF clearance-inflation contract')

def noise_clearance_target(noise,config=FlightConfig()):
    # Triangle inequality for CURRENT disk clearance: independent per-axis ego
    # and obstacle position bounds plus radius error, including15% innovations.
    # This is a desired planning clearance, not a future reachable-tube bound.
    return config.robot.clearance_buffer+1.15*(jnp.sqrt(jnp.asarray(2.,noise.dtype))*(noise[0]+noise[4])+noise[6])

def predictive_flight_control(x,goal,obs,mask,gains,points,rm,cursor,config=FlightConfig(),guidance=GuidanceConfig(),noise=None,forecast_obstacles=None):
    """Receding finite-input guidance followed by the unchanged hard CBF-QP.

    Rotation is relative to the observed route direction, not a rotation of
    gravity. Every branch uses genuine six-state flight and all moving disks.
    A prediction failure is returned separately from the instantaneous QP.
    """
    bank=jnp.asarray([(math.radians(a),s) for s in guidance.speed_scales for a in guidance.angles_degrees]+[(0.,0.)],x.dtype)
    guarded=isinstance(guidance,GuardedGuidanceConfig)
    inflation=0.
    if isinstance(guidance,InflatedGuidanceConfig):
        from .quad2d_guidance import current_clearance_uncertainty
        if noise is None:raise ValueError('Inflated guidance requires declared noise')
        inflation=current_clearance_uncertainty(noise)
    tracked=isinstance(guidance,ForecastMotionGuidanceConfig)
    if tracked != (forecast_obstacles is not None):
        raise ValueError('Forecast motion requires its explicit history-aware contract and observation')
    prediction_obs=forecast_obstacles if tracked else obs
    current_obs=forecast_obstacles if isinstance(guidance,ObservedMotionGuidanceConfig) else obs
    prediction=jax.vmap(lambda a:guidance_branch(x,goal,prediction_obs,mask,gains,points,rm,cursor,a[0],a[1],config,guidance.horizon,guarded,inflation,current_obs if tracked else None))(bank)
    progress=prediction['progress']
    if isinstance(guidance,TerminalGuidanceConfig):
        _,_,current_remaining=route_target_from_position(x[:2],jnp.linalg.norm(x[3:5]),points,rm,cursor)
        terminal_blend=jnp.clip(1-current_remaining/guidance.terminal_transition_distance,0.,1.)
        goal_progress=jnp.linalg.norm(x[:2]-goal)-jnp.linalg.norm(prediction['final_state'][:,:2]-goal,axis=-1)
        progress=(1-terminal_blend)*progress+terminal_blend*goal_progress
    score=progress/(guidance.horizon*config.robot.dt*config.cruise_speed)
    score+=guidance.clearance_reward*jnp.minimum(prediction['clearance'],.5)/.5
    score-=guidance.deviation_penalty*(jnp.abs(bank[:,0])/jnp.pi+1-bank[:,1])
    if isinstance(guidance,NoiseClearanceGuidanceConfig):
        if noise is None:raise ValueError('Noise-aware guidance requires the declared observation ranges')
        target_clearance=noise_clearance_target(noise,config)
        deficit=jnp.maximum(target_clearance-prediction['clearance'],0.)/target_clearance
        # Exact zero-noise compatibility; do not penalize FP32 boundary roundoff.
        penalty=jnp.where(jnp.any(noise>0),guidance.noise_clearance_weight*deficit**2,0.)
        score-=penalty
    valid=prediction['valid']&jnp.isfinite(score)
    if guarded:
        from .quad2d_guidance import command_clearance_bound, current_clearance_uncertainty
        guards=jax.vmap(lambda u:command_clearance_bound(x,u,obs,mask,noise,config))(prediction['first_control'])
        prediction_minimum=current_clearance_uncertainty(noise) if guidance.hard_prediction_clearance else jnp.float64(0.)
        valid &= (guards['lower_clearance']>0)&(prediction['clearance']>prediction_minimum)
    index=jnp.argmax(jnp.where(valid,score,-jnp.inf));approved=jnp.any(valid)
    angle,speed_scale=bank[index]
    target,proposed,remaining=route_target_from_position(x[:2],jnp.linalg.norm(x[3:5]),points,rm,cursor)
    delta=target-x[:2];direction=delta/jnp.maximum(jnp.linalg.norm(delta),1e-8)
    rotated=jnp.stack((jnp.cos(angle)*direction[0]-jnp.sin(angle)*direction[1],jnp.sin(angle)*direction[0]+jnp.cos(angle)*direction[1]))
    distance=jnp.maximum(jnp.linalg.norm(goal-x[:2])-.08,0.)
    speed=jnp.minimum(config.cruise_speed,jnp.minimum(1.5*distance,jnp.sqrt(1.2*distance)))*speed_scale
    reference=nominal_flight_velocity(x,speed*rotated,config)
    A,b,h,psi,domain=flight_rows(x,current_obs,mask,gains,config,inflation);qp=reduced_flight_qp(reference,A,b,len(obs),config)
    info=dict(approved=approved,index=index,valid_candidates=jnp.sum(valid).astype(jnp.int32),score=score[index],angle=angle,speed_scale=speed_scale)
    if isinstance(guidance,NoiseClearanceGuidanceConfig):
        info.update(clearance_target=target_clearance,clearance_penalty=penalty[index],predicted_clearance=prediction['clearance'][index],predicted_progress=prediction['progress'][index])
    if isinstance(guidance,TerminalGuidanceConfig):
        info.update(terminal_blend=terminal_blend,terminal_goal=goal,terminal_predicted_state=prediction['final_state'][index],
                    terminal_goal_progress=goal_progress[index],effective_progress=progress[index])
    if guarded:
        # Bind the check to the actual returned QP command as well as the bank's
        # first action; no reliance on agreement of separate compiled paths.
        current=command_clearance_bound(x,qp.control,obs,mask,noise,config)
        info.update({'guard_'+k:v for k,v in current.items()},guard_prediction_minimum=prediction_minimum)
        info['approved'] &= current['lower_clearance']>0
    if isinstance(guidance,InflatedGuidanceConfig):info['cbf_clearance_inflation']=inflation
    return (qp,h,psi,domain,proposed,remaining,target),info

def terminal_guidance_from_contract(contract):
    from .quad2d_guidance import TerminalGuidanceConfig
    fields=dict(contract)
    for key in ('angles_degrees','speed_scales'):
        fields[key]=tuple(fields[key])
    return TerminalGuidanceConfig(**fields)


NUMERICAL_ALLOWANCE = 2e-5

def current_clearance_uncertainty(noise):
    n=jnp.asarray(noise,jnp.float64)
    return 1.15*(jnp.sqrt(jnp.float64(2.))*(n[0]+n[4])+n[6])

def command_clearance_bound(x,u,obs,mask,noise,config=FlightConfig()):
    """Conservative disk separation through dt for a constant actual input.

    For time t, center error <= initial position + relative velocity error*t
    + F/m*[pitch_error*t²/2 + (|pitch_rate|+rate_error)*t³/6
    + |angular_acceleration|*t⁴/24]. Radius error is added separately.
    Each quadratic-center chord subtracts |initial acceleration|*subdt²/8.
    The fixed numerical allowance covers the tested FP32 physical replay
    tolerance; it is an engineering allowance, not an interval solver proof.
    """
    c=config.robot;x=jnp.asarray(x,jnp.float64);u=jnp.asarray(u,jnp.float64)
    obs=jnp.asarray(obs,jnp.float64);n=jnp.asarray(noise,jnp.float64)
    total=jnp.sum(jnp.abs(u))/c.mass
    acceleration=jnp.array([-jnp.sin(x[2])*jnp.sum(u)/c.mass,
                            jnp.cos(x[2])*jnp.sum(u)/c.mass-c.gravity])
    angular=jnp.abs(c.arm/c.inertia*(u[0]-u[1]))
    times=jnp.arange(c.integration_substeps+1,dtype=jnp.float64)*c.dt/c.integration_substeps
    centers=x[:2]+times[:,None]*x[3:5]+.5*times[:,None]**2*acceleration
    relative=centers[:,None,:]-obs[None,:,:2]-times[:,None,None]*obs[None,:,3:5]
    start=relative[:-1];delta=relative[1:]-start
    fraction=jnp.clip(-jnp.sum(start*delta,axis=-1)/jnp.maximum(jnp.sum(delta*delta,axis=-1),1e-24),0.,1.)
    distance=jnp.linalg.norm(start+fraction[...,None]*delta,axis=-1)-c.radius-obs[None,:,2]
    t=times[1:]
    error=current_clearance_uncertainty(n)+1.15*jnp.sqrt(jnp.float64(2.))*(n[2]+n[5])*t
    error+=total*(1.15*n[1]*t**2/2+(jnp.abs(x[5])+1.15*n[3])*t**3/6+angular*t**4/24)
    sag=jnp.linalg.norm(acceleration)*(c.dt/c.integration_substeps)**2/8
    lower=jnp.min(jnp.where(mask[None,:],distance-error[:,None]-sag-NUMERICAL_ALLOWANCE,jnp.inf))
    nominal=jnp.min(jnp.where(mask[None,:],distance,jnp.inf))
    return dict(lower_clearance=lower,nominal_clearance=nominal,uncertainty=error[-1],curvature_allowance=sag)

def numpy_command_clearance_bounds(x,u,obs,mask,noise,config=FlightConfig()):
    """Independent vectorized NumPy reconstruction for saved applied inputs."""
    import numpy as np
    c=config.robot;x=np.asarray(x,float);u=np.asarray(u,float);obs=np.asarray(obs,float);n=np.asarray(noise,float)
    times=np.linspace(0,c.dt,c.integration_substeps+1);t=times[1:]
    total=np.abs(u).sum(axis=-1)/c.mass
    acceleration=np.column_stack((-np.sin(x[:,2])*u.sum(axis=-1)/c.mass,
                                   np.cos(x[:,2])*u.sum(axis=-1)/c.mass-c.gravity))
    centers=x[:,None,:2]+times[None,:,None]*x[:,None,3:5]+.5*times[None,:,None]**2*acceleration[:,None,:]
    relative=centers[:,:,None,:]-obs[:,None,:,:2]-times[None,:,None,None]*obs[:,None,:,3:5]
    start=relative[:,:-1];delta=relative[:,1:]-start
    fraction=np.clip(-np.sum(start*delta,axis=-1)/np.maximum(np.sum(delta*delta,axis=-1),1e-24),0,1)
    distance=np.linalg.norm(start+fraction[...,None]*delta,axis=-1)-c.radius-obs[:,None,:,2]
    error=1.15*(np.sqrt(2)*(n[0]+n[4])+n[6])+1.15*np.sqrt(2)*(n[2]+n[5])*t[None,:]
    angular=np.abs(c.arm/c.inertia*(u[:,0]-u[:,1]))
    error=error+total[:,None]*(1.15*n[1]*t[None,:]**2/2+(np.abs(x[:,5])+1.15*n[3])[:,None]*t[None,:]**3/6+angular[:,None]*t[None,:]**4/24)
    sag=np.linalg.norm(acceleration,axis=-1)*(c.dt/c.integration_substeps)**2/8
    return dict(lower_clearance=np.min(np.where(mask[None,None,:],distance-error[:,:,None]-sag[:,None,None]-NUMERICAL_ALLOWANCE,np.inf),axis=(1,2)),
        nominal_clearance=np.min(np.where(mask[None,None,:],distance,np.inf),axis=(1,2)),uncertainty=error[:,-1],curvature_allowance=sag)

def check_guard_trace(data,guidance,config=FlightConfig(),noise=None,mask=None,gains=None):
    import numpy as np
    if guidance['clearance_guard']!='one_step_v1' or type(guidance['hard_prediction_clearance']) is not bool:
        raise ValueError('Unknown clearance-guard contract')
    active=np.asarray(data['active'],bool);noise=np.asarray(data.get('noise') if noise is None else noise,float)
    mask=np.asarray(data.get('obstacle_mask') if mask is None else mask,bool)
    if mask.shape!=(data['observed_obstacles'].shape[1],):raise ValueError('Missing observed obstacle mask for guard audit')
    expected=numpy_command_clearance_bounds(data['observed_state'][active],data['control'][active],
        data['observed_obstacles'][active],mask,noise,config)
    for key,value in expected.items():
        np.testing.assert_allclose(data['guidance_guard_'+key][active],value,atol=2e-9,rtol=2e-9)
    if not np.all(expected['lower_clearance']>0):raise ValueError('Applied command failed observation clearance bound')
    floor=1.15*(np.sqrt(2)*(noise[0]+noise[4])+noise[6]) if guidance['hard_prediction_clearance'] else 0.
    np.testing.assert_allclose(data['guidance_guard_prediction_minimum'][active],floor,atol=2e-9,rtol=2e-9)
    if not np.all(data['guidance_predicted_clearance'][active]>floor):raise ValueError('Applied profile failed declared predictive clearance filter')
    if 'cbf_clearance_inflation' in guidance:
        if guidance['cbf_clearance_inflation']!='current_position_radius_v1':raise ValueError('Unknown inflated CBF contract')
        inflation=1.15*(np.sqrt(2)*(noise[0]+noise[4])+noise[6])
        np.testing.assert_allclose(data['guidance_cbf_clearance_inflation'][active],inflation,atol=2e-9,rtol=2e-9)
        x=np.asarray(data['observed_state'][active],float);obs=np.asarray(data['observed_obstacles'][active],float)
        u=np.asarray(data['control'][active],float);gains=np.asarray(data.get('gain') if gains is None else gains,float);c=config.robot
        if gains.shape==(2,):gains=np.broadcast_to(gains,(len(active),2))
        if gains.shape!=(len(active),2):raise ValueError('Missing applied gains for inflated CBF audit')
        gains=gains[active]
        delta=x[:,None,:2]-obs[:,:,:2];relative_velocity=x[:,None,3:5]-obs[:,:,3:5]
        h=np.sum(delta**2,axis=-1)-(c.radius+c.clearance_buffer+obs[:,:,2]+inflation)**2
        hd=2*np.sum(delta*relative_velocity,axis=-1);psi=hd+gains[:,0,None]*h
        acceleration=np.column_stack((-np.sin(x[:,2])*u.sum(axis=-1)/c.mass,np.cos(x[:,2])*u.sum(axis=-1)/c.mass-c.gravity))
        hdd=2*np.sum(relative_velocity**2,axis=-1)+2*np.sum(delta*acceleration[:,None,:],axis=-1)
        residual=hdd+gains.sum(axis=-1)[:,None]*hd+gains.prod(axis=-1)[:,None]*h-c.cbf_margin
        hmin=np.min(np.where(mask[None,:],h,np.inf),axis=-1);pmin=np.min(np.where(mask[None,:],psi,np.inf),axis=-1)
        np.testing.assert_allclose(data['h'][active],hmin,atol=3e-6,rtol=2e-6)
        np.testing.assert_allclose(data['psi1'][active],pmin,atol=3e-6,rtol=2e-6)
        if np.any(hmin<-2e-5) or np.any(pmin<-2e-5) or np.any(residual[:,mask]<-2e-5):
            raise ValueError('Applied command fails inflated current CBF or hierarchy')


import numpy as np


from . import motion_observer

def update(memory, obstacles, mask, noise, dt, window=32):
    # Flight noise indices differ from the ground-robot observer's contract.
    mapped = jnp.zeros(6, noise.dtype).at[3].set(noise[4]).at[4].set(noise[5])
    return motion_observer.update(memory, obstacles, mask, mapped, dt, window)

def numpy_replay(raw, mask, noise, dt, window=32):
    """Independent NumPy reference using only the causal raw sensor sequence."""
    raw = np.asarray(raw, dtype=np.float64)
    mask = np.asarray(mask, bool)
    current = raw[0, :, :2].copy()
    previous = current.copy()
    estimates, bounds, lags, bads = [], [], [], []
    eps = np.finfo(np.float32).eps
    for k, reading in enumerate(raw):
        phase = k % window
        if k > 0 and phase == 0:
            previous, current = current.copy(), reading[:, :2].copy()
        anchor = previous if k >= window else current
        lag = window + phase if k >= window else k
        elapsed = max(lag, 1) * dt
        slope = (reading[:, :2] - anchor) / elapsed
        rounding = 16 * eps * (1 + abs(reading[:, :2]) + abs(anchor)) / elapsed
        error = .3 * noise[4] / elapsed + rounding
        sensor = np.full_like(slope, 1.15 * noise[5])
        lower = np.maximum(slope - error, reading[:, 3:5] - sensor)
        upper = np.minimum(slope + error, reading[:, 3:5] + sensor)
        bad = mask & (lag > 0) & np.any(lower > upper, axis=-1)
        use = mask & (lag > 0) & ~bad
        estimate = reading.copy()
        estimate[:, 3:5] = np.where(use[:, None], (lower + upper) / 2, reading[:, 3:5])
        estimate[~mask] = 0
        bound = np.where(use[:, None], (upper - lower) / 2, sensor)
        bound[~mask] = 0
        estimates.append(estimate); bounds.append(bound); lags.append(lag); bads.append(bad)
    return dict(obstacles=np.asarray(estimates), bound=np.asarray(bounds),
                lag=np.asarray(lags), inconsistent=np.asarray(bads))

def check_forecast_trace(data, guidance, dt=.05):
    """Reconstruct every recorded forecast; truth is used only for audit metrics."""
    reference = numpy_replay(data['observed_obstacles'], data['obstacle_mask'],
                             data['noise'], dt, guidance['motion_window'])
    for field, key in [('forecast_obstacles', 'obstacles'),
                       ('forecast_velocity_bound', 'bound')]:
        np.testing.assert_allclose(data[field], reference[key], rtol=1e-6, atol=2e-6)
    for field, key in [('forecast_lag', 'lag'), ('forecast_inconsistent', 'inconsistent')]:
        np.testing.assert_array_equal(data[field], reference[key])
    mask = data['obstacle_mask']; truth = data['true_obstacles'][mask, 3:5]
    residual = abs(data['forecast_obstacles'][:, mask, 3:5] - truth)
    excess = float(np.max(residual - data['forecast_velocity_bound'][:, mask], initial=0.))
    if excess > 3e-6:
        raise ValueError('Declared constant-velocity forecast interval missed physical truth')
    return dict(forecast_audit_passed=True, forecast_bound_excess=excess,
                forecast_inconsistent_tracks=int(data['forecast_inconsistent'].sum()))
