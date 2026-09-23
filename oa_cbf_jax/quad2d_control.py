"""OA planar flight nominal and exact collective/differential CBF-QP.

The complete six-state nonlinear plant remains in quad2d.py. This controller
does not stand in for any pinned baseline and has no trained gain policy yet.
Obstacle class-K parameters are queried explicitly; fixed flight envelope CBFs
preserve pitch/rate/velocity authority. All observed obstacle rows are checked.
"""
from dataclasses import dataclass,field,asdict
import math
import jax
import jax.numpy as jnp
from .quad2d import Quad2DConfig,quad2d_cbf_rows
from .controllers import solve_qp2,QPResult
from .routing import route_target_from_position


def normalize_flight_contract(value):
    """Interpret the historical missing motion flag without altering its file.

    Only this newly introduced field has a compatibility default. Every other
    configuration value remains subject to the existing strict comparisons.
    """
    result=dict(value)
    result.setdefault('stationary_obstacles',False)
    if not isinstance(result['stationary_obstacles'],bool):
        raise ValueError('Stationary-obstacle contract must be boolean')
    return result


@dataclass(frozen=True)
class FlightConfig:
    robot:Quad2DConfig=field(default_factory=lambda:Quad2DConfig(inertia=.05,force_min=2.5,force_max=5.5))
    velocity_limit:float=2.0  # componentwise world vx/vz envelope, not norm
    pitch_limit:float=.55
    pitch_rate_limit:float=2.
    cruise_speed:float=1.
    acceleration_x:float=1.2
    acceleration_z:float=.8
    attitude_kp:float=16.
    attitude_kd:float=8.
    envelope_gain:float=4.
    goal_tolerance:float=.25
    terminal_speed:float=.2
    terminal_pitch:float=.1
    terminal_pitch_rate:float=.2
    stationary_obstacles:bool=False

    def __post_init__(self):
        if not isinstance(self.stationary_obstacles,bool):raise ValueError('Stationary-obstacle contract must be boolean')
        values={k:v for k,v in asdict(self).items() if k not in ('robot','stationary_obstacles')}
        if not all(math.isfinite(v) and v>0 for v in values.values()):raise ValueError('Invalid flight envelope/nominal')
        if self.pitch_limit>=math.pi/2 or self.cruise_speed>self.velocity_limit:raise ValueError('Invalid hover-capable envelope')
        if not 2*self.robot.force_min<=self.robot.mass*self.robot.gravity<=2*self.robot.force_max:raise ValueError('Hover thrust outside actuator limits')


def flight_config_from_contract(value):
    """Load the complete saved plant/sensor contract without resetting defaults.

    Old manifests omitted only the newly added stationary-obstacle flag. They
    retain their historical moving-truth sensor model. Missing physical fields
    and unknown keys are errors, including inside the nested robot contract.
    """
    normalized=normalize_flight_contract(value)
    if set(normalized)!=set(asdict(FlightConfig())):
        raise ValueError('Incomplete or unknown flight configuration fields')
    robot=normalized['robot']
    if not isinstance(robot,dict) or set(robot)!=set(asdict(Quad2DConfig())):
        raise ValueError('Incomplete or unknown flight robot fields')
    config=FlightConfig(**dict(normalized,robot=Quad2DConfig(**robot)))
    if asdict(config)!=normalized:raise ValueError('Flight contract does not round trip')
    return config


def nominal_flight(x,goal,target,config=FlightConfig()):
    """Bounded outer velocity loop and true pitch/torque inner loop."""
    c=config.robot;delta=target-x[:2];distance=jnp.linalg.norm(delta)
    remaining=jnp.maximum(jnp.linalg.norm(goal-x[:2])-.08,0.)
    speed=jnp.minimum(config.cruise_speed,jnp.minimum(1.5*remaining,jnp.sqrt(1.2*remaining)))
    desired_velocity=speed*delta/jnp.maximum(distance,1e-8)
    return nominal_flight_velocity(x,desired_velocity,config)


def nominal_flight_velocity(x,desired_velocity,config=FlightConfig()):
    """Shared true pitch/rotor feedback for an explicitly commanded velocity."""
    c=config.robot
    acceleration=2*(desired_velocity-x[3:5])
    acceleration=jnp.clip(acceleration,jnp.array([-config.acceleration_x,-config.acceleration_z]),jnp.array([config.acceleration_x,config.acceleration_z]))
    desired_force=acceleration+jnp.array([0.,c.gravity])
    pitch=-jnp.arctan2(desired_force[0],desired_force[1])
    pitch=jnp.clip(pitch,-.8*config.pitch_limit,.8*config.pitch_limit)
    error=jnp.arctan2(jnp.sin(pitch-x[2]),jnp.cos(pitch-x[2]))
    torque=c.inertia*(config.attitude_kp*error-config.attitude_kd*x[5])
    thrust_axis=jnp.stack((-jnp.sin(x[2]),jnp.cos(x[2])))
    total=jnp.clip(c.mass*jnp.dot(desired_force,thrust_axis),2*c.force_min,2*c.force_max)
    differential_limit=jnp.minimum(total-2*c.force_min,2*c.force_max-total)
    differential=jnp.clip(torque/c.arm,-differential_limit,differential_limit)
    return .5*jnp.stack((total+differential,total-differential))


def envelope_rows(x,config=FlightConfig()):
    """Linear tilt HOCBF and rate/velocity CBFs; no physical state clipping."""
    c=config.robot;k=config.envelope_gain;sign=jnp.array([1.,-1.],x.dtype)
    torque_axis=c.arm/c.inertia*jnp.array([1.,-1.],x.dtype)
    angle_a=sign[:,None]*torque_axis
    angle_b=k*k*(config.pitch_limit-sign*x[2])-2*k*sign*x[5]
    rate_a=angle_a;rate_b=k*(config.pitch_rate_limit-sign*x[5])
    thrust_axis=jnp.stack((-jnp.sin(x[2]),jnp.cos(x[2])))/c.mass
    velocity_a=(sign[None,:]*thrust_axis[:,None]).reshape(4)
    gravity=jnp.array([0.,-c.gravity],x.dtype)
    velocity_b=((k/2)*(config.velocity_limit-sign[None,:]*x[3:5,None])-sign[None,:]*gravity[:,None]).reshape(4)
    A=jnp.concatenate((angle_a,rate_a,jnp.repeat(velocity_a[:,None],2,axis=1)))
    b=jnp.concatenate((angle_b,rate_b,velocity_b))
    domain=jnp.min(jnp.concatenate((config.pitch_limit-sign*x[2],
        -sign*x[5]+k*(config.pitch_limit-sign*x[2]),config.pitch_rate_limit-sign*x[5],
        (config.velocity_limit-sign[None,:]*x[3:5,None]).reshape(4))))
    return A,b,domain


def flight_rows(x,obstacles,mask,gains,config=FlightConfig(),clearance_uncertainty=0.):
    # Preserve precision in small geometric rows; neural/plant storage is FP32.
    dtype=x.dtype;s=x.astype(jnp.float64);o=obstacles.astype(jnp.float64);g=gains.astype(jnp.float64)
    A,b,h,psi=quad2d_cbf_rows(s,o,mask,g,config.robot,clearance_uncertainty)
    guard_A,guard_b,domain=envelope_rows(s,config)
    return jnp.concatenate((A,guard_A)),jnp.concatenate((b,guard_b)),jnp.min(jnp.where(mask,h,jnp.inf)).astype(dtype),jnp.min(jnp.where(mask,psi,jnp.inf)).astype(dtype),domain.astype(dtype)


def reduced_flight_qp(reference,A,b,obstacle_count,config=FlightConfig()):
    """Exact row reduction, not hazard truncation or an approximate safe set.

    Every obstacle/velocity row is parallel to[1,1], every attitude/rate row to
    [1,-1]. Reduce each family to its tightest interval, retain four rotor
    bounds, solve the resulting eight-row QP, then check EVERY original row.
    """
    c=config.robot;reference=reference.astype(jnp.float64)
    obs=A[:obstacle_count,0];obs_b=b[:obstacle_count]
    # Layout is obstacle rows,4input bounds,4angle/rate rows,4velocity rows.
    collective=jnp.concatenate((obs,A[-4:,0]));collective_b=jnp.concatenate((obs_b,b[-4:]))
    differential=A[-8:-4,0];differential_b=b[-8:-4]
    def interval(a,rhs,lo,hi):
        denominator=jnp.where(a!=0,a,1.);threshold=rhs/denominator
        lower=jnp.maximum(lo,jnp.max(jnp.where(a<0,threshold,-jnp.inf)))
        upper=jnp.minimum(hi,jnp.min(jnp.where(a>0,threshold,jnp.inf)))
        consistent=jnp.all((a!=0)|(rhs>=-c.qp_tolerance))
        return lower,upper,consistent
    low_t,high_t,valid_t=interval(collective,collective_b,2*c.force_min,2*c.force_max)
    span=c.force_max-c.force_min
    low_d,high_d,valid_d=interval(differential,differential_b,-span,span)
    reduced_A=jnp.array([[1.,1.],[-1.,-1.],[1.,-1.],[-1.,1.],[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],jnp.float64)
    reduced_b=jnp.stack((high_t,-low_t,high_d,-low_d,jnp.float64(c.force_max),jnp.float64(-c.force_min),jnp.float64(c.force_max),jnp.float64(-c.force_min)))
    result=solve_qp2(reference,reduced_A,reduced_b,jnp.ones(2,jnp.float64),c.qp_tolerance)
    # Check the actual stored actuator command, including every original CBF.
    control=result.control.astype(jnp.float32)
    violation=jnp.max(jnp.matmul(A,control.astype(A.dtype),precision='highest')-b)
    valid=result.feasible&valid_t&valid_d&jnp.isfinite(violation)&(violation<=c.qp_tolerance)
    return QPResult(jnp.where(valid,control,jnp.full(2,jnp.nan,jnp.float32)),valid,violation.astype(jnp.float32),result.objective.astype(jnp.float32))


def flight_control(x,goal,obstacles,mask,gains,points,route_mask,cursor,config=FlightConfig()):
    target,proposed,remaining=route_target_from_position(x[:2],jnp.linalg.norm(x[3:5]),points,route_mask,cursor)
    nominal=nominal_flight(x,goal,target,config)
    A,b,h,psi,domain=flight_rows(x,obstacles,mask,gains,config)
    result=reduced_flight_qp(nominal,A,b,obstacles.shape[0],config)
    return result,h,psi,domain,proposed,remaining,target


def flight_arrived(x,goal,config=FlightConfig()):
    return ((jnp.linalg.norm(x[:2]-goal)<=config.goal_tolerance)&(jnp.linalg.norm(x[3:5])<=config.terminal_speed)&
            (jnp.abs(x[2])<=config.terminal_pitch)&(jnp.abs(x[5])<=config.terminal_pitch_rate))


def physical_envelope_violation(x,config=FlightConfig()):
    return jnp.maximum(jnp.max(jnp.abs(x[3:5]))-config.velocity_limit,
                       jnp.maximum(jnp.abs(x[2])-config.pitch_limit,jnp.abs(x[5])-config.pitch_rate_limit))
