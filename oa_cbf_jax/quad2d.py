"""Planar quadrotor numerical foundation, separate from unicycle bundles.

State [x,z,pitch,vx,vz,pitch_rate], inputs [right_thrust,left_thrust]. Gravity
remains world vertical. Matches the pinned repository's continuous equations;
RK4 replaces Euler integration. No state clipping or hidden coordinate resets.
This module is not yet a route controller, trained policy or evaluated robot path.
"""
from dataclasses import dataclass,asdict
from functools import partial
import math
import jax
import jax.numpy as jnp
from .controllers import solve_qp2


@dataclass(frozen=True)
class Quad2DConfig:
    dt:float=.05
    mass:float=1.
    inertia:float=.01
    arm:float=.3
    radius:float=.3
    gravity:float=9.81
    force_min:float=1.
    force_max:float=10.
    clearance_buffer:float=.05
    cbf_margin:float=0.
    qp_tolerance:float=1e-5
    integration_substeps:int=8

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):raise ValueError('Nonfinite quadrotor config')
        if any(getattr(self,k)<=0 for k in ['dt','mass','inertia','arm','radius','gravity','force_max','qp_tolerance']):
            raise ValueError('Positive physical constants required')
        if self.radius<self.arm or not 0<=self.force_min<self.force_max or self.clearance_buffer<0:
            raise ValueError('Invalid rotor bounds or circumscribed collision radius')
        if isinstance(self.integration_substeps,bool) or not isinstance(self.integration_substeps,int) or self.integration_substeps<1:
            raise ValueError('Positive integer integration substeps required')


def quad2d_flow(x,u,config=Quad2DConfig()):
    force=jnp.sum(u)/config.mass
    return jnp.stack((x[3],x[4],x[5],-jnp.sin(x[2])*force,
                      jnp.cos(x[2])*force-config.gravity,config.arm/config.inertia*(u[0]-u[1])))


@partial(jax.jit,static_argnames=('config',))
def integrate_quad2d(x,u,config=Quad2DConfig()):
    dt=config.dt/config.integration_substeps
    angular_acceleration=config.arm/config.inertia*(u[0]-u[1])
    def step(state,k):
        k1=quad2d_flow(state,u,config);k2=quad2d_flow(state+dt*k1/2,u,config)
        k3=quad2d_flow(state+dt*k2/2,u,config);k4=quad2d_flow(state+dt*k3,u,config)
        y=state+dt*(k1+2*k2+2*k3+k4)/6
        # Held thrust gives constant angular acceleration. Avoid cumulative
        # roundoff in attitude without changing the physical trajectory.
        time=(k+1)*dt
        y=y.at[2].set(x[2]+time*x[5]+.5*time**2*angular_acceleration)
        y=y.at[5].set(x[5]+time*angular_acceleration)
        return y,y
    # Position does not enter these translation-invariant free-air equations.
    # Accumulate the small displacement locally, then restore world coordinates
    # once per reported substate. Repeated FP32 additions to a ~32m coordinate
    # otherwise lose >1e-5m within one tick, despite accurate velocity/attitude.
    local=x.at[:2].set(0.)
    _,states=jax.lax.scan(step,local,jnp.arange(config.integration_substeps))
    states=states.at[:,:2].add(x[:2])
    angles=states[:,2]
    angles=jnp.where(jnp.abs(angles)<=jnp.pi,angles,jnp.arctan2(jnp.sin(angles),jnp.cos(angles)))
    states=states.at[:,2].set(angles)
    return states[-1],states


def quad2d_cbf_rows(x,obstacles,mask,gains,config=Quad2DConfig(),clearance_uncertainty=0.):
    """Joint second-order disk CBF rows for known constant-velocity obstacles.

    h=||p-o||²-R²; hdd+(k1+k2)hd+k1*k2*h>=margin. The rotor sum controls
    translational acceleration; the difference first controls pitch acceleration.
    Thus a valid instantaneous row does not imply future controllability or
    recursive feasibility. Return h/psi separately from QP feasibility.
    """
    delta=x[:2]-obstacles[:,:2];velocity=x[3:5]-obstacles[:,3:5]
    radius=config.radius+config.clearance_buffer+obstacles[:,2]+clearance_uncertainty
    h=jnp.sum(delta**2,axis=-1)-radius**2;hd=2*jnp.sum(delta*velocity,axis=-1)
    thrust_axis=jnp.stack((-jnp.sin(x[2]),jnp.cos(x[2])))/config.mass
    authority=2*jnp.sum(delta*thrust_axis,axis=-1)
    rhs=2*jnp.sum(velocity**2,axis=-1)-2*config.gravity*delta[:,1]+jnp.sum(gains)*hd+jnp.prod(gains)*h-config.cbf_margin
    a=jnp.where(mask[:,None],-jnp.stack((authority,authority),axis=-1),0.)
    b=jnp.where(mask,rhs,1.)
    bounds=jnp.asarray([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],x.dtype)
    limits=jnp.asarray([config.force_max,-config.force_min,config.force_max,-config.force_min],x.dtype)
    return jnp.concatenate((a,bounds)),jnp.concatenate((b,limits)),h,hd+gains[0]*h


@partial(jax.jit,static_argnames=('config',))
def filter_quad2d(x,reference,obstacles,mask,gains,config=Quad2DConfig(),clearance_uncertainty=0.):
    a,b,h,psi=quad2d_cbf_rows(x,obstacles,mask,gains,config,clearance_uncertainty)
    result=solve_qp2(reference,a,b,jnp.ones(2,x.dtype),config.qp_tolerance)
    return result,jnp.min(jnp.where(mask,h,jnp.inf)),jnp.min(jnp.where(mask,psi,jnp.inf))
