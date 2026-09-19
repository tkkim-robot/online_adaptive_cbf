"""Bounded isotropic sensor errors, persistent biases and causal observations.

The simulator owns biases and physical state. Only observe() outputs and known
noise ranges go to control/features. No latent posterior claim is implied.
"""
import jax
import jax.numpy as jnp
import numpy as np
from .bicycle_control import observed32,constant

SCHEMA='bicycle_acquired_isotropic_bias_innovation_v67'
NOISE_FIELDS=('ego_position_disk','ego_heading','ego_speed','obstacle_position_disk','obstacle_velocity_disk','obstacle_radius')
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
