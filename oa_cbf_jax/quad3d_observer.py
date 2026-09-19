"""Causal measurement-bias estimate used only in nominal flight feedback.

For y=x+b+e and exact held F,G: y_next-F y-G u=(I-F)b+e_next-F e.
Position and yaw biases are unobservable here and remain identically zero in
the estimate. Only the eight identifiable velocity/tilt/rate components are
estimated. The plant, raw CBF inputs, noise bounds and arrival check are untouched.
"""
import numpy as np
import jax.numpy as jnp
from functools import lru_cache
from .quad3d import matrices

IDENTIFIABLE=np.array([3,4,6,7,8,9,10,11])


def coefficients(robot):
    a,b=matrices(robot);dt=robot.dt;identity=np.eye(12)
    aa=a@a;aaa=aa@a
    f=identity+dt*a+dt**2/2*aa+dt**3/6*aaa
    g=(dt*identity+dt**2/2*a+dt**3/6*aa+dt**4/24*aaa)@b
    design=(identity-f)[:,IDENTIFIABLE]
    assert np.linalg.matrix_rank(design)==len(IDENTIFIABLE)
    return f,g,np.linalg.pinv(design)


def update_bias(estimate,previous,current,applied,noise,config):
    f,g,inverse=(jnp.asarray(v,current.dtype) for v in coefficients(config.robot))
    residual=current-f@previous-g@applied
    sample=jnp.zeros(12,current.dtype).at[IDENTIFIABLE].set(inverse@residual)
    alpha=jnp.asarray(np.asarray(-np.expm1(-config.robot.dt/config.observer_time_constant),np.float64),current.dtype)
    updated=estimate+alpha*(sample-estimate)
    # Project the estimated persistent bias onto its public sensor support.
    # This is estimator regularization, never physical-state clipping.
    updated=updated.at[3:5].set(jnp.clip(updated[3:5],-noise[1],noise[1]))
    updated=updated.at[6:9].set(updated[6:9]*jnp.minimum(1.,noise[2]/jnp.maximum(jnp.linalg.norm(updated[6:9]),1e-24)))
    updated=updated.at[9:12].set(jnp.clip(updated[9:12],-noise[3],noise[3]))
    return updated.at[jnp.asarray([0,1,2,5])].set(0.)


@lru_cache(maxsize=8)
def numpy_coefficients(robot):
    from scipy.linalg import expm
    a,b=matrices(robot);aug=np.zeros((16,16));aug[:12,:12]=a;aug[:12,12:]=b
    fg=expm(aug*robot.dt)
    return fg[:12,:12],fg[:12,12:]


def numpy_update_bias(estimate,previous,current,applied,noise,config):
    """Independent reconstruction using scipy matrix exponential and least squares."""
    f,g=numpy_coefficients(config.robot)
    residual=np.asarray(current)-f@previous-g@applied
    value=np.linalg.lstsq((np.eye(12)-f)[:,IDENTIFIABLE],residual,rcond=None)[0]
    sample=np.zeros(12);sample[IDENTIFIABLE]=value
    updated=np.asarray(estimate)+(-np.expm1(-config.robot.dt/config.observer_time_constant))*(sample-estimate)
    updated[3:5]=np.clip(updated[3:5],-noise[1],noise[1])
    updated[6:9]*=min(1.,noise[2]/max(np.linalg.norm(updated[6:9]),1e-24))
    updated[9:12]=np.clip(updated[9:12],-noise[3],noise[3]);updated[[0,1,2,5]]=0.
    return updated
