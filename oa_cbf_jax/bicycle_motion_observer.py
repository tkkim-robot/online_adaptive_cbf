"""Causal obstacle velocity estimate from two observed positions.

Fixed persistent position bias cancels in a secant. Fuse that measurement with
the current velocity sensor using the declared marginal sensor variances. This
is a constant-velocity estimate, not a safety bound or a calibrated posterior.
"""
import jax.numpy as jnp
import numpy as np

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
