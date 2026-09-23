"""Physical scalar gain coordinate for the shared learned candidate head.

The DPCBF right-hand side is drift + alpha*h. Exposing alpha alongside
log(alpha) and log(alpha)**2 is an inductive bias, not a feasibility test.
No observation, label, candidate controller solve or rollout enters here.
"""
import math
import jax.numpy as jnp


def contract():
    return dict(schema='bicycle_log_quadratic_affine_gain',gain_dimension=1,
        fields=['log(alpha/2)/log(4)','(log(alpha/2)/log(4))^2','alpha/2-1'],
        unchanged_scene_encoding=True,controller_solve=False,training_labels_used=False,
        explanation='Fixed alpha=2 reference. Last coordinate exposes the exact linear gain dependence in drift+alpha*h.')


def coordinates(gains):
    if gains.shape[-1]!=1:raise ValueError('Bicycle affine gain requires one scalar gain')
    value=(jnp.log(jnp.maximum(gains,1e-6))-math.log(2.))/math.log(4.)
    return jnp.concatenate((value,value**2,gains/2.-1.),axis=-1)
