"""Replay the failed held QP behind an extra neural query; never search gains."""
import numpy as np
import jax
from .quad3d_observed_rollout import observed_control

_CACHE={}


def verify_held(x,goal,o,mask,gain,points,rm,cursor,noise,bias,c,trace,k):
    """Separate single-state AOT replay of the *one* previously held gain.

    This checks the controller failure signal, not a mathematical infeasibility
    theorem. Independent polynomial/physical audits still check applied control.
    No new observation, future noise, model prediction or alternate gain enters.
    """
    from .quad3d_foundation_experiment import to_device
    args=to_device(tuple(np.asarray(a) for a in (x,goal,o,mask,gain,points,rm,cursor,noise,bias)))
    key=(c,len(mask))
    if key not in _CACHE:
        fn=jax.jit(lambda x,g,o,m,a,p,rm,t,n,b:observed_control(x,g,o,m,a,p,rm,t,n,c,True,b)[:6])
        _CACHE[key]=(fn,fn.lower(*args).compile())
    fn,exe=_CACHE[key];actual=jax.device_get(exe(*args))
    for name,value in zip(('held_proposed','held_feasible','held_psi','held_domain','held_residual','held_iterations'),actual,strict=True):
        if name=='held_feasible':assert bool(trace[name][k])==bool(value)
        elif name!='held_iterations':np.testing.assert_allclose(trace[name][k],value,atol=1e-8,rtol=1e-9,equal_nan=True,err_msg=name)
    assert fn._cache_size()==0
