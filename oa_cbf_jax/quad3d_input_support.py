"""Necessary actuator-box admission before learned class-K ranking.

For each linear row A_i u <= b_i, reject a gain if even the minimum over the
actual motor box violates that row. This is not joint QP feasibility, a gain
optimizer, or a rollout search. Accepted gains are still ranked only by the
unchanged neural score and must pass the original hard QP before application.
"""
import itertools
import numpy as np
import jax
import jax.numpy as jnp
from .quad3d_observed_rollout import observed_problem


def box_row_margin(a, b, lower, upper):
    minimum=jnp.sum(jnp.where(a>=0,a*lower,a*upper),axis=-1)
    return jnp.min(b-minimum,axis=-1)


def candidate_rows(x,goal,o,mask,points,rm,cursor,noise,bias,bank,c):
    return jax.vmap(lambda gain:observed_problem(x,goal,o,mask,gain,points,rm,cursor,
        noise,c,True,nominal_bias=bias)[1:3])(bank.astype(x.dtype))


def apply_support(prediction,previous,bank,margin,tolerance):
    result=dict(prediction)
    support=jnp.isfinite(margin)&(margin>=-tolerance)
    admissible=prediction['admissible']&support
    accepted=prediction['screened']&admissible
    index=jnp.argmax(jnp.where(accepted,prediction['ranking_score'],-jnp.inf))
    any_=jnp.any(accepted)
    result.update(network_gain=jnp.where(any_,bank[index].astype(previous.dtype),previous),
        selected_index=jnp.where(any_,index,-1),admissible=admissible,accepted=accepted,
        admission_fallback=jnp.any(prediction['screened'])&~any_,
        candidate_input_margin=margin,input_admissible=support)
    return result


class InputSupportSelector:
    """Wrap the frozen predictor without changing features, scores or cadence."""
    def __init__(self, original):
        if original.reference:raise ValueError('This development variant requires a frozen adaptive gate')
        self.__dict__.update(original.__dict__)
        predict=original.predict;c=self.robot;bank=jnp.asarray(self.bank,jnp.float32)
        def with_support(params,x,goal,o,mask,points,rm,cursor,previous_u,previous_gain,noise,nominal_bias=None):
            result=predict(params,x,goal,o,mask,points,rm,cursor,previous_u,previous_gain,noise,nominal_bias)
            bias=jnp.zeros(12,x.dtype) if nominal_bias is None else nominal_bias
            a,b=candidate_rows(x,goal,o,mask,points,rm,cursor,noise,bias,bank,c)
            margin=box_row_margin(a,b,c.robot.input_min,c.robot.input_max)
            return apply_support(result,previous_gain,bank,margin,c.qp_tolerance)
        self.predict=with_support


_AUDIT_CACHE={}


def audit_margin(x,goal,o,mask,points,rm,cursor,noise,bias,bank,c):
    """Separate AOT row replay; independently enumerate all16 box vertices.

    Constraint construction is shared with the controller; vertex reduction is
    independent NumPy arithmetic. Existing separate polynomial audits verify
    every actually applied constraint and physical hold.
    """
    from .quad3d_foundation_experiment import to_device
    args=to_device(tuple(np.asarray(a) for a in (x,goal,o,mask,points,rm,cursor,noise,bias)))
    key=(c,tuple(map(tuple,bank)),len(mask))
    if key not in _AUDIT_CACHE:
        fn=jax.jit(lambda *v:candidate_rows(*v,jnp.asarray(bank,jnp.float64),c))
        _AUDIT_CACHE[key]=(fn,fn.lower(*args).compile())
    fn,exe=_AUDIT_CACHE[key];a,b=map(np.asarray,exe(*args))
    vertices=np.asarray(list(itertools.product((c.robot.input_min,c.robot.input_max),repeat=4)))
    minimum=np.min(a@vertices.T,axis=-1)
    assert fn._cache_size()==0
    return np.min(b-minimum,axis=-1)
