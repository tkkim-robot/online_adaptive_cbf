"""Exact diagonal two-input QP by clipping each active line to its feasible interval.

O(rows²) work instead of checking O(rows²) vertices against every row. All
original rows are rechecked on the stored command. This is an OA solver, not a
change to any native paper comparator's settings or implementation.
"""
import jax
import jax.numpy as jnp
from .controllers import QPResult


@jax.jit
def solve_qp2_intervals(reference, a, b, weights, tolerance=1e-5):
    invw=1/weights
    length=jnp.sqrt(jnp.sum(a*a,axis=1));nonzero=length>0
    scale=jnp.where(nonzero,length,1.)
    rows=a/scale[:,None];rhs=b/scale
    denominator=jnp.sum(rows*rows*invw,axis=1)
    residual=jnp.matmul(rows,reference,precision='highest')-rhs
    origins=reference-(residual/jnp.maximum(denominator,1e-30))[:,None]*rows*invw
    tangent=jnp.stack((-rows[:,1],rows[:,0]),axis=1)
    coefficient=jnp.matmul(tangent,rows.T,precision='highest')
    available=rhs[None]-jnp.matmul(origins,rows.T,precision='highest')
    parallel=jnp.abs(coefficient)<=1e-12
    bound=available/jnp.where(parallel,1.,coefficient)
    low=jnp.max(jnp.where(coefficient < -1e-12,bound,-jnp.inf),axis=1)
    high=jnp.min(jnp.where(coefficient > 1e-12,bound,jnp.inf),axis=1)
    compatible=jnp.all(~parallel | (available>=-1e-12*(1+jnp.abs(rhs[None]))),axis=1)
    parameter=jnp.maximum(low,jnp.minimum(jnp.zeros_like(high),high))
    face=origins+parameter[:,None]*tangent
    candidates=jnp.concatenate((reference[None],face))
    geometry=jnp.concatenate((jnp.ones(1,bool),nonzero&compatible&(low<=high+1e-12)))
    violation=jnp.max(jnp.matmul(candidates,a.T,precision='highest')-b[None],axis=1)
    valid=geometry&jnp.all(jnp.isfinite(candidates),axis=1)&(violation<=tolerance)
    cost=.5*jnp.sum(weights*(candidates-reference)**2,axis=1)
    candidate_cost=jnp.where(valid,cost,jnp.inf)
    index=jnp.argmax(candidate_cost==jnp.min(candidate_cost));feasible=jnp.any(valid)
    return QPResult(jnp.where(feasible,candidates[index],jnp.full_like(reference,jnp.nan)),feasible,
        jnp.where(feasible,violation[index],jnp.inf),jnp.where(feasible,cost[index],jnp.inf))
