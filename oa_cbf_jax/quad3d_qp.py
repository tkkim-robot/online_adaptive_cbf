"""Quad3d qp functions and shared contracts."""

import jax

import jax.numpy as jnp

from qpax.explicit import pdip as kernels

def kkt_residual(reference, a, b, x, s, z):
    stationarity=x-reference+a.T@z
    primal=a@x+s-b
    return jnp.max(jnp.abs(jnp.concatenate((stationarity,primal,s*z))))

def solve_actual_iterate(reference, a, b, tolerance=2e-7, max_iterations=60):
    if reference.dtype!=jnp.float64:
        raise ValueError('Quad3D numerical driver requires explicit FP64')
    eye=jnp.eye(4,dtype=reference.dtype);empty_a=jnp.zeros((0,4),reference.dtype);empty_b=jnp.zeros(0,reference.dtype)
    data=kernels.QPData(eye,-reference,empty_a,empty_b,a,b)
    initial=kernels.initialize(data)
    floor=jnp.sqrt(jnp.finfo(reference.dtype).eps)
    def finite(*arrays):return jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in arrays]))
    def condition(carry):
        x,s,z,converged,bad,count=carry
        return (count<max_iterations)&~converged&~bad
    def step(carry):
        x,s,z,converged,bad,count=carry
        residual=kkt_residual(reference,a,b,x,s,z)
        valid=finite(x,s,z)&jnp.all(s>=0)&jnp.all(z>=0)&(residual<tolerance)
        converged=converged|valid
        # Preserve raw s,z for the stopping test above. Floors are only local
        # inputs to a proposed Newton step and are never applied to the plant.
        sf=jnp.maximum(s,floor);zf=jnp.maximum(z,floor)
        dual=x-reference+a.T@zf;primal=a@x+sf-b;complement=sf*zf
        factor=kernels.factorize_kkt(eye,a,empty_a,sf,zf)
        dx,ds,dz,_=kernels.solve_kkt_rhs(a,empty_a,sf,zf,*factor,-dual,-complement,-primal,empty_b)
        sigma,mu=kernels.centering_params(sf,zf,ds,dz)
        corrected=complement+ds*dz-sigma*mu
        dx,ds,dz,_=kernels.solve_kkt_rhs(a,empty_a,sf,zf,*factor,-dual,-corrected,-primal,empty_b)
        alpha=.99*jnp.minimum(kernels.ort_linesearch(sf,ds),kernels.ort_linesearch(zf,dz))
        next_x=x+alpha*dx;next_s=sf+alpha*ds;next_z=zf+alpha*dz
        take=~converged&~bad
        next_bad=bad|(take&~finite(next_x,next_s,next_z))
        return (jnp.where(take,next_x,x),jnp.where(take,next_s,s),jnp.where(take,next_z,z),
                converged,next_bad,count+take.astype(jnp.int32))
    x,s,z,converged,bad,count=jax.lax.while_loop(condition,step,
        (initial.x,initial.s,initial.z,jnp.bool_(False),jnp.bool_(False),jnp.int32(0)))
    # A valid last permitted Newton update may satisfy convergence immediately.
    valid=finite(x,s,z)&jnp.all(s>=0)&jnp.all(z>=0)&(kkt_residual(reference,a,b,x,s,z)<tolerance)
    return x,s,z,(converged|valid)&~bad,count


import itertools

import numpy as np


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
    from .quad3d_data import to_device
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


_FACES=[()] + [c for n in range(1,5) for c in itertools.combinations(range(8),n)]

FACE_INDICES=np.array([list(c)+[0]*(4-len(c)) for c in _FACES],np.int32)

FACE_MASK=np.array([[i<len(c) for i in range(4)] for c in _FACES])

def polish(reference,a,b,multipliers):
    """Return a candidate plus a fresh certificate; caller keeps old on failure."""
    _,top=jax.lax.top_k(multipliers,8)
    mask=jnp.asarray(FACE_MASK);idx=top[jnp.asarray(FACE_INDICES)]
    rows=jnp.where(mask[...,None],a[idx],0.)
    rhs=jnp.where(mask,b[idx],0.)
    gram=rows@jnp.swapaxes(rows,-1,-2)+jnp.eye(4,dtype=a.dtype)[None]*~mask[...,None]
    dual=jnp.linalg.solve(gram,(rows@reference-rhs)[...,None])[...,0]
    nonnegative=jnp.all(dual>=-1e-12,axis=-1)
    dual=jnp.maximum(dual,0.)
    atz=jnp.einsum('fij,fi->fj',rows,dual)
    candidate=reference-atz
    face_error=jnp.max(jnp.abs(jnp.einsum('fij,fj->fi',rows,candidate)-rhs),axis=-1)
    valid=nonnegative&jnp.all(jnp.isfinite(candidate),axis=-1)&(face_error<=1e-10)
    lower_bound=jnp.sum(dual*(rows@reference-rhs),axis=-1)-.5*jnp.sum(atz**2,axis=-1)
    best=jnp.max(jnp.where(valid,lower_bound,-jnp.inf))
    # Keep the arg-reduction integer: JAX's mixed float64/index argmax identity
    # otherwise becomes float32 when only explicit x64 dtypes are enabled.
    index=jnp.min(jnp.where(valid&(lower_bound==best),jnp.arange(len(_FACES)),len(_FACES)-1))
    u=candidate[index];z=dual[index];selected=rows[index];selected_b=rhs[index]
    stationarity=u-reference+selected.T@z
    complement=z*(selected@u-selected_b)
    certificate=(valid[index]&jnp.all(jnp.isfinite(u))&(jnp.max(a@u-b)<=1e-9)
        &(jnp.max(jnp.abs(stationarity))<=1e-9)&(jnp.max(jnp.abs(complement))<=1e-9))
    return u,certificate
