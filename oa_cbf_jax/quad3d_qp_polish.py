"""Optional OA-only equality-face polishing of a converged four-input QP.

For min .5||u-reference||² with A u <= b, each nonnegative face multiplier
supplies a dual lower bound. Enumerate faces of the eight largest interior-point
multipliers, select the largest valid bound, then independently require all
inequalities and KKT residuals. A missed/degenerate face cannot certify itself.
No barrier, input bound, or baseline solver setting is changed.
"""
import itertools
import numpy as np
import jax.numpy as jnp
import jax

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
