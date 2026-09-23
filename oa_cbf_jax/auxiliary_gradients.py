"""Optional first-order interference guard for the training-only reserve head."""
import jax
import jax.numpy as jnp


def contract():
    return dict(schema='bicycle_reserve_shared_gradient_guard',
        scope='Shared encoder and candidate hidden layers only; separate reserve output keeps its own gradient.',
        rule='Add auxiliary shared gradient only when its cosine with adverse-BCE gradient exceeds1e-4; otherwise zero that contribution.',
        agreement_floor=1e-4,weight=.25,primary_gradient_changed=False,
        limitation='Local raw-gradient heuristic before the unchanged AdamW transform; no Adam-step, validation or physical-safety guarantee.')


def combine(primary, adverse, auxiliary, weight=.25):
    shared=[k for k in primary if k not in ('output','prefix_reserve')]
    dot=sum(jnp.vdot(a,b) for k in shared for a,b in zip(jax.tree.leaves(adverse[k]),jax.tree.leaves(auxiliary[k])))
    aa=sum(jnp.vdot(a,a) for k in shared for a in jax.tree.leaves(adverse[k]))
    bb=sum(jnp.vdot(a,a) for k in shared for a in jax.tree.leaves(auxiliary[k]))
    cosine=dot/jnp.sqrt(jnp.maximum(aa*bb,1e-24))
    # A near-zero event gradient provides no reliably agreeing direction.
    allow=(aa>1e-12)&(bb>1e-12)&(cosine>1e-4)
    guarded={k:jax.tree.map(lambda a:a if k=='prefix_reserve' else jnp.where(allow,a,jnp.zeros_like(a)),v) for k,v in auxiliary.items()}
    result=jax.tree.map(lambda p,a:p+weight*a,primary,guarded)
    return result,dict(auxiliary_shared_gradient_allowed=allow.astype(jnp.float32),auxiliary_adverse_cosine=cosine)
