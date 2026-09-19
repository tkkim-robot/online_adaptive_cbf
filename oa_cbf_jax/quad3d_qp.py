"""OA-only FP64 inequality-QP driver with convergence checked on actual iterates.

The installed qpax Newton kernels floor slack/dual variables for stable
factorization. Those temporary values must not replace the original iterate in
its KKT stopping test: many inactive rows otherwise create artificial residuals.
This driver uses qpax's initialization/direction kernels without modifying the
installed package or any comparator. Mathematical QP and tolerances are retained.
"""
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
