"""Affine sufficient conditions for an entire held-input CBF interval.

Under the full linearized plant, horizontal position is quartic in time. The
distance HOCBF's psi3 is a degree-eight polynomial. Its control-quadratic term
is a nonnegative scalar times ||horizontal_snap(u)||² for all t >= 0. Its tangent
in u is therefore a global lower bound at every time. Nonnegative Bernstein
coefficients of that tangent give linear, sufficient whole-interval QP rows.

Initial psi0..psi3/domain checks remain mandatory. With positive gains and an
exact constant-velocity observation model, psi3 >= 0 preserves lower stages on
the interval. This does not certify unknown sensor/plant errors, future recursive
feasibility, or nonlinear flight. Numerical tolerances remain explicit.
"""

import math

import numpy as np

import jax.numpy as jnp

from .quad3d import matrices

def power_to_bernstein(degree, duration):
    return np.array([[math.comb(i,k)/math.comb(degree,k)*duration**k if k<=i else 0.
                      for k in range(degree+1)] for i in range(degree+1)],np.float64)

def derivative_matrix(degree):
    d=np.zeros((degree+1,degree+1),np.float64)
    for k in range(degree):d[k,k+1]=k+1
    return d

def obstacle_hold_rows(x,reference,obstacles,mask,gains,config):
    r=config.robot;dtype=x.dtype
    a,b=matrices(r);snap=jnp.asarray(r.gravity*np.stack((b[9],-b[10])),dtype)
    gravity=jnp.asarray(np.asarray(r.gravity,np.float64),dtype)
    # p0..p4 coefficients of relative position at the tangent command.
    p=jnp.zeros((len(obstacles),5,2),dtype)
    p=p.at[:,0].set(x[:2]-obstacles[:,:2]).at[:,1].set(x[6:8]-obstacles[:,3:5])
    p=p.at[:,2].set(gravity*jnp.stack((x[3],-x[4]))/2)
    p=p.at[:,3].set(gravity*jnp.stack((x[9],-x[10]))/6)
    p=p.at[:,4].set((snap@reference)/24)
    hc=jnp.stack([sum(jnp.sum(p[:,j]*p[:,k-j],axis=-1)
                      for j in range(max(0,k-4),min(4,k)+1)) for k in range(9)],axis=1)
    radius=jnp.asarray(np.asarray(r.radius+config.clearance_buffer,np.float64),dtype)+obstacles[:,2]
    hc=hc.at[:,0].add(-radius**2)
    # Only coefficients 4..8 depend on the command. At coefficient8 the same
    # expression differentiates both copies of p4, giving the correct tangent.
    jac=jnp.zeros((len(obstacles),9,4),dtype)
    jac=jac.at[:,4:].set(jnp.einsum('nki,ij->nkj',p,snap)/12)
    intercept=hc-jnp.einsum('nki,i->nk',jac,reference)
    d=jnp.asarray(derivative_matrix(8),dtype);ident=jnp.eye(9,dtype=dtype)
    k1,k2,k3=gains[:3]
    operator=d@d@d+(k1+k2+k3)*(d@d)+(k1*k2+k1*k3+k2*k3)*d+k1*k2*k3*ident
    transform=jnp.asarray(power_to_bernstein(8,r.dt),dtype)@operator
    beta=jnp.einsum('ij,nj->ni',transform,intercept)
    beta_jac=jnp.einsum('ij,njk->nik',transform,jac)
    # Time-zero coefficient is the separately checked initial psi3 and has no
    # control authority; omitting it is not dropping a hazard/time interval.
    aa=-beta_jac[:,1:];bb=beta[:,1:]
    aa=jnp.where(mask[:,None,None],aa,0.);bb=jnp.where(mask[:,None],bb,1.)
    return aa.reshape(-1,4),bb.ravel()

def envelope_hold_rows(x,config):
    c=config;r=c.robot;dtype=x.dtype;a,b=matrices(r)
    # Exact affine coefficients of the full state trajectory.
    cx=jnp.asarray(np.stack([np.linalg.matrix_power(a,k)/math.factorial(k) for k in range(5)]),dtype)@x
    cu=jnp.asarray(np.stack([np.zeros_like(b)]+[np.linalg.matrix_power(a,k-1)@b/math.factorial(k) for k in range(1,5)]),dtype)
    d=jnp.asarray(derivative_matrix(4),dtype);ident=jnp.eye(5,dtype=dtype)
    bern=jnp.asarray(power_to_bernstein(4,r.dt),dtype);ops=[ident,d+c.envelope_gain*ident,(d+c.envelope_gain*ident)@(d+c.envelope_gain*ident)]
    aa=[];bb=[]
    barriers=[]
    for idx,limit,order in ((3,c.tilt_limit,2),(4,c.tilt_limit,2),(5,c.yaw_limit,2),
        (6,c.velocity_limit,3),(7,c.velocity_limit,3),(8,c.velocity_limit,1),
        (9,c.rate_limit,1),(10,c.rate_limit,1),(11,c.rate_limit,1)):
        for sign in (1,-1):barriers.append((idx,sign,limit,order))
    barriers.extend(((2,1,c.altitude_max,2),(2,-1,-c.altitude_min,2)))
    for idx,sign,limit,order in barriers:
        constant=(-sign*cx[:,idx]).at[0].add(jnp.asarray(np.asarray(limit,np.float64),dtype))
        grad=-sign*cu[:,idx,:];transform=bern@ops[order-1]
        aa.append(-(transform@grad)[1:]);bb.append((transform@constant)[1:])
    return jnp.concatenate(aa),jnp.concatenate(bb)

def hold_rows(x,reference,obstacles,mask,gains,config):
    a,b=obstacle_hold_rows(x,reference,obstacles,mask,gains,config)
    ea,eb=envelope_hold_rows(x,config)
    return jnp.concatenate((a,ea)),jnp.concatenate((b,eb))
