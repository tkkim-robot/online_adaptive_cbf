"""Affine sufficient next-observation obstacle-domain constraints.

The sensor has a constant bias plus bounded independent innovations. Predicting
the current measurement with the exact held-input plant does not predict the
next measurement exactly. Bound this discrepancy and protect ALL four lower
CBF cascades at the next query. This does not assert recursive QP feasibility.
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
from .quad3d import matrices, held_quad3d_state, cylinder_hocbf


def flow_matrices(config):
    a,b=matrices(config.robot);dt=config.robot.dt
    f=sum(np.linalg.matrix_power(a,k)*dt**k/math.factorial(k) for k in range(4))
    g=sum(np.linalg.matrix_power(a,k)@b*dt**(k+1)/math.factorial(k+1) for k in range(4))
    return f,g


def discrepancy_bounds(noise,config):
    """Component bounds for next raw measurement minus propagated current one.

    delta_x=(I-F)bias + innovation_next-F innovation_now. Position/rate vector
    ball supports imply these conservative component supports. Bias cancels
    where it should; using independent full errors twice would be much looser.
    """
    f,_=flow_matrices(config);dt=config.robot.dt;dtype=noise.dtype
    xs=jnp.repeat(noise[:4],3)
    mx=np.abs(np.eye(12)-f)+.15*(np.eye(12)+np.abs(f))
    dx=jnp.asarray(mx,dtype)@xs
    op=jnp.asarray(np.asarray(dt,np.float64),dtype)*noise[5]
    op+=jnp.asarray(np.asarray(.15,np.float64),dtype)*(2*noise[4]+jnp.asarray(np.asarray(dt,np.float64),dtype)*noise[5])
    return dx,op,jnp.asarray(np.asarray(.3,np.float64),dtype)*noise[5],jnp.asarray(np.asarray(.3,np.float64),dtype)*noise[6]


def cascade_error_bound(x,obstacles,noise,gains,config):
    """Uniform bound on changes of psi0..3 for ANY box-bounded command.

    Bound endpoint r,v,a,j over the entire actuator box, then expand all scalar
    products by triangle/Cauchy-Schwarz inequalities. This makes the tightening
    independent of the QP decision while keeping the final constraints affine.
    """
    f,g=flow_matrices(config);r=config.robot;dtype=x.dtype
    center=(r.input_min+r.input_max)/2;radius=(r.input_max-r.input_min)/2
    middle=jnp.asarray(f,dtype)@x+jnp.asarray(g@np.full(4,center),dtype)
    width=jnp.asarray(np.abs(g)@np.full(4,radius),dtype)
    future=obstacles[:,:2]+jnp.asarray(np.asarray(r.dt,np.float64),dtype)*obstacles[:,3:5]
    rn=jnp.linalg.norm(middle[:2]-future,axis=-1)+jnp.linalg.norm(width[:2])
    vn=jnp.linalg.norm(middle[6:8]-obstacles[:,3:5],axis=-1)+jnp.linalg.norm(width[6:8])
    gravity=jnp.asarray(np.asarray(r.gravity,np.float64),dtype)
    an=gravity*(jnp.linalg.norm(middle[3:5])+jnp.linalg.norm(width[3:5]))
    jn=gravity*(jnp.linalg.norm(middle[9:11])+jnp.linalg.norm(width[9:11]))
    dx,op,ov,rr=discrepancy_bounds(noise,config)
    er=jnp.linalg.norm(dx[:2]+op);ev=jnp.linalg.norm(dx[6:8]+ov)
    ea=gravity*jnp.linalg.norm(dx[3:5]);ej=gravity*jnp.linalg.norm(dx[9:11])
    radius_obs=obstacles[:,2]+jnp.asarray(np.asarray(r.radius+config.clearance_buffer,np.float64),dtype)
    dh=2*rn*er+er**2+2*jnp.abs(radius_obs)*rr+rr**2
    drv=rn*ev+vn*er+er*ev
    dvv=2*vn*ev+ev**2
    dra=rn*ea+an*er+er*ea
    dva=vn*ea+an*ev+ev*ea
    drj=rn*ej+jn*er+er*ej
    k1,k2,k3=gains[:3]
    return jnp.stack((dh,2*drv+k1*dh,2*dvv+2*dra+2*(k1+k2)*drv+k1*k2*dh,
        6*dva+2*drj+2*(k1+k2+k3)*(dvv+dra)+2*(k1*k2+k1*k3+k2*k3)*drv+k1*k2*k3*dh),axis=-1)


def endpoint_cascades(x,u,obstacles,mask,gains,config):
    dt=jnp.asarray(np.asarray(config.robot.dt,np.float64),x.dtype)
    endpoint=held_quad3d_state(x,u,dt,config.robot)
    future=obstacles.at[:,:2].add(dt*obstacles[:,3:5])
    return cylinder_hocbf(endpoint,future,mask,gains,config.robot,config.clearance_buffer)[2]


def transition_rows(x,reference,obstacles,mask,gains,noise,config):
    # For a held input, each r/v/a/j control contribution is a positive scalar
    # multiple of the same horizontal snap. For positive gains each lower
    # cascade is convex quadratic in u. Its tangent is a GLOBAL lower bound.
    fun=lambda u:endpoint_cascades(x,u,obstacles,mask,gains,config)
    psi=fun(reference);jac=jax.jacfwd(fun)(reference)
    error=cascade_error_bound(x,obstacles,noise,gains,config)
    rhs=psi-jnp.einsum('nki,i->nk',jac,reference)-error
    aa=jnp.where(mask[:,None,None],-jac,0.);bb=jnp.where(mask[:,None],rhs,1.)
    return aa.reshape(-1,4),bb.ravel()


def numpy_transition_certificate(x,u,obstacles,mask,gains,noise,config):
    """Independent NumPy endpoint/cascade-error check for actual applied u."""
    from .quad3d_audit import coefficients
    r=config.robot;dt=r.dt
    # Independent component polynomials also derive the reachable endpoint box.
    zero=np.zeros(12);f=np.column_stack([np.polynomial.polynomial.polyval(dt,coefficients(e,np.zeros(4),config)) for e in np.eye(12)])
    g=np.column_stack([np.polynomial.polynomial.polyval(dt,coefficients(zero,e,config)) for e in np.eye(4)])
    xs=np.repeat(noise[:4],3);dx=(np.abs(np.eye(12)-f)+.15*(np.eye(12)+np.abs(f)))@xs
    op=dt*noise[5]+.15*(2*noise[4]+dt*noise[5]);ov=.3*noise[5];dr=.3*noise[6]
    mid=f@x+g@np.full(4,(r.input_min+r.input_max)/2);width=np.abs(g)@np.full(4,(r.input_max-r.input_min)/2)
    o=obstacles[mask].copy();o[:,:2]+=dt*o[:,3:5]
    rn=np.linalg.norm(mid[:2]-o[:,:2],axis=-1)+np.linalg.norm(width[:2]);vn=np.linalg.norm(mid[6:8]-o[:,3:5],axis=-1)+np.linalg.norm(width[6:8])
    an=r.gravity*(np.linalg.norm(mid[3:5])+np.linalg.norm(width[3:5]));jn=r.gravity*(np.linalg.norm(mid[9:11])+np.linalg.norm(width[9:11]))
    er=np.linalg.norm(dx[:2]+op);ev=np.linalg.norm(dx[6:8]+ov);ea=r.gravity*np.linalg.norm(dx[3:5]);ej=r.gravity*np.linalg.norm(dx[9:11])
    rad=o[:,2]+r.radius+config.clearance_buffer
    dh=2*rn*er+er*er+2*abs(rad)*dr+dr*dr
    drv=rn*ev+vn*er+er*ev;dvv=2*vn*ev+ev*ev;dra=rn*ea+an*er+er*ea;dva=vn*ea+an*ev+ev*ea;drj=rn*ej+jn*er+er*ej
    k1,k2,k3=gains[:3]
    bound=np.stack((dh,2*drv+k1*dh,2*dvv+2*dra+2*(k1+k2)*drv+k1*k2*dh,
        6*dva+2*drj+2*(k1+k2+k3)*(dvv+dra)+2*(k1*k2+k1*k3+k2*k3)*drv+k1*k2*k3*dh),axis=-1)
    end=np.polynomial.polynomial.polyval(dt,coefficients(x,u,config));rr=end[:2]-o[:,:2];v=end[6:8]-o[:,3:5]
    a=r.gravity*np.array([end[3],-end[4]]);j=r.gravity*np.array([end[9],-end[10]])
    h=np.sum(rr*rr,-1)-rad*rad;d1=2*np.sum(rr*v,-1);d2=2*np.sum(v*v,-1)+2*np.sum(rr*a,-1);d3=6*np.sum(v*a,-1)+2*np.sum(rr*j,-1)
    psi=np.stack((h,d1+k1*h,d2+(k1+k2)*d1+k1*k2*h,d3+(k1+k2+k3)*d2+(k1*k2+k1*k3+k2*k3)*d1+k1*k2*k3*h),axis=-1)
    return float(np.min(psi-bound,initial=np.inf)),psi,bound
