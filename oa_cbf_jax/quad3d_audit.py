"""Independent NumPy polynomial audit of full linearized quadrotor motion.

The position path is quartic under a held motor command. Cylinder clearance
extrema are obtained from polynomial stationary points, in addition to dense
samples. This numerical check is not a proof for a nonlinear physical vehicle.
"""
import numpy as np
from numpy.polynomial import Polynomial
from .quad3d_control import Quad3DControlConfig


def coefficients(x, u, config=Quad3DControlConfig()):
    """Explicit component equations, independent of the JAX matrix/flow code."""
    r=config.robot; p=np.zeros((5,12),np.float64); p[0]=x
    az=np.sum(u)/r.mass
    qdot=r.arm/r.inertia_y*(u[1]-u[3]); pdot=r.arm/r.inertia_x*(u[0]-u[2])
    rdot=r.yaw_coefficient/r.inertia_z*(u[0]-u[1]+u[2]-u[3])
    p[1]=np.r_[x[6:9],x[9:12],r.gravity*x[3],-r.gravity*x[4],az,qdot,pdot,rdot]
    p[2,:3]=[r.gravity*x[3]/2,-r.gravity*x[4]/2,az/2]
    p[2,3:6]=[qdot/2,pdot/2,rdot/2]
    p[2,6:8]=[r.gravity*x[9]/2,-r.gravity*x[10]/2]
    p[3,:2]=[r.gravity*x[9]/6,-r.gravity*x[10]/6]
    p[3,6:8]=[r.gravity*qdot/6,-r.gravity*pdot/6]
    p[4,:2]=[r.gravity*qdot/24,-r.gravity*pdot/24]
    return p


def extrema(poly, duration):
    roots=poly.deriv().trim().roots()
    # Include the real part of nearly real roots. Additional points can only
    # tighten the sampled bound; do not discard tangent/double roots by sign.
    ts=[0.,duration,*[float(z.real) for z in roots if abs(z.imag)<1e-6 and 0<z.real<duration]]
    ts.extend(np.linspace(0,duration,17))
    values=poly(np.asarray(ts))
    if not np.isfinite(values).all():raise ValueError('Nonfinite polynomial physical audit')
    return float(np.min(values)),float(np.max(values))


def audit_hold(x,u,next_state,obstacles,mask,config=Quad3DControlConfig()):
    c=config; p=coefficients(np.asarray(x),np.asarray(u),c); dt=c.robot.dt
    expected=np.polynomial.polynomial.polyval(dt,p)
    error=float(np.max(abs(expected-next_state)))
    if error>2e-10:raise AssertionError(f'Full-state held-input replay mismatch: {error}')
    input_violation=float(max(np.max(u-c.robot.input_max),np.max(c.robot.input_min-u)))
    if input_violation>c.qp_tolerance:raise AssertionError('Applied actuator bounds violated')
    clearance=float('inf')
    for o in obstacles[mask]:
        px=Polynomial(p[:,0])-Polynomial([o[0],o[3]])
        py=Polynomial(p[:,1])-Polynomial([o[1],o[4]])
        minimum,_=extrema(px*px+py*py,dt)
        clearance=min(clearance,np.sqrt(max(minimum,0.))-c.robot.radius-o[2])
    violation=-float('inf')
    for i,limit in ((3,c.tilt_limit),(4,c.tilt_limit),(5,c.yaw_limit),
                    *[(k,c.velocity_limit) for k in range(6,9)],*[(k,c.rate_limit) for k in range(9,12)]):
        lo,hi=extrema(Polynomial(p[:,i]),dt);violation=max(violation,hi-limit,-lo-limit)
    lo,hi=extrema(Polynomial(p[:,2]),dt)
    violation=max(violation,hi-c.altitude_max,c.altitude_min-lo)
    return dict(replay_error=error,minimum_clearance=clearance,envelope_violation=violation,input_violation=input_violation)


def polynomial_obstacle_values(x,u,obstacles,mask,gains,config=Quad3DControlConfig()):
    """Derivatives from actual independent polynomial coefficients at t=0."""
    p=coefficients(x,u,config); residual=[]; cascade=[]
    for o in obstacles[mask]:
        px=Polynomial(p[:,0])-Polynomial([o[0],o[3]])
        py=Polynomial(p[:,1])-Polynomial([o[1],o[4]])
        h=px*px+py*py-(config.robot.radius+config.clearance_buffer+o[2])**2
        values=[h.deriv(k)(0.) for k in range(5)]; ps=[values[0]]
        for gain in gains:
            values=[values[j+1]+gain*values[j] for j in range(len(values)-1)]
            ps.append(values[0])
        cascade.extend(ps[:4]);residual.append(ps[4])
    return min(cascade,default=float('inf')),min(residual,default=float('inf'))


def polynomial_envelope_values(x,u,config=Quad3DControlConfig()):
    c=config;p=coefficients(x,u,c);cascade=[];residual=[];barriers=[]
    for i,limit,order in ((3,c.tilt_limit,2),(4,c.tilt_limit,2),(5,c.yaw_limit,2),
        (6,c.velocity_limit,3),(7,c.velocity_limit,3),(8,c.velocity_limit,1),
        (9,c.rate_limit,1),(10,c.rate_limit,1),(11,c.rate_limit,1)):
        for sign in (1,-1):barriers.append((limit-sign*Polynomial(p[:,i]),order))
    barriers.extend([(c.altitude_max-Polynomial(p[:,2]),2),(Polynomial(p[:,2])-c.altitude_min,2)])
    for h,order in barriers:
        vals=[h.deriv(k)(0.) for k in range(order+1)]
        for _ in range(order):
            cascade.append(vals[0]);vals=[vals[j+1]+c.envelope_gain*vals[j] for j in range(len(vals)-1)]
        residual.append(vals[0])
    return min(cascade),min(residual)


def independent_obstacle_values(x,u,obstacles,mask,gains,config=Quad3DControlConfig()):
    """Vectorized t=0 derivatives of the independent held-motion polynomial.

    Keep polynomial_obstacle_values as the separate object-based reference.
    Only these t=0 derivatives are batched; continuous extrema/root checks in
    audit_hold and independent_held_cascade_minimum are unchanged.
    """
    p=coefficients(x,u,config);obs=np.asarray(obstacles)[mask]
    if len(obs)==0:return float('inf'),float('inf')
    d0=p[0,:2]-obs[:,:2];d1=p[1,:2]-obs[:,3:5]
    d2,d3,d4=p[2,:2],p[3,:2],p[4,:2]
    dot=lambda a,b:np.sum(a*b,axis=-1)
    radius=config.robot.radius+config.clearance_buffer+obs[:,2]
    values=np.stack((dot(d0,d0)-radius**2,2*dot(d0,d1),
        2*(dot(d1,d1)+2*dot(d0,d2)),12*(dot(d0,d3)+dot(d1,d2)),
        24*(dot(d2,d2)+2*dot(d1,d3)+2*dot(d0,d4))),axis=-1)
    cascade=float(np.min(values[:,0]))
    for stage,gain in enumerate(gains):
        values=values[:,1:]+gain*values[:,:-1]
        if stage<3:cascade=min(cascade,float(np.min(values[:,0])))
    return cascade,float(np.min(values[:,0]))


def independent_envelope_values(x,u,config=Quad3DControlConfig()):
    """Batch the independent linear barrier derivatives by relative degree."""
    c=config;p=coefficients(x,u,c)
    indices=np.repeat(np.array([3,4,5,6,7,8,9,10,11,2]),2)
    signs=np.tile(np.array([1.,-1.]),10)
    limits=np.r_[np.repeat([c.tilt_limit,c.tilt_limit,c.yaw_limit,
        c.velocity_limit,c.velocity_limit,c.velocity_limit,c.rate_limit,c.rate_limit,c.rate_limit],2),
        c.altitude_max,-c.altitude_min]
    orders=np.repeat(np.array([2,2,2,3,3,1,1,1,1,2]),2)
    values=-signs[:,None]*p[:4,indices].T*np.array([1.,1.,2.,6.])
    values[:,0]+=limits;cascade=residual=float('inf')
    for stage in range(3):
        cascade=min(cascade,float(np.min(values[orders>stage,0])))
        values=values[:,1:]+c.envelope_gain*values[:,:-1]
        residual=min(residual,float(np.min(values[orders==stage+1,0])))
    return cascade,residual


def independent_held_cascade_minimum(x,u,obstacles,mask,gains,config=Quad3DControlConfig()):
    """Extrema of the ACTUAL final cascades, not the controller's tangent rows."""
    c=config;p=coefficients(x,u,c);polys=[]
    for o in obstacles[mask]:
        px=Polynomial(p[:,0])-Polynomial([o[0],o[3]])
        py=Polynomial(p[:,1])-Polynomial([o[1],o[4]])
        h=px*px+py*py-(c.robot.radius+c.clearance_buffer+o[2])**2
        for gain in gains[:3]:h=h.deriv()+gain*h
        polys.append(h)
    for idx,limit,order in ((3,c.tilt_limit,2),(4,c.tilt_limit,2),(5,c.yaw_limit,2),
        (6,c.velocity_limit,3),(7,c.velocity_limit,3),(8,c.velocity_limit,1),
        (9,c.rate_limit,1),(10,c.rate_limit,1),(11,c.rate_limit,1)):
        for sign in (1,-1):
            h=limit-sign*Polynomial(p[:,idx])
            for _ in range(order-1):h=h.deriv()+c.envelope_gain*h
            polys.append(h)
    for h in (c.altitude_max-Polynomial(p[:,2]),Polynomial(p[:,2])-c.altitude_min):
        polys.append(h.deriv()+c.envelope_gain*h)
    return min(extrema(h,c.robot.dt)[0] for h in polys)
