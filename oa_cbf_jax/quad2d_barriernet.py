"""Native Quad2D BarrierNet in JAX; pinned architecture and untuned settings.

The shared physical task changes robot radius/actuator bounds explicitly. The
native model's *inner* inference bounds remain 1..10 before wrapper clipping.
No OA guidance, obstacle observer, envelope rows or gain search is added.
"""
import jax
import jax.numpy as jnp
from .barriernet import BarrierNet, require_x64, features as unicycle_features
from .controllers import solve_qp2
from .quad2d_control import FlightConfig


def contract(config=FlightConfig()):
    c=config.robot
    if c.mass!=1. or c.gravity!=9.81:
        raise ValueError('Pinned native BarrierNet assumes mass1 and gravity9.81')
    return dict(method='native_quad2d_barriernet_v1',radius=c.radius,
        inner_bounds=[1.,10.],applied_bounds=[c.force_min,c.force_max],
        architecture='5x[5->256ReLU->64ReLU], 2sigmoid*4 gains per obstacle; mean pool, 74->64ReLU->2 residual',
        features='five nearest by clearance, raw speed norm, native absolute dummy(100,100)',
        geometry='static centers, beta1.01, actual radius, no additional OA clearance buffer',
        goal='common observed route lookahead target; ordered physical arrival remains separate',
        solver='default exact-JAX QP; native1e-6 diagonal, post-clip and actual FP32 residual check',
        deployment_slack=False,oa_guidance=False)


def features(state,goal,obstacles,mask,radius=.3):
    # The selection, dummy and bearing conventions are identical across native
    # 2D robots. Only the six-state context and speed feature differ.
    surrogate=jnp.concatenate((state[:3],jnp.linalg.norm(state[3:5],keepdims=True)))
    z,ctx=unicycle_features(surrogate,goal,obstacles,mask,radius)
    return z,jnp.concatenate((state,goal,ctx[6:]))


def nominal(state,goal,config=FlightConfig()):
    """Original Quad2D.nominal_input gains and actuator clipping."""
    c=config.robot
    acceleration=jnp.stack((3*(goal[0]-state[0])-.5*state[3],
                            .1*(goal[1]-state[1])-.5*state[4]+c.gravity))
    thrust=c.mass*jnp.linalg.norm(acceleration)
    error=-jnp.arctan2(acceleration[0],acceleration[1])-state[2]
    error=jnp.arctan2(jnp.sin(error),jnp.cos(error))
    torque=jnp.clip(.05*error-.05*state[5],-1.,1.)
    return jnp.clip(.5*jnp.stack((thrust+torque/c.radius,thrust-torque/c.radius)),c.force_min,c.force_max)


def constraints(state,obstacles,parameters,radius=.3):
    """Exact native static-center relative-degree-two flight rows."""
    require_x64()
    delta=state[:2]-obstacles[:,:2]
    barrier=jnp.sum(delta**2,axis=-1)-1.01*(obstacles[:,2]+radius)**2
    derivative=2*jnp.sum(delta*state[3:5],axis=-1)
    drift=2*jnp.sum(state[3:5]**2)-2*9.81*delta[:,1]
    coefficient=2*(-delta[:,0]*jnp.sin(state[2])+delta[:,1]*jnp.cos(state[2]))
    G=-jnp.repeat(coefficient[:,None],2,axis=1)
    rhs=drift+jnp.sum(parameters,axis=-1)*derivative+jnp.prod(parameters,axis=-1)*barrier
    return G,rhs


def soft_training_qp(u_nom,G,h):
    """Native slack objective, solved in collective/differential coordinates.

    Flight rows depend only on collective thrust. This diagonalization avoids
    subtracting nearly equal determinants in the native ill-conditioned soft
    QP; objective, slack penalty and selected-piece derivative are unchanged.
    All 32 stationary hinge patterns are scored against the actual objective.
    """
    require_x64()
    if G.shape!=(5,2) or h.shape!=(5,):raise ValueError('Native model requires five rows')
    patterns=((jnp.arange(32)[:,None]>>jnp.arange(5))&1).astype(G.dtype)
    rho=1e4+1e-6;diagonal=1+2e-6;a=G[:,0]
    total=(jnp.sum(u_nom)+2*rho*jnp.sum(patterns*a*h,axis=-1))/(diagonal+2*rho*jnp.sum(patterns*a*a,axis=-1))
    difference=(u_nom[0]-u_nom[1])/diagonal
    points=.5*jnp.column_stack((total+difference,total-difference))
    violation=jnp.maximum(total[:,None]*a-h,0.)
    cost=.5*diagonal*jnp.sum(points**2,axis=-1)-jnp.sum(points*u_nom,axis=-1)+.5*rho*jnp.sum(violation**2,axis=-1)
    return points[jnp.argmin(cost)]


def bounded_qp(u_nom,G,h,lower,upper,diagonal=1.):
    A=jnp.concatenate((G,jnp.eye(2),-jnp.eye(2)))
    b=jnp.concatenate((h,jnp.full(2,upper),jnp.full(2,-lower)))
    return solve_qp2(u_nom/diagonal,A,b,jnp.full(2,diagonal,G.dtype))


def deployment_qp(u_nom,G,h,config=FlightConfig()):
    c=config.robot;solved=bounded_qp(u_nom,G,h,1.,10.,1+1e-6)
    clipped=jnp.clip(solved.control,c.force_min,c.force_max)
    violation=jnp.maximum(jnp.max(jnp.matmul(G,clipped,precision='highest')-h),
                          jnp.max(jnp.concatenate((c.force_min-clipped,clipped-c.force_max))))
    valid=solved.feasible&jnp.all(jnp.isfinite(clipped))&(violation<=c.qp_tolerance)
    return clipped,valid,violation,solved.control,solved.feasible,solved.max_violation


def train_prediction(model,params,z,ctx,u_ref,mean,std,radius=.3):
    u_nom,p=model.apply({'params':params},(z-mean)/std,ctx[:6],ctx[6:8],u_ref)
    G,h=constraints(ctx[:6],ctx[8:].reshape(5,7),p,radius)
    return soft_training_qp(u_nom,G,h),p
