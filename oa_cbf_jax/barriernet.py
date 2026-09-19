"""JAX port of the pinned repository's unicycle BarrierNet formulation.

Native five-nearest-obstacle architecture and static-center HOCBF are retained.
No OA ranker, gain search, observer, detour, safety slack at deployment, or
baseline tuning is introduced. Run this module in an isolated x64 JAX process.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from .controllers import solve_qp2


def require_x64():
    if not jax.config.x64_enabled:
        raise ValueError('BarrierNet port requires JAX_ENABLE_X64=true; silent downcasting is forbidden')


def features(state,goal,obstacles,mask,radius=.25):
    """Pinned build_z_ctx_for: nearest five by clearance, absolute dummy(100,100)."""
    require_x64()
    dummy=jnp.array([100.,100.,0.,0.,0.,0.,0.],jnp.float64)
    obs=jnp.pad(obstacles,((0,0),(0,7-obstacles.shape[-1])))
    # Five dummies guarantee a valid static signature even for zero obstacles.
    distance=jnp.linalg.norm(obs[:,:2]-state[:2],axis=-1)-obs[:,2]-radius
    obs=jnp.concatenate((obs,jnp.tile(dummy,(5,1))))
    priority=jnp.concatenate((jnp.where(mask,distance,jnp.inf),jnp.full(5,jnp.inf)))
    indices=jnp.argsort(priority,stable=True)[:5]
    valid=jnp.concatenate((mask,jnp.zeros(5,bool)))[indices]
    selected=jnp.where(valid[:,None],obs[indices],dummy)
    delta=selected[:,:2]-state[:2]
    bearing=jnp.arctan2(delta[:,1],delta[:,0])-state[2]
    angle=(bearing+jnp.pi)%(2*jnp.pi)-jnp.pi
    clearance=jnp.linalg.norm(delta,axis=-1)-selected[:,2]-radius
    z=jnp.column_stack((delta,angle,clearance,jnp.full(5,state[3]))).reshape(25)
    ctx=jnp.concatenate((state[:4],goal[:2],selected.reshape(35)))
    return z,ctx


def nominal(state,goal,v_max=1.):
    """Original DynamicUnicycle2D.nominal_input, default gains/d_min."""
    delta=goal-state[:2];distance=jnp.maximum(jnp.linalg.norm(delta)-.05,0.)
    angle=(jnp.arctan2(delta[1],delta[0])-state[2]+jnp.pi)%(2*jnp.pi)-jnp.pi
    speed=jnp.where(jnp.abs(angle)>jnp.pi/2,0.,jnp.minimum(distance*jnp.cos(angle),v_max))
    return jnp.stack((speed-state[3],2*angle))


class BarrierNet(nn.Module):
    @nn.compact
    def __call__(self,z_normalized,state,goal,u_ref):
        require_x64()
        def layer(x,width,name):
            # Match torch.nn.Linear's default uniform initialization law;
            # random streams are explicitly JAX seeds, not claimed bit-identical.
            limit=x.shape[-1]**-.5
            def init(key,shape,dtype=jnp.float64):
                return jax.random.uniform(key,shape,dtype,minval=-limit,maxval=limit)
            return nn.Dense(width,dtype=jnp.float64,param_dtype=jnp.float64,
                            kernel_init=init,bias_init=init,name=name)(x)
        blocks=z_normalized.reshape(z_normalized.shape[:-1]+(5,5))
        encoded=nn.relu(layer(blocks,256,'obs_fc1'))
        encoded=nn.relu(layer(encoded,64,'obs_fc2'))
        parameters=4*jax.nn.sigmoid(layer(encoded,2,'fc_p'))
        pooled=jnp.mean(encoded,axis=-2)
        hidden=jnp.concatenate((pooled,state,goal,u_ref),axis=-1)
        hidden=nn.relu(layer(hidden,64,'u_fc1'))
        return u_ref+layer(hidden,2,'u_out'),parameters


def constraints(state,obstacles,parameters,radius=.25):
    """Exact pinned static-center unicycle HOCBF; ignores obstacle velocity."""
    delta=state[:2]-obstacles[:,:2];c=jnp.cos(state[2]);s=jnp.sin(state[2]);v=state[3]
    barrier=jnp.sum(delta**2,axis=-1)-1.01*(obstacles[:,2]+radius)**2
    along=delta[:,0]*c+delta[:,1]*s
    derivative=2*v*along
    G=-jnp.column_stack((2*along,2*v*(-delta[:,0]*s+delta[:,1]*c)))
    h=2*v*v+jnp.sum(parameters,axis=-1)*derivative+jnp.prod(parameters,axis=-1)*barrier
    return G,h


def soft_training_qp(u_nom,G,h):
    """Solve the pinned training QP exactly by eliminating its five slacks.

    min .5(1+2e-6)||u||²-u_nom.u + .5(1e4+1e-6)||max(G.u-h,0)||².
    Each of32 active hinge sets has a two-dimensional stationary point. The
    global minimizer occurs in this set; evaluate the actual convex objective
    at every point. Autodiff through the chosen solve is the piecewise implicit
    QP derivative, with the usual nondifferentiability at active-set changes.
    """
    require_x64()
    if G.shape!=(5,2) or h.shape!=(5,):raise ValueError('Pinned BarrierNet uses exactly five rows')
    patterns=((jnp.arange(32)[:,None]>>jnp.arange(5))&1).astype(G.dtype)
    rho=1e4+1e-6;diagonal=1+2e-6
    H=diagonal*jnp.eye(2)+rho*jnp.einsum('pk,ki,kj->pij',patterns,G,G,precision='highest')
    rhs=u_nom+rho*jnp.einsum('pk,ki,k->pi',patterns,G,h,precision='highest')
    det=H[:,0,0]*H[:,1,1]-H[:,0,1]*H[:,1,0]
    points=jnp.column_stack(((rhs[:,0]*H[:,1,1]-rhs[:,1]*H[:,0,1])/det,
                            (H[:,0,0]*rhs[:,1]-H[:,1,0]*rhs[:,0])/det))
    violation=jnp.maximum(jnp.einsum('ki,pi->pk',G,points,precision='highest')-h,0.)
    costs=.5*diagonal*jnp.sum(points**2,axis=-1)-jnp.sum(points*u_nom,axis=-1)+.5*rho*jnp.sum(violation**2,axis=-1)
    return points[jnp.argmin(costs)]


def hard_deployment_qp(u_nom,G,h,a_max=.5,w_max=.5,return_diagnostics=False):
    """Original input bounds plus default exact-JAX solve and post-clip audit."""
    A=jnp.concatenate((G,jnp.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],G.dtype)))
    b=jnp.concatenate((h,jnp.array([a_max,a_max,w_max,w_max],h.dtype)))
    diagonal=1+1e-6
    solved=solve_qp2(u_nom/diagonal,A,b,jnp.full(2,diagonal,G.dtype))
    clipped=jnp.clip(solved.control,jnp.array([-a_max,-w_max]),jnp.array([a_max,w_max]))
    violation=jnp.max(jnp.matmul(A,clipped,precision='highest')-b)
    valid=solved.feasible&jnp.all(jnp.isfinite(clipped))&(violation<=1e-5)
    # Missing/infeasible actions remain invalid; no fake optimal zero fallback.
    if return_diagnostics:
        return clipped,valid,violation,solved.control,solved.feasible,solved.max_violation
    return clipped,valid,violation


def train_prediction(model,params,z,ctx,u_ref,mean,std,radius=.25):
    u_nom,p=model.apply({'params':params},(z-mean)/std,ctx[:4],ctx[4:6],u_ref)
    G,h=constraints(ctx[:4],ctx[6:].reshape(5,7),p,radius)
    return soft_training_qp(u_nom,G,h),p
