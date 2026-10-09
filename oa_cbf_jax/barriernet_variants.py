"""Pinned bicycle and Quad3D BarrierNet components, ported to JAX.

These reproduce the original repository's approximations, including ignored
steering in the bicycle constraint and the direct-acceleration Quad3D proxy.
They are native baseline components, not the OA physical CBFs or safety proofs.
Training, deployment bounds and common-plant adapters are separate consumers.
"""

import jax

import jax.numpy as jnp

import numpy as np

from flax import linen as nn

from .barriernet import require_x64, features as ground_features, soft_training_qp

from .bicycle import BicycleConfig

from .quad3d import Quad3DConfig, matrices

class BarrierNetVariant(nn.Module):
    robot_model:str

    @nn.compact
    def __call__(self,z_normalized,state,goal,u_ref):
        require_x64()
        if self.robot_model not in ('Quad3D','KinematicBicycle2D_DPCBF'):
            raise ValueError('Expected an original bicycle or Quad3D BarrierNet variant')
        flight=self.robot_model=='Quad3D'
        activation=jnp.tanh if flight else nn.relu
        first,width=(512,128) if flight else (256,64)
        nu=4 if flight else 2
        if state.shape[-1]!=(6 if flight else 4) or goal.shape[-1]!=(3 if flight else 2) or u_ref.shape[-1]!=nu:
            raise ValueError('Wrong native reduced-state/goal/control shape')
        def layer(x,size,name):
            limit=x.shape[-1]**-.5
            def init(key,shape,dtype=jnp.float64):
                return jax.random.uniform(key,shape,dtype,minval=-limit,maxval=limit)
            return nn.Dense(size,dtype=jnp.float64,param_dtype=jnp.float64,
                kernel_init=init,bias_init=init,name=name)(x)
        blocks=z_normalized.reshape(z_normalized.shape[:-1]+(5,5))
        encoded=activation(layer(blocks,first,'obs_fc1'))
        encoded=activation(layer(encoded,width,'obs_fc2'))
        if flight:encoded=activation(layer(encoded,width,'obs_fcm'))
        parameters=4*jax.nn.sigmoid(layer(encoded,2 if flight else 1,'fc_p'))
        hidden=jnp.concatenate((jnp.mean(encoded,axis=-2),state,goal,u_ref),axis=-1)
        hidden=activation(layer(hidden,width,'u_fc1'))
        if flight:hidden=activation(layer(hidden,width,'u_fcm'))
        return u_ref+layer(hidden,nu,'u_out'),parameters

def bicycle_features(state,goal,obstacles,mask,radius=.3):
    return ground_features(state,goal,obstacles,mask,radius)

def quad3d_features(state,goal,obstacles,mask,radius=.25):
    """Original full-state input, six-state context, z=0 obstacle assumption."""
    require_x64()
    if state.shape!=(12,) or goal.shape!=(3,):raise ValueError('Native Quad3D features require full12 state and 3D goal')
    obs=jnp.pad(obstacles,((0,0),(0,7-obstacles.shape[-1])))
    dummy=jnp.array([100.,100.,0.,0.,0.,0.,0.],jnp.float64)
    clearance=jnp.sqrt(jnp.sum((obs[:,:2]-state[:2])**2,axis=-1)+state[2]**2)-obs[:,2]-radius
    rows=jnp.concatenate((obs,jnp.tile(dummy,(5,1))))
    priority=jnp.concatenate((jnp.where(mask,clearance,jnp.inf),jnp.full(5,jnp.inf)))
    indices=jnp.argsort(priority,stable=True)[:5]
    valid=jnp.concatenate((mask,jnp.zeros(5,bool)))[indices]
    selected=jnp.where(valid[:,None],rows[indices],dummy)
    delta=selected[:,:2]-state[:2]
    bearing=(jnp.arctan2(delta[:,1],delta[:,0])-state[5]+jnp.pi)%(2*jnp.pi)-jnp.pi
    distance=jnp.sqrt(jnp.sum(delta**2,axis=-1)+state[2]**2)-selected[:,2]-radius
    z=jnp.column_stack((delta,bearing,distance,jnp.full(5,jnp.linalg.norm(state[6:9])))).reshape(25)
    context=jnp.concatenate((state[:3],state[6:9],goal,selected.reshape(35)))
    return z,context

def bicycle_nominal(state,goal,config=BicycleConfig()):
    """Original BaseRobot wrapper defaults: k_theta=2, k_a=k_v=1."""
    distance=jnp.maximum(jnp.linalg.norm(goal-state[:2])-.05,.05)
    error=(jnp.arctan2(goal[1]-state[1],goal[0]-state[0])-state[2]+jnp.pi)%(2*jnp.pi)-jnp.pi
    delta=jnp.clip(2*error,-config.steering_max,config.steering_max)
    slip=jnp.arctan(config.rear_axle_distance/config.wheel_base*jnp.tan(delta))
    speed=jnp.clip(distance*jnp.maximum(jnp.cos(error),0.),config.speed_min,config.speed_max)
    return jnp.stack((speed-state[3],slip))

def quad3d_nominal(state,goal,config=Quad3DConfig()):
    """Original Quad3D nominal_input defaults, shared actual motor bounds."""
    c=config;acceleration=goal-state[:3]-2*state[6:9]
    wrench=jnp.stack((c.mass*acceleration[2],
        c.inertia_y*(5*(acceleration[0]/c.gravity-state[3])-2*state[9]),
        c.inertia_x*(5*(-acceleration[1]/c.gravity-state[4])-2*state[10]),
        c.inertia_z*(-5*state[5]-2*state[11])))
    _,b=matrices(c)
    allocation=np.diag([c.mass,c.inertia_y,c.inertia_x,c.inertia_z])@b[8:12]
    inverse=jnp.asarray(np.linalg.pinv(allocation),state.dtype)
    return jnp.clip(inverse@wrench,c.input_min,c.input_max)

def quad3d_constraints(state,obstacles,parameters,radius=.25):
    """Exact original reduced-state proxy: fourth control and p2 are unused."""
    delta=jnp.column_stack((state[:2]-obstacles[:,:2],jnp.full(len(obstacles),state[2])))
    barrier=jnp.sum(delta**2,axis=-1)-1.01*(obstacles[:,2]+radius)**2
    drift=2*jnp.sum(delta*state[3:6],axis=-1)
    return -jnp.column_stack((2*delta,jnp.zeros(len(obstacles)))),drift+parameters[:,0]*barrier

def bicycle_constraints(state,obstacles,parameters,radius=.3):
    """Original line-of-sight barrier and its acceleration-only derivative proxy."""
    relative=obstacles[:,:2]-state[:2]
    velocity=obstacles[:,3:5]-state[3]*jnp.array([jnp.cos(state[2]),jnp.sin(state[2])])
    rotation=jnp.arctan2(relative[:,1],relative[:,0]);cr=jnp.cos(rotation);sr=jnp.sin(rotation)
    vx=cr*velocity[:,0]+sr*velocity[:,1];vy=-sr*velocity[:,0]+cr*velocity[:,1]
    distance=jnp.sqrt(jnp.sum(relative**2,axis=-1)+1e-12)
    speed=jnp.sqrt(jnp.sum(velocity**2,axis=-1)+1e-12)
    dimension=1.1*(obstacles[:,2]+radius)
    clearance=jnp.sqrt(jnp.maximum(distance**2-dimension**2,1e-6))
    k=jnp.sqrt(jnp.float64(1.1)**2-1.)/dimension
    barrier=vx+.5*k*clearance/speed*vy**2+k*clearance
    along=cr*jnp.cos(state[2])+sr*jnp.sin(state[2])
    return jnp.column_stack((along,jnp.zeros_like(along))),parameters[:,0]*barrier

def variant_training_qp(reference,G,h):
    """Original five independent training slacks, no hard training input bounds.

    Enumerate the32 hinge regions after eliminating slacks, retaining the exact
    original convex objective. The two-control variant uses the existing port.
    """
    require_x64()
    if G.shape==(5,2):return soft_training_qp(reference,G,h)
    if G.shape!=(5,4) or h.shape!=(5,) or reference.shape!=(4,):
        raise ValueError('Native Quad3D training expects five rows and four controls')
    patterns=((jnp.arange(32)[:,None]>>jnp.arange(5))&1).astype(G.dtype)
    rho=1e4+1e-6;diagonal=1+2e-6
    matrices=diagonal*jnp.eye(4)+rho*jnp.einsum('pk,ki,kj->pij',patterns,G,G,precision='highest')
    rhs=reference+rho*jnp.einsum('pk,ki,k->pi',patterns,G,h,precision='highest')
    candidates=jnp.linalg.solve(matrices,rhs[...,None])[...,0]
    violation=jnp.maximum(jnp.einsum('ki,pi->pk',G,candidates,precision='highest')-h,0.)
    objective=.5*diagonal*jnp.sum(candidates**2,axis=-1)-jnp.sum(candidates*reference,axis=-1)+.5*rho*jnp.sum(violation**2,axis=-1)
    return candidates[jnp.argmin(objective)]

def train_prediction(model,params,z,ctx,u_ref,mean,std,radius=.3):
    """Original native soft-QP training, including expert-reference convention."""
    flight=model.robot_model=='Quad3D';ns,ng=(6,3) if flight else (4,2)
    u_nom,p=model.apply({'params':params},(z-mean)/std,ctx[:ns],ctx[ns:ns+ng],u_ref)
    rows=quad3d_constraints if flight else bicycle_constraints
    G,h=rows(ctx[:ns],ctx[ns+ng:].reshape(5,7),p,radius)
    return variant_training_qp(u_nom,G,h),p
