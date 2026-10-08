"""Default native bicycle/Quad3D BarrierNet deployment components.

The neural features and nominal/constraint arithmetic compile together in JAX.
The hard QP uses untouched OSQP defaults; the original inner bounds and wrapper
post-clip are distinct. No OA guidance, observer, solver polish or rescue is used.
"""
import time
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from .bicycle import BicycleConfig
from .quad3d import Quad3DConfig
from .barriernet_variants import (bicycle_features,quad3d_features,bicycle_nominal,
    quad3d_nominal,bicycle_constraints,quad3d_constraints,require_x64,BarrierNetVariant)
from .dataset import sha256


def native_variant_contract(robot_model,config):
    flight=robot_model=='Quad3D'
    if not ((flight and isinstance(config,Quad3DConfig)) or
            (robot_model=='KinematicBicycle2D_DPCBF' and isinstance(config,BicycleConfig))):
        raise ValueError('Mismatched native model and physical config')
    return dict(robot_model=robot_model,physics=asdict(config),obstacles=5,
        state_dimension=6 if flight else 4,goal_dimension=3 if flight else 2,
        control_dimension=4 if flight else 2,
        constraint=('Original direct-acceleration spherical-distance proxy, z=0 obstacle centers; p2 and fourth control unused'
                    if flight else 'Original parabolic line-of-sight proxy; steering coefficient and drift set to zero'),
        physical_evaluation='Full shared plant, all obstacles, actual actuator limits; native proxy residual is not a physical safety certificate',
        inference='Original observed-goal nominal and neural residual, default OSQP hard rows and original inner bounds, actual wrapper post-clip',
        training='Original single-model architecture and loss; default50epochs,64batch,Adam1e-3,patience10')


def load_bundle(bundle,robot_model,config):
    """Reload native parameters; fail on wrong physics or modified artifacts."""
    require_x64();root=Path(bundle);m=json.loads((root/'manifest.json').read_text())
    contract=native_variant_contract(robot_model,config)
    name='quad3d' if robot_model=='Quad3D' else 'bicycle'
    if (m['schema']!=f'barriernet_{name}_jax_v1' or m['task_contract']!=contract
            or m['radius']!=config.radius):
        raise ValueError('Wrong native model/task contract')
    defaults=dict(epochs=50,batch_size=64,adam_lr=.001,patience=10,
                  p_regularizer=.01,train_slack_rho=1e4,eps_q=1e-6)
    if m['settings']!=defaults:raise ValueError('Changed native training defaults')
    for file,key in (('weights.msgpack','weights_sha256'),('normalization.npz','normalization_sha256')):
        if sha256(root/file)!=m[key]:raise ValueError('Changed native weights/normalization')
    model=BarrierNetVariant(robot_model)
    template=model.init(jax.random.PRNGKey(0),jnp.zeros(25),
        jnp.zeros(contract['state_dimension']),jnp.zeros(contract['goal_dimension']),
        jnp.zeros(contract['control_dimension']))['params']
    params=serialization.from_bytes(template,(root/'weights.msgpack').read_bytes())
    if not all(np.isfinite(v).all() for v in jax.tree.leaves(params)):
        raise ValueError('Nonfinite native weights')
    with np.load(root/'normalization.npz') as f:mean,std=f['mean'],f['std']
    if (mean.shape!=(25,) or std.shape!=(25,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or not np.all(std>0)):
        raise ValueError('Invalid native normalization')
    return m,model,params,jnp.asarray(mean),jnp.asarray(std)


def network_problem(model,params,mean,std,robot_model,config):
    """Return one fixed-shape NN/nominal/constraint function; caller AOT compiles."""
    require_x64()
    if robot_model=='Quad3D':
        if not isinstance(config,Quad3DConfig):raise ValueError('Expected Quad3D physics')
        feature,nominal,rows=quad3d_features,quad3d_nominal,quad3d_constraints
        state_dim,goal_dim=6,3
    elif robot_model=='KinematicBicycle2D_DPCBF':
        if not isinstance(config,BicycleConfig):raise ValueError('Expected bicycle physics')
        feature,nominal,rows=bicycle_features,bicycle_nominal,bicycle_constraints
        state_dim,goal_dim=4,2
    else:raise ValueError('Unsupported native BarrierNet variant')
    mean=jnp.asarray(mean,jnp.float64);std=jnp.asarray(std,jnp.float64)
    def problem(state,goal,obstacles,mask):
        z,ctx=feature(state,goal,obstacles,mask,config.radius)
        ref=nominal(state,goal,config)
        u_nom,p=model.apply({'params':params},(z-mean)/std,ctx[:state_dim],goal,ref)
        G,h=rows(ctx[:state_dim],ctx[state_dim+goal_dim:].reshape(5,7),p,config.radius)
        return u_nom,G,h,p,ref,z,ctx
    return jax.jit(problem)


class NativeVariantQP:
    """One episode's default OSQP instance, original inner bounds then post-clip.

    Match native optimal-status acceptance. Residuals, including any violation
    introduced by wrapper clipping, are recorded separately and never hidden.
    The common physical evaluator still checks every obstacle and plant bound.
    """
    def __init__(self,robot_model,config):
        import osqp
        if robot_model=='Quad3D' and isinstance(config,Quad3DConfig):
            self.nu=4;self.inner_upper=np.full(4,10.);self.inner_lower=-self.inner_upper
            self.applied_lower=np.full(4,config.input_min);self.applied_upper=np.full(4,config.input_max)
        elif robot_model=='KinematicBicycle2D_DPCBF' and isinstance(config,BicycleConfig):
            self.nu=2;self.inner_upper=np.array([5.,np.deg2rad(32.)]);self.inner_lower=-self.inner_upper
            self.applied_upper=np.array([config.acceleration_max,config.slip_max]);self.applied_lower=-self.applied_upper
        else:raise ValueError('Mismatched native model and physical config')
        self.solver=osqp.OSQP();self.initialized=False
        self.contract=dict(robot_model=robot_model,solver='OSQP',solver_version=osqp.__version__,
            solver_options={'verbose':False},diagonal=1+1e-6,slack=False,
            inner_lower=self.inner_lower.tolist(),inner_upper=self.inner_upper.tolist(),
            applied_lower=self.applied_lower.tolist(),applied_upper=self.applied_upper.tolist(),
            acceptance='Native solved status1 and finite; wrapper clip, residuals recorded without extra veto',
            deviations='Default OSQP replaces qpth hard inference QP; same objective, rows and inner bounds. Shared actual plant actuator bounds used for original wrapper clipping.')

    def solve(self,reference,G,h):
        from scipy import sparse
        reference=np.asarray(reference,float);G=np.asarray(G,float);h=np.asarray(h,float)
        if reference.shape!=(self.nu,) or G.shape!=(5,self.nu) or h.shape!=(5,):
            raise ValueError('Expected five native obstacle rows')
        A=np.concatenate((G,np.eye(self.nu),-np.eye(self.nu)))
        b=np.r_[h,self.inner_upper,-self.inner_lower]
        if not all(np.isfinite(v).all() for v in (reference,A,b)):
            return dict(control=np.full(self.nu,np.nan),raw_control=np.full(self.nu,np.nan),
                accepted=False,status='nonfinite_problem',status_value=-1,iterations=0,
                raw_residual=np.nan,applied_residual=np.nan,dual=np.full(len(b),np.nan),seconds=0.)
        if not self.initialized:
            n,m=A.shape
            pattern=sparse.csc_matrix((A.ravel(order='F'),np.tile(np.arange(n),m),np.arange(0,n*m+1,n)),shape=A.shape)
            self.solver.setup(P=sparse.eye(self.nu,format='csc')*(1+1e-6),q=-reference,
                A=pattern,l=np.full(n,-np.inf),u=b,verbose=False)
            self.initialized=True
        else:self.solver.update(q=-reference,Ax=A.ravel(order='F'),u=b)
        start=time.perf_counter();result=self.solver.solve(raise_error=False);seconds=time.perf_counter()-start
        raw=np.full(self.nu,np.nan) if result.x is None else np.asarray(result.x).copy()
        control=np.clip(raw,self.applied_lower,self.applied_upper)
        finite=np.isfinite(raw).all() and np.isfinite(control).all()
        return dict(control=control,raw_control=raw,accepted=bool(result.info.status_val==1 and finite),
            status=result.info.status,status_value=int(result.info.status_val),iterations=int(result.info.iter),
            raw_residual=float(np.max(A@raw-b)),applied_residual=float(np.max(A@control-b)),
            dual=np.asarray(result.y).copy() if result.y is not None else np.full(len(b),np.nan),seconds=seconds)
