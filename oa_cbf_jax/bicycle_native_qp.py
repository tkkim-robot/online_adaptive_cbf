"""Registered bicycle fixed/optimal-decay QPs with untouched OSQP defaults.

The original continuous barrier is differentiated automatically: this corrects
the independently demonstrated hand-gradient error, without changing h or gains.
The dynamic tracker passes five nearest centers into a ten-slot fixed QP; OD
uses only the first center. Moving-obstacle rows include the obstacle-position
derivative, as corrected upstream in safe_control commit 27d8569.
"""
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp

from .bicycle_control import BicycleControlConfig, constant
from .dataset import sha256

METHODS={'fixed_low':.1,'fixed_high':70.,'optimal_decay':.1}
DPCBF_DERIVATIVE_CONTRACT='total_position_drift_constant_obstacle_velocity'
UPSTREAM_DRIFT_FIX='27d856980435e374534cd158f3514eceb1433199'
SOURCE_FILES=('online_cbf_config.py','safe_control/dynamic_env/main.py',
    'safe_control/tracking.py','safe_control/robots/robot.py',
    'safe_control/robots/kinematic_bicycle2D.py','safe_control/robots/kinematic_bicycle2D_dpcbf.py',
    'safe_control/position_control/cbf_qp.py','safe_control/position_control/optimal_decay_cbf_qp.py')


def barrier(x,o,c=BicycleControlConfig()):
    p=o[:2]-x[:2];v=o[3:5]-x[3]*jnp.stack((jnp.cos(x[2]),jnp.sin(x[2])))
    unit=p/jnp.linalg.norm(p);radial=jnp.dot(unit,v);lateral=unit[0]*v[1]-unit[1]*v[0]
    radius=(o[2]+constant(c.radius,x.dtype))*constant(1.05,x.dtype)
    root=jnp.sqrt(jnp.maximum(jnp.dot(p,p)-radius**2,constant(1e-6,x.dtype)))
    shape=jnp.sqrt(constant(1.05**2-1,x.dtype))/radius
    return radial+constant(.5,x.dtype)*shape*root*lateral**2/jnp.linalg.norm(v)+shape*root


def nominal(x,goal,method,c=BicycleControlConfig()):
    # BaseRobot.nominal_input overrides the bicycle class signature defaults.
    # OD tracking passes (k_omega,k_a,k_v)=(3,.5,.5); fixed uses (2,1,1).
    kt,ka,kv=(3.,.5,.5) if method=='optimal_decay' else (2.,1.,1.)
    distance=max(np.linalg.norm(x[:2]-goal)-.05,.05)
    error=(np.arctan2(goal[1]-x[1],goal[0]-x[0])-x[2]+np.pi)%(2*np.pi)-np.pi
    delta=np.clip(kt*error,-c.robot.steering_max,c.robot.steering_max)
    beta=np.arctan(c.robot.rear_axle_distance/c.robot.wheel_base*np.tan(delta))
    speed=np.clip(kv*distance*max(0.,np.cos(error)),c.robot.speed_min,c.robot.speed_max)
    return np.array([ka*(speed-x[3]),beta])


def nearest(x,obs,mask,method):
    eligible=np.flatnonzero(mask)
    order=np.argsort(np.linalg.norm(obs[eligible,:2]-x[:2],axis=1))
    selected=eligible[order[:1 if method=='optimal_decay' else 5]]
    return selected


def contract(method,c=BicycleControlConfig(),acceptance='strict_rows'):
    import osqp
    if method not in METHODS:raise ValueError('Unknown registered method')
    if acceptance not in ('strict_rows','native_status'):raise ValueError('Unknown acceptance rule')
    return dict(schema='bicycle_native_continuous_qp_total_drift',method=method,alpha=METHODS[method],
        solver='OSQP',solver_version=osqp.__version__,solver_options={'verbose':False},
        source_sha256={f:sha256(f) for f in SOURCE_FILES},config=asdict(c),
        obstacle_rows=1 if method=='optimal_decay' else 10,
        used_obstacles='Single nearest center' if method=='optimal_decay' else 'Five nearest centers from dynamic tracker; ten original QP slots with unused rows zero',
        nominal='Native BaseRobot wrapper gains (3,.5,.5) for OD or (2,1,1) fixed; original nominal clipping only',
        barrier='Original beta1.05, shape.5/1, max(domain,1e-6), no OA buffer or relative-speed regularizer; total robot and obstacle-position drift at constant obstacle velocity',
        derivative_contract=DPCBF_DERIVATIVE_CONTRACT,upstream_drift_fix=UPSTREAM_DRIFT_FIX,
        derivative='Exact JAX derivative of native scalar h replaces erroneous hand gradient; no h/gain/objective tuning',
        decay='Unrestricted omega with reference1 and10000*(omega-1)^2' if method=='optimal_decay' else 'Fixed configured alpha',
        physical_adapter='Shared actual unclipped affine-slip plant, FP32 observations/actuators, common sensor streams and rolling goal condition. Original acceleration/slip bounds only; actual speed violations end and fail the task.',
        acceptance=('Solved/solved-inaccurate and finite plus original QP rows on raw and FP32 actuator within common1e-5tolerance. Violations are numerical residual stops, not certified infeasibility. No retry/tolerance change/rescue.' if acceptance=='strict_rows' else 'Native tracker optimal status only (OSQP solved, status1) and finite. No additional row-residual veto, clipping, retry or solver tuning. Record actual residual and input-bound violations separately from navigation outcomes.'),
        initialization='New OSQP instance per episode, setup on first real QP; default warm start thereafter')


class BarrierRows:
    def __init__(self,c=BicycleControlConfig()):
        self.config=c
        self.function=jax.jit(jax.vmap(jax.value_and_grad(lambda x,o:barrier(x,o,c),argnums=0),in_axes=(None,0)))
        x=jnp.asarray(np.array([0.,0.,0.,.5]),dtype=jnp.float64)
        obs=jnp.asarray(np.tile([10.,10.,.3,0.,0.],(64,1)),dtype=jnp.float64)
        start=time.perf_counter();self.execute=self.function.lower(x,obs).compile()
        jax.block_until_ready(self.execute(x,obs));self.compile_seconds=time.perf_counter()-start

    def terms(self,x,obs,mask):
        safe=obs.astype(float).copy();safe[~mask]=[x[0]+10.,x[1]+10.,.3,0.,0.]
        h,grad=jax.device_get(self.execute(jnp.asarray(x,dtype=jnp.float64),jnp.asarray(safe,dtype=jnp.float64)))
        return h,grad


def problem(x,goal,obs,mask,method,h,grad,c=BicycleControlConfig()):
    x=np.asarray(x,float);obs=np.asarray(obs,float);od=method=='optimal_decay'
    slots=1 if od else 10;columns=3 if od else 2;selected=nearest(x,obs,mask,method)
    A=np.zeros((slots+4,columns));b=np.zeros(slots+4);v=x[3];theta=x[2]
    drift=np.array([v*np.cos(theta),v*np.sin(theta),0.,0.])
    authority=np.array([[0.,-v*np.sin(theta)],[0.,v*np.cos(theta)],[0.,v/c.robot.rear_axle_distance],[1.,0.]])
    # Translation invariance gives dh/dp_obs = -dh/dp_robot. Velocity already
    # appearing inside h does NOT replace this explicit obstacle-motion term.
    n=len(selected);A[:n,:2]=-grad[selected]@authority
    b[:n]=grad[selected]@drift-np.sum(grad[selected,:2]*obs[selected,3:5],axis=1)
    if od:A[:n,2]=-METHODS[method]*h[selected]
    else:b[:n]+=METHODS[method]*h[selected]
    A[slots:,:2]=[[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]
    b[slots:]=[c.robot.acceleration_max,c.robot.acceleration_max,c.robot.slip_max,c.robot.slip_max]
    ref=nominal(x,np.asarray(goal,float),method,c)
    return np.r_[ref,1.] if od else ref,A,b,selected


class NativeBicycleQP:
    def __init__(self,method,rows,c=BicycleControlConfig(),acceptance='strict_rows'):
        import osqp
        if method not in METHODS:raise ValueError('Unknown native method')
        if acceptance not in ('strict_rows','native_status'):raise ValueError('Unknown acceptance rule')
        self.method=method;self.rows=rows;self.config=c;self.initialized=False;self.acceptance=acceptance
        self.solver=osqp.OSQP();self.weights=np.array([1.,1.,1e4]) if method=='optimal_decay' else np.ones(2)

    def solve(self,x,goal,obs,mask):
        h,grad=self.rows.terms(x,obs,mask);ref,A,b,selected=problem(x,goal,obs,mask,self.method,h,grad,self.config)
        indices=np.full(10,-1,np.int32);indices[:len(selected)]=selected
        base=dict(reference=ref,selected=indices,barrier=h,gradient=grad)
        if not all(np.isfinite(v).all() for v in (ref,A,b)):
            return dict(base,solution=np.full(len(ref),np.nan),status='nonfinite_problem',status_value=-1,iterations=0,seconds=0.,raw_residual=np.nan,stored_residual=np.nan,feasible=False)
        q=-2*self.weights*ref
        if not self.initialized:
            from scipy import sparse
            n,m=A.shape
            pattern=sparse.csc_matrix((A.ravel(order='F'),np.tile(np.arange(n),m),np.arange(0,n*m+1,n)),shape=A.shape)
            self.solver.setup(P=sparse.diags(2*self.weights,format='csc'),q=q,A=pattern,l=np.full(n,-np.inf),u=b,verbose=False)
            self.initialized=True
        else:self.solver.update(q=q,Ax=A.ravel(order='F'),u=b)
        start=time.perf_counter();result=self.solver.solve(raise_error=False);seconds=time.perf_counter()-start
        solution=np.full(len(ref),np.nan) if result.x is None else np.asarray(result.x).copy()
        finite=np.isfinite(solution).all();raw=float(np.max(A@solution-b)) if finite else np.nan
        stored=solution.copy();stored[:2]=solution[:2].astype(np.float32)
        residual=float(np.max(A@stored-b)) if finite else np.nan
        accepted=(result.info.status_val==1 and finite) if self.acceptance=='native_status' else (result.info.status_val in (1,2) and finite and max(raw,residual)<=self.config.qp_tolerance)
        return dict(base,solution=solution,status=result.info.status,status_value=int(result.info.status_val),iterations=int(result.info.iter),seconds=seconds,
            raw_residual=raw,stored_residual=residual,feasible=bool(accepted))
