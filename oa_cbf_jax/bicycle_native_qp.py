"""Bicycle native qp functions and shared contracts."""

from dataclasses import asdict

from pathlib import Path

import time

import numpy as np

import jax

import jax.numpy as jnp

from .bicycle_control import BicycleControlConfig, constant

from .io import sha256

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


from .bicycle import integrate_bicycle, bicycle_state_violation


from .bicycle_observation import observe, unit_errors

from .bicycle_rollout import NAMES, GOAL, COLLISION, INFEASIBLE, TIMEOUT, STATE_BOUND

from .dynamics import swept_disk_clearance

class PhysicalKernels:
    def __init__(self,c=BicycleControlConfig()):
        self.config=c;r=c.robot
        def sense(x,original,mask,bx,bo,noise,key,k,first_x,first_o):
            current=original.at[:,:2].add(k.astype(jnp.float64)*constant(r.dt,jnp.float64)*original[:,3:5])
            ix,io=unit_errors(jax.random.fold_in(key,k),64)
            ix=jnp.where(k==0,0.,ix);io=jnp.where(k==0,0.,io)
            sx,so=observe(x,current,mask,bx,bo,noise,ix,io)
            return jnp.where(k==0,first_x,sx),jnp.where(k==0,first_o,so),ix,io
        def advance(x,u,original,mask,k):
            y,sub=integrate_bicycle(x,u,r);starts=jnp.concatenate((x[None],sub[:-1]))
            dt=constant(r.dt,jnp.float64)
            times=k.astype(jnp.float64)*dt+jnp.arange(r.integration_substeps,dtype=jnp.float64)*dt/r.integration_substeps
            clearance=jnp.min(jax.vmap(lambda a,b,t:swept_disk_clearance(a,b,original,mask,constant(r.radius,jnp.float64),t,t+dt/r.integration_substeps))(starts,sub,times))
            violation=jnp.max(jax.vmap(lambda z:bicycle_state_violation(z,r))(jnp.concatenate((x[None],sub))))
            return y,clearance,violation
        self.sense_function=jax.jit(sense);self.advance_function=jax.jit(advance)
        x=jnp.asarray(np.zeros(4),dtype=jnp.float64);o=jnp.asarray(np.zeros((64,5)),dtype=jnp.float64);mask=jnp.zeros(64,bool)
        # Sensor and actuator precision is explicit even when a native neural
        # controller enables global FP64 defaults for its own network.
        args=(x,o,mask,jnp.zeros(4,jnp.float32),jnp.zeros((64,5),jnp.float32),
            jnp.zeros(6,jnp.float32),jax.random.PRNGKey(1),jnp.int32(0),
            jnp.zeros(4,jnp.float32),jnp.zeros((64,5),jnp.float32))
        start=time.perf_counter();self.sense=self.sense_function.lower(*args).compile()
        self.advance=self.advance_function.lower(x,jnp.zeros(2,jnp.float32),o,mask,jnp.int32(0)).compile()
        self.rows=BarrierRows(c);self.compile_seconds=time.perf_counter()-start

    def cache_sizes(self):
        return dict(sense=self.sense_function._cache_size(),advance=self.advance_function._cache_size(),rows=self.rows.function._cache_size())

def episode(parent,method,kernels,steps=1600,acceptance='strict_rows'):
    c=kernels.config;r=c.robot;solver=NativeBicycleQP(method,kernels.rows,c,acceptance)
    x=np.asarray(parent['initial'],float);original=np.asarray(parent['obstacles'],float);mask=np.asarray(parent['mask'],bool)
    goal=np.asarray(parent['goal'],np.float32);noise=np.asarray(parent['noise'],np.float32)
    bx,bo,fx,fo=[np.asarray(parent[k],np.float32) for k in ('bias_x','bias_o','first_x','first_o')]
    key=np.asarray(jax.random.PRNGKey(parent['seed']+4));history=[];count=0
    minimum=float(np.min(np.where(mask,np.linalg.norm(original[:,:2]-x[:2],axis=1)-r.radius-original[:,2],np.inf)))
    arrived=lambda z:np.linalg.norm(z[:2]-goal)<=c.goal_tolerance and z[3]<=c.terminal_speed
    status=GOAL if arrived(x) else 0
    if max(r.speed_min-x[3],x[3]-r.speed_max)>c.qp_tolerance:status=STATE_BOUND
    if minimum<=0:status=COLLISION
    start=time.perf_counter();reason=NAMES[status]
    immutable=(jnp.asarray(original,dtype=jnp.float64),jnp.asarray(mask),jnp.asarray(bx),jnp.asarray(bo),jnp.asarray(noise),jnp.asarray(key))
    for tick in range(steps):
        sx,so,ix,io=map(np.asarray,kernels.sense(jnp.asarray(x,dtype=jnp.float64),*immutable,np.int32(tick),fx,fo))
        attempted=status==0;accepted=False;u=np.zeros(2,np.float32);before=x.copy();clear=violation=np.nan
        result=dict(solution=np.full(3 if method=='optimal_decay' else 2,np.nan),reference=np.full(3 if method=='optimal_decay' else 2,np.nan),selected=np.full(10,-1,np.int32),barrier=np.full(64,np.nan),gradient=np.full((64,4),np.nan),status='not_attempted',status_value=0,iterations=0,seconds=0.,raw_residual=np.nan,stored_residual=np.nan,feasible=False)
        if attempted:
            result=solver.solve(sx,goal,so,mask);accepted=result['feasible']
            if accepted:
                u=result['solution'][:2].astype(np.float32)
                x,clear,violation=map(np.asarray,kernels.advance(jnp.asarray(x,dtype=jnp.float64),u,immutable[0],immutable[1],np.int32(tick)))
                clear=float(clear);violation=float(violation);count+=1;minimum=min(minimum,clear)
                if arrived(x):status=GOAL
                if violation>c.qp_tolerance:status=STATE_BOUND
                if clear<=0:status=COLLISION
                reason=NAMES[status]
            else:
                status=INFEASIBLE
                reason=('stored_qp_residual_rejected' if acceptance=='strict_rows' and result['status_value'] in (1,2) else 'solver_or_nonfinite_failure:'+result['status'])
        history.append(dict(state_before=before,state=x.copy(),control=u,active=accepted,status=np.int32(status),
            observed_state=sx,observed_obstacles=so,innovation_x=ix,innovation_o=io,clearance=clear,state_violation=violation,
            attempted=attempted,**{'qp_'+k:v for k,v in result.items()}))
        if status:break
    if status==0:status=TIMEOUT;reason=NAMES[status]
    payload={k:np.asarray([h[k] for h in history]) for k in history[0]}
    payload.update({k:np.asarray(parent[k]) for k in ('initial','goal','obstacles','mask','bias_x','bias_o','first_x','first_o','noise')})
    payload.update(key=key,final_status=np.int32(status),expected_steps=np.int32(count),horizon=np.int32(steps))
    row=dict(group_id=parent['group_id'],family=parent['family'],method=method,acceptance=acceptance,status=NAMES[status],status_code=status,steps=count,
        termination_reason=reason,min_clearance=None if not np.isfinite(minimum) else minimum,execution_seconds=time.perf_counter()-start,
        solver_attempts=int(payload['attempted'].sum()),solver_seconds=float(payload['qp_seconds'].sum()),
        applied_row_violation_ticks=int(np.sum(payload['active']&(payload['qp_stored_residual']>c.qp_tolerance))),
        applied_input_violation_ticks=int(np.sum(payload['active']&np.any(np.abs(payload['control'])>np.array([r.acceleration_max,r.slip_max])+c.qp_tolerance,axis=1))))
    return row,payload
