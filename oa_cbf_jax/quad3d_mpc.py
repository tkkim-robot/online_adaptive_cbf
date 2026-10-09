"""Quad3d mpc functions and shared contracts."""

from dataclasses import asdict

from functools import lru_cache

import time

import numpy as np

from .quad3d import matrices

from .quad3d_control import Quad3DControlConfig

DEFAULTS={'fixed_low':.01,'fixed_high':.99,'optimal_decay':.01}

Q=np.array([30.,30.,5.,20.,20.,1.,10.,10.,10.,20.,20.,1.])

SOURCE_FILES=('online_cbf_config.py','safe_control/robots/quad3D.py',
    'safe_control/position_control/mpc_cbf.py','safe_control/position_control/optimal_decay_mpc_cbf.py')

@lru_cache(maxsize=8)
def flow(robot):
    a,b=matrices(robot);dt=robot.dt;eye=np.eye(12);aa=a@a;aaa=aa@a
    return a,b,eye+dt*a+dt**2/2*aa+dt**3/6*aaa,(dt*eye+dt**2/2*a+dt**3/6*aa+dt**4/24*aaa)@b

def numpy_barrier(states,controls,obs,alpha,omegas,config,shared_contract=True,coupled_decay=True):
    """Vectorized independent RK4 stages, separate from the CasADi matrix form."""
    x=np.asarray(states,float);u=np.asarray(controls,float);o=np.asarray(obs,float)
    a,b=matrices(config.robot);dt=config.robot.dt
    k1=x@a.T+u@b.T;k2=(x+dt/2*k1)@a.T+u@b.T
    k3=(x+dt/2*k2)@a.T+u@b.T;k4=(x+dt*k3)@a.T+u@b.T
    y=x+dt/6*(k1+2*k2+2*k3+k4)
    radius=config.robot.radius+(config.clearance_buffer if shared_contract else 0.)
    def h(z):return np.sum((z[...,None,:2]-o[:,:2])**2,axis=-1)-1.01*(radius+o[:,2])**2
    gain=alpha*(np.asarray(omegas)[...,0] if coupled_decay else np.ones(x.shape[:-1]))
    return h(y)-h(x)+gain[...,None]*h(x)

def prediction_residual(states,controls,omegas,x,obs,mask,alpha,c,shared_contract=True,coupled_decay=True):
    if not all(np.isfinite(v).all() for v in (states,controls,omegas,x,obs)):return np.inf,np.inf
    a,b=matrices(c.robot)
    expected=states[:-1]+c.robot.dt*(states[:-1]@a.T+controls@b.T)
    eq=max(float(np.max(abs(states[0]-x))),float(np.max(abs(states[1:]-expected))))
    barrier=numpy_barrier(states[:-1],controls,obs,alpha,omegas,c,shared_contract,coupled_decay)
    violation=max(float(np.max(-barrier[:,mask],initial=-np.inf)),
        float(np.max(c.robot.input_min-controls)),float(np.max(controls-c.robot.input_max)))
    if shared_contract:
        limits=np.array([c.tilt_limit,c.tilt_limit,c.yaw_limit]+[c.velocity_limit]*3+[c.rate_limit]*3)
        violation=max(violation,float(np.max(abs(states[1:,3:])-limits)),
            float(np.max(c.altitude_min-states[1:,2])),float(np.max(states[1:,2]-c.altitude_max)))
    return eq,violation

class Quad3DMPC:
    def __init__(self,capacity=64,method='fixed_low',config=Quad3DControlConfig(hold_guard='bernstein_v87'),*,shared_contract=True,coupled_decay=True):
        import casadi as ca
        if method not in DEFAULTS or type(capacity) is not int or capacity<1:raise ValueError('Invalid Quad3D MPC method/capacity')
        if config.nominal_bias_observer!='none' or config.qp_refinement!='none':raise ValueError('Native baselines do not inherit OA improvements')
        self.capacity=capacity;self.method=method;self.config=config;self.horizon=10;self.alpha=DEFAULTS[method]
        self.decay=method=='optimal_decay';self.shared_contract=shared_contract;self.coupled_decay=coupled_decay
        self.last_omega=np.zeros(2);H=self.horizon;C=capacity;c=config
        X=ca.SX.sym('x',12,H+1);U=ca.SX.sym('u',4,H);W=ca.SX.sym('omega',2,H) if self.decay else ca.DM.ones(2,H)
        initial=ca.SX.sym('initial',12);goal=ca.SX.sym('goal',3);obs=ca.SX.sym('obs',C,5)
        active=ca.SX.sym('active',C);previous=ca.SX.sym('previous',4)
        parameters=ca.vertcat(initial,goal,ca.vec(obs),active,previous)
        variables=ca.vertcat(ca.vec(X),ca.vec(U),ca.vec(W)) if self.decay else ca.vertcat(ca.vec(X),ca.vec(U))
        a,b,f,g=map(ca.DM,flow(c.robot));target=ca.vertcat(goal,ca.DM.zeros(9));weights=ca.DM(Q)
        def barrier(x,u,w):
            radius=c.robot.radius+(c.clearance_buffer if shared_contract else 0.)
            def h(z):return (obs[:,0]-z[0])**2+(obs[:,1]-z[1])**2-1.01*(obs[:,2]+radius)**2
            gain=self.alpha*(w[0] if coupled_decay else 1.)
            return h(f@x+g@u)-h(x)+gain*h(x)
        equalities=[X[:,0]-initial];inequalities=[];cost=0
        limits=ca.DM([c.tilt_limit,c.tilt_limit,c.yaw_limit]+[c.velocity_limit]*3+[c.rate_limit]*3)
        for k in range(H):
            cost+=ca.dot(weights,(X[:,k]-target)**2)
            # Effective do-mpc5.1.2 objective: the second custom OD rterm
            # replaces the first. Preserve its existing omega penalties.
            cost+=10*ca.sumsqr(W[:,k]-1) if self.decay else ca.sumsqr(U[:,k]-(previous if k==0 else U[:,k-1]))
            equalities.append(X[:,k+1]-X[:,k]-c.robot.dt*(a@X[:,k]+b@U[:,k]))
            inequalities.append(active*barrier(X[:,k],U[:,k],W[:,k])+(1-active))
            if shared_contract:
                inequalities.append(ca.vertcat(limits-X[3:,k+1],limits+X[3:,k+1],X[2,k+1]-c.altitude_min,c.altitude_max-X[2,k+1]))
        cost+=ca.dot(weights,(X[:,H]-target)**2)
        constraints=ca.vertcat(*equalities,*inequalities);self.equalities=12*(H+1)
        self.lbg=np.zeros(int(constraints.numel()));self.ubg=np.r_[np.zeros(self.equalities),np.full(len(self.lbg)-self.equalities,np.inf)]
        self.lbx=np.full(int(variables.numel()),-np.inf);self.ubx=-self.lbx.copy()
        self.lbx[12*(H+1):12*(H+1)+4*H]=c.robot.input_min;self.ubx[12*(H+1):12*(H+1)+4*H]=c.robot.input_max
        self.solver_options={'ipopt.print_level':0,'print_time':False};start=time.perf_counter()
        self.solver=ca.nlpsol('quad3d_native_mpc','ipopt',dict(x=variables,p=parameters,f=cost,g=constraints),self.solver_options)
        self.setup_seconds=time.perf_counter()-start
        sx=ca.SX.sym('state',12);su=ca.SX.sym('control',4);sw=ca.SX.sym('omega',2)
        self.barrier_expression=ca.Function('quad3d_native_barrier',[sx,su,obs,sw],[barrier(sx,su,sw)])

    def contract(self):
        from .io import sha256
        import casadi
        return dict(schema='quad3d_default_discrete_mpc_v101',method=self.method,alpha=self.alpha,horizon=self.horizon,Q=Q.tolist(),
            solver='CasADi/IPOPT',casadi_version=casadi.__version__,solver_options=self.solver_options,
            source_files={n:sha256(n) for n in SOURCE_FILES},config=asdict(self.config),capacity=self.capacity,
            input_cost='10*sum((omega-1)^2), pinned effective OD rterm' if self.decay else 'sum((u[k]-u[k-1])^2)',
            omega_bounds='unrestricted; second omega retains original penalty only' if self.decay else 'fixed1',
            coupled_decay=self.coupled_decay,shared_contract=self.shared_contract,
            validity_correction='Quad3D original OD CBF omits omega entirely; valid OD connects alpha*omega1 to h. No penalty/gain/solver tuning.' if self.decay and self.coupled_decay else None,
            prediction='Euler12state horizon, native RK4 one-step cylinder CBF, beta1.01, observed centers fixed over horizon',
            task_adapter='All active observed cylinders, shared radius/noise inflation/clearance and predicted full-state envelope, common causal route target. Not the literal five-obstacle original simulation.',
            acceptance='Default solver success plus finite solution and independently checked prediction equality/inequality residual <=1e-5. No rescue or solver retries.',
            initialization='Repeat current raw observed state and previous applied input, previous omega; zero input/omega on episode start; no tuned warm-start settings.')

    def solve(self,x,goal,obs,mask,previous):
        x,goal,obs,previous=map(lambda v:np.asarray(v,float),(x,goal,obs,previous));mask=np.asarray(mask,bool);H=self.horizon
        if x.shape!=(12,) or goal.shape!=(3,) or obs.shape!=(self.capacity,5) or mask.shape!=(self.capacity,) or previous.shape!=(4,):raise ValueError('Quad3D MPC input shape mismatch')
        parameters=np.r_[x,goal,obs.ravel(order='F'),mask.astype(float),previous]
        if not np.isfinite(parameters).all():raise ValueError('Nonfinite Quad3D MPC input')
        guess=np.r_[np.tile(x,H+1),np.tile(previous,H)]
        if self.decay:guess=np.r_[guess,np.tile(self.last_omega,H)]
        start=time.perf_counter();answer=self.solver(x0=guess,p=parameters,lbx=self.lbx,ubx=self.ubx,lbg=self.lbg,ubg=self.ubg)
        elapsed=time.perf_counter()-start;stats=self.solver.stats();v=np.asarray(answer['x']).ravel()
        X=v[:12*(H+1)].reshape(H+1,12);U=v[12*(H+1):12*(H+1)+4*H].reshape(H,4)
        W=v[-2*H:].reshape(H,2) if self.decay else np.ones((H,2))
        eq,violation=prediction_residual(X,U,W,x,obs,mask,self.alpha,self.config,self.shared_contract,self.coupled_decay)
        feasible=bool(stats.get('success',False) and eq<=self.config.qp_tolerance and violation<=self.config.qp_tolerance)
        return dict(control=U[0],omega=W[0],states=X,controls=U,omegas=W,feasible=feasible,
            solver_success=bool(stats.get('success',False)),solver_status=str(stats.get('return_status')),iterations=int(stats.get('iter_count',-1)),
            solve_seconds=elapsed,max_equality_error=eq,max_constraint_violation=violation,objective=float(answer['f']))


import jax

import jax.numpy as jnp

from .quad3d import integrate_quad3d


from .quad3d_observation import observe, observed_arrived, controller_obstacles, guidance_obstacles, unit_tape

from .quad3d_routing import flight_target

C=Quad3DControlConfig(hold_guard='bernstein_v87')

def initial_physical(x,o,mask,c=C):
    clearance=float(np.min(np.where(mask,np.linalg.norm(x[:2]-o[:,:2],axis=-1)-o[:,2]-c.robot.radius,np.inf)))
    limits=np.array([c.tilt_limit,c.tilt_limit,c.yaw_limit]+[c.velocity_limit]*3+[c.rate_limit]*3)
    envelope=max(float(np.max(abs(x[3:])-limits)),c.altitude_min-x[2],x[2]-c.altitude_max)
    return clearance,envelope

class PhysicalKernels:
    def __init__(self,c=C,capacity=64):
        x=jnp.asarray(np.zeros(12),jnp.float64);o=jnp.asarray(np.zeros((capacity,5)),jnp.float64)
        mask=jnp.zeros(capacity,bool);noise=jnp.asarray(np.zeros(7),jnp.float64);goal=jnp.asarray(np.zeros(3),jnp.float64)
        p=jnp.asarray(np.zeros((64,2)),jnp.float64);rm=jnp.zeros(64,bool);cursor=jnp.asarray(np.asarray(0.,np.float64),jnp.float64)
        def sense(x,o,mask,bx,bo,noise,ix,io):return observe(x,o,mask,bx,bo,noise,ix,io)
        def route(x,goal,o,mask,p,rm,cursor,noise):
            target,progress,remaining,visible=flight_target(x,goal,guidance_obstacles(o,mask,noise),mask,p,rm,cursor,c)
            return target,progress,remaining,visible,controller_obstacles(o,mask,noise)
        def advance(x,u,o,mask):
            y,sub=integrate_quad3d(x,u,c.robot)
            t=jnp.asarray(np.arange(1,c.robot.integration_substeps+1)/c.robot.integration_substeps*c.robot.dt,x.dtype)
            centers=o[None,:,:2]+t[:,None,None]*o[None,:,3:5]
            clear=jnp.min(jnp.where(mask[None],jnp.linalg.norm(sub[:,None,:2]-centers,axis=-1)-c.robot.radius-o[None,:,2],jnp.inf))
            limits=jnp.asarray(np.array([c.tilt_limit,c.tilt_limit,c.yaw_limit]+[c.velocity_limit]*3+[c.rate_limit]*3),x.dtype)
            envelope=jnp.maximum(jnp.max(abs(sub[:,3:])-limits),jnp.maximum(jnp.max(c.altitude_min-sub[:,2]),jnp.max(sub[:,2]-c.altitude_max)))
            return y,clear,envelope
        self.functions={};start=time.perf_counter()
        for name,fn,args in [('sense',sense,(x,o,mask,x,o,noise,x,o)),('route',route,(x,goal,o,mask,p,rm,cursor,noise)),
                            ('advance',advance,(x,jnp.asarray(np.zeros(4),jnp.float64),o,mask)),
                            ('arrived',lambda x,g,n:observed_arrived(x,g,n,c),(x,goal,noise))]:
            f=jax.jit(fn);self.functions[name]=f;setattr(self,name,f.lower(*args).compile())
        self.compile_seconds=time.perf_counter()-start

    def cache_sizes(self):return {k:f._cache_size() for k,f in self.functions.items()}

def execute64(executable,*values):
    # NumPy float64 arguments otherwise canonicalize to FP32 on an AOT call
    # when global x64 is disabled. Explicit device dtypes preserve the contract.
    arrays=[np.asarray(v) for v in values]
    return executable(*(jnp.asarray(v,dtype=jnp.bool_ if v.dtype==bool else jnp.float64) for v in arrays))

def empty_result():
    return dict(control=np.zeros(4),omega=np.zeros(2),states=np.full((11,12),np.nan),controls=np.full((10,4),np.nan),omegas=np.full((10,2),np.nan),
        feasible=False,solver_success=False,solver_status='not_attempted',iterations=0,solve_seconds=0.,
        max_equality_error=np.inf,max_constraint_violation=np.inf,objective=np.nan)

def episode(p,solver,kernels,steps=1600,ordered=False):
    c=solver.config;mask=np.asarray(p['mask'],bool);original=np.asarray(p['obstacles'],float);x=np.asarray(p['x'],float)
    goal=np.asarray(p['goal'],float);noise=np.asarray(p['noise'],float);points=np.asarray(p['route']['points'],float);rm=np.asarray(p['route']['mask'],bool)
    leg=0
    if ordered:
        goals=np.asarray(p['waypoint_goals'],float);total=p['waypoint_count'];goal=goals[0]
        routes=np.asarray(p['waypoint_routes']['points'],float);route_masks=np.asarray(p['waypoint_routes']['mask'],bool);points=routes[0];rm=route_masks[0]
    bx,bo,ix,io=unit_tape(p['sensor_seed'],steps,len(mask));cursor=0.;previous=np.zeros(4);solver.last_omega=np.zeros(2)
    clear,bound=initial_physical(x,original,mask,c);status=4 if clear<=0 else 5 if bound>c.qp_tolerance else 0
    count=0;records=[];start=time.perf_counter()
    for k in range(steps):
        truth=original.copy();truth[:,:2]+=k*c.robot.dt*original[:,3:5]
        seen,so=map(np.asarray,execute64(kernels.sense,x,truth,mask,bx,bo,noise,ix[k],io[k]))
        mission_info={}
        if ordered:
            handoff=status==0 and leg<total-1 and bool(execute64(kernels.arrived,seen,goals[leg],noise))
            if handoff:leg+=1;cursor=0.
            goal=goals[leg];points=routes[leg];rm=route_masks[leg]
            mission_info=dict(waypoint_index=leg,waypoint_handoff=handoff,mission_goal=goal)
        target,progress,remaining,visible,controlled=map(np.asarray,execute64(kernels.route,seen,goal,so,mask,points,rm,np.asarray(cursor,np.float64),noise))
        if status==0 and (not ordered or leg==total-1) and bool(execute64(kernels.arrived,seen,goal,noise)):status=1
        result=empty_result();attempted=status==0;active=False;before=x.copy();cursor_before=cursor;previous_before=previous.copy();omega_before=solver.last_omega.copy()
        if attempted:
            try:result=solver.solve(seen,target,controlled,mask,previous)
            except RuntimeError as error:result['solver_status']='exception:'+str(error)
            if result['feasible']:
                active=True;previous=result['control'];solver.last_omega=result['omega'];cursor=float(progress)
                x,clear,bound=map(np.asarray,execute64(kernels.advance,x,previous,truth,mask));count+=1
                if clear<=0:status=4
                elif bound>c.qp_tolerance:status=5
            else:status=3
        records.append(dict(state=before,next_state=x.copy(),observed=seen,control=previous.copy() if active else np.zeros(4),
            active=active,status=status,attempted=attempted,clearance=clear,envelope=bound,route_target=target,
            route_cursor_before=cursor_before,route_progress=cursor,route_remaining=remaining,route_visible=visible,
            previous_control=previous_before,previous_omega=omega_before,**mission_info,**{'mpc_'+key:value for key,value in result.items()}))
        if status:break
    if status==0:
        truth=original.copy();truth[:,:2]+=steps*c.robot.dt*original[:,3:5]
        seen,_=execute64(kernels.sense,x,truth,mask,bx,bo,noise,ix[steps],io[steps]);status=1 if (not ordered or leg==total-1) and bool(execute64(kernels.arrived,seen,goal,noise)) else 6
    assert all(v==0 for v in kernels.cache_sizes().values())
    trace={k:np.asarray([r[k] for r in records]) for k in records[0]}
    row=dict(id=p['id'],family=p['family'],noise_level=p['noise_level'],method=solver.method,status=status,steps=count,
        final_state=x.tolist(),elapsed_seconds=time.perf_counter()-start,solver_attempts=int(trace['attempted'].sum()),
        solver_seconds=float(trace['mpc_solve_seconds'].sum()),solver_iterations=int(trace['mpc_iterations'].sum()),
        termination_solver_status=str(trace['mpc_solver_status'][-1]),implicit_jit_cache_entries=kernels.cache_sizes())
    if ordered:row.update(waypoint_index=leg,waypoints_visited=leg+int(status==1))
    return row,trace
