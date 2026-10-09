"""Discrete mpc functions and shared contracts."""

from dataclasses import asdict

import time

import numpy as np

from .config import UnicycleConfig

DEFAULTS={'fixed_low':(.01,.01),'fixed_high':(.35,.35),'optimal_decay':(.01,.01)}

def numpy_euler(x,u,dt):
    x=np.asarray(x,float);u=np.asarray(u,float)
    return x+dt*np.asarray([x[3]*np.cos(x[2]),x[3]*np.sin(x[2]),u[1],u[0]])

def numpy_barrier(x,u,obs,gains,omega,robot,buffer=None):
    """Independent NumPy version of the two repeated-input Euler differences."""
    obs=np.asarray(obs,float);a=np.asarray(gains)*np.asarray(omega)
    radius=robot.radius+(robot.clearance_buffer if buffer is None else buffer)
    x1=numpy_euler(x,u,robot.dt);x2=numpy_euler(x1,u,robot.dt)
    h=lambda y:np.sum((y[:2]-obs[:,:2])**2,axis=-1)-1.01*(radius+obs[:,2])**2
    h0,h1,h2=h(x),h(x1),h(x2)
    return h2-2*h1+h0+a.sum()*(h1-h0)+a.prod()*h0

class DiscreteMPC:
    def __init__(self,capacity,method='fixed_low',robot=UnicycleConfig(),*,shared_contract=True):
        import casadi as ca
        if method not in DEFAULTS or isinstance(capacity,bool) or not isinstance(capacity,int) or capacity<1:
            raise ValueError('Unknown discrete MPC method/capacity')
        self.capacity=capacity;self.method=method;self.robot=robot;self.horizon=10
        self.gains=np.asarray(DEFAULTS[method]);self.decay=method=='optimal_decay';self.shared_contract=shared_contract
        self.last_omega=np.zeros(2);H=self.horizon;C=capacity
        X=ca.SX.sym('x',4,H+1);U=ca.SX.sym('u',2,H)
        W=ca.SX.sym('omega',2,H) if self.decay else ca.DM.ones(2,H)
        initial=ca.SX.sym('initial',4);goal=ca.SX.sym('goal',2);obs=ca.SX.sym('obs',C,5)
        active=ca.SX.sym('active',C);previous=ca.SX.sym('previous',2);speed_error=ca.SX.sym('speed_error')
        parameters=ca.vertcat(initial,goal,ca.vec(obs),active,previous,speed_error)
        variables=ca.vertcat(ca.vec(X),ca.vec(U),ca.vec(W)) if self.decay else ca.vertcat(ca.vec(X),ca.vec(U))
        goal_state=ca.vertcat(goal,0,0);Q=ca.DM([50,50,.01,30])
        equality=[X[:,0]-initial];inequality=[];cost=0
        def step(x,u):return x+robot.dt*ca.vertcat(x[3]*ca.cos(x[2]),x[3]*ca.sin(x[2]),u[1],u[0])
        def barrier(x,u,omega):
            radius=robot.radius+(robot.clearance_buffer if shared_contract else 0.)
            h=lambda y:(obs[:,0]-y[0])**2+(obs[:,1]-y[1])**2-1.01*(obs[:,2]+radius)**2
            x1=step(x,u);x2=step(x1,u);h0,h1,h2=h(x),h(x1),h(x2)
            a1,a2=self.gains[0]*omega[0],self.gains[1]*omega[1]
            return h2-2*h1+h0+(a1+a2)*(h1-h0)+a1*a2*h0
        for k in range(H):
            error=X[:,k]-goal_state;cost+=ca.dot(Q,error**2)
            if self.decay:
                # Pinned OptimalDecayMPCCBF calls set_rterm twice. do-mpc5.1.2
                # replaces the first custom term: its effective default is
                # the two omega penalties only, not an extra control cost.
                cost+=10*ca.sumsqr(W[:,k]-1)
            else:
                last=previous if k==0 else U[:,k-1]
                cost+=.5*ca.sumsqr(U[:,k]-last)
            equality.append(X[:,k+1]-step(X[:,k],U[:,k]))
            inequality.append(active*barrier(X[:,k],U[:,k],W[:,k])+(1-active))
            lower=initial[3]-ca.fmax(0.,initial[3]-speed_error) if shared_contract else -robot.v_max
            upper=robot.v_max+initial[3]-ca.fmin(robot.v_max,initial[3]+speed_error) if shared_contract else robot.v_max
            inequality.append(ca.vertcat(X[3,k+1]-lower,upper-X[3,k+1]))
        cost+=ca.dot(Q,(X[:,H]-goal_state)**2)
        constraints=ca.vertcat(*equality,*inequality);self.equalities=4*(H+1)
        self.lbg=np.r_[np.zeros(self.equalities),np.zeros(int(constraints.numel())-self.equalities)]
        self.ubg=np.r_[np.zeros(self.equalities),np.full(int(constraints.numel())-self.equalities,np.inf)]
        self.lbx=np.full(int(variables.numel()),-np.inf);self.ubx=-self.lbx.copy()
        lower_u=np.tile([-robot.a_max,-robot.w_max],H);self.lbx[4*(H+1):4*(H+1)+2*H]=lower_u
        self.ubx[4*(H+1):4*(H+1)+2*H]=-lower_u
        self.solver_options={'ipopt.print_level':0,'print_time':False}
        start=time.perf_counter()
        self.solver=ca.nlpsol('discrete_mpc','ipopt',dict(x=variables,p=parameters,f=cost,g=constraints),self.solver_options)
        self.setup_seconds=time.perf_counter()-start
        self.expression=ca.Function('discrete_mpc_expressions',[variables,parameters],[cost,constraints])
        sx=ca.SX.sym('state',4);su=ca.SX.sym('control',2);sw=ca.SX.sym('omega',2)
        self.barrier_expression=ca.Function('discrete_barrier',[sx,su,obs,sw],[barrier(sx,su,sw)])

    def contract(self):
        return dict(method=self.method,horizon=self.horizon,gains=self.gains.tolist(),Q=[50,50,.01,30],
            input_cost='10*(omega1-1)^2+10*(omega2-1)^2; pinned effective custom rterm' if self.decay else '.5*sum((u[k]-u[k-1])^2)',
            omega_bounds='unrestricted' if self.decay else 'fixed1',obstacle_capacity=self.capacity,
            barrier='h[k+2]-2*h[k+1]+h[k]+(alpha1*omega1+alpha2*omega2)*(h[k+1]-h[k])+alpha1*alpha2*omega1*omega2*h[k]',
            predictor='Euler, same control repeated in the two-step barrier difference, observed obstacle centers held fixed across horizon as in the pinned repository. All active rows included; inactive rows1>=0.',
            radius_beta=1.01,shared_contract=self.shared_contract,robot=asdict(self.robot),solver='CasADi3.7.2/IPOPT',solver_options=self.solver_options,
            adaptation='All-obstacle masked capacity and shared0..v_max speed contract; future speed tightened by the sensed speed interval intersected with physical bounds. Shared CBF clearance_buffer added to robot radius. Initial measurement is not projected. These declared common-contract changes are not a literal five-obstacle legacy run.',
            initialization='Repeat current sensed state and previous applied command, as repository set_initial_guess; omega starts0 at each new episode, then previous applied omega. No multiplier warm-start or solver tuning.')

    def pack(self,x,goal,obs,mask,previous,speed_error):
        x=np.asarray(x,float);goal=np.asarray(goal,float);obs=np.asarray(obs,float);mask=np.asarray(mask,bool);previous=np.asarray(previous,float)
        if x.shape!=(4,) or goal.shape!=(2,) or obs.shape!=(self.capacity,5) or mask.shape!=(self.capacity,) or previous.shape!=(2,):
            raise ValueError('Discrete MPC input shape mismatch')
        packed=np.r_[x,goal,obs.ravel(order='F'),mask.astype(float),previous,speed_error]
        if not np.isfinite(packed).all() or speed_error<0 or speed_error>=self.robot.v_max/2:raise ValueError('Invalid MPC observation/uncertainty')
        return packed

    def solve(self,x,goal,obs,mask,previous,speed_error=0.):
        H=self.horizon;p=self.pack(x,goal,obs,mask,previous,speed_error)
        guess=np.r_[np.tile(np.asarray(x,float),H+1),np.tile(previous,H)]
        if self.decay:guess=np.r_[guess,np.tile(self.last_omega,H)]
        start=time.perf_counter();answer=self.solver(x0=guess,p=p,lbx=self.lbx,ubx=self.ubx,lbg=self.lbg,ubg=self.ubg)
        elapsed=time.perf_counter()-start;stats=self.solver.stats();v=np.asarray(answer['x']).reshape(-1)
        X=v[:4*(H+1)].reshape(H+1,4);U=v[4*(H+1):4*(H+1)+2*H].reshape(H,2)
        W=v[-2*H:].reshape(H,2) if self.decay else np.ones((H,2))
        finite=np.isfinite(v).all();eq=np.inf;ineq=np.inf
        if finite:
            eq=max(float(np.max(np.abs(X[0]-x))),max(float(np.max(np.abs(X[k+1]-numpy_euler(X[k],U[k],self.robot.dt)))) for k in range(H)))
            residuals=np.stack([numpy_barrier(X[k],U[k],obs,self.gains,W[k],self.robot,buffer=None if self.shared_contract else 0.) for k in range(H)])
            cbf=float(np.max(-residuals[:,np.asarray(mask,bool)],initial=-np.inf))
            low=x[3]-max(0.,x[3]-speed_error) if self.shared_contract else -self.robot.v_max
            high=self.robot.v_max+x[3]-min(self.robot.v_max,x[3]+speed_error) if self.shared_contract else self.robot.v_max
            ineq=max(cbf,float(np.max(low-X[1:,3])),float(np.max(X[1:,3]-high)),float(np.max(np.abs(U)-[self.robot.a_max,self.robot.w_max])))
        feasible=bool(stats.get('success',False) and finite and eq<=self.robot.qp_tolerance and ineq<=self.robot.qp_tolerance)
        return dict(control=U[0],omega=W[0],states=X,controls=U,omegas=W,feasible=feasible,
            solver_success=bool(stats.get('success',False)),
            solver_status=str(stats.get('return_status')),iterations=int(stats.get('iter_count',-1)),solve_seconds=elapsed,
            max_equality_error=eq,max_constraint_violation=ineq,objective=float(answer['f']))


import hashlib


import jax


from .simulation import RUNNING, GOAL, COLLISION, INFEASIBLE, TIMEOUT, STATUS_NAMES

from .stochastic import STATE_BOUND_VIOLATION

PLANNER_FAILURE=7

NAMES={**STATUS_NAMES,PLANNER_FAILURE:'planner_failure',STATE_BOUND_VIOLATION:'state_bound_violation'}

def check_applied(sensed,control,obstacles,mask,omega,solver,speed_error):
    """Recheck the actual FP32 command; reject rather than clip a bad action."""
    robot=solver.robot
    residual=numpy_barrier(sensed,control,obstacles,solver.gains,omega,robot)
    low=max(0.,float(sensed[3])-speed_error)+robot.dt*float(control[0])
    high=min(robot.v_max,float(sensed[3])+speed_error)+robot.dt*float(control[0])
    violation=max(float(np.max(-residual[np.asarray(mask,bool)],initial=-np.inf)),
        float(np.max(np.abs(control)-[robot.a_max,robot.w_max])),-low,high-robot.v_max)
    return bool(np.isfinite(control).all() and np.isfinite(omega).all() and violation<=robot.qp_tolerance),violation

def episode(record,solver,kernels,steps,noise_scale,ordered=False):
    robot=solver.robot;scene=record['scene'];f=lambda a:np.asarray(a,np.float32)
    mask=np.asarray(scene['obstacle_mask'],bool);observed=f(scene['initial_state']);obstacles=f(scene['obstacles'])
    if ordered:
        count=record['waypoint_count'];goals=f(record['waypoint_goals']);routes=record['waypoint_routes']
        points=f(routes['points']);rmask=np.asarray(routes['mask'],bool);ready=np.asarray(routes['ready'],bool)
    else:
        count=1;goals=f([scene['goal']]);points=f([record['route']['points']])
        rmask=np.asarray([record['route']['mask']],bool);ready=np.asarray([record['route']['status']=='ready'])
    noise=noise_scale*np.array([.02,.03,.02,.03,.025,.01]);fn=f(noise)
    seed=int.from_bytes(hashlib.sha256(scene['scene_id'].encode()).digest()[:4],'little');key=np.asarray(jax.random.PRNGKey(seed))
    x,truth_obs,xb,ob,xs,os,innovations=kernels.prepare(observed,obstacles,mask,fn,key)
    initial=np.asarray(x);truth=np.asarray(truth_obs);minimum=float(kernels.clearance(x,truth_obs,mask))
    status=PLANNER_FAILURE if not ready[0] else COLLISION if minimum<=0 else GOAL if count==1 and bool(kernels.arrived(x,goals[0],np.zeros(6,np.float32))) else RUNNING
    leg=0;cursor=np.float32(0);previous=np.zeros(2,np.float32);solver.last_omega=np.zeros(2)
    traces=[];applied=0;solve_times=[];step_times=[];solver_statuses={};start=time.perf_counter()
    for k in range(steps):
        tick=time.perf_counter();sx,so=kernels.sense(x,truth_obs,xb,ob,xs,os,innovations[k],np.int32(k))
        sx,so=np.asarray(sx),np.asarray(so)
        handoff=status==RUNNING and leg<count-1 and bool(kernels.arrived(sx,goals[leg],fn))
        if handoff:leg+=1;cursor=np.float32(0)
        if status==RUNNING and not ready[leg]:status=PLANNER_FAILURE
        attempt=status==RUNNING;result=None;u=np.zeros(2,np.float32);omega=np.ones(2);clear=bound=violation=np.nan
        target,proposal,remaining=kernels.target(sx,points[leg],rmask[leg],cursor)
        if attempt:
            # Pinned tracking.update_goal and MPCCBF.solve_control_problem use
            # the current mandatory waypoint, not a receding route lookahead.
            result=solver.solve(sx,goals[leg],so,mask,previous,speed_error=1.15*float(fn[2]))
            solve_times.append(result['solve_seconds']);label=result['solver_status'];solver_statuses[label]=solver_statuses.get(label,0)+1
            candidate=f(result['control']);omega=result['omega']
            accepted,violation=check_applied(sx,candidate,so,mask,omega,solver,1.15*float(fn[2]))
            accepted=accepted and result['feasible']
            if accepted:
                u=candidate;x,clear,bound=kernels.advance(x,u,truth_obs,mask,np.int32(k))
                clear,bound=float(clear),float(bound);minimum=min(minimum,clear)
                cursor=np.float32(proposal);previous=u.copy();solver.last_omega=omega.copy();applied+=1
                if leg==count-1 and bool(kernels.arrived(x,goals[leg],np.zeros(6,np.float32))):status=GOAL
                if bound>robot.qp_tolerance:status=STATE_BOUND_VIOLATION
                if clear<=0:status=COLLISION
            else:status=INFEASIBLE
        else:accepted=False
        traces.append(dict(state=np.asarray(x),control=u,active=accepted,status=status,gains=solver.gains,
            omega=omega,observed_state=sx,observed_obstacles=so,clearance=clear,state_bound_violation=bound,
            qp_violation=violation,route_target=np.asarray(target),mpc_reference=goals[leg],route_remaining=float(remaining),route_progress=cursor,
            selection_tick=attempt,selection_accepted=accepted,solver_status='' if result is None else result['solver_status'],
            solver_success=False if result is None else result['solver_success'],
            solver_iterations=-1 if result is None else result['iterations'],solver_seconds=np.nan if result is None else result['solve_seconds'],
            predicted_states=np.full((11,4),np.nan) if result is None else result['states'],
            predicted_controls=np.full((10,2),np.nan) if result is None else result['controls'],
            predicted_omegas=np.full((10,2),np.nan) if result is None else result['omegas'],
            solver_equality_error=np.nan if result is None else result['max_equality_error'],
            solver_constraint_violation=np.nan if result is None else result['max_constraint_violation'],
            waypoint_index=leg,waypoint_handoff=handoff,waypoints_visited=leg+int(status==GOAL),mission_goal=goals[leg],
            collision_event=accepted and clear<=0,state_bound_event=accepted and bound>robot.qp_tolerance))
        step_times.append(time.perf_counter()-tick)
        if status!=RUNNING:break
    if status==RUNNING:status=TIMEOUT
    trace={k:np.stack([t[k] for t in traces]) for k in traces[0]}
    row=dict(scene_id=scene['scene_id'],family=scene['family'],mode='discrete_mpc_'+solver.method,status=NAMES[status],steps=applied,
        min_clearance=minimum,goal_progress=float(np.linalg.norm(initial[:2]-goals[count-1])-np.linalg.norm(np.asarray(x)[:2]-goals[count-1])),
        final_route_coordinate=float(kernels.coordinate(x[:2],points[leg],rmask[leg],cursor)),
        applied_source_counts={'default_discrete_mpc':applied},selection_attempts=len(solve_times),
        selection_rejections=int(status==INFEASIBLE),reactive_reselections=0,gate_stage_totals=[],gain_total_variation=0.,
        max_physical_bound_violation=float(np.max(trace['state_bound_violation'][trace['active']])) if applied else None,
        solver_status_counts=solver_statuses,elapsed_seconds=time.perf_counter()-start,
        solver_seconds=sum(solve_times),tick_seconds_p50_p99_max=np.quantile(step_times,[.5,.99,1]).tolist(),
        collision_event=bool(np.any(trace['collision_event'])) or minimum<=0,state_bound_event=bool(np.any(trace['state_bound_event'])))
    if status==INFEASIBLE:
        row['termination_reason']=('solver_reported_failure:'+result['solver_status'] if not result['solver_success'] else
            'independent_prediction_residual_rejection' if not result['feasible'] else 'applied_command_residual_rejection')
    if ordered:row.update(waypoint_index=leg,waypoints_visited=leg+int(status==GOAL),required_waypoints=count,waypoint_handoffs=int(trace['waypoint_handoff'].sum()))
    return row,trace,dict(true_initial_state=initial,true_obstacles=truth,scene_id=scene['scene_id'],noise=noise,key=key)


import jax.numpy as jnp


from .dynamics import integrate_unicycle, signed_clearance, swept_disk_clearance

from .routing import route_target, physical_route_coordinate

from .stochastic import conditioned_sensor_model

class PhysicalKernels:
    def __init__(self,capacity,route_capacity,steps,robot=UnicycleConfig()):
        # Keep the common physical prior, RNG and plant identical when an FP64
        # controller is hosted in this process. Only AOT compilation uses this
        # context; runtime executes the fixed FP32 signatures.
        with jax.enable_x64(False):
            self._compile(capacity,route_capacity,steps,robot)

    def _compile(self,capacity,route_capacity,steps,robot):
        x=jnp.zeros(4);obs=jnp.zeros((capacity,5));mask=jnp.zeros(capacity,bool)
        noise=jnp.zeros(6);key=jax.random.PRNGKey(0);points=jnp.zeros((route_capacity,2));rmask=jnp.ones(route_capacity,bool)
        begin=time.perf_counter()
        def prepare(x,obs,mask,noise,key):
            return conditioned_sensor_model(x,obs,mask,noise,key,robot,steps)
        def sense(x,truth_obs,xb,ob,xs,os,innovation,k):
            physical_obs=truth_obs.at[:,:2].set(truth_obs[:,:2]+k*robot.dt*truth_obs[:,3:5])
            return x-xb+.15*xs*innovation[:4],physical_obs-ob+.15*os*innovation[4:].reshape(obs.shape)
        def advance(x,u,truth_obs,mask,k):
            y,sub=integrate_unicycle(x,u,robot.dt,robot.integration_substeps)
            starts=jnp.concatenate((x[None],sub[:-1]));times=k*robot.dt+jnp.arange(robot.integration_substeps)*robot.dt/robot.integration_substeps
            clear=jnp.min(jax.vmap(lambda a,b,t:swept_disk_clearance(a,b,truth_obs,mask,robot.radius,t,t+robot.dt/robot.integration_substeps))(starts,sub,times))
            return y,clear,jnp.maximum(-y[3],y[3]-robot.v_max)
        def arrived(x,goal,noise):
            return (jnp.linalg.norm(x[:2]-goal)+jnp.sqrt(2.)*1.15*noise[0]<=robot.goal_tolerance)&(jnp.abs(x[3])+1.15*noise[2]<=.2)
        self.prepare=jax.jit(prepare).lower(x,obs,mask,noise,key).compile()
        self.sense=jax.jit(sense).lower(x,obs,x,obs,x,jnp.zeros(5),jnp.zeros(4+capacity*5),jnp.int32(0)).compile()
        self.advance=jax.jit(advance).lower(x,jnp.zeros(2),obs,mask,jnp.int32(0)).compile()
        self.target=jax.jit(lambda x,p,m,c:route_target(x,p,m,c,robot)).lower(x,points,rmask,jnp.float32(0)).compile()
        self.coordinate=jax.jit(physical_route_coordinate).lower(x[:2],points,rmask,jnp.float32(0)).compile()
        self.clearance=jax.jit(lambda x,o,m:jnp.min(signed_clearance(x[:2],o,m,robot.radius))).lower(x,obs,mask).compile()
        self.arrived=jax.jit(arrived).lower(x,jnp.zeros(2),noise).compile()
        self.compile_seconds=time.perf_counter()-begin
