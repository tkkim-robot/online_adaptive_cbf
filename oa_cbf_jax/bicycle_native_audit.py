"""Independent NumPy complex-step rows and SciPy physical replay for native QPs."""
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from .bicycle_audit import reference_flow
from .bicycle_control import BicycleControlConfig
from .bicycle_experiment import control_config
from .bicycle_rollout import GOAL,COLLISION,INFEASIBLE,TIMEOUT,STATE_BOUND
from .dataset import sha256


def reference_barrier(x,o,c=BicycleControlConfig()):
    p=o[:2]-x[:2];v=o[3:5]-x[3]*np.array([np.cos(x[2]),np.sin(x[2])])
    distance=np.sqrt(p@p);unit=p/distance
    radial=unit@v;lateral=unit[0]*v[1]-unit[1]*v[0]
    radius=(c.radius+o[2])*1.05;domain=p@p-radius**2
    # Native clamp is part of h. Its derivative is zero on the interior branch.
    root=np.sqrt(domain if domain.real>1e-6 else 1e-6)
    shape=np.sqrt(1.05**2-1)/radius
    return radial+.5*shape*root*lateral*lateral/np.sqrt(v@v)+shape*root


def reference_gradient(x,o,c=BicycleControlConfig()):
    return np.array([reference_barrier(x.astype(complex)+1j*1e-25*np.eye(4)[i],o,c).imag/1e-25 for i in range(4)])


def reference_obstacle_gradient(x,o,c=BicycleControlConfig()):
    # Differentiate obstacle coordinates independently; do not assume the
    # negative-robot-gradient identity used by the runtime implementation.
    return np.array([reference_barrier(x,o.astype(complex)+1j*1e-25*np.eye(5)[i],c).imag/1e-25 for i in range(2)])


def reference_problem(x,goal,obs,mask,method,c=BicycleControlConfig()):
    """Rebuild the original inequalities without the runtime AD/row builder."""
    od=method=='optimal_decay';alpha=.1 if method!='fixed_high' else 70.
    selected=np.flatnonzero(mask)
    selected=selected[np.argsort(np.sqrt(np.sum((obs[selected,:2]-x[:2])**2,axis=1)))[:1 if od else 5]]
    slots=1 if od else 10;A=np.zeros((slots+4,3 if od else 2));b=np.zeros(slots+4)
    h=np.array([reference_barrier(x,o,c) for o in obs[selected]])
    grad=np.array([reference_gradient(x,o,c) for o in obs[selected]]).reshape(-1,4)
    f=reference_flow(0,x,np.zeros(2),c)
    g=np.stack([reference_flow(0,x,u,c)-f for u in np.eye(2)],axis=1)
    obstacle_grad=np.array([reference_obstacle_gradient(x,o,c) for o in obs[selected]]).reshape(-1,2)
    A[:len(selected),:2]=-grad@g
    b[:len(selected)]=grad@f+np.sum(obstacle_grad*obs[selected,3:5],axis=1)
    if od:A[:len(selected),2]=-alpha*h
    else:b[:len(selected)]+=alpha*h
    A[slots:,:2]=[[1,0],[-1,0],[0,1],[0,-1]]
    b[slots:]=[c.robot.acceleration_max,c.robot.acceleration_max,c.robot.slip_max,c.robot.slip_max]
    error=(np.arctan2(goal[1]-x[1],goal[0]-x[0])-x[2]+np.pi)%(2*np.pi)-np.pi
    steering=np.clip((3. if od else 2.)*error,-c.robot.steering_max,c.robot.steering_max)
    slip=np.arctan(c.robot.rear_axle_distance/c.robot.wheel_base*np.tan(steering))
    distance=max(np.sqrt(np.sum((goal-x[:2])**2))-.05,.05)
    desired=np.clip((.5 if od else 1.)*distance*max(np.cos(error),0.),c.robot.speed_min,c.robot.speed_max)
    ref=np.array([(.5 if od else 1.)*(desired-x[3]),slip]+([1.] if od else []))
    return ref,A,b,selected,h,grad


def audit_episode(parent,d,row,c=BicycleControlConfig()):
    r=c.robot;method=row['method'];physical=np.asarray(parent['initial'],float)
    acceptance=row.get('acceptance','strict_rows');assert acceptance in ('strict_rows','native_status')
    obs=np.asarray(parent['obstacles'],float);mask=np.asarray(parent['mask'],bool)
    goal=np.asarray(parent['goal'],np.float32).astype(float);noise=np.asarray(parent['noise'],np.float32).astype(float)
    bx,bo=[np.asarray(parent[k],float) for k in ('bias_x','bias_o')]
    for key in ('initial','goal','obstacles','mask','bias_x','bias_o','first_x','first_o','noise'):
        np.testing.assert_array_equal(d[key],np.asarray(parent[key]))
    np.testing.assert_array_equal(d['key'],[0,(parent['seed']+4)%2**32])
    np.testing.assert_array_equal(d['observed_state'][0],parent['first_x'])
    np.testing.assert_array_equal(d['observed_obstacles'][0],parent['first_o'])
    n=len(d['active']);assert n>=1 and not np.any(np.diff(d['active'].astype(int))>0)
    status=int(d['final_status']);assert status==row['status_code']
    minimum=float(np.min(np.where(mask,np.linalg.norm(obs[:,:2]-physical[:2],axis=1)-r.radius-obs[:,2],np.inf)))
    count=0;worst=-np.inf;max_error=0.;numerical_stops=0;row_violations=0;input_violations=0
    for k,applied in enumerate(d['active']):
        np.testing.assert_allclose(d['state_before'][k],physical,atol=3e-5,rtol=2e-6)
        sx=d['observed_state'][k].astype(float);so=d['observed_obstacles'][k].astype(float)
        ix,io=d['innovation_x'][k],d['innovation_o'][k]
        assert np.linalg.norm(ix[:2])<=1+1e-6 and np.max(np.abs(ix[2:]))<=1+1e-6
        assert np.max(np.linalg.norm(io[:,:2],axis=1))<=1+1e-6 and np.max(np.linalg.norm(io[:,3:5],axis=1))<=1+1e-6 and np.max(np.abs(io[:,2]))<=1+1e-6
        current=obs.copy();current[:,:2]+=k*r.dt*current[:,3:5]
        if k:
            np.testing.assert_allclose(sx,(physical+bx+.15*noise[[0,0,1,2]]*ix).astype(np.float32),atol=3e-6,rtol=2e-6)
            expected=(current+bo+.15*noise[[3,3,5,4,4]]*io).astype(np.float32);expected[~mask]=0.
            np.testing.assert_allclose(so,expected,atol=3e-6,rtol=2e-6)
        else:
            np.testing.assert_array_equal(ix,0.);np.testing.assert_array_equal(io,0.)
            assert np.linalg.norm(sx[:2]-physical[:2])<=noise[0]+5e-6
        if not bool(d['attempted'][k]):
            assert k==0 and not applied and status in (GOAL,COLLISION,STATE_BOUND)
            np.testing.assert_array_equal(d['control'][k],0.);continue
        with np.errstate(divide='ignore',invalid='ignore'):
            target=d['mission_goals'][int(d['mission_leg'][k])].astype(float) if 'mission_leg' in d else goal
            ref,A,b,selected,h,grad=reference_problem(sx,target,so,mask,method,c)
        slots=np.full(10,-1);slots[:len(selected)]=selected
        np.testing.assert_array_equal(d['qp_selected'][k],slots)
        np.testing.assert_allclose(d['qp_reference'][k],ref,atol=3e-12,rtol=3e-12)
        np.testing.assert_allclose(d['qp_barrier'][k,selected],h,atol=3e-9,rtol=3e-10,equal_nan=True)
        np.testing.assert_allclose(d['qp_gradient'][k,selected],grad,atol=3e-8,rtol=3e-10,equal_nan=True)
        solution=d['qp_solution'][k].astype(float);stored=solution.copy();stored[:2]=solution[:2].astype(np.float32)
        finite=all(np.isfinite(v).all() for v in (A,b,solution))
        if finite:
            raw=float(np.max(A@solution-b));residual=float(np.max(A@stored-b))
            np.testing.assert_allclose([d['qp_raw_residual'][k],d['qp_stored_residual'][k]],[raw,residual],atol=3e-8,rtol=3e-10)
        expected_accept=bool(d['qp_status_value'][k]==1 and finite) if acceptance=='native_status' else bool(d['qp_status_value'][k] in (1,2) and finite and max(d['qp_raw_residual'][k],d['qp_stored_residual'][k])<=c.qp_tolerance)
        assert bool(applied)==expected_accept==bool(d['qp_feasible'][k])
        if not applied:
            assert status==INFEASIBLE and k==n-1
            np.testing.assert_array_equal(d['control'][k],0.)
            np.testing.assert_allclose(d['state'][k],physical,atol=3e-5,rtol=2e-6)
            numerical_stops+=int(d['qp_status_value'][k] in (1,2));break
        if acceptance=='strict_rows':assert residual<=c.qp_tolerance+3e-8
        worst=max(worst,residual)
        row_violations+=int(d['qp_stored_residual'][k]>c.qp_tolerance)
        input_violations+=int(np.any(np.abs(d['control'][k])>np.array([r.acceleration_max,r.slip_max])+c.qp_tolerance))
        np.testing.assert_array_equal(d['control'][k],solution[:2].astype(np.float32))
        u=d['control'][k];times=np.linspace(0,r.dt,101)
        sol=solve_ivp(lambda t,s:reference_flow(t,s,u,c),(0,r.dt),physical,method='DOP853',rtol=1e-11,atol=1e-12,t_eval=times)
        assert sol.success;error=float(np.max(np.abs(sol.y[:,-1]-d['state'][k])));max_error=max(max_error,error)
        np.testing.assert_allclose(d['state'][k],sol.y[:,-1],atol=3e-5,rtol=2e-6)
        centers=current[None,:,:2]+times[:,None,None]*current[None,:,3:5]
        clear=float(np.min(np.where(mask,np.linalg.norm(sol.y[:2].T[:,None]-centers,axis=-1)-r.radius-current[None,:,2],np.inf)))
        if np.isfinite(clear):np.testing.assert_allclose(clear,d['clearance'][k],atol=1e-4,rtol=0)
        violation=max(r.speed_min-sol.y[3].min(),sol.y[3].max()-r.speed_max)
        np.testing.assert_allclose(violation,d['state_violation'][k],atol=2e-6,rtol=0)
        if clear<-1e-4:assert status==COLLISION and k==n-1
        if violation>c.qp_tolerance+1e-6:assert status in (STATE_BOUND,COLLISION) and k==n-1
        physical=sol.y[:,-1];minimum=min(minimum,clear);count+=1
    assert count==row['steps']==int(d['expected_steps'])
    if status==TIMEOUT:assert count==int(d['horizon'])
    if status==GOAL:assert np.linalg.norm(physical[:2]-goal)<=c.goal_tolerance+1e-4 and physical[3]<=c.terminal_speed+1e-5
    if status==COLLISION:assert minimum<=1e-4
    if status==STATE_BOUND:assert max(r.speed_min-physical[3],physical[3]-r.speed_max)>c.qp_tolerance-1e-6
    if row['min_clearance'] is not None:np.testing.assert_allclose(minimum,row['min_clearance'],atol=1e-4,rtol=0)
    if 'applied_row_violation_ticks' in row:assert row_violations==row['applied_row_violation_ticks']
    if 'applied_input_violation_ticks' in row:assert input_violations==row['applied_input_violation_ticks']
    return dict(audit_passed=True,steps=count,max_state_error=max_error,
        min_clearance=None if not np.isfinite(minimum) else minimum,max_qp_residual=None if not np.isfinite(worst) else worst,numerical_residual_stops=numerical_stops,
        applied_row_violation_ticks=row_violations,applied_input_violation_ticks=input_violations)


def audit_one(task):
    directory,row,m,parent=task;path=Path(directory)/row['file']
    assert sha256(path)==row['sha256'] and row['group_id']==parent['group_id'] and row['family']==parent['family']
    assert row['method']==m['method'] and parent['partition']=='policy_audit'
    assert row.get('acceptance','strict_rows')==m.get('acceptance','strict_rows')
    with np.load(path) as z:d={k:z[k] for k in z.files}
    assert int(d['horizon'])==m['steps']
    return dict(group_id=row['group_id'],trace_sha256=row['sha256'],**audit_episode(parent,d,row,control_config(m['config'])))
