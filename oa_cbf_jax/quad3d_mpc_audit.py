"""Independent raw-sensor, NLP-prediction and continuous-motion MPC audit."""
from pathlib import Path
import numpy as np
from .quad3d_control import control_config
from .quad3d_observation import unit_tape,numpy_observe,numpy_obstacles,numpy_arrived
from .quad3d_routing import numpy_flight_target
from .quad3d_audit import audit_hold,coefficients
from .quad3d_mpc import DEFAULTS,Q,prediction_residual
from .quad3d_learning_contract import read
from .dataset import sha256


def audit_one(task):
    p,row,directory,m=task;out=Path(directory);c=control_config(m['config']);path=out/row['file']
    assert p['id']==row['id'] and sha256(path)==row['sha256']
    assert m['contract_sha256']==sha256(out/'contract.json');contract=read(out/'contract.json')
    assert contract['shared_contract'] and contract['coupled_decay'] and contract['method']==row['method']==m['method']
    assert contract['solver_options']=={'ipopt.print_level':0,'print_time':False}
    with np.load(path) as z:d=dict(z)
    original=np.asarray(p['obstacles']);mask=np.asarray(p['mask'],bool);noise=np.asarray(p['noise']);goal=np.asarray(p['goal'])
    points=np.asarray(p['route']['points']);rm=np.asarray(p['route']['mask'],bool)
    ordered=m.get('ordered_mission',False);leg=0
    if ordered:
        goals=np.asarray(p['waypoint_goals']);total=p['waypoint_count'];goal=goals[0]
        routes=np.asarray(p['waypoint_routes']['points']);route_masks=np.asarray(p['waypoint_routes']['mask']);points=routes[0];rm=route_masks[0]
    bx,bo,ix,io=unit_tape(p['sensor_seed'],m['steps'],len(mask))
    x=np.asarray(p['x']);cursor=0.;previous=np.zeros(4);previous_omega=np.zeros(2);count=0
    limits=np.array([c.tilt_limit,c.tilt_limit,c.yaw_limit]+[c.velocity_limit]*3+[c.rate_limit]*3)
    minimum=float(np.min(np.where(mask,np.linalg.norm(x[:2]-original[:,:2],axis=-1)-original[:,2]-c.robot.radius,np.inf)))
    envelope=max(float(np.max(abs(x[3:])-limits)),c.altitude_min-x[2],x[2]-c.altitude_max)>c.qp_tolerance
    collision=minimum<=0;status=4 if collision else 5 if envelope else 0;first_physical=status if status else None;first_step=0 if status else None
    max_replay=0.;max_equality=0.;max_violation=-np.inf
    for k in range(len(d['active'])):
        np.testing.assert_array_equal(d['state'][k],x);truth=original.copy();truth[:,:2]+=k*c.robot.dt*original[:,3:5]
        seen,so=numpy_observe(x,truth,mask,bx,bo,noise,ix[k],io[k]);np.testing.assert_allclose(d['observed'][k],seen,atol=2e-12,rtol=1e-12)
        if ordered:
            handoff=status==0 and leg<total-1 and numpy_arrived(seen,goals[leg],noise,c)
            if handoff:
                assert numpy_arrived(x,goals[leg],np.zeros(7),c),'Waypoint handoff was not true full-state arrival'
                leg+=1;cursor=0.
            goal=goals[leg];points=routes[leg];rm=route_masks[leg]
            assert d['waypoint_index'][k]==leg and d['waypoint_handoff'][k]==handoff
            np.testing.assert_array_equal(d['mission_goal'][k],goal)
        target,progress,remaining,visible=numpy_flight_target(seen,goal,numpy_obstacles(so,mask,noise,guidance=True),mask,points,rm,cursor,c)
        for name,value in [('route_target',target),('route_cursor_before',cursor),('route_remaining',remaining)]:
            np.testing.assert_allclose(d[name][k],value,atol=1e-8,rtol=1e-10)
        assert d['route_visible'][k]==visible
        np.testing.assert_array_equal(d['previous_control'][k],previous);np.testing.assert_array_equal(d['previous_omega'][k],previous_omega)
        if status==0 and (not ordered or leg==total-1) and numpy_arrived(seen,goal,noise,c):status=1
        assert d['attempted'][k]==(status==0)
        if d['attempted'][k]:
            states,controls,omegas=(d['mpc_'+name][k] for name in ('states','controls','omegas'))
            eq,violation=prediction_residual(states,controls,omegas,seen,numpy_obstacles(so,mask,noise),mask,DEFAULTS[row['method']],c)
            np.testing.assert_allclose(d['mpc_max_equality_error'][k],eq,atol=2e-10,rtol=1e-8)
            np.testing.assert_allclose(d['mpc_max_constraint_violation'][k],violation,atol=2e-10,rtol=1e-8)
            feasible=bool(d['mpc_solver_success'][k] and eq<=c.qp_tolerance and violation<=c.qp_tolerance)
            assert d['mpc_feasible'][k]==feasible
            if not feasible:status=3
            if np.isfinite(states).all() and np.isfinite(controls).all() and np.isfinite(omegas).all():
                objective=float(np.sum((states-np.r_[target,np.zeros(9)])**2*Q))
                objective+=float(10*np.sum((omegas-1)**2)) if row['method']=='optimal_decay' else float(np.sum(np.diff(np.vstack((previous,controls)),axis=0)**2))
                np.testing.assert_allclose(d['mpc_objective'][k],objective,atol=1e-7,rtol=1e-8)
            if row['method']!='optimal_decay' and np.isfinite(omegas).all():np.testing.assert_array_equal(omegas,np.ones((10,2)))
        active=status==0;assert d['active'][k]==active
        if active:
            u=d['control'][k];np.testing.assert_array_equal(u,d['mpc_control'][k]);np.testing.assert_array_equal(u,d['mpc_controls'][k,0])
            actual=audit_hold(x,u,d['next_state'][k],truth,mask,c);count+=1
            minimum=min(minimum,actual['minimum_clearance']);max_replay=max(max_replay,actual['replay_error'])
            collision|=actual['minimum_clearance']<=0;envelope|=actual['envelope_violation']>c.qp_tolerance
            if first_physical is None and (collision or envelope):first_physical=4 if collision else 5;first_step=count
            sub=np.polynomial.polynomial.polyval(np.arange(1,c.robot.integration_substeps+1)*c.robot.dt/c.robot.integration_substeps,coefficients(x,u,c)).T
            centers=truth[None,:,:2]+(np.arange(1,c.robot.integration_substeps+1)*c.robot.dt/c.robot.integration_substeps)[:,None,None]*truth[None,:,3:5]
            clear=float(np.min(np.where(mask[None],np.linalg.norm(sub[:,None,:2]-centers,axis=-1)-truth[None,:,2]-c.robot.radius,np.inf)))
            bound=max(float(np.max(abs(sub[:,3:])-limits)),float(np.max(c.altitude_min-sub[:,2])),float(np.max(sub[:,2]-c.altitude_max)))
            np.testing.assert_allclose(d['clearance'][k],clear,atol=2e-10,rtol=1e-9);np.testing.assert_allclose(d['envelope'][k],bound,atol=2e-10,rtol=1e-9)
            if clear<=0:status=4
            elif bound>c.qp_tolerance:status=5
            previous=u;previous_omega=d['mpc_omega'][k];cursor=progress
            np.testing.assert_array_equal(previous_omega,d['mpc_omegas'][k,0])
            max_equality=max(max_equality,eq);max_violation=max(max_violation,violation)
        else:
            np.testing.assert_array_equal(d['control'][k],np.zeros(4));np.testing.assert_array_equal(d['next_state'][k],x)
        np.testing.assert_allclose(d['route_progress'][k],cursor,atol=1e-8,rtol=1e-10)
        assert d['status'][k]==status
        if status:assert k==len(d['active'])-1
        x=d['next_state'][k]
    if status==0:
        assert len(d['active'])==m['steps']
        truth=original.copy();truth[:,:2]+=m['steps']*c.robot.dt*original[:,3:5]
        seen,_=numpy_observe(x,truth,mask,bx,bo,noise,ix[m['steps']],io[m['steps']]);status=1 if (not ordered or leg==total-1) and numpy_arrived(seen,goal,noise,c) else 6
    assert row['status']==status and row['steps']==count and not any(row['implicit_jit_cache_entries'].values())
    np.testing.assert_array_equal(row['final_state'],x)
    assert row['solver_attempts']==int(d['attempted'].sum())
    np.testing.assert_allclose(row['solver_seconds'],d['mpc_solve_seconds'].sum(),atol=1e-10)
    audited_status=first_physical if first_physical is not None else status
    if audited_status==1:assert numpy_arrived(x,goal,np.zeros(7),c),'Not a true full-state arrival'
    if ordered:
        assert row['waypoint_index']==leg and row['waypoints_visited']==leg+int(status==1)
        assert int(d['waypoint_handoff'].sum())==leg and (audited_status!=1 or leg==total-1)
    return dict(**row,audited_status=audited_status,physical_collision=bool(collision),envelope_exit=bool(envelope),
        first_physical_stop_status=first_physical,first_physical_stop_step=first_step,minimum_clearance=minimum,max_replay_error=max_replay,
        maximum_applied_prediction_equality_error=max_equality,maximum_applied_prediction_inequality_violation=max_violation,
        all_sensor_route_prediction_and_continuous_physics_verified=True)
