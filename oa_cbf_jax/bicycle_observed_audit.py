"""Independent NumPy/SciPy checks for acquired-observation bicycle labels."""
import numpy as np
from scipy.integrate import solve_ivp
from .bicycle_audit import reference_rows,reference_flow,polygon_qp
from .bicycle_control import BicycleControlConfig
from .bicycle_rollout import GOAL,COLLISION,INFEASIBLE,TIMEOUT,INADMISSIBLE,PLANNER_FAILURE,STATE_BOUND


def numpy_graph(x,goal,obs,mask,points,rm,noise,c,cursor,previous_control,previous_gain,projection_index=None):
    x=np.asarray(x,float);obs=np.asarray(obs,float);points=np.asarray(points,float);goal=np.asarray(goal,float);robot=c.robot
    valid=rm[:-1]&rm[1:];vectors=np.diff(points,axis=0);length=np.where(valid,np.linalg.norm(vectors,axis=1),0.);cumulative=np.r_[0.,np.cumsum(length)]
    fraction=np.clip(np.sum((x[:2]-points[:-1])*vectors,axis=1)/np.maximum(length**2,1e-12),0,1);projected=cumulative[:-1]+fraction*length
    eligible=valid&(projected>=cursor-.05)&(projected<=cursor+1.);nearest=points[:-1]+fraction[:,None]*vectors
    index=int(np.argmin(np.where(eligible,np.sum((nearest-x[:2])**2,axis=1),np.inf))) if projection_index is None else projection_index
    updated=max(cursor,projected[index]) if eligible.any() else cursor
    desired=min(updated+np.clip(.45+.65*x[3],.35,1.1),cumulative[-1]);segment=np.argmax(valid&(cumulative[1:]>=desired-1e-6))
    target=points[segment]+np.clip((desired-cumulative[segment])/max(length[segment],1e-12),0,1)*vectors[segment]
    rotation=np.array([[np.cos(x[2]),-np.sin(x[2])],[np.sin(x[2]),np.cos(x[2])]])
    positions=(np.vstack((x[:2],goal,obs[:,:2]))-x[:2])@rotation
    velocity=x[3]*np.array([np.cos(x[2]),np.sin(x[2])]);velocities=(np.vstack((velocity,[0,0],obs[:,3:5]))-velocity)@rotation
    radii=np.r_[robot.radius,0.,obs[:,2]];clearance=np.sqrt(np.sum(positions**2,axis=1)+1e-12)-robot.radius-radii
    types=np.vstack(([1,0,0],[0,0,1],np.tile([0,1,0],(len(obs),1))))
    ego=[x[3]/robot.speed_max,robot.radius,robot.wheel_base,robot.rear_axle_distance,robot.acceleration_max,robot.slip_max,robot.speed_min,robot.speed_max,robot.dt,c.clearance_buffer,c.barrier_inflation,c.relative_speed_epsilon]
    context=np.r_[(goal-x[:2])@rotation/5,(target-x[:2])@rotation/5,max(cumulative[-1]-updated,0)/10,np.asarray(previous_control)/[robot.acceleration_max,robot.slip_max],np.log(previous_gain),noise]
    n=len(obs)+2;features=np.column_stack((types,positions/5,velocities/robot.speed_max,radii,clearance/3,np.tile(ego,(n,1)),np.tile(context,(n,1))))
    node_mask=np.r_[True,True,mask];features[~node_mask]=0.;return features,node_mask


def check_graph(features,node_mask,args):
    expected,mask=numpy_graph(*args);np.testing.assert_array_equal(node_mask,mask)
    if np.allclose(features,expected,atol=3e-6,rtol=2e-6):return
    other=np.ones(35,bool);other[23:26]=False
    np.testing.assert_allclose(features[:,other],expected[:,other],atol=3e-6,rtol=2e-6)
    from .route_audit import distance_intervals
    x,_,_,_,points,rm,_,_,cursor,_,_=args;points=np.asarray(points,float);x=np.asarray(x,float)
    valid=rm[:-1]&rm[1:];v=np.diff(points,axis=0);length=np.where(valid,np.linalg.norm(v,axis=1),0.);cs=np.r_[0.,np.cumsum(length)]
    t=np.clip(np.sum((x[:2]-points[:-1])*v,axis=1)/np.maximum(length**2,1e-12),0,1);s=cs[:-1]+t*length;eligible=valid&(s>=cursor-.05)&(s<=cursor+1.)
    low,high=distance_intervals(x[:2],points)
    for i in np.flatnonzero(eligible&(low<=np.min(high[eligible],initial=np.inf))):
        candidate,_=numpy_graph(*args,projection_index=int(i))
        if np.allclose(features,candidate,atol=3e-6,rtol=2e-6):return
    raise ValueError('Observed bicycle graph differs from independent reference')


def reference_route_coordinate(position,points,mask,cursor,projection_index=None):
    points=np.asarray(points,float);position=np.asarray(position,float);v=np.diff(points,axis=0);valid=mask[:-1]&mask[1:]
    length=np.where(valid,np.linalg.norm(v,axis=1),0.);cs=np.r_[0.,np.cumsum(length)]
    lo=np.maximum(cs[:-1],cursor-1.);hi=np.minimum(cs[1:],cursor+1.);eligible=valid&(lo<=hi)
    fraction=np.sum((position-points[:-1])*v,axis=1)/np.maximum(length**2,1e-12)
    coordinate=np.clip(cs[:-1]+fraction*length,lo,np.maximum(lo,hi))
    projected=points[:-1]+((coordinate-cs[:-1])/np.maximum(length,1e-12))[:,None]*v
    index=np.argmin(np.where(eligible,np.sum((projected-position)**2,axis=1),np.inf)) if projection_index is None else projection_index
    return float(coordinate[index])


def route_coordinate_candidates(position,points,mask,cursor):
    """Possible nearest segments under FP32 clipped arclength projection.

    Outward operation intervals cover both fused and separate arithmetic.
    Positive prefix sums use a gamma_n bound valid for any summation tree.
    Only numerically overlapping distances may supply an alternate segment;
    the returned coordinates are still recomputed independently in FP64.
    """
    from .route_audit import _exact,_outward,_add,_sub,_mul,_square,_sum2
    points=np.asarray(points,float);position=np.asarray(position,float);mask=np.asarray(mask,bool)
    valid=mask[:-1]&mask[1:];p=_exact(points[:-1]);v=_sub(_exact(points[1:]),p)
    squared=_sum2(_square(v))
    length=_outward(np.sqrt(np.maximum(squared[0],0.)),np.sqrt(np.maximum(squared[1],0.)))
    length=tuple(np.where(valid,a,0.) for a in length)
    count=np.arange(1,len(points));unit=np.finfo(np.float32).eps/2
    gamma=count*unit/(1-count*unit)
    cumulative=_outward(np.r_[0.,np.cumsum(length[0])*(1-gamma)],np.r_[0.,np.cumsum(length[1])*(1+gamma)])
    # The first prefix is exactly the explicitly stored zero, not a sum.
    for a in cumulative:a[0]=0.
    start=tuple(a[:-1] for a in cumulative);end=tuple(a[1:] for a in cumulative)
    window_low=_sub(_exact(cursor),_exact(1.));window_high=_add(_exact(cursor),_exact(1.))
    low=tuple(np.maximum(a,b) for a,b in zip(start,window_low))
    high=tuple(np.minimum(a,b) for a,b in zip(end,window_high))
    possible=valid&(low[0]<=high[1]);certain=valid&(low[1]<=high[0])
    def divide(a,b):
        b=tuple(np.maximum(x,1e-12) for x in b)
        return _mul(a,_outward(1/b[1],1/b[0]))
    fraction=divide(_sum2(_mul(_sub(_exact(position),p),v)),_square(length))
    coordinate=_add(start,_mul(fraction,length))
    upper=tuple(np.maximum(a,b) for a,b in zip(low,high))
    coordinate=tuple(np.minimum(np.maximum(a,b),c) for a,b,c in zip(coordinate,low,upper))
    t=divide(_sub(coordinate,start),length)
    projected=_add(p,_mul(tuple(a[:,None] for a in t),v))
    distance=_sum2(_square(_sub(projected,_exact(position))))
    indices=np.flatnonzero(possible&(distance[0]<=np.min(distance[1][certain],initial=np.inf)))
    return [reference_route_coordinate(position,points,mask,cursor,int(i)) for i in indices]


def check_route_progress(saved,initial,final,points,mask,initial_cursor,final_cursor):
    """Check a retained progress label, including genuine nearest-segment ties."""
    args=(points,mask)
    expected=reference_route_coordinate(final,*args,final_cursor)-reference_route_coordinate(initial,*args,initial_cursor)
    if np.isclose(saved,expected,atol=3e-5,rtol=2e-6):return False
    starts=route_coordinate_candidates(initial,*args,initial_cursor)
    ends=route_coordinate_candidates(final,*args,final_cursor)
    if any(np.isclose(saved,end-start,atol=3e-5,rtol=2e-6) for start in starts for end in ends):return True
    raise ValueError('Route progress differs from every numerically admissible projection')


def recorded_gain(data,mask):
    """Preserve legacy scalar traces; require explicit offline vector semantics."""
    value=np.asarray(data['alpha'])
    if not bool(data.get('per_obstacle_gain',False)):
        if value.ndim:raise ValueError('Vector bicycle gain requires an explicit per-obstacle declaration')
        return float(value)
    if (value.shape!=np.asarray(mask).shape or not np.isfinite(value).all()
            or np.any((value<.5)|(value>8.)) or 'controller_gain' in data):
        raise ValueError('Invalid or mixed fixed per-obstacle gain contract')
    return value.astype(float)


def audit_trace(d,config=BicycleControlConfig()):
    from .route_audit import check_transition
    c=config.robot;physical=d['initial'].astype(float);obs=d['obstacles'].astype(float);mask=d['mask'].astype(bool);noise=d['noise'].astype(float)
    bias_x=d['bias_x'].astype(float);bias_o=d['bias_o'].astype(float);alpha=recorded_gain(d,mask);status=int(d['final_status']);count=int(d['expected_steps']);horizon=int(d['horizon'])
    error=1.15*noise[2]+(5e-7 if noise[2]>0 else 0.)
    xs=noise[[0,0,1,2]];os=noise[[3,3,5,4,4]]
    assert np.linalg.norm(bias_x[:2])<=noise[0]+1e-7 and abs(bias_x[2])<=noise[1]+1e-7 and abs(bias_x[3])<=noise[2]+1e-7
    assert np.all(np.linalg.norm(bias_o[mask,:2],axis=1)<=noise[3]+1e-7) and np.all(np.linalg.norm(bias_o[mask,3:5],axis=1)<=noise[4]+1e-7) and np.all(np.abs(bias_o[mask,2])<=noise[5]+1e-7)
    assert not np.any(bias_o[~mask])
    np.testing.assert_array_equal(d['observed_state'][0],d['first_x']);np.testing.assert_array_equal(d['observed_obstacles'][0],d['first_o'])
    if np.any(np.diff(d['active'].astype(int))>0):raise ValueError('Post-terminal physical command')
    minimum=np.min(np.where(mask,np.linalg.norm(obs[:,:2]-physical[:2],axis=1)-c.radius-obs[:,2],np.inf));worst=-np.inf;max_error=0.;checked=0;feasible_rejection=False
    for k,accepted in enumerate(d['active']):
        if 'controller_gain' in d:
            alpha=float(d['controller_gain'][k])
            if not np.isfinite(alpha) or not .5<=alpha<=8.:raise ValueError('Invalid per-tick bicycle gain')
        current=obs.copy();current[:,:2]+=k*c.dt*current[:,3:5]
        sensed=d['observed_state'][k].astype(float);seen=d['observed_obstacles'][k].astype(float)
        ix=d['innovation_x'][k];io=d['innovation_o'][k]
        assert np.linalg.norm(ix[:2])<=1+1e-6 and np.max(np.abs(ix[2:]))<=1+1e-6
        assert np.all(np.linalg.norm(io[:,:2],axis=1)<=1+1e-6) and np.all(np.linalg.norm(io[:,3:5],axis=1)<=1+1e-6) and np.max(np.abs(io[:,2]))<=1+1e-6
        if k:
            np.testing.assert_allclose(sensed,(physical+bias_x+.15*xs*ix).astype(np.float32),atol=3e-6,rtol=2e-6)
            expected=(current+bias_o+.15*os*io).astype(np.float32);expected[~mask]=0.
            np.testing.assert_allclose(seen,expected,atol=3e-6,rtol=2e-6)
        else:
            # Later acquired observations contain their actual acquisition
            # innovation. The branch does not substitute a new latent draw.
            assert np.linalg.norm(sensed[:2]-physical[:2])<=1.15*noise[0]+5e-6
            assert abs(sensed[2]-physical[2])<=1.15*noise[1]+5e-6
            assert np.all(np.linalg.norm(seen[mask,:2]-current[mask,:2],axis=1)<=1.15*noise[3]+5e-6)
        assert abs(sensed[3]-physical[3])<=error+5e-7
        np.testing.assert_allclose(d['state_before'][k],physical,atol=3e-5,rtol=2e-6)
        points,route_mask=(d['mission_points'][int(d['mission_leg'][k])],d['mission_route_mask'][int(d['mission_leg'][k])]) if 'mission_leg' in d else (d['points'],d['route_mask'])
        check_transition(sensed,points,route_mask,float(d['cursor_before'][k]),d['route_target'][k],float(d['route_progress'][k]),bool(accepted))
        if not accepted:
            np.testing.assert_array_equal(d['control'][k],0.)
            if status in (INFEASIBLE,INADMISSIBLE):
                domain=np.sum((seen[:,:2]-sensed[:2])**2,axis=1)-((c.radius+config.clearance_buffer+seen[:,2])*config.barrier_inflation)**2
                if np.all(domain[mask]>0):
                    a,b,h,_=reference_rows(sensed,seen,mask,alpha,config)
                    b[-4]=min(c.acceleration_max,(c.speed_max-sensed[3]-error)/c.dt);b[-3]=-max(-c.acceleration_max,(c.speed_min-sensed[3]+error)/c.dt)
                    if status==INADMISSIBLE and np.min(h[mask],initial=np.inf)>=-config.qp_tolerance:raise ValueError('False barrier rejection')
                    if status==INFEASIBLE:feasible_rejection=polygon_qp([0,0],a,b,[1,1],[-c.acceleration_max,-c.slip_max],[c.acceleration_max,c.slip_max]) is not None
            break
        a,b,h,domain=reference_rows(sensed,seen,mask,alpha,config)
        b[-4]=min(c.acceleration_max,(c.speed_max-sensed[3]-error)/c.dt);b[-3]=-max(-c.acceleration_max,(c.speed_min-sensed[3]+error)/c.dt)
        if np.min(h[mask],initial=np.inf)<-config.qp_tolerance:raise ValueError('Applied negative observed barrier')
        u=d['control'][k];residual=float(np.max(a@u-b));worst=max(worst,residual)
        if residual>config.qp_tolerance:raise ValueError(f'Original observed row residual {residual} at {k}')
        times=np.linspace(0,c.dt,101);sol=solve_ivp(lambda t,s:reference_flow(t,s,u,config),(0,c.dt),physical,method='DOP853',rtol=1e-11,atol=1e-12,t_eval=times)
        assert sol.success;max_error=max(max_error,float(np.max(np.abs(sol.y[:,-1]-d['state'][k]))));np.testing.assert_allclose(d['state'][k],sol.y[:,-1],atol=3e-5,rtol=2e-6)
        centers=current[None,:,:2]+times[:,None,None]*current[None,:,3:5]
        clear=float(np.min(np.where(mask,np.linalg.norm(sol.y[:2].T[:,None]-centers,axis=-1)-c.radius-current[None,:,2],np.inf)))
        if np.isfinite(clear) and abs(clear-float(d['clearance'][k]))>1e-4:raise ValueError('Physical clearance mismatch')
        if clear<-1e-4 and status!=COLLISION:raise ValueError('Unreported physical collision')
        violation=max(c.speed_min-sol.y[3].min(),sol.y[3].max()-c.speed_max)
        if violation>config.qp_tolerance+1e-6 and status!=STATE_BOUND:raise ValueError('Unreported physical speed violation')
        minimum=min(minimum,clear);physical=sol.y[:,-1];checked+=1
    assert checked==count
    if status==TIMEOUT:assert count==horizon
    if status==GOAL:assert np.linalg.norm(physical[:2]-d['goal'])<=config.goal_tolerance+1e-4 and physical[3]<=config.terminal_speed+1e-5
    if status==COLLISION:assert minimum<=1e-4
    if status==STATE_BOUND:assert c.speed_min-physical[3]>config.qp_tolerance-1e-6 or physical[3]-c.speed_max>config.qp_tolerance-1e-6
    if status==PLANNER_FAILURE:
        ready=d['mission_ready'][int(d['mission_leg'][-1])] if 'mission_leg' in d else d['ready']
        assert not bool(ready)
    return dict(audit_passed=True,steps=count,max_state_error=max_error,min_clearance=float(minimum),max_qp_residual=float(worst),feasible_qp_rejected=feasible_rejection)
