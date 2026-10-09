"""Quad2d odqp functions and shared contracts."""

import time

import numpy as np

from .quad2d_control import FlightConfig

def nominal(x, target, config=FlightConfig()):
    """Direct NumPy transcription of Quad2D.nominal_input's default gains."""
    x=np.asarray(x,float);target=np.asarray(target,float);c=config.robot
    acceleration=np.array([3*(target[0]-x[0])-.5*x[3],.1*(target[1]-x[1])-.5*x[4]+c.gravity])
    thrust=c.mass*np.linalg.norm(acceleration);desired=-np.arctan2(*acceleration)
    error=np.arctan2(np.sin(desired-x[2]),np.cos(desired-x[2]))
    torque=np.clip(.05*error-.05*x[5],-1.,1.)
    # The legacy nominal uses robot radius as its rotor lever arm. Shared
    # FlightConfig has equal radius/arm; no substitute gain tuning is applied.
    return np.clip(.5*np.array([thrust+torque/c.radius,thrust-torque/c.radius]),c.force_min,c.force_max)

def nearest(x,obs,mask):
    """Tracking's default Quad2D full-angle nearest-center selection."""
    if not np.any(mask):return -1
    return int(np.argmin(np.where(mask,np.linalg.norm(obs[:,:2]-x[:2],axis=1),np.inf)))

def problem(x,target,obs,mask,config=FlightConfig(),shared_buffer=True):
    x=np.asarray(x,float);obs=np.asarray(obs,float);mask=np.asarray(mask,bool);c=config.robot
    selected=nearest(x,obs,mask);A=np.zeros((5,4));b=np.zeros(5)
    if selected>=0:
        o=obs[selected];d=x[:2]-o[:2];v=x[3:5]
        h=d@d-1.01*(o[2]+c.radius+(c.clearance_buffer if shared_buffer else 0.))**2
        hd=2*d@v;drift=2*v@v-2*c.gravity*d[1]
        collective=2*(-d[0]*np.sin(x[2])+d[1]*np.cos(x[2]))/c.mass
        A[0]=[-collective,-collective,-hd,-.25*h];b[0]=drift
    A[1:,:2]=[[1,0],[-1,0],[0,1],[0,-1]];b[1:]=[c.force_max,-c.force_min,c.force_max,-c.force_min]
    return np.r_[nominal(x,target,config),1.,1.],A,b,selected

class Quad2DODQP:
    def __init__(self,config=FlightConfig(),shared_buffer=True):
        import osqp
        from scipy import sparse
        self.config=config;self.shared_buffer=shared_buffer;self.weights=np.array([1.,1.,1e4,1e4]);self.solver=osqp.OSQP();self.initialized=False;self.setup_seconds=0.

    def solve(self,x,target,obs,mask):
        ref,A,b,selected=problem(x,target,obs,mask,self.config,self.shared_buffer)
        if not all(np.isfinite(v).all() for v in [ref,A,b]):raise ValueError('Nonfinite original OD-QP inputs')
        if not self.initialized:
            from scipy import sparse
            start=time.perf_counter()
            # Default solver scaling must see the real first QP. Retain explicit
            # zeros in the CSC pattern so every later coefficient can update.
            pattern=sparse.csc_matrix((A.ravel(order='F'),np.tile(np.arange(5),4),np.arange(0,21,5)),shape=(5,4))
            self.solver.setup(P=sparse.diags(2*self.weights,format='csc'),q=-2*self.weights*ref,A=pattern,l=np.full(5,-np.inf),u=b,verbose=False)
            self.setup_seconds=time.perf_counter()-start;self.initialized=True
        else:self.solver.update(q=-2*self.weights*ref,Ax=A.ravel(order='F'),u=b)
        start=time.perf_counter();result=self.solver.solve(raise_error=False)
        solution=np.full(4,np.nan) if result.x is None else np.asarray(result.x).copy()
        success=bool(result.info.status_val in (1,2) and np.isfinite(solution).all())
        raw_violation=float(np.max(A@solution-b)) if np.isfinite(solution).all() else np.inf
        applied=solution.copy();applied[:2]=solution[:2].astype(np.float32)
        stored_violation=float(np.max(A@applied-b)) if np.isfinite(applied).all() else np.inf
        return dict(solution=solution,reference=ref,selected_obstacle=selected,solver_success=success,solver_status=result.info.status,status_value=int(result.info.status_val),
            iterations=int(result.info.iter),solve_seconds=time.perf_counter()-start,primal_residual=float(result.info.prim_res),dual_residual=float(result.info.dual_res),
            raw_violation=raw_violation,stored_violation=stored_violation,feasible=bool(success and max(raw_violation,stored_violation)<=self.config.robot.qp_tolerance))


import jax


from .quad2d_static_inputs import validate_parent, numpy_arrived

from .quad2d_rollout import NAMES

from .io import sanitize

def episode(parent,kernels,steps,config=FlightConfig()):
    c=config.robot;solver=Quad2DODQP(config);ordered='waypoint_count' in parent
    observed,final_goal,obs,mask,noise=(np.asarray(parent[k],bool if k=='obstacle_mask' else np.float32) for k in ['initial_state','goal','obstacles','obstacle_mask','noise'])
    if ordered:
        validate_parent(parent);goals=np.asarray(parent['waypoint_goals'],np.float32);total=parent['waypoint_count']
        routes=np.asarray(parent['waypoint_routes']['points'],np.float32);route_masks=np.asarray(parent['waypoint_routes']['mask'],bool);ready=np.asarray(parent['waypoint_routes']['ready'],bool)
    else:
        goals=final_goal[None];total=1;routes=np.asarray(parent['route']['points'],np.float32)[None];route_masks=np.asarray(parent['route']['mask'],bool)[None];ready=[parent['route']['status']=='ready']
    leg=0;key=np.asarray(jax.random.PRNGKey(parent['seed']+7193));initial,truth,xb,ob,xs,os,innovations=kernels.prepare(observed,obs,mask,noise,key)
    x=initial;previous=np.zeros(2,np.float32);cursor=np.float32(0);count=0;minimum=float(kernels.clearance(x,truth,mask))
    status=1 if total==1 and bool(kernels.arrived(x,goals[0])) else 0
    if float(kernels.bound(x))>c.qp_tolerance:status=8
    if minimum<=0:status=2
    if not ready[0]:status=7
    records=[];reason=NAMES[status];start=time.perf_counter()
    for k in range(steps):
        sensed,seen=map(np.asarray,kernels.sense(x,truth,xb,ob,xs,os,innovations[k],np.int32(k)))
        handoff=bool(status==0 and leg<total-1 and numpy_arrived(sensed.astype(float),goals[leg].astype(float),config,noise))
        if handoff:leg+=1;cursor=np.float32(0)
        goal=goals[leg];points=routes[leg];rm=route_masks[leg]
        if status==0 and not ready[leg]:status=7;reason=NAMES[status]
        before_cursor=cursor;before_control=previous.copy();target,proposed,remaining=map(np.asarray,kernels.target(sensed,points,rm,cursor))
        attempted=status==0;accepted=False;u=np.zeros(2,np.float32);clear=bound=np.nan
        result=dict(solution=np.full(4,np.nan),reference=np.r_[nominal(sensed,target,config),1.,1.],selected_obstacle=nearest(sensed,seen,mask),solver_success=False,solver_status='not_attempted',status_value=0,
            iterations=0,solve_seconds=0.,primal_residual=np.nan,dual_residual=np.nan,raw_violation=np.nan,stored_violation=np.nan,feasible=False)
        if attempted:
            result=solver.solve(sensed,target,seen,mask);accepted=result['feasible']
            if not accepted:
                status=3;reason=('solver_reported_failure:'+result['solver_status'] if not result['solver_success'] else 'independent_stored_qp_rejected')
            else:
                u=result['solution'][:2].astype(np.float32);x,clear,bound=kernels.advance(x,u,truth,mask,np.int32(k));clear=float(clear);bound=float(bound);minimum=min(minimum,clear)
                count+=1;cursor=np.float32(proposed);previous=u
                if leg==total-1 and bool(kernels.arrived(x,goal)):status=1
                if bound>c.qp_tolerance:status=8
                if clear<=0:status=2
                reason=NAMES[status]
        mission=dict(waypoint_index=leg,waypoint_handoff=handoff,mission_goal=goal,mission_previous_control=before_control,mission_route_cursor_before=before_cursor,waypoints_visited=leg+int(status==1)) if ordered else {}
        records.append(dict(state=np.asarray(x),control=u,active=accepted,status=status,observed_state=sensed,observed_obstacles=seen,clearance=clear,state_bound_violation=bound,
            route_progress=cursor,route_target=target,route_remaining=remaining,solver_attempted=attempted,**{'odqp_'+k:v for k,v in result.items()},**mission))
        if status:break
    if not status:status=4;reason=NAMES[status]
    data={k:np.asarray([r[k] for r in records]) for k in records[0]}
    data.update(true_initial_state=np.asarray(initial),true_obstacles=np.asarray(truth),initial_observation=observed,observed_obstacles_initial=obs,obstacle_mask=mask,noise=noise,goal=final_goal,key=key)
    if not ordered:data.update(points=routes[0],route_mask=route_masks[0])
    row=dict(group_id=parent['group_id'],family=parent['family'],obstacles=int(mask.sum()),noise_scale=float(parent.get('noise_scale',round(float(noise[0]/.015),6))),status=NAMES[status],status_code=status,
        steps=count,min_clearance=minimum,final_state=np.asarray(x).tolist(),route_progress=float(cursor),termination_reason=reason,execution_seconds=time.perf_counter()-start,
        solver_attempts=int(data['solver_attempted'].sum()),solver_seconds=float(data['odqp_solve_seconds'].sum()),solver_setup_seconds=solver.setup_seconds,
        waypoint_index=leg,waypoints_visited=leg+int(status==1),required_waypoints=total,waypoint_handoffs=sum(int(r.get('waypoint_handoff',False)) for r in records))
    if ordered:row.update(original_kind=parent['original_kind'],variant_index=parent['variant_index'])
    return sanitize(row),data
