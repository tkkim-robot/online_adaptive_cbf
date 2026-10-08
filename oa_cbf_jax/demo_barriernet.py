"""Native BarrierNet navigation on the shared bicycle and Quad3D plants.

These controllers retain each baseline's original proxy constraints, default
OSQP settings and actuator clipping. The simulation checks the full plant.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

from .barriernet_variant_inference import load_bundle, network_problem, NativeVariantQP
from .bicycle import BicycleConfig
from .bicycle_rollout import NAMES, GOAL, COLLISION, INFEASIBLE, TIMEOUT, STATE_BOUND
from .comparison_contracts import physical_obstacle_scope
from .quad3d import Quad3DConfig
from .quad3d_mpc_experiment import execute64, initial_physical
from .quad3d_observation import unit_tape

KIND = 'KinematicBicycle2D_DPCBF'



class BicycleController:
    def __init__(self,bundle,config=BicycleConfig(),capacity=64):
        self.config,self.capacity = config,capacity
        self.manifest,model,self.params,self.mean,self.std = load_bundle(bundle,KIND,config)
        begin = time.monotonic()
        self.fn = network_problem(model,self.params,self.mean,self.std,KIND,config)
        args = (jnp.zeros(4,jnp.float64),jnp.zeros(2,jnp.float64),
            jnp.zeros((capacity,5),jnp.float64),jnp.zeros(capacity,bool))
        self.execute = self.fn.lower(*args).compile()
        self.compile_seconds = time.monotonic()-begin
        self.reset()

    def reset(self):
        self.qp = NativeVariantQP(KIND,self.config)

    def solve(self,state,goal,obstacles,mask):
        if (np.shape(state)!=(4,) or np.shape(goal)!=(2,)
                or np.shape(obstacles)!=(self.capacity,5) or np.shape(mask)!=(self.capacity,)):
            raise ValueError('Unsupported unwarmed native inference shape')
        begin = time.monotonic()
        args = tuple(jnp.asarray(v,jnp.bool_ if i==3 else jnp.float64)
            for i,v in enumerate((state,goal,obstacles,mask)))
        reference,G,h,gains,nominal,z,ctx = map(np.asarray,jax.device_get(self.execute(*args)))
        result = self.qp.solve(reference,G,h)
        result.update(network_reference=reference,G=G,h=h,gains=gains,nominal=nominal,z=z,ctx=ctx,
            total_seconds=time.monotonic()-begin)
        if self.fn._cache_size()!=0:
            raise ValueError('Unexpected implicit native inference compilation')
        return result


def bicycle_empty_result():
    return dict(control=np.zeros(2),raw_control=np.full(2,np.nan),dual=np.full(9,np.nan),
        accepted=False,status_value=0,status='not_attempted',iterations=0,seconds=0.,
        raw_residual=np.nan,applied_residual=np.nan,network_reference=np.full(2,np.nan),
        G=np.full((5,2),np.nan),h=np.full(5,np.nan),gains=np.full((5,1),np.nan),
        nominal=np.full(2,np.nan),z=np.full(25,np.nan),ctx=np.full(41,np.nan),total_seconds=0.)


def bicycle_episode(parent, controller, kernels, steps=1600):
    if steps<1:
        raise ValueError('Positive simulation horizon required')
    controller.reset()
    c,r=kernels.config,kernels.config.robot
    x=np.asarray(parent['initial'],float).copy()
    original=np.asarray(parent['obstacles'],float);mask=np.asarray(parent['mask'],bool)
    goal,noise=[np.asarray(parent[k],np.float32) for k in ('goal','noise')]
    bx,bo,fx,fo=[np.asarray(parent[k],np.float32) for k in ('bias_x','bias_o','first_x','first_o')]
    key=np.asarray(jax.random.PRNGKey(parent['seed']+4))
    immutable=(jnp.asarray(original,jnp.float64),jnp.asarray(mask),jnp.asarray(bx),
        jnp.asarray(bo),jnp.asarray(noise),jnp.asarray(key))
    arrived=lambda z: np.linalg.norm(z[:2]-goal)<=c.goal_tolerance and z[3]<=c.terminal_speed
    minimum=float(np.min(np.where(mask,np.linalg.norm(original[:,:2]-x[:2],axis=1)-r.radius-original[:,2],np.inf)))
    status=GOAL if arrived(x) else 0
    if max(r.speed_min-x[3],x[3]-r.speed_max)>c.qp_tolerance: status=STATE_BOUND
    if minimum<=0: status=COLLISION
    history=[];count=0;start=time.perf_counter()
    for k in range(steps):
        sx,so,ix,io=map(np.asarray,kernels.sense(jnp.asarray(x,jnp.float64),*immutable,np.int32(k),fx,fo))
        attempted=status==0;accepted=False;u=np.zeros(2,np.float32)
        before=x.copy();clear=violation=np.nan;result=bicycle_empty_result()
        if attempted:
            result=controller.solve(sx,goal,so,mask);accepted=result['accepted']
            if accepted:
                u=result['control'].astype(np.float32)
                x,clear,violation=map(np.asarray,kernels.advance(jnp.asarray(x,jnp.float64),u,
                    immutable[0],immutable[1],np.int32(k)))
                clear,violation=float(clear),float(violation)
                count+=1;minimum=min(minimum,clear)
                if arrived(x): status=GOAL
                if violation>c.qp_tolerance: status=STATE_BOUND
                if clear<=0: status=COLLISION
            else: status=INFEASIBLE
        history.append(dict(state_before=before,state=x.copy(),control=u,active=accepted,
            attempted=attempted,status=np.int32(status),observed_state=sx,observed_obstacles=so,
            innovation_x=ix,innovation_o=io,clearance=clear,state_violation=violation,
            **{'nn_'+k:v for k,v in result.items()}))
        if status: break
    if status==0: status=TIMEOUT
    data={k:np.asarray([h[k] for h in history]) for k in history[0]}
    data.update({k:np.asarray(parent[k]) for k in
        ('initial','goal','obstacles','mask','bias_x','bias_o','first_x','first_o','noise')})
    data.update(key=key,final_status=np.int32(status),expected_steps=np.int32(count),horizon=np.int32(steps))
    row=dict(group_id=parent['group_id'],family=parent['family'],method='barriernet',status=NAMES[status],
        status_code=status,steps=count,min_clearance=minimum if np.isfinite(minimum) else None,
        execution_seconds=time.perf_counter()-start,solver_attempts=int(data['attempted'].sum()),
        solver_seconds=float(data['nn_seconds'].sum()),
        applied_proxy_violation_ticks=int(np.sum(data['active']&(data['nn_applied_residual']>c.qp_tolerance))))
    return row,data


class Quad3DController:
    def __init__(self,bundle,config=Quad3DConfig(),capacity=64):
        self.config=config
        self.manifest,model,self.params,self.mean,self.std=load_bundle(bundle,'Quad3D',config)
        start=time.perf_counter()
        self.fn=network_problem(model,self.params,self.mean,self.std,'Quad3D',config)
        args=(jnp.zeros(12,jnp.float64),jnp.zeros(3,jnp.float64),
              jnp.zeros((capacity,5),jnp.float64),jnp.zeros(capacity,bool))
        self.execute=self.fn.lower(*args).compile();self.compile_seconds=time.perf_counter()-start
        self.reset()

    def reset(self):self.qp=NativeVariantQP('Quad3D',self.config)

    def solve(self,state,goal,obstacles,mask):
        begin=time.perf_counter()
        args=tuple(jnp.asarray(v,jnp.bool_ if i==3 else jnp.float64)
                   for i,v in enumerate((state,goal,obstacles,mask)))
        reference,G,h,gains,nominal,z,ctx=map(np.asarray,jax.device_get(self.execute(*args)))
        result=self.qp.solve(reference,G,h)
        result.update(network_reference=reference,G=G,h=h,gains=gains,nominal=nominal,z=z,ctx=ctx,
                      total_seconds=time.perf_counter()-begin)
        if self.fn._cache_size()!=0:raise ValueError('Unexpected native inference tracing')
        return result


def quad3d_empty_result():
    return dict(control=np.full(4,np.nan),raw_control=np.full(4,np.nan),accepted=False,
        status='not_attempted',status_value=-1,iterations=0,raw_residual=np.nan,applied_residual=np.nan,
        dual=np.full(13,np.nan),seconds=0.,network_reference=np.full(4,np.nan),G=np.full((5,4),np.nan),
        h=np.full(5,np.nan),gains=np.full((5,2),np.nan),nominal=np.full(4,np.nan),z=np.full(25,np.nan),
        ctx=np.full(44,np.nan),total_seconds=0.)


def quad3d_episode(parent,kernels,solver,steps,config):
    c=config;mask=np.asarray(parent['mask'],bool);original=np.asarray(parent['obstacles'],float)
    physical_obstacle_scope('quad3d',original,mask)
    x=np.asarray(parent['x'],float);noise=np.asarray(parent['noise'],float)
    ordered='waypoint_count' in parent;leg=0
    goals=np.asarray(parent['waypoint_goals'] if ordered else [parent['goal']],float)
    total=parent['waypoint_count'] if ordered else 1
    routes=np.asarray(parent['waypoint_routes']['points'] if ordered else [parent['route']['points']],float)
    route_masks=np.asarray(parent['waypoint_routes']['mask'] if ordered else [parent['route']['mask']],bool)
    bx,bo,ix,io=unit_tape(parent['sensor_seed'],steps,len(mask));cursor=0.;solver.reset()
    clear,bound=initial_physical(x,original,mask,c);status=4 if clear<=0 else 5 if bound>c.qp_tolerance else 0
    count=0;records=[];start=time.perf_counter()
    for tick in range(steps):
        seen,obs=map(np.asarray,execute64(kernels.sense,x,original,mask,bx,bo,noise,ix[tick],io[tick]))
        handoff=status==0 and leg<total-1 and bool(execute64(kernels.arrived,seen,goals[leg],noise))
        if handoff:leg+=1;cursor=0.
        goal=goals[leg]
        target,proposed,remaining,visible,controlled=map(np.asarray,execute64(kernels.route,
            seen,goal,obs,mask,routes[leg],route_masks[leg],np.asarray(cursor,np.float64),noise))
        if status==0 and leg==total-1 and bool(execute64(kernels.arrived,seen,goal,noise)):status=1
        attempted=status==0;active=False;result=quad3d_empty_result();before=x.copy();before_cursor=cursor
        u=np.zeros(4)
        if attempted:
            result=solver.solve(seen,target,controlled,mask)
            if result['accepted']:
                active=True;u=result['control'];cursor=float(proposed)
                x,clear,bound=map(np.asarray,execute64(kernels.advance,x,u,original,mask));count+=1
                if clear<=0:status=4
                elif bound>c.qp_tolerance:status=5
            else:status=3
        records.append(dict(state=before,next_state=x.copy(),observed=seen,control=u,active=active,
            attempted=attempted,status=status,clearance=clear,envelope=bound,route_target=target,
            route_cursor_before=before_cursor,route_progress=cursor,route_remaining=remaining,route_visible=visible,
            waypoint_index=leg,waypoint_handoff=handoff,mission_goal=goal,
            **{'native_'+k:v for k,v in result.items()}))
        if status:break
    if status==0:
        seen,_=execute64(kernels.sense,x,original,mask,bx,bo,noise,ix[steps],io[steps])
        status=1 if leg==total-1 and bool(execute64(kernels.arrived,seen,goals[leg],noise)) else 6
    if any(kernels.cache_sizes().values()) or solver.fn._cache_size()!=0:
        raise ValueError('Runtime tracing in native episode')
    trace={k:np.asarray([r[k] for r in records]) for k in records[0]}
    row=dict(id=parent['id'],family=parent['family'],noise_level=parent['noise_level'],method='barriernet',
        status=status,steps=count,final_state=x.tolist(),waypoint_index=leg,waypoints_visited=leg+int(status==1),
        solver_attempts=int(trace['attempted'].sum()),solver_seconds=float(trace['native_seconds'].sum()),
        execution_seconds=time.perf_counter()-start,implicit_jit_cache_entries=0)
    return row,trace
