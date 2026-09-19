"""Default Quad3D MPC comparison on frozen policy parents; no outcome filtering."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import multiprocessing
from pathlib import Path
import os
import shutil
import time
import json
import numpy as np
import jax
import jax.numpy as jnp
from .quad3d import integrate_quad3d
from .quad3d_control import Quad3DControlConfig
from .quad3d_observation import observe,observed_arrived,controller_obstacles,guidance_obstacles,unit_tape
from .quad3d_routing import flight_target
from .quad3d_mpc import Quad3DMPC,DEFAULTS,SOURCE_FILES
from .quad3d_learning_contract import read
from .dataset import sha256
from .io import write_json
from .cli import sanitize

C=Quad3DControlConfig(hold_guard='bernstein_v87')
RUNTIME=('quad3d_mpc.py','quad3d_mpc_experiment.py','quad3d_mpc_audit.py','quad3d.py','quad3d_control.py',
    'quad3d_observation.py','quad3d_routing.py','quad3d_audit.py','routing.py')


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


_KERNELS=None
_SOLVER=None


def worker_init(method,cpus):
    global _KERNELS,_SOLVER
    identity=multiprocessing.current_process()._identity[-1]
    os.sched_setaffinity(0,{cpus[(identity-1)%len(cpus)]})
    _KERNELS=PhysicalKernels();_SOLVER=Quad3DMPC(64,method)


def worker(task):
    p,index,output,steps,storage_floor=task;out=Path(output);row,trace=episode(p,_SOLVER,_KERNELS,steps)
    if shutil.disk_usage(out).free/2**30<storage_floor:raise RuntimeError(f'{storage_floor}GiB free storage reserve reached')
    filename=f'{index:03d}.npz';np.savez_compressed(out/filename,**trace)
    if 'motion_stratum' in p:row['motion_stratum']=p['motion_stratum']
    row.update(file=filename,sha256=sha256(out/filename),kernel_compile_seconds=_KERNELS.compile_seconds,solver_setup_seconds=_SOLVER.setup_seconds)
    write_json(out/f'{index:03d}.json',sanitize(row));return row


def collect(source,output,method,slot=0,workers=12,limit=None):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=False);sm=read(source/'manifest.json')
    assert sm['schema'] in ('quad3d_fresh_learned_policy_v100','quad3d_fresh_learned_policy_v108') and not sm['weight_fit_authorized']
    version=109 if sm['schema']=='quad3d_fresh_learned_policy_v108' else 101
    storage_floor=100. if version==109 else 125.
    assert sm['parents_sha256']==sha256(source/'parents.json')
    pp=[p for p in read(source/'parents.json') if p['partition']=='policy_audit']
    if limit is not None:pp=pp[:limit]
    parents=[p for p in pp if (p['index']//12)%4==slot]
    assert parents and 0<=slot<4 and 1<=workers<=12
    contract=Quad3DMPC(64,method).contract();write_json(out/'contract.json',contract)
    cpus=list(range(slot*14,slot*14+workers));rows=[];start=time.perf_counter()
    from .quad3d_mpc_audit import audit_one
    # Collection and audit have separate implementations. Each saved trace is
    # read back and checked in its own worker after the rollout pool finishes.
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(workers,mp_context=context,initializer=worker_init,initargs=(method,cpus)) as pool:
        for row in pool.map(worker,[(p,i,str(out),sm['steps'],storage_floor) for i,p in enumerate(parents)]):
            rows.append(row);progress=dict(parents_complete=len(rows),parents_total=len(parents),physical_steps=sum(r['steps'] for r in rows),elapsed_seconds=time.perf_counter()-start)
            write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    write_json(out/'index.json',sanitize(rows))
    manifest=dict(schema=f'quad3d_native_mpc_trace_v{version}',source=str(source.resolve()),source_sha256=sha256(source/'manifest.json'),
        source_files={n:sha256(Path(__file__).parent/n) for n in RUNTIME},legacy_source_files={n:sha256(n) for n in SOURCE_FILES},
        method=method,slot=slot,steps=sm['steps'],storage_floor_gib=storage_floor,config=asdict(C),contract_sha256=sha256(out/'contract.json'),
        index_sha256=sha256(out/'index.json'),parents=len(rows),workers=workers,worker_cpu_affinities=cpus,
        whole_goal_complete=False,elapsed_seconds=time.perf_counter()-start)
    write_json(out/'manifest.json',manifest)
    results=[]
    with ProcessPoolExecutor(workers,mp_context=context) as pool:
        for row in pool.map(audit_one,[(p,r,str(out),manifest) for p,r in zip(parents,rows,strict=True)]):
            results.append(row);write_json(out/'audit_progress.json',dict(parents_audited=len(results),parents_total=len(rows),physical_steps=sum(r['steps'] for r in results)))
    write_json(out/'independent_audit.json',sanitize(dict(audit_passed=True,rows=results,index_sha256=sha256(out/'index.json'),
        manifest_sha256=sha256(out/'manifest.json'),physical_steps=sum(r['steps'] for r in results),all_continuous_physics_and_nlp_predictions_checked=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--method',choices=DEFAULTS,required=True)
    p.add_argument('--slot',type=int,default=0);p.add_argument('--workers',type=int,default=12);p.add_argument('--limit',type=int);a=p.parse_args()
    collect(a.source,a.output,a.method,a.slot,a.workers,a.limit)
