"""Default bicycle QPs on shared physical parents; every stop is retained."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
import json
import shutil
import time
import numpy as np
import jax
import jax.numpy as jnp

from .bicycle import integrate_bicycle, bicycle_state_violation
from .bicycle_control import BicycleControlConfig, constant, bicycle_arrived
from .bicycle_observation import observe, unit_errors
from .bicycle_native_qp import BarrierRows, NativeBicycleQP, METHODS, contract
from .bicycle_experiment import read,control_config
from .bicycle_rollout import NAMES,GOAL,COLLISION,INFEASIBLE,TIMEOUT,STATE_BOUND
from .dynamics import signed_clearance,swept_disk_clearance
from .dataset import sha256,source_fingerprint
from .io import write_json


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
        args=(x,o,mask,jnp.zeros(4),jnp.zeros((64,5)),jnp.zeros(6),jax.random.PRNGKey(1),jnp.int32(0),jnp.zeros(4),jnp.zeros((64,5)))
        start=time.perf_counter();self.sense=self.sense_function.lower(*args).compile()
        self.advance=self.advance_function.lower(x,jnp.zeros(2),o,mask,jnp.int32(0)).compile()
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


def collect(source,output,method,shard=0,shards=4,steps=1600,limit=None,acceptance='strict_rows'):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);source=Path(source);sm=read(source/'manifest.json');parents=read(source/'scenes.json')
    if sm['weight_fit_authorized'] is not False or sm['scenes_sha256']!=sha256(source/'scenes.json'):raise ValueError('Changed evaluation source')
    indices=sm['phase_order']['policy_audit'][shard::shards]
    if limit is not None:indices=indices[:limit]
    if not indices or not 0<=shard<shards or steps<1:raise ValueError('Invalid native cohort')
    c=control_config(sm['config']);start=time.perf_counter();kernels=PhysicalKernels(c)
    manifest=dict(source=str(source.resolve()),source_manifest_sha256=sha256(source/'manifest.json'),source_fingerprint=source_fingerprint(),
        selected_indices=indices,method=method,contract=contract(method,c,acceptance),acceptance=acceptance,steps=steps,smoke_only=limit is not None,config=asdict(c),
        source_task='Exact V80 physical parents/goal/noise/bias/first observations. Native nominal follows the task goal directly; no OA route or predictive guidance.',
        sensor_key='parent.seed+4; same global-tick innovation rule as OA/FC',device=str(jax.devices()[0]),whole_goal_complete=False)
    write_json(root/'manifest.json',manifest);index=[]
    print(json.dumps(dict(stage='ready',compiled_signatures=3,compile_seconds=kernels.compile_seconds,device=str(jax.devices()[0]))),flush=True)
    for i in indices:
        if shutil.disk_usage(root).free/2**30<150.35:raise ValueError('Native evaluation disk safety buffer reached')
        row,payload=episode(parents[i],method,kernels,steps,acceptance);file=f'parent_{i:04d}.npz'
        np.savez_compressed(root/file,**payload);row.update(source_index=i,file=file,sha256=sha256(root/file));index.append(row)
        write_json(root/'index.json',index)
        if len(index)%8==0:print(json.dumps(dict(completed_parents=len(index),total_parents=len(indices),physical_steps=sum(r['steps'] for r in index),elapsed_seconds=time.perf_counter()-start)),flush=True)
    caches=kernels.cache_sizes()
    if any(caches.values()):raise ValueError('Unexpected runtime compilation')
    write_json(root/'summary.json',dict(complete=True,parents=len(index),physical_steps=sum(r['steps'] for r in index),compiled_signatures=3,
        implicit_jit_cache_entries=caches,compile_seconds=kernels.compile_seconds,execution_seconds=time.perf_counter()-start))


def audit(directory,workers=12):
    from .bicycle_native_audit import audit_one
    root=Path(directory);m=read(root/'manifest.json');source=Path(m['source']);sm=read(source/'manifest.json');parents=read(source/'scenes.json');index=read(root/'index.json')
    if m['source_manifest_sha256']!=sha256(source/'manifest.json') or sm['scenes_sha256']!=sha256(source/'scenes.json'):raise ValueError('Changed native evaluation source')
    if [r['source_index'] for r in index]!=m['selected_indices']:raise ValueError('Incomplete native cohort')
    with ProcessPoolExecutor(workers,mp_context=get_context('spawn')) as pool:
        results=list(pool.map(audit_one,[(str(root),r,m,parents[r['source_index']]) for r in index]))
    proof=dict(audit_passed=True,all_source_sensor_rows_physics_checked=True,manifest_sha256=sha256(root/'manifest.json'),index_sha256=sha256(root/'index.json'),
        parents=len(index),physical_steps=sum(r['steps'] for r in results),rows=results)
    write_json(root/'independent_replay.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('collect')
    for k in ('source','output','method'):q.add_argument('--'+k,required=True)
    q.add_argument('--shard',type=int,default=0);q.add_argument('--shards',type=int,default=4)
    q.add_argument('--steps',type=int,default=1600);q.add_argument('--limit',type=int)
    q.add_argument('--acceptance',choices=('strict_rows','native_status'),default='strict_rows')
    q=s.add_parser('audit');q.add_argument('--directory',required=True);q.add_argument('--workers',type=int,default=12)
    a=vars(p.parse_args());cmd=a.pop('command');dict(collect=collect,audit=audit)[cmd](**a)
