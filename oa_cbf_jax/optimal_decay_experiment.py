"""Matched noisy physical evaluation of native optimal-decay QPs.

JAX observation/assembly/integration kernels compile once per fixed batch shape.
Persistent native OSQP instances solve the three/four-variable programs on the
host. This is the authentic instantaneous baseline, without OA-CBF's predictive
gain search. Every method receives the same prior, innovations and route.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np

from .cli import sanitize
from .config import UnicycleConfig
from .controllers import NativeOSQP
from .native_proxqp import NativeProxQP
from .native_clarabel import NativeClarabel
from .dataset import source_fingerprint,sha256
from .dynamics import integrate_unicycle,signed_clearance,swept_disk_clearance
from .io import write_json
from .optimal_decay import OptimalDecayConfig,optimal_decay_problem
from .routing import physical_route_coordinate
from .simulation import RUNNING,GOAL,COLLISION,INFEASIBLE,TIMEOUT,STATUS_NAMES
from .route_control import INADMISSIBLE
from .closed_loop import PLANNER_FAILURE
from .stochastic import conditioned_sensor_model,STATE_BOUND_VIOLATION


def make_kernels(robot,config,steps):
    def initialize(x,goal,obs,mask,noise,key,ready):
        truth=conditioned_sensor_model(x,obs,mask,noise,key,robot,steps)
        initial,physical_obs,*_=truth
        clearance=jnp.min(signed_clearance(initial[:2],physical_obs,mask,robot.radius))
        at_goal=(jnp.linalg.norm(initial[:2]-goal)<=robot.goal_tolerance)&(jnp.abs(initial[3])<=.2)
        status=jnp.where(~ready,PLANNER_FAILURE,jnp.where(clearance<=0,COLLISION,jnp.where(at_goal,GOAL,RUNNING)))
        return truth,clearance,status

    def assemble(k,x,truth_obs,x_bias,obs_bias,x_scale,obs_scale,innovation,goal,mask,points,route_mask,cursor,noise):
        sensed_x=x-x_bias+.15*x_scale*innovation[:4]
        physical_obs=truth_obs.at[:,:2].set(truth_obs[:,:2]+k*robot.dt*truth_obs[:,3:5])
        sensed_obs=physical_obs-obs_bias+.15*obs_scale*innovation[4:].reshape(truth_obs.shape)
        problem=optimal_decay_problem(sensed_x,goal,sensed_obs,mask,points,route_mask,cursor,robot,config,1.15*noise[2])
        return problem,sensed_x,sensed_obs

    def advance(k,x,goal,truth_obs,mask,u,status,qp_ok,admissible):
        active=status==RUNNING;can_step=active&qp_ok&admissible
        u=jnp.where(can_step,u,jnp.zeros(2,x.dtype))
        y,sub=integrate_unicycle(x,u,robot.dt,robot.integration_substeps)
        starts=jnp.concatenate((x[None],sub[:-1]));times=k*robot.dt+jnp.arange(robot.integration_substeps)*robot.dt/robot.integration_substeps
        clear=jnp.min(jax.vmap(lambda a,b,t:swept_disk_clearance(a,b,truth_obs,mask,robot.radius,t,
                              t+robot.dt/robot.integration_substeps))(starts,sub,times))
        bound=jnp.maximum(-y[3],y[3]-robot.v_max)
        reached=can_step&(jnp.linalg.norm(y[:2]-goal)<=robot.goal_tolerance)&(jnp.abs(y[3])<=.2)
        status=jnp.where(active&~qp_ok,INFEASIBLE,status)
        status=jnp.where(active&~admissible,INADMISSIBLE,status)
        status=jnp.where(reached,GOAL,status)
        status=jnp.where(can_step&(bound>robot.qp_tolerance),STATE_BOUND_VIOLATION,status)
        status=jnp.where(can_step&(clear<=0),COLLISION,status)
        return jnp.where(can_step,y,x),status,can_step,jnp.where(can_step,clear,jnp.nan),jnp.where(can_step,bound,jnp.nan)

    return (jax.jit(jax.vmap(initialize)),
            jax.jit(jax.vmap(assemble,in_axes=(None,)+(0,)*13)),
            jax.jit(jax.vmap(advance,in_axes=(None,)+(0,)*8)))


def run(source,output,robot_config,formulation='repo',alpha1=.5,alpha2=.5,penalty=1e4,
        noise_scale=1.,batch=32,steps=800,limit=None,backend='osqp'):
    if backend not in ('osqp','proxqp','clarabel'):raise ValueError('Unknown native solver')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    robot=UnicycleConfig(**json.loads(Path(robot_config).read_text()))
    config=OptimalDecayConfig(formulation,alpha1,alpha2,penalty)
    records=json.loads((Path(source)/'scenes.json').read_text())
    if limit is not None:
        records=[r for r in records if not r['scene']['scene_id'].startswith('fixture:')][:limit]+[
                    r for r in records if r['scene']['scene_id'].startswith('fixture:')]
    write_json(root/'manifest.json',dict(stage='development_optimal_decay_closed_loop',final_test=False,
        source_fingerprint=source_fingerprint(),policy=asdict(config),robot=asdict(robot),noise_scale=noise_scale,steps=steps,batch=batch,
        input_scene_sha256=sha256(Path(source)/'scenes.json'),input_scenes=str(Path(source).resolve()),
        conditions='Identical observed initial scenes, routes, physical prior, sensor noise stream and limits for every method. Synchronous compute; no real-delay or final-test claim.',
        admissibility='No additional fixed-gain psi rejection for the repository coefficient variant.' if formulation=='repo' else 'Require h>=0 and fixed psi=hdot+alpha1*h>=0; optimize omega>=1.',
        modifications='All obstacle rows jointly; live gain coefficients; shared nominal guidance, moving-obstacle derivatives and known-error speed bounds. No post-solve clipping.',
        solver_backend=backend,
        solver=dict(osqp='Native OSQP with library default solver settings (only console verbosity disabled).',
                    proxqp='Native ProxQP FP64 with clean initialization, sqrt(weight) scaling, eps_abs 1e-9, eps_rel 0, 10000 iterations.',
                    clarabel='Native Clarabel FP64, centered sqrt(weight) scaling, gap_abs/rel and feasibility tolerances 1e-9, 200 iterations.')[backend]+
                ' Validate original QP after FP32 applied-command conversion.',
        interpretation='Instantaneous optimal-decay comparator; no OA-CBF predictive validator/backup imposed. Physical collisions, bounds and censored solver failures are measured. No continuous-time guarantee.'))
    init_fn,assemble_fn,advance_fn=make_kernels(robot,config,steps)
    compiled={};compile_seconds={};rows=[];timings=[];start=time.perf_counter()
    coordinate_fn=jax.jit(jax.vmap(physical_route_coordinate))
    def execute(name,fn,args):
        if name not in compiled:
            begin=time.perf_counter();compiled[name]=fn.lower(*args).compile();compile_seconds[name]=time.perf_counter()-begin
        return jax.device_get(compiled[name](*args))
    f32=lambda value:np.asarray(value,np.float32)
    names={**STATUS_NAMES,INFEASIBLE:'solver_rejected',INADMISSIBLE:'hocbf_inadmissible',PLANNER_FAILURE:'planner_failure',STATE_BOUND_VIOLATION:'state_bound_violation'}
    for offset in range(0,len(records),batch):
        chosen=records[offset:offset+batch];valid=len(chosen);chosen+=[chosen[-1]]*(batch-valid)
        x0,goal,obs=[f32([r['scene'][key] for r in chosen]) for key in ('initial_state','goal','obstacles')]
        mask=np.asarray([r['scene']['obstacle_mask'] for r in chosen],bool)
        points=f32([r['route']['points'] for r in chosen]);route_mask=np.asarray([r['route']['mask'] for r in chosen],bool)
        ready=np.asarray([r['route']['status']=='ready' for r in chosen])
        noise=np.broadcast_to(f32(noise_scale*np.array([.02,.03,.02,.03,.025,.01])),(batch,6))
        seeds=[int.from_bytes(hashlib.sha256(r['scene']['scene_id'].encode()).digest()[:4],'little') for r in chosen]
        keys=np.stack([np.asarray(jax.random.PRNGKey(seed)) for seed in seeds])
        truth,minimum,status=execute('initialize',init_fn,(x0,goal,obs,mask,noise,keys,ready))
        x,physical_obs,x_bias,obs_bias,x_scale,obs_scale,innovations=truth
        initial=x.copy();cursor=np.zeros(batch,np.float32);count=np.zeros(batch,int)
        solvers=None;trace=[];control_seconds=[];batch_begin=time.perf_counter()
        for k in range(steps):
            active=status==RUNNING
            if not np.any(active[:valid]):break
            begin=time.perf_counter()
            assembled=execute('assemble',assemble_fn,(np.int32(k),x,physical_obs,x_bias,obs_bias,x_scale,obs_scale,
                        innovations[:,k],goal,mask,points,route_mask,cursor,noise))
            (ref,a,b,h,psi,proposed,remaining,target),sensed_x,sensed_obs=assembled
            if solvers is None:
                factory=dict(osqp=NativeOSQP,proxqp=NativeProxQP,clarabel=NativeClarabel)[backend]
                solvers=[factory(b.shape[1],config.weights,**({'solver_defaults':True} if backend=='osqp' else {})) for _ in range(batch)]
            solution=np.zeros_like(ref);ok=np.zeros(batch,bool);residual=np.full(batch,np.nan,np.float32)
            solver_status=np.full(batch,'inactive',dtype='<U40');iterations=np.zeros(batch,np.int32)
            admissible=np.all(~mask|((h>=-robot.qp_tolerance)&(psi>=-robot.qp_tolerance)),axis=1) if formulation=='hocbf' else np.ones(batch,bool)
            for index in np.flatnonzero(active):
                result=solvers[index].solve(ref[index],a[index],b[index]);solver_status[index]=result['status'];iterations[index]=result['iterations']
                if result['feasible']:
                    applied=np.asarray(result['control'],np.float32)
                    error=np.max(np.asarray(a[index],float)@applied-np.asarray(b[index],float))
                    residual[index]=error
                    if np.isfinite(applied).all() and error<=robot.qp_tolerance:
                        solution[index]=applied;ok[index]=True
                    else:solver_status[index]='applied_precision_rejected'
            control_seconds.append(time.perf_counter()-begin)
            x,status,can_step,clear,bound=execute('advance',advance_fn,(np.int32(k),x,goal,physical_obs,mask,solution[:,:2],status,ok,admissible))
            cursor=np.where(can_step,proposed,cursor);count+=can_step
            minimum=np.minimum(minimum,np.where(can_step,clear,np.inf))
            trace.append(dict(state=x,control=np.where(can_step[:,None],solution[:,:2],0.),active=can_step,status=status,
                observed_state=sensed_x,observed_obstacles=sensed_obs,omega=solution[:,2:],route_progress=cursor,
                route_remaining=remaining,route_target=target,clearance=clear,state_bound_violation=bound,
                qp_violation=residual,psi1=np.min(np.where(mask,psi,np.inf),axis=1),
                solver_status=solver_status,solver_iterations=iterations,admissible=admissible))
        traces={key:np.stack([t[key] for t in trace],axis=1) for key in trace[0]} if trace else {}
        status=np.where(status==RUNNING,TIMEOUT,status)
        coordinates=execute('coordinate',coordinate_fn,(x[:,:2],points,route_mask,cursor))
        for index in range(valid):
            bound=traces.get('state_bound_violation',np.empty((batch,0)))[index]
            attempts=np.flatnonzero(traces['solver_status'][index]!='inactive') if trace else []
            row=dict(scene_id=chosen[index]['scene']['scene_id'],family=chosen[index]['scene']['family'],
                mode='optimal_decay_'+formulation,status=names[int(status[index])],steps=int(count[index]),
                min_clearance=float(minimum[index]),goal_progress=float(np.linalg.norm(initial[index,:2]-goal[index])-np.linalg.norm(x[index,:2]-goal[index])),
                final_route_coordinate=float(coordinates[index]),applied_source_counts={'optimal_decay':int(count[index])},
                max_physical_bound_violation=float(np.nanmax(bound)) if np.any(np.isfinite(bound)) else None,
                final_solver_status=str(traces['solver_status'][index,attempts[-1]]) if len(attempts) else None)
            rows.append(row)
        np.savez_compressed(root/f'traces_{offset:05d}.npz',**{k:v[:valid] for k,v in traces.items()},
            true_initial_state=initial[:valid],true_obstacles=physical_obs[:valid],noise=noise[:valid],key=keys[:valid],
            scene_id=np.array([r['scene']['scene_id'] for r in chosen[:valid]]))
        timings.append(dict(offset=offset,seconds=time.perf_counter()-batch_begin,control_batch_seconds=control_seconds))
        write_json(root/'results.json',sanitize(rows))
        write_json(root/'progress.json',dict(completed_groups=offset+valid,total_groups=len(records),elapsed_seconds=time.perf_counter()-start))
        print(json.dumps(dict(completed_groups=offset+valid,seconds=timings[-1]['seconds'])),flush=True)
    evaluated=[r for r in rows if not r['scene_id'].startswith('fixture:')]
    aggregate=dict(groups=len(evaluated),outcomes={s:sum(r['status']==s for r in evaluated) for s in sorted({r['status'] for r in evaluated})})
    write_json(root/'summary.json',dict(aggregate=aggregate,timings=timings,compile_seconds=compile_seconds,
        compiled_programs=list(compiled),elapsed_seconds=time.perf_counter()-start,device=str(jax.devices()[0])))
    print(json.dumps(aggregate),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','output','robot-config'):p.add_argument('--'+name,required=True)
    p.add_argument('--formulation',choices=['repo','hocbf'],default='repo')
    p.add_argument('--alpha1',type=float,default=.5);p.add_argument('--alpha2',type=float,default=.5);p.add_argument('--penalty',type=float,default=1e4)
    p.add_argument('--noise-scale',type=float,default=1.);p.add_argument('--batch',type=int,default=32)
    p.add_argument('--backend',choices=['osqp','proxqp','clarabel'],default='osqp')
    p.add_argument('--steps',type=int,default=800);p.add_argument('--limit',type=int)
    run(**vars(p.parse_args()))
