"""Optimal decay functions and shared contracts."""

from dataclasses import dataclass

import math

import jax.numpy as jnp

from .config import UnicycleConfig

from .route_control import route_problem

@dataclass(frozen=True)
class OptimalDecayConfig:
    formulation: str = 'repo'
    alpha1: float = .5
    alpha2: float = .5
    penalty: float = 1e4

    def __post_init__(self):
        if self.formulation not in ('repo','hocbf'):
            raise ValueError('Unknown optimal-decay formulation')
        if any(not math.isfinite(v) or v<=0 for v in (self.alpha1,self.alpha2,self.penalty)):
            raise ValueError('Gains and decay penalty must be finite and positive')

    @property
    def weights(self):
        # A common factor of 1/2 leaves the repository minimizer unchanged.
        return (1.,1.)+(self.penalty,)*(2 if self.formulation=='repo' else 1)

def optimal_decay_problem(x,goal,obstacles,mask,points,route_mask,progress,
                          robot=UnicycleConfig(),config=OptimalDecayConfig(),speed_uncertainty=0.):
    reference,a,b,h,hdot,cursor,remaining,target=route_problem(
        x,goal,obstacles,mask,jnp.zeros(2,x.dtype),points,route_mask,progress,robot,speed_uncertainty)
    n=obstacles.shape[0]
    psi=hdot+config.alpha1*h
    if config.formulation=='repo':
        # hddot + (a1+a2)*omega1*hdot + a1*a2*omega2*h >= margin.
        coefficients=jnp.stack(((config.alpha1+config.alpha2)*hdot,config.alpha1*config.alpha2*h),axis=-1)
        extra=jnp.concatenate((-jnp.where(mask[:,None],coefficients,0.),jnp.zeros((4,2),x.dtype)))
        matrix=jnp.concatenate((a,extra),axis=-1)
        ref=jnp.concatenate((reference,jnp.ones(2,x.dtype)))
    else:
        # psidot + omega*a2*psi >= margin, omega>=1. alpha1 remains fixed.
        extra=jnp.concatenate((-jnp.where(mask,config.alpha2*psi,0.),jnp.zeros(4,x.dtype)))[:,None]
        matrix=jnp.concatenate((a,extra),axis=-1)
        b=b.at[:n].add(jnp.where(mask,config.alpha1*hdot,0.))
        matrix=jnp.concatenate((matrix,jnp.array([[0.,0.,-1.]],x.dtype)))
        b=jnp.concatenate((b,jnp.array([-1.],x.dtype)))
        ref=jnp.concatenate((reference,jnp.ones(1,x.dtype)))
    return ref,matrix,b,h,psi,cursor,remaining,target


from dataclasses import asdict

import json

from pathlib import Path


from .io import sha256

from .comparison_contracts import physical_obstacle_scope

SCHEMA='oa_cbf_unicycle_static_inputs'

PHYSICAL_FIELDS=tuple(k for k in asdict(UnicycleConfig()) if not k.startswith('guidance_'))

def source_robot(directory,requested=None):
    """Bind new evaluations to the frozen task; preserve historical sources.

    OA guidance belongs to the method, not the shared physical task. It may
    differ from a native baseline; the full plant, sensor and arrival contract
    must match exactly. A learned model's supplied config is never overwritten.
    """
    root=Path(directory);path=root/'manifest.json'
    if not path.exists():
        if requested is not None and requested.stationary_obstacles:
            raise ValueError('Stationary unicycle inputs require the explicit source schema')
        return requested if requested is not None else UnicycleConfig()
    m=json.loads(path.read_text())
    if m.get('schema')!=SCHEMA:
        if m.get('stationary_physical_obstacles') is True or (requested is not None and requested.stationary_obstacles):
            raise ValueError('Stationary unicycle inputs require the explicit source schema')
        return requested if requested is not None else UnicycleConfig()
    saved=m['robot']
    if set(saved)!=set(asdict(UnicycleConfig())):raise ValueError('Incomplete unicycle physical source contract')
    robot=UnicycleConfig(**saved)
    if not robot.stationary_obstacles or m.get('stationary_physical_obstacles') is not True:
        raise ValueError('Static unicycle source must bind stationary physical truth')
    if sha256(root/'scenes.json')!=m['scenes_sha256']:raise ValueError('Changed unicycle scenes')
    rows=json.loads((root/'scenes.json').read_text())
    ids=[r['scene']['scene_id'] for r in rows]
    if len(rows)!=m['groups'] or len(set(ids))!=len(ids):raise ValueError('Missing or duplicate unicycle parents')
    for r in rows:physical_obstacle_scope('unicycle',r['scene']['obstacles'],r['scene']['obstacle_mask'])
    if requested is not None:
        mismatched=[k for k in PHYSICAL_FIELDS if getattr(requested,k)!=getattr(robot,k)]
        if mismatched:raise ValueError('Unicycle source physical mismatch: '+', '.join(mismatched))
        return requested
    return robot


import hashlib


import time

import jax


import numpy as np

from .io import sanitize


from .controllers import NativeOSQP

from .controllers import NativeProxQP

from .controllers import NativeClarabel

from .io import source_fingerprint

from .dynamics import integrate_unicycle, signed_clearance, swept_disk_clearance

from .io import write_json

from .routing import physical_route_coordinate

from .simulation import RUNNING, GOAL, COLLISION, INFEASIBLE, TIMEOUT, STATUS_NAMES

from .route_control import INADMISSIBLE

from .closed_loop import PLANNER_FAILURE

from .stochastic import conditioned_sensor_model, STATE_BOUND_VIOLATION

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

def run(source,output,robot_config=None,formulation='repo',alpha1=.5,alpha2=.5,penalty=1e4,
        noise_scale=1.,batch=32,steps=800,limit=None,backend='osqp'):
    if backend not in ('osqp','proxqp','clarabel'):raise ValueError('Unknown native solver')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    requested=UnicycleConfig(**json.loads(Path(robot_config).read_text())) if robot_config else None
    robot=source_robot(source,requested)
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
