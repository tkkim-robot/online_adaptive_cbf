"""Versioned perfect-observation bicycle mechanics and gain-sensitivity pilot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict,replace
import json
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
from .dataset import sha256,source_fingerprint
from .io import write_json
from .cli import sanitize
from .bicycle_control import BicycleControlConfig
from .bicycle import BicycleConfig
from .bicycle_rollout import make_bicycle_episode,NAMES


def read(path):return json.loads(Path(path).read_text())


def control_config(value):
    """Restore the complete versioned physical/control contract, no defaults substitution."""
    value=dict(value);value['robot']=BicycleConfig(**value['robot'])
    config=BicycleControlConfig(**value)
    if asdict(config)!=dict(value,robot=asdict(value['robot'])):raise ValueError('Noncanonical bicycle configuration')
    return config


def prepare(output,groups=128,seed=6651):
    from .multiscale_scenes import scene
    from .scenes import DIVERSE_FAMILIES
    from .routing import plan_route
    if groups%8 or groups<8:raise ValueError('Balanced eight-family parents required')
    p=Path(output);p.mkdir(parents=True,exist_ok=False);config=BicycleControlConfig()
    def make(i):
        family=DIVERSE_FAMILIES[i%8];s=scene(seed*100000+i,family);rng=np.random.default_rng(s.seed+337)
        state=s.initial_state.astype(np.float32);state[3]=rng.uniform(.25,1.)
        route=plan_route(state[:2],s.goal,s.obstacles,s.obstacle_mask,config,capacity=64,visibility_batch_nodes=32)
        return dict(group_id=f'bicycle_v63:{family}:{s.seed}',seed=s.seed,family=family,initial_state=state.tolist(),goal=s.goal.astype(np.float32).tolist(),
            obstacles=s.obstacles.astype(np.float32).tolist(),obstacle_mask=s.obstacle_mask.tolist(),
            route={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in asdict(route).items()},solvability='unknown')
    with ThreadPoolExecutor(32) as pool:rows=list(pool.map(make,range(groups)))
    write_json(p/'scenes.json',rows)
    write_json(p/'manifest.json',dict(schema='oa_cbf_affine_bicycle_pilot_v63',source_fingerprint=source_fingerprint(),groups=groups,seed=seed,
        config=asdict(config),capacity=64,route_capacity=64,scenes_sha256=sha256(p/'scenes.json'),training_use=False,final_test=False,
        observation='Perfect current state and moving-obstacle observations. No learned weights, noisy-observation calibration or adaptive performance claim.',
        initial_speed='Independent prespecified uniform.25..1 within the retained.2..3.5 envelope; not clipped during simulation.',
        task='Reach goal disk.25m with speed<=.35m/s; rolling arrival, no standstill/post-terminal claim.',
        sampling=f'{groups}newparents over8multiscale families, rotations/translations,<=16/32/48/64obstacles. All outcomes retained; geometric route is not a feasible-control witness.',
        gains=np.geomspace(.5,8.,8).astype(np.float32).tolist()))


def arrays(records):
    return tuple(np.asarray([r[k] for r in records],np.float32 if k!='obstacle_mask' else bool) for k in ('initial_state','goal','obstacles','obstacle_mask'))+(
        np.asarray([r['alpha'] for r in records],np.float32),np.asarray([r['route']['points'] for r in records],np.float32),
        np.asarray([r['route']['mask'] for r in records],bool),np.asarray([r['route']['status']=='ready' for r in records],bool))


def run(source,output,shard=0,shards=4,batch=32,steps=1600,solver='interval',guidance_horizon=0):
    source=Path(source);p=Path(output);p.mkdir(parents=True,exist_ok=False)
    sm=read(source/'manifest.json');config=replace(control_config(sm['config']),solver=solver)
    from .bicycle_guidance import BicycleGuidanceConfig
    guidance=BicycleGuidanceConfig(guidance_horizon) if guidance_horizon else None
    if sm['scenes_sha256']!=sha256(source/'scenes.json'):raise ValueError('Changed bicycle source')
    parents=read(source/'scenes.json');rows=[]
    for i,r in enumerate(parents):
        if i%shards==shard:rows.extend(dict(r,alpha=g,candidate=k) for k,g in enumerate(sm['gains']))
    if not rows or len(rows)%batch:raise ValueError('Fixed batch must divide retained branches')
    manifest=dict(schema='oa_cbf_affine_bicycle_episodes_v63',config=asdict(config),source=str(source.resolve()),
        source_manifest_sha256=sha256(source/'manifest.json'),source_fingerprint=source_fingerprint(),shard=shard,shards=shards,
        batch=batch,steps=steps,parents=len(rows)//len(sm['gains']),branches=len(rows),gains=sm['gains'],
        observation=sm['observation'],task=sm['task'],device=str(jax.devices()[0]),final_test=False,
        precision='Accumulated physical64, explicitly reduced current observations32 and actuators32, actual centers logged, row construction/solve64',
        observation_schema='bicycle_rounded_current_centers_v63')
    write_json(p/'manifest.json',manifest)
    if guidance is not None:
        manifest.update(guidance=asdict(guidance),guidance_schema='bicycle_constrained_nominal_profiles_v65',
            guidance_contract='34nominal profiles, supplied gain unchanged, constrained exact-bicycle previews. First applied input is the previewed projected command; incomplete previews rank by retained prefix. Current observations only, no robustness certificate.')
        write_json(p/'manifest.json',manifest)
    fn=jax.jit(jax.vmap(make_bicycle_episode(config,steps,guidance)));dummy=tuple(jnp.asarray(x) for x in arrays(rows[:batch]))
    t=time.monotonic();executable=fn.lower(*dummy).compile();compile_seconds=time.monotonic()-t
    print(json.dumps(dict(stage='compiled',compile_seconds=compile_seconds,device=str(jax.devices()[0]))),flush=True)
    index=[];start=time.monotonic()
    for offset in range(0,len(rows),batch):
        group=rows[offset:offset+batch];args=tuple(jnp.asarray(x) for x in arrays(group));t=time.monotonic()
        summaries,traces=jax.device_get(executable(*args));seconds=time.monotonic()-t
        for i,row in enumerate(group):
            status=int(summaries['status'][i]);count=int(summaries['steps'][i]);length=max(1,min(steps,count+int(status not in (1,2,4,7))))
            trace={k:v[i,:length] for k,v in traces.items()};trace.update(initial_state=np.asarray(row['initial_state'],np.float32),
                obstacles=np.asarray(row['obstacles'],np.float32),obstacle_mask=np.asarray(row['obstacle_mask'],bool),goal=np.asarray(row['goal'],np.float32),
                points=np.asarray(row['route']['points'],np.float32),route_mask=np.asarray(row['route']['mask'],bool),alpha=np.float32(row['alpha']))
            file=f'episode_{offset+i:05d}.npz';np.savez_compressed(p/file,**trace)
            index.append(sanitize(dict(group_id=row['group_id'],family=row['family'],candidate=row['candidate'],alpha=row['alpha'],
                obstacles=int(sum(row['obstacle_mask'])),file=file,sha256=sha256(p/file),status_code=status,status=NAMES[status],steps=count,
                min_clearance=float(summaries['min_clearance'][i]),goal_progress=float(summaries['goal_progress'][i]))))
        write_json(p/'index.json',index)
        print(json.dumps(dict(completed=len(index),total=len(rows),batch_seconds=seconds,physical_steps=int(summaries['steps'].sum()))),flush=True)
    write_json(p/'summary.json',dict(complete=True,compiled_signatures=1,compile_seconds=compile_seconds,execution_seconds=time.monotonic()-start,
        branches=len(index),physical_steps=sum(r['steps'] for r in index),outcomes={k:sum(r['status']==k for r in index) for k in sorted({r['status'] for r in index})}))


def audit_episode(payload):
    from scipy.integrate import solve_ivp
    from .bicycle_audit import reference_flow,reference_rows,polygon_qp
    p,m,original,row=payload
    config=control_config(m['config']);c=config.robot
    if sha256(p/row['file'])!=row['sha256']:raise ValueError('Changed saved bicycle episode')
    with np.load(p/row['file']) as f:d={k:f[k] for k in f}
    for field in ('initial_state','goal','obstacles','obstacle_mask'):
        np.testing.assert_array_equal(d[field],np.asarray(original[field],d[field].dtype))
    for field,key in [('points','points'),('route_mask','mask')]:
        np.testing.assert_array_equal(d[field],np.asarray(original['route'][key],d[field].dtype))
    assert float(d['alpha'])==row['alpha']==m['gains'][row['candidate']]
    if np.any(np.diff(d['active'].astype(int))>0):raise ValueError('Applied control after terminal rejection')
    physical=d['initial_state'].astype(float)
    minimum=float(np.min(np.where(d['obstacle_mask'],np.linalg.norm(physical[:2]-d['obstacles'][:,:2],axis=1)-c.radius-d['obstacles'][:,2],np.inf)))
    max_error=0.;checked=0;worst=-np.inf;feasible_rejection=False
    for k,accepted in enumerate(d['active']):
        if not accepted:continue
        observation=d['observed_state'][k].astype(float)
        np.testing.assert_allclose(observation,physical,atol=3e-5,rtol=2e-6)
        current=d['obstacles'].astype(float).copy();current[:,:2]+=k*c.dt*current[:,3:5]
        seen=current.copy()
        if m.get('observation_schema')=='bicycle_rounded_current_centers_v63':
            np.testing.assert_array_equal(d['observed_centers'][k],current[:,:2].astype(np.float32))
            seen[:,:2]=d['observed_centers'][k]
        a,b,h,domain=reference_rows(observation,seen,d['obstacle_mask'],float(d['alpha']),config)
        if np.any(h[d['obstacle_mask']] < -config.qp_tolerance) or np.any(domain[d['obstacle_mask']]<=0):raise ValueError('Applied inadmissible barrier')
        residual=float(np.max(a@d['control'][k].astype(float)-b));worst=max(worst,residual)
        if residual>config.qp_tolerance:raise ValueError(f'Applied command violates original CBF/input row: {row["group_id"]},candidate{row["candidate"]},tick{k},residual{residual}')
        times=np.linspace(0,c.dt,101)
        sol=solve_ivp(lambda t,x:reference_flow(t,x,d['control'][k],config),(0,c.dt),physical,method='DOP853',rtol=1e-11,atol=1e-12,t_eval=times)
        if not sol.success:raise ValueError('Independent integration failed')
        error=float(np.max(np.abs(sol.y[:,-1]-d['state'][k])));max_error=max(max_error,error)
        np.testing.assert_allclose(d['state'][k],sol.y[:,-1],atol=3e-5,rtol=2e-6)
        centers=current[None,:,:2]+times[:,None,None]*current[None,:,3:5]
        distances=np.linalg.norm(sol.y[:2].T[:,None,:]-centers,axis=-1)-c.radius-current[None,:,2]
        clear=float(np.min(np.where(d['obstacle_mask'][None],distances,np.inf)));minimum=min(minimum,clear)
        speed_violation=float(max(c.speed_min-np.min(sol.y[3]),np.max(sol.y[3])-c.speed_max))
        if abs(clear-float(d['clearance'][k]))>1e-4 and np.isfinite(clear):raise ValueError('Physical clearance replay mismatch')
        if clear < -1e-4 and row['status_code']!=2:raise ValueError('Unreported physical collision')
        if speed_violation>config.qp_tolerance+1e-6 and row['status_code']!=7:raise ValueError('Unreported physical speed violation')
        physical=sol.y[:,-1];checked+=1
    if checked!=row['steps']:raise ValueError('Missing physical steps')
    if row['status_code'] in (3,5):
        k=checked;x=d['observed_state'][-1].astype(float)
        np.testing.assert_allclose(x,physical,atol=3e-5,rtol=2e-6)
        current=d['obstacles'].astype(float).copy();current[:,:2]+=k*c.dt*current[:,3:5]
        if m.get('observation_schema')=='bicycle_rounded_current_centers_v63':
            np.testing.assert_array_equal(d['observed_centers'][-1],current[:,:2].astype(np.float32))
            current[:,:2]=d['observed_centers'][-1]
        radii=(c.radius+config.clearance_buffer+current[:,2])*config.barrier_inflation
        domain=np.sum((current[:,:2]-x[:2])**2,axis=1)-radii**2
        if np.all(domain[d['obstacle_mask']]>0):
            a,b,h,_=reference_rows(x,current,d['obstacle_mask'],float(d['alpha']),config)
            if row['status_code']==5 and np.all(h[d['obstacle_mask']]>=-config.qp_tolerance):raise ValueError('False barrier rejection')
            if row['status_code']==3:
                feasible_rejection=polygon_qp([0.,0.],a,b,[1.,1.],[-c.acceleration_max,-c.slip_max],[c.acceleration_max,c.slip_max]) is not None
        elif row['status_code']==3:raise ValueError('Geometric rejection mislabeled as QP rejection')
    if row['status_code']==4 and checked!=m['steps']:raise ValueError('Premature timeout')
    if row['status_code']==6 and original['route']['status']=='ready':raise ValueError('False planner failure')
    if row['status_code']==2 and minimum>1e-4:raise ValueError('False physical collision label')
    if row['status_code']==1 and not (np.linalg.norm(physical[:2]-d['goal'])<=config.goal_tolerance+1e-4 and physical[3]<=config.terminal_speed+1e-5):raise ValueError('False rolling arrival')
    return sanitize(dict(group_id=row['group_id'],candidate=row['candidate'],steps=checked,min_clearance=minimum,max_state_error=max_error,max_qp_residual=worst,feasible_qp_rejected=feasible_rejection,audit_passed=True))


def audit(directory,workers=1,audit_file='independent_replay.json'):
    p=Path(directory);m=read(p/'manifest.json');rows=read(p/'index.json');config=control_config(m['config']);c=config.robot
    if asdict(config)!=m['config'] or sha256(Path(m['source'])/'manifest.json')!=m['source_manifest_sha256']:raise ValueError('Changed physical contract')
    sm=read(Path(m['source'])/'manifest.json');parents=read(Path(m['source'])/'scenes.json')
    if sha256(Path(m['source'])/'scenes.json')!=sm['scenes_sha256']:raise ValueError('Changed parent scenes')
    source={r['group_id']:r for r in parents}
    expected=[(r['group_id'],k) for i,r in enumerate(parents) if i%m['shards']==m['shard'] for k in range(len(m['gains']))]
    if [(r['group_id'],r['candidate']) for r in rows]!=expected:raise ValueError('Missing, duplicate or reordered branches')
    if workers<1:raise ValueError('Positive audit worker count required')
    payloads=[(p,m,source[row['group_id']],row) for row in rows]
    if workers==1:
        reports=list(map(audit_episode,payloads))
    else:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        with ProcessPoolExecutor(workers,mp_context=get_context('spawn')) as pool:
            reports=list(pool.map(audit_episode,payloads,chunksize=1))
    result=dict(audit_passed=True,episodes=len(rows),steps=sum(r['steps'] for r in reports),manifest_sha256=sha256(p/'manifest.json'),index_sha256=sha256(p/'index.json'),
        source_fingerprint=source_fingerprint(),workers=workers,all_physical_prefixes_replayed=True,all_applied_rows_checked_by_complex_step=True,rows=reports,
        limitation='Perfect-observation fixed-gain development pilot, rolling arrival. No learned/generalization or post-stop safety claim. SciPy trajectory samples are an independent numerical audit, not a continuous-time theorem.')
    write_json(p/audit_file,result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--output',required=True);q.add_argument('--groups',type=int,default=128);q.add_argument('--seed',type=int,default=6651)
    r=sub.add_parser('run');r.add_argument('--source',required=True);r.add_argument('--output',required=True)
    for k,v in [('shard',0),('shards',4),('batch',32),('steps',1600),('guidance-horizon',0)]:r.add_argument('--'+k,type=int,default=v)
    r.add_argument('--solver',choices=['interval','enumeration'],default='interval')
    a=sub.add_parser('audit');a.add_argument('--directory',required=True);a.add_argument('--workers',type=int,default=1);a.add_argument('--audit-file',default='independent_replay.json')
    args=vars(p.parse_args());command=args.pop('command');dict(prepare=prepare,run=run,audit=audit)[command](**args)
