"""V85 fixed-shape backend qualification and untrained full-state OA pilot.

All sources are frozen before timing. No learned/gain-search/baseline superiority
claim. Compiled executables are called directly, never implicit runtime jit.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import time
import importlib.metadata
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from .quad3d import integrate_quad3d
from .quad3d_control import Quad3DControlConfig, quad3d_control, quad3d_problem, control_config, arrived
from .quad3d_audit import audit_hold, independent_obstacle_values, independent_envelope_values, independent_held_cascade_minimum
from .io import write_json
from .cli import sanitize
from .dataset import sha256


def read(path):return json.loads(Path(path).read_text())


def frozen_source(output,hold_guard='none'):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(6851)
    families=('free_3d','single_blocker','crossing','staggered')
    parents=[]
    for capacity in (8,16,32,64):
        for i in range(16):
            x=np.zeros(12);x[:3]=[0.,0.,1.];x[3:6]=rng.uniform(-.025,.025,3)
            goal=np.array([rng.uniform(5,8),rng.uniform(-2,2),rng.uniform(.2,2.8)])
            family=families[i%4];o=np.zeros((capacity,5));mask=np.zeros(capacity,bool)
            if family!='free_3d':
                mask[:]=True;o[:,0]=rng.uniform(2,10,capacity);o[:,1]=rng.choice([-1,1],capacity)*rng.uniform(2,5,capacity)
                o[:,2]=rng.uniform(.1,.35,capacity)
                o[0]=[3.,0.,.35,0.,0.]
                if family=='crossing':o[0]=[3.,-1.,.25,0.,.18]
                if family=='staggered':o[0,:2]=[2.5,.45];o[1]=[4.,-.45,.3,0.,0.]
            parents.append(dict(id=f'quad3d_v85:{capacity}:{i}',capacity=capacity,family=family,
                                x=x.tolist(),goal=goal.tolist(),obstacles=o.tolist(),mask=mask.tolist(),gains=[2.]*4))
    write_json(out/'parents.json',parents)
    manifest=dict(schema='quad3d_full_linearized_v85',seed=6851,parents=64,
        parents_sha256=sha256(out/'parents.json'),config=asdict(Quad3DControlConfig(hold_guard=hold_guard)),
        perfect_current_observations=True,training_use=False,final_test=False,
        note='Initial controller/backend diagnosis. No source filtering or learned superiority claim. Full 12-state plant; infinite vertical cylinders; three-dimensional goals.')
    if hold_guard=='bernstein_v87':
        # Prespecified numerical qualification probes: LAST APPLIED state of
        # EVERY V86 staggered parent, whether it succeeded or failed. These do
        # not enter training/evaluation denominators or replace any pilot scene.
        probes=[];old=Path('artifacts/experiments/quad3d_v86_stable_nominal')
        review=read('reports/quad3d_v86_review.json')
        assert review['all_trace_audit_source_benchmark_bindings_verified'] and review['report_sha256']==sha256('reports/quad3d_v86_stable_nominal.json')
        for parent in parents:
            if parent['family']!='staggered':continue
            directory=old/f'capacity{parent["capacity"]}'/'pilot'
            row=next(v for v in read(directory/'index.json') if v['id']==parent['id'])
            assert sha256(directory/row['file'])==row['sha256'] and row['steps']>0
            tick=row['steps']-1
            with np.load(directory/row['file']) as z:x=z['state'][tick].copy()
            obs=np.asarray(parent['obstacles']);obs[:,:2]+=tick*.05*obs[:,3:5]
            probes.append(dict(parent,x=x.tolist(),obstacles=obs.tolist(),probe_tick=tick,
                original_trace_sha256=row['sha256'],original_status=row['status']))
        write_json(out/'probes.json',probes)
        manifest.update(probes=16,probes_sha256=sha256(out/'probes.json'),probe_rule='Last applied state of every V86 staggered parent; numerical solver qualification only',
                        prior_review_sha256=sha256('reports/quad3d_v86_review.json'))
    write_json(out/'manifest.json',manifest)


def source(output,capacity):
    p=Path(output);m=read(p/'manifest.json');assert m['parents_sha256']==sha256(p/'parents.json')
    rows=[r for r in read(p/'parents.json') if r['capacity']==capacity];assert rows
    return m,rows


def arrays(rows):
    return tuple(np.asarray([r[k] for r in rows],bool if k=='mask' else np.float64) for k in ('x','goal','obstacles','mask','gains'))


def to_device(values):
    # device_put alone canonicalizes host float64 to float32 when global x64 is
    # disabled. Explicit typed conversion preserves this numerical contract
    # without changing the defaults of future neural inference.
    result=tuple(jnp.asarray(v,dtype=jnp.bool_ if v.dtype==bool else jnp.float64) for v in values)
    for host,device in zip(values,result):np.testing.assert_array_equal(host,np.asarray(device))
    return result


def compile_fn(fn,args):
    f=jax.jit(fn);start=time.perf_counter();exe=f.lower(*args).compile()
    return f,exe,time.perf_counter()-start


def measure(exe,args,repeats=25,host_result=False):
    jax.block_until_ready(exe(*args));samples=[];result=None
    for _ in range(repeats):
        begin=time.perf_counter();result=exe(*args)
        result=jax.device_get(result) if host_result else jax.block_until_ready(result)
        samples.append(time.perf_counter()-begin)
    return result,dict(p50_seconds=float(np.median(samples)),p95_seconds=float(np.quantile(samples,.95)),
                       p99_seconds=float(np.quantile(samples,.99)),repeats=repeats,host_result=host_result)


def benchmark(src,output,capacity):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);m,parents=source(src,capacity)
    c=control_config(m['config']);host=arrays(parents);args=to_device(host)
    for a,b in zip(host,args):np.testing.assert_array_equal(a,np.asarray(b))
    records=[];proof={};compiled=[]
    for batch in (1,16):
        inputs=tuple(a[:batch] for a in args)
        f,exe,cold=compile_fn(jax.vmap(lambda x,g,o,mask,k:quad3d_control(x,g,o,mask,k,c)),inputs)
        result,timing=measure(exe,inputs,host_result=batch==1)
        compiled.append(f);records.append(dict(kind='complete_controller',batch=batch,compile_seconds=cold,**timing))
        if batch==16:proof=dict(control=np.asarray(result[0]),accepted=np.asarray(result[1]),psi=np.asarray(result[2]),
                                domain=np.asarray(result[3]),violation=np.asarray(result[4]),iterations=np.asarray(result[5]))
    f,exe,cold=compile_fn(jax.vmap(make_rollout(40,c)),args)
    _,timing=measure(exe,args,repeats=3)
    compiled.append(f);records.append(dict(kind='rollout_40',batch=16,compile_seconds=cold,**timing))
    probes=[]
    if m.get('probes'):
        assert m['probes_sha256']==sha256(Path(src)/'probes.json')
        probes=[p for p in read(Path(src)/'probes.json') if p['capacity']==capacity];assert len(probes)==4
        probe_args=to_device(arrays(probes))
        f,exe,cold=compile_fn(jax.vmap(lambda x,g,o,mask,k:quad3d_control(x,g,o,mask,k,c)),probe_args)
        result,timing=measure(exe,probe_args);compiled.append(f)
        records.append(dict(kind='boundary_controller',batch=4,compile_seconds=cold,**timing))
        for key,value in zip(('control','accepted','psi','domain','violation','iterations'),result):
            proof[key]=np.concatenate((proof[key],np.asarray(value)))
    oracle_args=to_device(arrays(parents+probes))
    # Numerical oracle only. This is NOT a tuned paper baseline.
    f,exe,cold=compile_fn(jax.vmap(lambda x,g,o,mask,k:quad3d_problem(x,g,o,mask,k,c)[:3]),oracle_args)
    compiled.append(f);ref,a,b=map(np.asarray,exe(*oracle_args));reference=[]
    import osqp
    import scipy.sparse as sp
    native_times=[];native_counts={'solved':0,'raw_residual_accepted':0}
    for i in range(len(ref)):
        opt=minimize(lambda u:.5*np.sum((u-ref[i])**2),np.zeros(4),jac=lambda u:u-ref[i],
                     constraints={'type':'ineq','fun':lambda u:b[i]-a[i]@u,'jac':lambda u:-a[i]},
                     method='SLSQP',options={'ftol':1e-11,'maxiter':1000})
        violation=float(np.max(a[i]@opt.x-b[i]));oracle=bool(opt.success and violation<1e-7)
        err=float(np.max(abs(opt.x-proof['control'][i]))) if oracle and proof['accepted'][i] else None
        reference.append(dict(oracle_success=oracle,oracle_violation=violation,solver_accepted=bool(proof['accepted'][i]),action_error=err))
        begin=time.perf_counter();solver=osqp.OSQP();solver.setup(P=sp.eye(4,format='csc'),q=-ref[i],A=sp.csc_matrix(a[i]),
                                      l=np.full(len(b[i]),-np.inf),u=b[i],verbose=False);result=solver.solve()
        native_times.append(time.perf_counter()-begin)
        native_counts['solved']+=int(result.info.status_val==1)
        native_counts['raw_residual_accepted']+=int(result.info.status_val==1 and np.max(a[i]@result.x-b[i])<=c.qp_tolerance)
    np.savez_compressed(out/'numerics.npz',**proof,reference=ref,a=a,b=b)
    eligible=all(r['oracle_success'] and r['solver_accepted'] and r['action_error'] is not None and r['action_error']<3e-5 for r in reference)
    report=dict(device=str(jax.devices()[0]),device_kind=jax.devices()[0].device_kind,capacity=capacity,source_sha256=sha256(Path(src)/'manifest.json'),
                numerics_sha256=sha256(out/'numerics.npz'),records=records,reference=reference,accuracy_eligible=eligible,
                implicit_jit_cache_entries=[f._cache_size() for f in compiled],
                native_osqp_default=dict(**native_counts,p50_setup_solve_seconds=float(np.median(native_times)),cases=len(ref),
                    note='Timing reference for the SAME OA QP, not a paper baseline; default OSQP options except quiet output.'),
                versions={k:importlib.metadata.version(k) for k in ('jax','jaxlib','qpax','numpy','scipy','osqp')})
    import qpax
    dependency_root=Path(qpax.__file__).parent
    report['qpax_sources']={str(p.relative_to(dependency_root)):sha256(p) for p in sorted(dependency_root.rglob('*.py'))}
    write_json(out/'report.json',sanitize(report));print(json.dumps(sanitize(report)),flush=True)


def make_rollout(steps=600,config=Quad3DControlConfig()):
    c=config
    def rollout(x,goal,obs,mask,gains):
        dt=jnp.asarray(np.asarray(c.robot.dt,np.float64),x.dtype)
        def tick(carry,k):
            state,status,count=carry
            seen=obs.at[:,:2].add(k.astype(state.dtype)*dt*obs[:,3:5])
            u,feasible,psi,domain,residual,it=quad3d_control(state,goal,seen,mask,gains,c)
            at_goal=arrived(state,goal,c)
            status=jnp.where((status==0)&at_goal,1,status)
            status=jnp.where((status==0)&((psi < -c.qp_tolerance)|(domain < -c.qp_tolerance)),2,status)
            status=jnp.where((status==0)&~feasible,3,status)
            active=status==0
            applied=jnp.where(active,u,jnp.zeros(4,state.dtype))
            yy,sub=integrate_quad3d(state,applied,c.robot)
            next_state=jnp.where(active,yy,state)
            # Runtime sampled physical screen. Independent polynomial audit
            # below checks the whole held interval and cannot erase a failure.
            times=(k+jnp.asarray(np.arange(1,c.robot.integration_substeps+1)/c.robot.integration_substeps,state.dtype))*dt
            centers=obs[None,:,:2]+times[:,None,None]*obs[None,:,3:5]
            distances=jnp.linalg.norm(sub[:,None,:2]-centers,axis=-1)-c.robot.radius-obs[None,:,2]
            clear=jnp.min(jnp.where(mask[None,:],distances,jnp.inf))
            limits=jnp.asarray(np.array([c.tilt_limit,c.tilt_limit,c.yaw_limit,c.velocity_limit,c.velocity_limit,c.velocity_limit,c.rate_limit,c.rate_limit,c.rate_limit]),state.dtype)
            violation=jnp.maximum(jnp.max(jnp.abs(sub[:,3:])-limits),jnp.maximum(jnp.max(sub[:,2]-c.altitude_max),jnp.max(c.altitude_min-sub[:,2])))
            status=jnp.where(active&(clear<=0),4,status)
            status=jnp.where(active&(status==0)&(violation>c.qp_tolerance),5,status)
            data=dict(state=state,next_state=next_state,control=applied,proposed=u,active=active,status=status,
                      psi=psi,domain=domain,residual=residual,iterations=it,clearance=clear,envelope=violation)
            return (next_state,status,count+active.astype(jnp.int32)),data
        result,trace=jax.lax.scan(tick,(x,jnp.int32(0),jnp.int32(0)),jnp.arange(steps,dtype=jnp.int32))
        state,status,count=result
        status=jnp.where((status==0)&arrived(state,goal,c),1,status)
        status=jnp.where(status==0,6,status)
        return dict(final_state=state,status=status,steps=count),trace
    return rollout


def collect(src,output,capacity,steps=600):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);m,parents=source(src,capacity)
    c=control_config(m['config']);args=to_device(arrays(parents));f,exe,cold=compile_fn(jax.vmap(make_rollout(steps,c)),args)
    start=time.perf_counter();summary,trace=jax.device_get(exe(*args));elapsed=time.perf_counter()-start
    rows=[]
    for i,p in enumerate(parents):
        data={k:v[i] for k,v in trace.items()};file=f'{i:03d}.npz';np.savez_compressed(out/file,**data)
        rows.append(dict(id=p['id'],family=p['family'],file=file,sha256=sha256(out/file),
                         status=int(summary['status'][i]),steps=int(summary['steps'][i]),final_state=summary['final_state'][i].tolist()))
    write_json(out/'index.json',rows)
    write_json(out/'manifest.json',dict(source=str(Path(src).resolve()),source_sha256=sha256(Path(src)/'manifest.json'),capacity=capacity,
        index_sha256=sha256(out/'index.json'),steps=steps,compile_seconds=cold,execute_seconds=elapsed,
        physical_steps=sum(r['steps'] for r in rows),implicit_jit_cache_entries=f._cache_size(),device=str(jax.devices()[0]),
        config=asdict(c),untrained_fixed_gain_diagnosis=True))
    print(json.dumps(dict(stage='collected',capacity=capacity,seconds=elapsed,statuses=[r['status'] for r in rows])),flush=True)


def audit_parent(arguments):
    """One complete independently replayed parent, suitable for spawned CPU workers."""
    p,row,directory,m=arguments;path=Path(directory);c=control_config(m['config'])
    assert p['id']==row['id'] and sha256(path/row['file'])==row['sha256']
    with np.load(path/row['file']) as z:d={k:z[k] for k in z.files}
    np.testing.assert_array_equal(d['state'][0],p['x']);mask=np.asarray(p['mask']);obs=np.asarray(p['obstacles'])
    np.testing.assert_array_equal(np.flatnonzero(d['active']),np.arange(row['steps']))
    np.testing.assert_array_equal(d['control'][~d['active']],np.zeros((m['steps']-row['steps'],4)))
    initial=np.asarray(p['x']);limits=np.array([c.tilt_limit,c.tilt_limit,c.yaw_limit,
        c.velocity_limit,c.velocity_limit,c.velocity_limit,c.rate_limit,c.rate_limit,c.rate_limit])
    min_clear=float(np.min(np.linalg.norm(initial[:2]-obs[mask,:2],axis=-1)-c.robot.radius-obs[mask,2],initial=np.inf))
    physical_collision=min_clear<=0
    envelope_exit=bool(np.any(np.abs(initial[3:])-limits>c.qp_tolerance) or
        initial[2]>c.altitude_max+c.qp_tolerance or initial[2]<c.altitude_min-c.qp_tolerance)
    max_error=0.;applied=0;min_cascade=float('inf')
    for k in range(m['steps']):
        if k:np.testing.assert_array_equal(d['state'][k],d['next_state'][k-1])
        if not d['active'][k]:
            np.testing.assert_array_equal(d['state'][k],d['next_state'][k]);continue
        seen=obs.copy();seen[:,:2]+=k*c.robot.dt*obs[:,3:5]
        r=audit_hold(d['state'][k],d['control'][k],d['next_state'][k],seen,mask,c)
        psi,residual=independent_obstacle_values(d['state'][k],d['control'][k],seen,mask,p['gains'],c)
        if psi < -c.qp_tolerance or residual < -c.qp_tolerance-1e-8:raise AssertionError('Applied independent obstacle CBF failed')
        domain,er=independent_envelope_values(d['state'][k],d['control'][k],c)
        if domain < -c.qp_tolerance or er < -c.qp_tolerance-1e-8:raise AssertionError('Applied independent envelope CBF failed')
        np.testing.assert_allclose(d['psi'][k],psi,atol=1e-8,rtol=1e-10)
        np.testing.assert_allclose(d['domain'][k],domain,atol=1e-8,rtol=1e-10)
        if c.hold_guard!='none':
            held=independent_held_cascade_minimum(d['state'][k],d['control'][k],seen,mask,p['gains'],c)
            if held < -c.qp_tolerance-1e-8:raise AssertionError(f'Actual held CBF cascade became negative: {held}')
            min_cascade=min(min_cascade,held)
        max_error=max(max_error,r['replay_error']);min_clear=min(min_clear,r['minimum_clearance'])
        physical_collision|=r['minimum_clearance']<=0;envelope_exit|=r['envelope_violation']>c.qp_tolerance
        applied+=1
    assert applied==row['steps']
    np.testing.assert_array_equal(d['next_state'][-1],row['final_state'])
    # Every continuous-time miss remains an evaluation failure, even if the
    # cheaper runtime sample grid did not detect it. Never silently certify.
    audited_status=4 if physical_collision else 5 if envelope_exit else row['status']
    if audited_status==1:
        xx=np.asarray(row['final_state']);assert np.linalg.norm(xx[:3]-p['goal'])<=c.goal_tolerance and np.linalg.norm(xx[6:9])<=c.terminal_speed
        assert np.max(abs(xx[3:6]))<=c.terminal_attitude and np.max(abs(xx[9:12]))<=c.terminal_rate
    return dict(**row,audited_status=audited_status,minimum_clearance=min_clear,max_replay_error=max_error,
                        physical_collision=physical_collision,envelope_exit=envelope_exit,minimum_held_cascade=min_cascade)


def audit(directory,workers=1):
    path=Path(directory);m=read(path/'manifest.json');sm,parents=source(m['source'],m['capacity']);rows=read(path/'index.json')
    assert m['config']==sm['config']
    assert sha256(Path(m['source'])/'manifest.json')==m['source_sha256'] and sha256(path/'index.json')==m['index_sha256']
    if 'parent_ids' in m:
        by_id={p['id']:p for p in parents};assert len(by_id)==len(parents)
        assert len(set(m['parent_ids']))==len(m['parent_ids'])
        parents=[by_id[i] for i in m['parent_ids']]
    arguments=[(p,r,str(path),m) for p,r in zip(parents,rows,strict=True)]
    if workers==1:results=[audit_parent(a) for a in arguments]
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            results=[]
            for result in pool.map(audit_parent,arguments):
                results.append(result)
                if len(results)%8==0 or len(results)==len(arguments):
                    write_json(path/'audit_progress.json',dict(parents_audited=len(results),parents_total=len(arguments),
                        physical_steps=sum(r['steps'] for r in results)))
    total=sum(r['steps'] for r in results)
    write_json(path/'independent_audit.json',sanitize(dict(audit_passed=True,manifest_sha256=sha256(path/'manifest.json'),
        index_sha256=sha256(path/'index.json'),physical_steps=total,rows=results,whole_project_complete=False)))
    print(json.dumps(dict(stage='audited',capacity=m['capacity'],physical_steps=total)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','benchmark','collect','audit']);ap.add_argument('--source');ap.add_argument('--output',required=True);ap.add_argument('--capacity',type=int);ap.add_argument('--steps',type=int,default=600);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--hold-guard',choices=['none','bernstein_v87'],default='none');args=ap.parse_args()
    if args.action=='prepare':frozen_source(args.output,args.hold_guard)
    elif args.action=='benchmark':benchmark(args.source,args.output,args.capacity)
    elif args.action=='collect':collect(args.source,args.output,args.capacity,args.steps)
    else:audit(args.output,args.workers)
