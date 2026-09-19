"""V92 bounded noisy-observation qualification on fresh, outcome-independent parents."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import multiprocessing
import json
import shutil
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import linprog, minimize
from . import quad3d_observation as sensor
from .quad3d_control import Quad3DControlConfig, control_config, quad3d_problem
from .quad3d_routing import observed_route, flight_target
from .quad3d_observed_rollout import make_observed_rollout, observed_control, observed_problem
from .quad3d_observation_audit import audit_parent
from .quad3d_foundation_experiment import source, read, to_device, compile_fn, measure
from .quad3d_diverse_experiment import FAMILIES
from .scenes import DIVERSE_FAMILIES
from .multiscale_scenes import scene as diverse_scene, contract
from .generalization_scenes import scene as structural_scene
from .routing import segment_clearances
from .dataset import sha256
from .io import write_json
from .cli import sanitize


def plan_parent(p):
    c = Quad3DControlConfig(hold_guard='bernstein_v87')
    x, o, mask, noise = [np.asarray(p[k]) for k in ('x', 'obstacles', 'mask', 'noise')]
    bx, bo, ix, io = sensor.unit_tape(p['sensor_seed'], 0, len(mask))
    seen, obs = sensor.numpy_observe(x, o, mask, bx, bo, noise, ix[0], io[0])
    guidance = sensor.numpy_obstacles(obs, mask, noise, True)
    route = observed_route(seen, p['goal'], guidance, mask, c)
    pp = route.points[route.mask]
    np.testing.assert_array_equal(pp[0], seen[:2]); np.testing.assert_array_equal(pp[-1], p['goal'][:2])
    static = mask & (np.linalg.norm(guidance[:, 3:5], axis=-1) < 1e-10)
    if route.status == 'ready':
        gap = segment_clearances(pp[:-1], pp[1:], guidance[static, :2], guidance[static, 2]+c.robot.radius+c.clearance_buffer+route.planning_margin)
        assert np.min(gap, initial=np.inf) >= -1e-8
    return dict(p, route=dict(points=route.points.tolist(), mask=route.mask.tolist(), status=route.status,
                             length=route.length, planning_margin=route.planning_margin))


def prepare(output):
    out = Path(output); out.mkdir(parents=True, exist_ok=False); physical = []
    for i in range(48):
        seed = 692100+i; family = FAMILIES[i % 12]
        scene = diverse_scene(seed, family) if family in DIVERSE_FAMILIES else structural_scene(seed, family, 64)
        rng = np.random.default_rng(seed+11007); x = np.zeros(12); x[:2] = scene.initial_state[:2]
        x[2] = rng.uniform(.4, 2.4); x[3:5] = rng.uniform(-.025, .025, 2); x[5] = rng.uniform(-.2, .2)
        x[6:8] = rng.uniform(-.08, .08, 2); x[8] = rng.uniform(-.05, .05); x[9:12] = rng.uniform(-.025, .025, 3)
        physical.append(dict(group_id=f'quad3d_v92:{family}:{seed}', index=i, seed=seed, sensor_seed=seed+1000000,
            family=family, capacity=64, x=x.tolist(), goal=np.r_[scene.goal, rng.uniform(.2, 2.8)].tolist(),
            obstacles=scene.obstacles.astype(np.float64).tolist(), mask=scene.obstacle_mask.tolist(), gains=[2.]*4,
            active_obstacles=int(scene.obstacle_mask.sum()), solvability='unknown; no resampling or outcome filtering'))
    # No pilot outcome has been seen before this frozen list is written.
    values = [dict(p, id=p['group_id']+f':noise{level}', noise_level=level, noise=(level*sensor.BASE_NOISE).tolist())
              for level in (0, 1, 2) for p in physical]
    write_json(out/'preplanning_parents.json', values)
    with ProcessPoolExecutor(12, mp_context=multiprocessing.get_context('spawn')) as pool:
        parents = list(pool.map(plan_parent, values))
    write_json(out/'parents.json', parents)
    write_json(out/'manifest.json', dict(schema=sensor.SCHEMA, parents=144, independent_physical_parents=48,
        capacity=64, families=list(FAMILIES), seed=692100, steps=1600, batch=12, noise_levels=[0, 1, 2],
        noise_fields=sensor.FIELDS, base_noise=sensor.BASE_NOISE.tolist(), innovation_fraction=.15,
        parents_sha256=sha256(out/'parents.json'), preplanning_sha256=sha256(out/'preplanning_parents.json'),
        config=asdict(Quad3DControlConfig(hold_guard='bernstein_v87')), geometry_distribution=contract(),
        route_statuses=dict(Counter(p['route']['status'] for p in parents)), training_use=False, final_test=False,
        information='Current full-state/cylinder observations and public noise support only. Persistent bias plus paired innovations.',
        safety='Geometry inflation1.15*(position_ball+cylinder_position_disk+radius_error). Not a derivative or recursive safety proof.',
        termination='Observed full-state arrival with worst-case public sensor error bounds; independent true arrival verification.',
        pairing='Same48physicalparents and unit sensor tapes at noise0/1/2. All parent-noise rows retained; no independent-count inflation.',
        policy='Untrained fixed[2,2,2,2] V91 QP; noisy observation and causal static-compatible route adapter.',
        source_files={name:sha256(Path(__file__).parent/name) for name in (
            'quad3d_observation.py','quad3d_observed_rollout.py','quad3d_observation_audit.py',
            'quad3d_observation_experiment.py','quad3d_control.py','quad3d_qp.py','quad3d.py','quad3d_held.py','quad3d_routing.py','quad3d_audit.py')}))
    print(json.dumps(dict(stage='prepared',parents=len(parents))), flush=True)


def parents_for_shard(src, shard):
    if not 0 <= shard < 4: raise ValueError('Expected one of four balanced lanes')
    m, parents = source(src, 64)
    parents = [p for p in parents if p['index']//12 == shard]
    assert len(parents) == 36
    return m, parents


def prepare_v93(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    prior=Path('artifacts/experiments/quad3d_v92_frozen_inputs');m,parents=source(prior,64)
    reviewed=read('reports/quad3d_v92_review.json');assert reviewed['all_trace_audit_source_runtime_report_bindings_verified']
    assert reviewed['report_sha256']==sha256('reports/quad3d_v92_noisy_observations.json')
    for name in ('parents.json','preplanning_parents.json'):shutil.copyfile(prior/name,out/name)
    assert sha256(out/'parents.json')==m['parents_sha256']
    files=list(m['source_files'])+['quad3d_transition.py']
    write_json(out/'manifest.json',dict(m,schema='quad3d_observation_transition_v93',transition_guard=True,
        prior_source=str(prior.resolve()),prior_source_sha256=sha256(prior/'manifest.json'),
        prior_review_sha256=sha256('reports/quad3d_v92_review.json'),
        source_files={n:sha256(Path(__file__).parent/n) for n in files},
        policy='V92 fixed gains/physics/sensor/routes plus affine sufficient next-measurement psi0..3 domain guard.',
        limitation='Same48physicalparents/144noisevariants for paired development. Not fresh generalization, recursive feasibility or learned superiority.'))
    print(json.dumps(dict(stage='prepared_paired',parents=len(parents))),flush=True)


def observation_arrays(parents, steps):
    fixed = tuple(np.asarray([p[k] for p in parents], bool if k == 'mask' else np.float64)
                  for k in ('x','goal','obstacles','mask','gains'))
    route = tuple(np.asarray([p['route'][k] for p in parents], bool if k == 'mask' else np.float64) for k in ('points','mask'))
    noise = np.asarray([p['noise'] for p in parents], np.float64)
    tapes = [sensor.unit_tape(p['sensor_seed'], steps, len(p['mask'])) for p in parents]
    return (*fixed, *route, noise, *[np.stack([t[i] for t in tapes]) for i in range(4)])


def initial_observed(x, g, o, m, gain, p, rm, n, bx, bo, ix, io, c, guard=False):
    xx, oo = sensor.observe(x, o, m, bx, bo, n, ix[0], io[0])
    return observed_control(xx,g,oo,m,gain,p,rm,jnp.zeros((),x.dtype),n,c,guard)


def benchmark(src, output, shard):
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    m, pp = parents_for_shard(src, shard); c = control_config(m['config']); guard=m.get('transition_guard',False)
    # All twelve families at the middle sensor scale; no favorable prefix filter.
    parents = [p for p in pp if p['noise_level'] == 1]
    args = to_device(observation_arrays(parents, 40)); records=[]; compiled=[]
    for batch in (1, 12):
        inp = tuple(a[:batch] for a in args)
        f, exe, cold = compile_fn(jax.vmap(lambda *a: initial_observed(*a,c,guard)), inp)
        result, timing = measure(exe, inp, repeats=15, host_result=batch==1); compiled.append(f)
        records.append(dict(kind='complete_observed_control',batch=batch,compile_seconds=cold,**timing))
        if batch==12: proof={k:np.asarray(v) for k,v in zip(('control','accepted','psi','domain','residual','iterations'),result[:6])}
    f, exe, cold = compile_fn(jax.vmap(make_observed_rollout(40,c,guard)), args)
    _, timing = measure(exe,args,repeats=3); compiled.append(f)
    records.append(dict(kind='observed_rollout_40',batch=12,compile_seconds=cold,**timing))
    def problem(x,g,o,m,gain,p,rm,n,bx,bo,ix,io):
        xx,oo=sensor.observe(x,o,m,bx,bo,n,ix[0],io[0])
        return observed_problem(xx,g,oo,m,gain,p,rm,jnp.zeros((),x.dtype),n,c,guard)[:3]
    f, exe, _=compile_fn(jax.vmap(problem),args);compiled.append(f)
    refs,aa,bb=map(np.asarray,exe(*args));oracles=[]
    for i in range(12):
        lp=linprog(np.zeros(4),A_ub=aa[i],b_ub=bb[i],bounds=[(None,None)]*4,method='highs')
        record=dict(feasible=bool(lp.success),lp_status=int(lp.status),action_error=None)
        if lp.success:
            opt=minimize(lambda u:.5*np.sum((u-refs[i])**2),lp.x,jac=lambda u:u-refs[i],
                constraints={'type':'ineq','fun':lambda u:bb[i]-aa[i]@u,'jac':lambda u:-aa[i]},
                method='SLSQP',options={'ftol':1e-11,'maxiter':1000})
            record.update(oracle_converged=bool(opt.success and np.max(aa[i]@opt.x-bb[i])<1e-7),
                          action_error=float(np.max(abs(opt.x-proof['control'][i]))))
        else: assert lp.status==2, 'Independent feasibility oracle did not resolve'
        oracles.append(record)
    eligible=all(bool(proof['accepted'][i])==r['feasible'] and (not r['feasible'] or r['oracle_converged'] and r['action_error']<3e-5)
                 for i,r in enumerate(oracles))
    np.savez_compressed(out/'numerics.npz',**proof,reference=refs,a=aa,b=bb)
    from qpax.explicit import pdip
    write_json(out/'report.json',sanitize(dict(source_sha256=sha256(Path(src)/'manifest.json'),records=records,oracles=oracles,
        accuracy_eligible=eligible,numerics_sha256=sha256(out/'numerics.npz'),device=str(jax.devices()[0]),
        qpax_kernel_sha256=sha256(pdip.__file__),implicit_jit_cache_entries=[f._cache_size() for f in compiled])))
    print(json.dumps(dict(stage='benchmarked',accuracy_eligible=eligible,records=records)),flush=True)


def collect(src, output, shard):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);m,parents=parents_for_shard(src,shard);c=control_config(m['config'])
    args=to_device(observation_arrays(parents[:12],m['steps']))
    f,exe,cold=compile_fn(jax.vmap(make_observed_rollout(m['steps'],c,m.get('transition_guard',False))),args)
    manifest=dict(source=str(Path(src).resolve()),source_sha256=sha256(Path(src)/'manifest.json'),config=m['config'],
        steps=m['steps'],batch=12,shard=shard,compile_seconds=cold,device=str(jax.devices()[0]),
        parent_ids=[p['id'] for p in parents],trace_storage='Complete applied prefix plus one termination tick; no counterfactual inactive suffix.',
        runtime_source_files={n:sha256(Path(__file__).parent/n) for n in m['source_files']},transition_guard=m.get('transition_guard',False))
    assert manifest['runtime_source_files']==m['source_files']
    write_json(out/'pending_manifest.json',manifest);rows=[];duration=0.
    for start in range(0,36,12):
        if shutil.disk_usage(out).free/2**30<150.35:raise ValueError('Retained storage buffer reached')
        ps=parents[start:start+12];args=to_device(observation_arrays(ps,m['steps']));begin=time.perf_counter()
        summaries,traces=jax.device_get(exe(*args));elapsed=time.perf_counter()-begin;duration+=elapsed
        for i,p in enumerate(ps):
            count=int(summaries['steps'][i]);length=min(m['steps'],count+1);file=f'{start+i:03d}.npz'
            np.savez_compressed(out/file,**{k:v[i,:length] for k,v in traces.items()})
            rows.append(dict(id=p['id'],group_id=p['group_id'],family=p['family'],noise_level=p['noise_level'],file=file,
                sha256=sha256(out/file),steps=count,status=int(summaries['status'][i]),final_state=summaries['final_state'][i].tolist()))
        write_json(out/'progress.json',dict(parents_complete=len(rows),parents_total=36,physical_steps=sum(r['steps'] for r in rows),
                                          execute_seconds=duration,implicit_jit_cache_entries=f._cache_size()))
        print(json.dumps(dict(stage='collected',parents=len(rows),seconds=elapsed)),flush=True)
    write_json(out/'index.json',rows);assert f._cache_size()==0
    manifest.update(index_sha256=sha256(out/'index.json'),physical_steps=sum(r['steps'] for r in rows),execute_seconds=duration,implicit_jit_cache_entries=0)
    write_json(out/'manifest.json',manifest)


def audit(output,workers=12):
    out=Path(output);m=read(out/'manifest.json');sm,parents=source(m['source'],64);by={p['id']:p for p in parents};index=read(out/'index.json')
    assert m['source_sha256']==sha256(Path(m['source'])/'manifest.json') and m['config']==sm['config']
    assert m['index_sha256']==sha256(out/'index.json')
    assert [r['id'] for r in index]==m['parent_ids'] and len(set(m['parent_ids']))==36
    assert m['runtime_source_files']==sm['source_files']
    args=[(by[r['id']],r,str(out),m) for r in index];rows=[]
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(audit_parent,args):
            rows.append(result)
            if len(rows)%4==0:write_json(out/'audit_progress.json',dict(parents_audited=len(rows),parents_total=len(index),physical_steps=sum(r['steps'] for r in rows)))
    write_json(out/'independent_audit.json',sanitize(dict(audit_passed=True,rows=rows,physical_steps=sum(r['steps'] for r in rows),
        manifest_sha256=sha256(out/'manifest.json'),index_sha256=sha256(out/'index.json'),
        audit_runtime_source_files={n:sha256(Path(__file__).parent/n) for n in sm['source_files']},quad3d_milestone_passed=False)))
    print(json.dumps(dict(stage='audited',parents=len(rows))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','prepare_v93','benchmark','collect','audit']);p.add_argument('--source');p.add_argument('--output',required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--workers',type=int,default=12);a=p.parse_args()
    if a.action=='prepare':prepare(a.output)
    elif a.action=='prepare_v93':prepare_v93(a.output)
    elif a.action=='benchmark':benchmark(a.source,a.output,a.shard)
    elif a.action=='collect':collect(a.source,a.output,a.shard)
    else:audit(a.output,a.workers)
