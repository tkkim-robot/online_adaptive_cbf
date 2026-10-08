"""Frozen Quad3D policies on every previously reserved random obstacle field.

The geometry inventory is shared with Quad2D, without using its outcomes for
selection. Altitudes, initial motions and sensor seeds are freshly reserved.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

from .dataset import sha256
from .io import write_json
from .quad3d_learning_contract import read

BASE = Path('artifacts/experiments/quad2d_large_random_density_comparison/source')
ORIGINAL = Path('artifacts/experiments/matched_quad3d_static_trained_inputs')
FC = Path('artifacts/experiments/paper_nearest_fc_quad3d/inputs')
OA_COMPARISON = Path('artifacts/experiments/quad3d_static_all_method_comparison/all_methods/comparison.json')
FC_COMPARISON = Path('artifacts/experiments/paper_nearest_fc_quad3d/evaluation/comparison.json')
SCHEMA = 'quad3d_frozen_random_density_policy_v1'
METHODS = ('gat','nearest_fc')
SEED = 2609288100
GROUPS = 2048
RESERVE = 30.


def project_parent(row, index, initial_gain=4., *, seed=SEED, namespace='quad3d_random_density'):
    from .quad3d_observation import BASE_NOISE
    rng=np.random.default_rng(seed+index)
    x=np.zeros(12);x[:2]=row['initial_state'][:2];x[2]=rng.uniform(.4,2.4)
    broad=bool(row['replica']%2)
    tilt,velocity,vertical,rate=(.1,.4,.2,.15) if broad else (.025,.08,.05,.025)
    x[3:5]=rng.uniform(-tilt,tilt,2);x[5]=rng.uniform(-.2,.2)
    x[6:8]=rng.uniform(-velocity,velocity,2);x[8]=rng.uniform(-vertical,vertical)
    x[9:12]=rng.uniform(-rate,rate,3)
    ident=f'{namespace}:{seed+index}'
    return dict(id=ident,group_id=ident,index=index,study_slot=index%4,seed=seed+index,
        sensor_seed=seed+index+1000000,layout_group_id=row['group_id'],family=row['family'],
        density=row['density'],obstacle_count=row['obstacle_count'],replica=row['replica'],
        noise_level=row['declared_noise_scale'],noise=(row['declared_noise_scale']*BASE_NOISE).tolist(),
        motion_stratum='broad' if broad else 'small',partition='policy_audit',capacity=64,
        x=x.tolist(),goal=[*row['goal'],float(rng.uniform(.2,2.8))],obstacles=row['obstacles'],
        mask=row['obstacle_mask'],gains=[initial_gain]*4,
        solvability='unknown; no route retries, outcome selection or exclusion')


def frozen_inputs():
    from .nearest_fc import validate_metadata
    from .comparison_contracts import matched_controller_settings
    a,b=read(ORIGINAL/'manifest.json'),read(FC/'manifest.json')
    matched_controller_settings(a['comparison_contracts']['gat'],a['comparison_contracts']['matched_fc'])
    for k in ('config','policy_config','steps','capacity'):
        if a[k]!=b[k]:raise ValueError('Changed shared retained controller: '+k)
    if a['steps']!=1600 or a['capacity']!=64:raise ValueError('Wrong retained runtime dimensions')
    models={'gat':a['models']['gat'],'nearest_fc':b['models']['nearest_fc']}
    contracts={m:read(Path(v['bundle'])/'manifest.json') for m,v in models.items()}
    validate_metadata(contracts['nearest_fc'])
    for k in ('controller','quad3d_contract','gain_domain','gain_dimension','normalization','targets','events','dataset_manifest_sha256'):
        if contracts['gat'][k]!=contracts['nearest_fc'][k]:raise ValueError('Unmatched learning contract: '+k)
    gates={};bindings={}
    for m,path in zip(METHODS,(OA_COMPARISON,FC_COMPARISON)):
        report=read(path);gate=Path(report['gate_paths'][m])
        if sha256(gate)!=report['gate_sha256'][m]:raise ValueError('Changed retained gate')
        gates[m]=dict(path=str(gate.resolve()),sha256=sha256(gate))
        info=models[m];bundle=Path(info['bundle'])
        for p,d in ((bundle/'manifest.json',info['bundle_manifest_sha256']),
                    (bundle/'weights.msgpack',info['weights_sha256']),
                    (Path(info['prediction_fit']),info['prediction_fit_sha256'])):
            if sha256(p)!=d:raise ValueError('Changed retained trained artifact')
            bindings[str(p.resolve())]=d
        for p in (path,gate):bindings[str(p.resolve())]=sha256(p)
    for p in (ORIGINAL/'manifest.json',FC/'manifest.json',BASE/'manifest.json',BASE/'scenes.json'):
        bindings[str(p.resolve())]=sha256(p)
    return a,models,gates,bindings


def verify_source(root, spec=None):
    from .comparison_contracts import physical_obstacle_scope
    root=Path(root);s=read(root/'manifest.json') if spec is None else spec
    if (s['schema']!=SCHEMA or s['training_use'] or s['weight_fit_authorized'] or s['final_test']
            or set(s['models'])!=set(METHODS) or s['reference_parents']!=0):
        raise ValueError('Changed frozen evaluation scope')
    for p,h in s['frozen_files'].items():
        if sha256(p)!=h:raise ValueError('Changed frozen density input: '+p)
    for f,k in (('parents.json','parents_sha256'),('preplanning_parents.json','preplanning_sha256')):
        if sha256(root/f)!=s[k]:raise ValueError('Changed parent reservation')
    parents=read(root/'parents.json');raw=read(root/'preplanning_parents.json');base=read(BASE/'scenes.json')
    if len(parents)!=s['adaptive_parents'] or len(parents)!=len(raw):raise ValueError('Lost physical parents')
    expected_indices=s['pilot_indices'] if s['pilot'] else list(range(GROUPS))
    if [p['index'] for p in parents]!=expected_indices:raise ValueError('Changed parent selection')
    for p,r in zip(parents,raw,strict=True):
        wanted=project_parent(base[p['index']],p['index'],s['policy_config']['initial_gain'])
        if s['pilot']:wanted['study_slot']=0
        if r!=wanted or any(p[k]!=v for k,v in r.items()):raise ValueError('Changed projected scene')
        physical_obstacle_scope('quad3d',p['obstacles'],p['mask'])
    if [sum(p['study_slot']==i for p in parents) for i in range(4)]!=s['slot_counts']:
        raise ValueError('Lost evaluation partition')
    return parents


def prepare(root):
    from .quad2d_random_density import verify
    from .quad3d_observation_experiment import plan_parent
    from .quad3d_policy_experiment import runtime_names,assert_disjoint_parents
    root=Path(root);base=verify(BASE);original,models,gates,bound=frozen_inputs()
    if len(base)!=GROUPS:raise ValueError('Incomplete geometry inventory')
    parents=[project_parent(p,i,original['policy_config']['initial_gain']) for i,p in enumerate(base)]
    # Check all earlier Quad3D learning and evaluation roles, not just ID strings.
    for area in ('datasets','experiments'):
        for pattern in ('quad3d*/parents.json','matched_quad3d*/parents.json','paper_nearest_fc_quad3d/inputs/parents.json'):
            for path in sorted(Path('artifacts',area).glob(pattern)):
                old=read(path);assert_disjoint_parents(parents,old)
                bound[str(path.resolve())]=sha256(path)
    source=root/'inputs';source.mkdir(parents=True,exist_ok=False)
    write_json(source/'preplanning_parents.json',parents)
    with ProcessPoolExecutor(28,mp_context=multiprocessing.get_context('spawn')) as pool:
        parents=list(pool.map(plan_parent,parents))
    write_json(source/'parents.json',parents)
    spec=dict(schema=SCHEMA,config=original['config'],policy_config=original['policy_config'],models=models,gates=gates,
        frozen_files=bound,parents=GROUPS,adaptive_parents=GROUPS,reference_parents=0,capacity=64,steps=1600,batch=12,
        pilot=False,pilot_indices=[],slot_counts=[GROUPS//4]*4,storage_floor_gib=RESERVE,
        training_use=False,weight_fit_authorized=False,final_test=False,stationary_physical_obstacles=True,
        source_files={n:sha256(Path(__file__).parent/n) for n in runtime_names(SCHEMA)},
        parents_sha256=sha256(source/'parents.json'),preplanning_sha256=sha256(source/'preplanning_parents.json'),
        geometry_source=str(BASE.resolve()),geometry_source_sha256=sha256(BASE/'scenes.json'),
        allocation='All2048 original random fields, unfiltered. Fresh seeded altitude/motion/sensor tapes; equal small/broad motion within each family/count/noise cell.',
        route_statuses=dict(Counter(p['route']['status'] for p in parents)),
        scope='Static vertical cylinders, retained linearized12-state model, controller/actuator limits/speed and calibrated neural policies unchanged. No density-shift coverage guarantee.')
    write_json(source/'manifest.json',spec);verify_source(source)
    # Twelve qualification parents span every family and density, include all
    # noise levels and both initial-motion strata. Indices fixed before outcomes.
    indices=sorted(f*512+d*128+((f+d)%4)*32+((f+d)%2) for f in range(4) for d in (0,1,3))
    pilot=root/'pilot';pilot.mkdir();pp=[dict(parents[i],study_slot=0) for i in indices]
    write_json(pilot/'parents.json',pp);write_json(pilot/'preplanning_parents.json',[{k:v for k,v in p.items() if k!='route'} for p in pp])
    ps=deepcopy(spec);ps.update(pilot=True,pilot_indices=indices,parents=12,adaptive_parents=12,slot_counts=[12,0,0,0],
        parents_sha256=sha256(pilot/'parents.json'),preplanning_sha256=sha256(pilot/'preplanning_parents.json'))
    write_json(pilot/'manifest.json',ps);verify_source(pilot)
    return spec


def destination(root,method,slot=0,pilot=False,backend='cuda'):
    root=Path(root)
    return root/'qualification'/f'{method}_{backend}' if pilot else root/'policy'/method/f'part{slot}'


def checked(root,method,slot=0,pilot=False,backend='cuda'):
    root=Path(root);source=root/('pilot' if pilot else 'inputs');s=read(source/'manifest.json')
    expected=[p for p in verify_source(source,s) if p['study_slot']==slot]
    out=destination(root,method,slot,pilot,backend);m=read(out/'manifest.json');a=read(out/'independent_audit.json')
    if (not a['audit_passed'] or not a['all_model_predictions_recomputed'] or not a['all_continuous_physics_checked']
            or a['manifest_sha256']!=sha256(out/'manifest.json') or a['index_sha256']!=sha256(out/'index.json')
            or m['source_sha256']!=sha256(source/'manifest.json') or m['config']!=s['config']
            or m['policy_config']!=s['policy_config'] or m['steps']!=s['steps'] or m['encoder']!=method
            or m['weights_sha256']!=s['models'][method]['weights_sha256'] or m['gate_sha256']!=s['gates'][method]['sha256']
            or m['implicit_jit_cache_entries']!=0):raise ValueError('Changed density physical/model audit')
    rows=a['rows']
    if [r['id'] for r in rows]!=[p['id'] for p in expected]:raise ValueError('Missing or reordered physical parent')
    if a['physical_steps']!=sum(r['steps'] for r in rows):raise ValueError('Missing audited steps')
    for r in rows:
        for f,h in (('file','sha256'),('query_file','query_sha256')):
            if sha256(out/r[f])!=r[h]:raise ValueError('Changed physical/query trace')
    proof=dict(directory=str(out.resolve()),manifest_sha256=sha256(out/'manifest.json'),
               audit_sha256=sha256(out/'independent_audit.json'),prediction_replay_sha256=sha256(out/'prediction_replay.json'))
    return [dict(r,directory=str(out.resolve()),density=p['density'],obstacle_count=p['obstacle_count'],motion_stratum=p['motion_stratum'])
            for r,p in zip(rows,expected,strict=True)],proof


def parity(root,method):
    left=destination(root,method,pilot=True,backend='cpu');right=destination(root,method,pilot=True,backend='cuda')
    try:
        a,b=read(left/'index.json'),read(right/'index.json')
        if [r['id'] for r in a]!=[r['id'] for r in b]:raise AssertionError('Different qualification parents')
        for p,q in zip(a,b,strict=True):
            if (p['status'],p['steps'])!=(q['status'],q['steps']):raise AssertionError('Different complete mission outcome')
            with np.load(left/p['file']) as x,np.load(right/q['file']) as y:
                for k in ('active','requery','controller_gain','previous_gain'):np.testing.assert_array_equal(x[k],y[k])
                for k in ('control','next_state'):np.testing.assert_allclose(x[k],y[k],atol=2e-7,rtol=2e-7)
            with np.load(left/p['query_file']) as x,np.load(right/q['query_file']) as y:
                for k in ('selected_index','query_tick','accepted'):np.testing.assert_array_equal(x[k],y[k])
                for k in ('prediction_mean','prediction_variance','prediction_event_logits'):np.testing.assert_allclose(x[k],y[k],atol=3e-5,rtol=3e-5)
        return dict(passed=True)
    except AssertionError as e:return dict(passed=False,error=str(e))


def review(root):
    from .matched_quad3d_evaluation import summarize
    from .paper_comparison import paired
    root=Path(root);parents=verify_source(root/'inputs');raw={};proofs={};ids=[p['id'] for p in parents]
    for method in METHODS:
        rows=[];proofs[method]=[]
        for slot in range(4):
            r,p=checked(root,method,slot);rows.extend(r);proofs[method].append(p)
        mapping={r['id']:r for r in rows}
        if len(rows)!=GROUPS or len(mapping)!=GROUPS or set(mapping)!=set(ids):raise ValueError('Incomplete paired denominator')
        raw[method]=[mapping[i] for i in ids]
    strata={field:{str(value):{m:summarize([r for r in rows if r[field]==value],1600) for m,rows in raw.items()}
        for value in sorted({r[field] for r in raw['gat']})} for field in ('family','density','noise_level','motion_stratum')}
    write_json(root/'outcomes.json',raw)
    result=dict(methods={m:summarize(rows,1600) for m,rows in raw.items()},paired=paired(raw['gat'],raw['nearest_fc']),
        strata=strata,parts=proofs,qualification=read(root/'qualification.json'),source_manifest_sha256=sha256(root/'inputs/manifest.json'),
        outcomes_sha256=sha256(root/'outcomes.json'),all2048parents_retained=True,native_baselines_pending=True,whole_goal_complete=False,
        scope='All shared randomized fields, frozen GAT/nearestFC and gates; development-density evidence. No fitting or outcome filtering; no automatic calibration guarantee under shift.')
    write_json(root/'comparison.json',result);return result


def run(job,root):
    job,root=Path(job),Path(root);start=time.monotonic()
    if shutil.disk_usage('.').free/2**30<RESERVE+6:raise ValueError('Need6GiB forecast above30GiB reserve')
    root.mkdir(parents=True,exist_ok=False)
    write_json(job/'progress.json',dict(stage='reserve_complete_random_fields',estimated_remaining_seconds=2400))
    spec=prepare(root)
    write_json(job/'protocol.json',dict(source=str((root/'inputs').resolve()),source_manifest_sha256=sha256(root/'inputs/manifest.json'),
        families=sorted({p['family'] for p in read(root/'inputs/parents.json')}),parents_per_method=GROUPS,
        methods=METHODS,reserve_gib=RESERVE,forecast_gib=6,cpu_lanes=[[14*i,14*i+13] for i in range(4)],
        weights_or_gates_refitted=False,whole_goal_complete=False))
    def execute(method,slot=0,pilot=False,backend='cuda',lane=0):
        if shutil.disk_usage('.').free/2**30<RESERVE:raise ValueError('30GiB reserve reached; retain partial evidence')
        out=destination(root,method,slot,pilot,backend);source=root/('pilot' if pilot else 'inputs')
        env=os.environ.copy();env.pop('JAX_ENABLE_X64',None)
        env.update(JAX_PLATFORMS=backend,CUDA_VISIBLE_DEVICES=str(lane),JAX_EXPLICIT_X64_DTYPES='allow',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
        name=f'{method}_{backend}_pilot' if pilot else f'{method}_part{slot}';tick=time.monotonic()
        common=['--source',str(source),'--encoder',method,'--phase','policy_audit','--slot',str(slot),
                '--gate',spec['gates'][method]['path'],'--output',str(out)]
        for label,args in ((name,['collect',*common]),(name+'_audit',['audit','--output',str(out),'--workers','12'])):
            with (job/(label+'.log')).open('w') as log:
                subprocess.run(['taskset','-c',f'{14*lane}-{14*lane+13}',sys.executable,'-m','oa_cbf_jax.quad3d_policy_experiment',*args],
                    env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        checked(root,method,slot,pilot,backend)
        elapsed=time.monotonic()-tick
        print(dict(stage=name,status='completed',seconds=elapsed),flush=True)
        return elapsed
    write_json(job/'progress.json',dict(stage='full_horizon_cpu_cuda_qualification',estimated_remaining_seconds=2400))
    jobs=[('gat','cpu'),('gat','cuda'),('nearest_fc','cpu'),('nearest_fc','cuda')]
    def qualify(item):
        lane,(method,backend)=item
        return method,backend,execute(method,pilot=True,backend=backend,lane=lane)
    with ThreadPoolExecutor(4) as pool:measured=list(pool.map(qualify,enumerate(jobs)))
    timings={m:{b:s for mm,b,s in measured if mm==m} for m in METHODS};checks={m:parity(root,m) for m in METHODS}
    totals={b:sum(timings[m][b] for m in METHODS) for b in ('cpu','cuda')}
    backend=min(totals if all(p['passed'] for p in checks.values()) else ['cpu'],key=totals.get)
    write_json(root/'qualification.json',dict(actual_full1600tick_parents=12,timings=timings,parity=checks,combined_seconds=totals,selected=backend))
    for method in METHODS:
        write_json(job/'progress.json',dict(stage='full2048_'+method,backend=backend,estimated_remaining_seconds=2400))
        with ThreadPoolExecutor(4) as pool:list(pool.map(lambda slot:execute(method,slot,backend=backend,lane=slot),range(4)))
    write_json(job/'progress.json',dict(stage='final_complete_comparison_review',estimated_remaining_seconds=120))
    r=review(root)
    write_json(job/'complete.json',dict(status='completed',comparison=str((root/'comparison.json').resolve()),comparison_sha256=sha256(root/'comparison.json'),
        elapsed_seconds=time.monotonic()-start,whole_goal_complete=False))
    print(r['methods'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','review']);p.add_argument('--job');p.add_argument('--root',required=True)
    a=p.parse_args()
    if a.action=='run':run(a.job,a.root)
    else:review(a.root)
