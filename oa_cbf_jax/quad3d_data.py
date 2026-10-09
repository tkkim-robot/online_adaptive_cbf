"""Quad3d data functions and shared contracts."""

import itertools

import numpy as np

from .quad3d_observation import unit_tape, numpy_observe

from .quad3d_features import numpy_graph

from .quad3d_routing import numpy_flight_target

from .quad3d_observation import numpy_obstacles

SCHEMA='oa_cbf_quad3d_acquired_hurdle_v94'

GAIN_BANK=np.array([[a,a,b,b] for a,b in itertools.product((1.,2.,3.,4.),repeat=2)],np.float64)

REPLICAS=4

HORIZON=120

QUERY_TICKS=(0,80,240)

OBSERVER_SCHEMA='oa_cbf_quad3d_observer_history_hurdle_v98'

OBSERVER_QUERY_TICKS=(0,160,400,800,1200)

WIDE_SCHEMA='oa_cbf_quad3d_wide_gain_history_hurdle_v104'

WIDE_GAIN_BANK=np.array([[a,a,b,b] for a,b in itertools.product((2.,4.,6.,8.),repeat=2)],np.float64)

WIDE_QUERY_TICKS=(0,40,160,400,800)

def candidate_bank(schema):
    if schema==WIDE_SCHEMA:return WIDE_GAIN_BANK
    if schema in (SCHEMA,OBSERVER_SCHEMA):return GAIN_BANK
    raise ValueError('Unregistered Quad3D candidate bank')

def candidate_domain(schema):
    bank=candidate_bank(schema)
    return dict(lower=float(bank.min()),upper=float(bank.max()))

TARGETS=['conditional_negative_continuous_min_clearance_div_0.3_capped_below_minus_2',
         'physical_route_and_3d_terminal_blended_progress_plus_observed_early_arrival']

EVENTS=['collision_first','any_adverse_termination']

def query_context(parent,trace,tick,c,ticks=None):
    if ticks is None:ticks=OBSERVER_QUERY_TICKS if c.nominal_bias_observer=='innovation_ema_v97' else QUERY_TICKS
    if tick not in ticks or tick>=len(trace['active']) or not np.all(trace['active'][:tick]):
        raise ValueError('Query is outside the actually observed acquisition prefix')
    x=np.array(trace['state'][tick]);o=np.array(parent['obstacles']);o[:,:2]+=tick*c.robot.dt*o[:,3:5]
    noise=np.array(parent['noise']);mask=np.array(parent['mask']);bx,bo,ix,io=unit_tape(parent['sensor_seed'],tick,len(mask))
    seen,so=numpy_observe(x,o,mask,bx,bo,noise,ix[tick],io[tick])
    np.testing.assert_allclose(seen,trace['observed'][tick],atol=2e-12,rtol=1e-12)
    context=dict(x=x,obstacles=o,observed=seen,observed_obstacles=so,noise=noise,mask=mask,
        cursor=float(trace['route_cursor_before'][tick]),previous_u=np.array(trace['control'][tick-1]) if tick else np.zeros(4),
        previous_gain=np.array(trace['previous_gain'][tick] if 'previous_gain' in trace else parent['gains']),
        bias_x=bx,bias_o=bo,innovation_x=ix[tick],innovation_o=io[tick],tick=tick)
    if c.nominal_bias_observer=='innovation_ema_v97':context['nominal_bias']=np.array(trace['nominal_bias_estimate'][tick])
    return context

def future_seed(parent,tick,replica):
    # Disjoint deterministic query/replica seeds, paired across ALL gains.
    return int(parent['seed'])*10000+tick*REPLICAS+replica+9400000000

def branch_parent(parent,context,gain,replica):
    q=context['tick']
    branch=dict(parent,id=f"{parent['id']}:q{q}:gain"+','.join(map(str,gain))+f':rep{replica}',
        x=context['x'].tolist(),obstacles=context['obstacles'].tolist(),gains=np.asarray(gain).tolist(),
        initial_cursor=context['cursor'],branch=dict(query_tick=q,future_seed=future_seed(parent,q,replica)))
    if 'nominal_bias' in context:branch['initial_nominal_bias']=context['nominal_bias'].tolist()
    return branch

def branch_arrays(parent,context,horizon=HORIZON,bank=GAIN_BANK,replicas=REPLICAS):
    if replicas!=REPLICAS:raise ValueError('Frozen replica seed/layout contract')
    branches=[branch_parent(parent,context,gain,r) for gain in bank for r in range(replicas)]
    tapes=[]
    for r in range(replicas):
        _,_,ix,io=unit_tape(future_seed(parent,context['tick'],r),horizon,len(context['mask']))
        ix[0]=context['innovation_x'];io[0]=context['innovation_o'];tapes.append((ix,io))
    size=len(branches);repeat=lambda a:np.repeat(np.asarray(a)[None],size,axis=0)
    args=(repeat(context['x']),repeat(parent['goal']),repeat(context['obstacles']),repeat(context['mask']),
        np.repeat(bank,replicas,axis=0),repeat(parent['route']['points']),repeat(np.array(parent['route']['mask'],bool)),
        repeat(context['noise']),repeat(context['bias_x']),repeat(context['bias_o']),
        np.stack([tapes[r%replicas][0] for r in range(size)]),np.stack([tapes[r%replicas][1] for r in range(size)]),
        np.full(size,context['cursor'],np.float64))
    if 'nominal_bias' in context:args+= (repeat(context['nominal_bias']),)
    return branches,args

def graph_args(parent,context):
    return (context['observed'],np.array(parent['goal']),context['observed_obstacles'],context['mask'],
        np.array(parent['route']['points']),np.array(parent['route']['mask'],bool),np.array(context['cursor']),
        context['previous_u'],context['previous_gain'],context['noise'])

def route_coordinate(position,points,mask,cursor):
    """Independent physical projection near the recorded causal route cursor."""
    vectors=np.diff(points,axis=0);valid=mask[:-1]&mask[1:];length=np.where(valid,np.linalg.norm(vectors,axis=-1),0.)
    cum=np.r_[0.,np.cumsum(length)];frac=np.clip(np.sum((position-points[:-1])*vectors,axis=-1)/np.maximum(length**2,1e-12),0.,1.)
    arc=cum[:-1]+frac*length;eligible=valid&(arc>=cursor-1.)&(arc<=cursor+1.)
    # Routes always contain a segment. If the hint is beyond local projection
    # support, use nearest full segment and record physical progress honestly.
    eligible=eligible if np.any(eligible) else valid
    i=np.argmin(np.where(eligible,np.sum((points[:-1]+frac[:,None]*vectors-position)**2,axis=-1),np.inf))
    return float(arc[i])

def labels(parent,context,rows,traces,c,horizon=HORIZON):
    """Use independent continuous-time outcomes/clearance; never invent a future."""
    goal=np.array(parent['goal']);points=np.array(parent['route']['points']);rm=np.array(parent['route']['mask'],bool)
    _,_,remaining,_=numpy_flight_target(context['observed'],goal,numpy_obstacles(context['observed_obstacles'],context['mask'],context['noise'],True),
        context['mask'],points,rm,context['cursor'],c)
    blend=float(np.clip(1-remaining/2.5,0.,1.))
    start=route_coordinate(context['x'][:2],points,rm,context['cursor'])
    status=np.array([r['audited_status'] for r in rows]);raw_steps=np.array([r['steps'] for r in rows])
    steps=np.array([r['steps'] if r.get('first_physical_stop_step') is None else min(r['steps'],r['first_physical_stop_step']) for r in rows])
    clearance=np.array([r['minimum_clearance'] if r.get('first_physical_stop_step') is None else r['first_physical_stop_clearance'] for r in rows])
    final=np.array([r['final_state'] if r.get('first_physical_stop_step') is None else t['state'][0] if n==0 else t['next_state'][n-1]
                    for r,t,n in zip(rows,traces,steps,strict=True)])
    route_progress=np.array([route_coordinate(x[:2],points,rm,context['cursor'] if n==0 else float(t['route_progress'][min(n-1,len(t['route_progress'])-1)]))-start
                             for x,t,n in zip(final,traces,steps,strict=True)])
    goal_progress=np.linalg.norm(context['x'][:3]-goal)-np.linalg.norm(final[:,:3]-goal,axis=-1)
    performance=((1-blend)*route_progress+blend*goal_progress)/(horizon*c.robot.dt*c.cruise_speed)+(status==1)*(1-steps/horizon)
    observed=np.isin(status,[1,4,6]);collision=status==4;adverse=~np.isin(status,[1,6])
    risk=-np.minimum(clearance,.6)/.3
    target=np.stack((np.where(observed,risk,0.),performance),-1).astype(np.float32)
    target_mask=np.stack((observed,np.ones_like(observed)),-1)
    events=np.stack((collision,adverse),-1).astype(np.float32)
    event_mask=np.stack((observed,np.ones_like(observed)),-1)
    assert np.isfinite(target).all()
    return dict(target=target,target_mask=target_mask,events=events,event_mask=event_mask,status=status,steps=steps,recorded_physical_steps=raw_steps,
        min_clearance=clearance,final_state=final,physical_initial_state=np.repeat(context['x'][None],len(rows),axis=0),
        route_progress=route_progress,goal_progress=goal_progress,initial_terminal_blend=np.full(len(rows),blend))


from .scenes import DIVERSE_FAMILIES

from .scenes import FAMILIES as STRUCTURAL

FAMILIES=DIVERSE_FAMILIES+STRUCTURAL


from pathlib import Path

import json

import time


import jax

import jax.numpy as jnp

def read(path):return json.loads(Path(path).read_text())

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


from . import quad3d_observation as sensor

from .quad3d_control import Quad3DControlConfig

from .quad3d_routing import observed_route

from .quad3d_observed_rollout import observed_control

from .routing import segment_clearances

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


import argparse

from collections import Counter

from concurrent.futures import ProcessPoolExecutor

from dataclasses import asdict


import multiprocessing

import shutil


from scipy.optimize import linprog, minimize

from .quad3d_features import history_graph, numpy_history_graph, SCHEMA as GRAPH_SCHEMA, FEATURES, OBSERVER_SCHEMA as OBSERVER_GRAPH, OBSERVER_FEATURES

from .quad3d_control import control_config

from .quad3d_observed_rollout import make_observed_rollout, observed_problem

from .quad3d_observation import BASE_NOISE, SCHEMA as SENSOR_SCHEMA, observe

from .scenes import multiscale_scenes_scene as diverse_scene, contract

from .scenes import scene as structural_scene


from .io import sha256

from .io import write_json

from .io import sanitize

RUNTIME=('quad3d.py','quad3d_control.py','quad3d_qp.py','quad3d_held.py','quad3d_transition.py',
    'quad3d_observation.py','quad3d_routing.py','quad3d_observed_rollout.py','quad3d_audit.py',
    'quad3d_observation_audit.py','quad3d_features.py','quad3d_data.py','quad3d_data.py',
    'quad3d_data.py','quad3d_data.py','quad3d_data.py',
    'routing.py','scenes.py','scenes.py','scenes.py',
    'quad3d_observation.py','quad3d_data.py','quad3d_policy_rollout.py','quad3d_qp.py')

def parent_design(version):
    if version not in (94,95,98,99,104,106):raise ValueError('Unregistered data version')
    count=48 if version==94 else 1152 if version==106 else 288 if version==104 else 576
    for i in range(count):
        round=i//12
        if version==94:
            partition='train' if round<2 else 'validation' if round==2 else 'prediction_fit' if i%2==0 else 'prediction_audit'
        elif version==104:
            partition='train' if round<12 else 'validation' if round<16 else 'prediction_fit' if round<20 else 'prediction_audit'
        elif version==106:
            partition='train' if round<72 else 'validation' if round<80 else 'prediction_fit' if round<88 else 'prediction_audit'
        else:
            partition='train' if round<24 else 'validation' if round<32 else 'prediction_fit' if round<40 else 'prediction_audit'
        yield dict(index=i,seed=((98 if version==99 else version)*1000+600100)+i,family=FAMILIES[i%12],partition=partition,noise_level=(round+i%12)%3)

def prepare(output,version=94):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);parents=[]
    observer=version in (98,99,104,106);wide=version in (104,106)
    c=Quad3DControlConfig(hold_guard='bernstein_v87',nominal_bias_observer='innovation_ema_v97' if observer else 'none',
        qp_refinement='active_faces_v99' if version in (99,104,106) else 'none')
    schema=WIDE_SCHEMA if wide else OBSERVER_SCHEMA if observer else SCHEMA
    bank=candidate_bank(schema)
    ticks=WIDE_QUERY_TICKS if wide else OBSERVER_QUERY_TICKS if observer else QUERY_TICKS
    graph_schema=OBSERVER_GRAPH if observer else GRAPH_SCHEMA
    features=OBSERVER_FEATURES if observer else FEATURES
    for design in parent_design(version):
        i=design['index'];seed=design['seed'];family=design['family'];partition=design['partition']
        scene=diverse_scene(seed,family) if family in DIVERSE_FAMILIES else structural_scene(seed,family,64)
        rng=np.random.default_rng(seed+11007);x=np.zeros(12);x[:2]=scene.initial_state[:2]
        broad=wide and (i//24+i%12)%2==1
        tilt,velocity,vertical,rate=(.1,.4,.2,.15) if broad else (.025,.08,.05,.025)
        x[2]=rng.uniform(.4,2.4);x[3:5]=rng.uniform(-tilt,tilt,2);x[5]=rng.uniform(-.2,.2)
        x[6:8]=rng.uniform(-velocity,velocity,2);x[8]=rng.uniform(-vertical,vertical);x[9:12]=rng.uniform(-rate,rate,3)
        initial=(4. if (i//12+i%12)%2==0 else 8.) if wide else 2.
        level=design['noise_level'];gid=f'quad3d_v{version}:{family}:{seed}'
        parents.append(dict(id=gid,group_id=gid,index=i,seed=seed,sensor_seed=seed+1000000,family=family,partition=partition,capacity=64,
            x=x.tolist(),goal=np.r_[scene.goal,rng.uniform(.2,2.8)].tolist(),obstacles=scene.obstacles.astype(np.float64).tolist(),
            mask=scene.obstacle_mask.tolist(),gains=[initial]*4,noise_level=level,noise=(level*BASE_NOISE).tolist(),active_obstacles=int(scene.obstacle_mask.sum()),
            solvability='unknown; all parents retained, no outcome-dependent resampling'))
        if wide:parents[-1]['motion_stratum']='broad' if broad else 'small'
    if observer:
        from .quad3d_data import schedule
        for p in parents:
            p['exploratory']=bool((p['index']//36)%2)
            p['gain_schedule']=schedule(p['seed'],p['exploratory'],p['gains'][0],bank).tolist()
    previous={}
    if wide:
        from .quad3d_data import assert_disjoint_parents
        for area in ('datasets','experiments'):
            for path in sorted(Path('artifacts',area).glob('quad3d_v*_inputs/parents.json')):
                if path.parent.resolve()==out.resolve():continue
                assert_disjoint_parents(parents,read(path));previous[str(path)]=sha256(path)
    if version==99:
        prior=read('artifacts/datasets/quad3d_v98_inputs/preplanning_parents.json')
        for original,repeated in zip(prior,parents,strict=True):
            assert {k:v for k,v in original.items() if k not in ('id','group_id')}=={k:v for k,v in repeated.items() if k not in ('id','group_id')}
    write_json(out/'preplanning_parents.json',parents)
    with ProcessPoolExecutor(12,mp_context=multiprocessing.get_context('spawn')) as pool:parents=list(pool.map(plan_parent,parents))
    write_json(out/'parents.json',parents)
    m=dict(schema=schema,stage='quad3d_acquired_candidate_contract_pilot' if version==94 else f'quad3d_acquired_learning_v{version}',production_eligible=False,weight_fit_authorized=version in (95,98,99,104,106),
        training_use=True,final_test=False,version=version,parents=len(parents),groups=[{k:p[k] for k in ('group_id','partition','seed','family')} for p in parents],
        config=asdict(c),capacity=64,acquisition_steps=1600,transition_guard=True,
        parents_sha256=sha256(out/'parents.json'),preplanning_sha256=sha256(out/'preplanning_parents.json'),
        source_files={n:sha256(Path(__file__).parent/n) for n in RUNTIME},sensor_schema=SENSOR_SCHEMA,graph_schema=graph_schema,graph_features=features,
        snapshot_ticks=list(ticks),gain_bank=bank.tolist(),queries=16,replicas=REPLICAS,horizon_steps=HORIZON,
        gain_dimension=4,gain_domain=candidate_domain(schema),targets=TARGETS,events=EVENTS,
        geometry_distribution=contract(),partitions=dict(Counter(p['partition'] for p in parents)),route_statuses=dict(Counter(p['route']['status'] for p in parents)),
        controller=dict(dynamics='Quad3D',config=asdict(c),transition_guard=True,
            graph_schema=graph_schema,sensor_schema=SENSOR_SCHEMA,candidate_gains='[a,a,b,b],a,b in2,4,6,8' if wide else '[a,a,b,b],a,b in1,2,3,4',performance_target='physical_route_and_3d_terminal_blend_v94'),
        branch_contract='Authentic physical/query/cursor/bias history; current reading identical for every gain/replica. Future innovations paired across gains, fresh between replicas. No latent state resampling.'+(' Inherit the current causal nominal bias estimate, skip duplicate update at branch tick zero, then update from branch observations/applied controls.' if observer else ''),
        censoring='Risk and collision-negative mask on domain/QP/envelope stops. Prefix progress and any-adverse event remain observed. True continuous physical audit overrides sampled outcomes.',
        grouping='Every query/gain/replica of a physical parent shares its preassigned split and one statistical parent weight.',
        limitations='Reserved predictive groups remain isolated. V94 is a no-weight-fit pilot; V95 permits train/validation fitting only. V98/V99 use observer history and predeclared gain exploration; V99 repeats all V98 scene/sensor reservations with optional certified QP polishing. None is final test or learned-policy performance evidence.',
        acquisition_mode='paired_fixed_and_seeded_checked_gain_schedule_v98' if observer else 'fixed_2',
        adaptive_gain_trace=observer)
    if wide:m.update(prior_parent_files=previous,protocol_sha256=sha256(f'docs/quad3d_v{version}.md'),
        acquisition_mode='balanced_initial4_or8_fixed_or_seeded_checked_schedule',
        limitations='Fresh balanced small/broad initial motion; banks2/4/6/8 and snapshots0/40/160/400/800. No hero or prior development parent enters labels. Reserved predictive groups isolated; no policy performance claim.')
    write_json(out/'manifest.json',m);print(json.dumps(dict(stage='prepared',parents=len(parents),partitions=m['partitions'])),flush=True)

def source(src,slot):
    root=Path(src);m=read(root/'manifest.json');assert m['parents_sha256']==sha256(root/'parents.json')
    pp=read(root/'parents.json');assert len({p['id'] for p in pp})==m['parents']
    verify_runtime(m)
    return m,[p for p in pp if (p['index']//12)%4==slot]

def verify_runtime(manifest):
    files=manifest['source_files']
    if not set(RUNTIME)<=set(files) or any(Path(n).name!=n for n in files):
        raise ValueError('Incomplete or invalid acquisition runtime binding')
    if {n:sha256(Path(__file__).parent/n) for n in files}!=files:
        raise ValueError('Changed acquisition runtime')

def storage_floor(manifest):
    value=float(manifest.get('storage_floor_gib',100. if manifest.get('version')==106 else 125.))
    if not np.isfinite(value) or value<100.:raise ValueError('Invalid acquisition storage reserve')
    return value

def acquire(src,output,slot):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);m,pp=source(src,slot);c=control_config(m['config'])
    if len(pp)%12:raise ValueError('Acquisition must preserve the qualified B12 signature')
    fn=exe=None;duration=0.;rows=[]
    for start_index in range(0,len(pp),12):
        if shutil.disk_usage(out).free/2**30<storage_floor(m):raise ValueError('Learning storage reserve reached')
        batch=pp[start_index:start_index+12];host=observation_arrays(batch,m['acquisition_steps'])
        if m.get('adaptive_gain_trace',False):
            host+= (np.zeros(len(batch)),np.zeros((len(batch),12)),np.array([p['gain_schedule'] for p in batch]))
        args=to_device(host)
        if exe is None:fn,exe,cold=compile_fn(jax.vmap(make_observed_rollout(m['acquisition_steps'],c,True)),args)
        start=time.perf_counter();summaries,traces=jax.device_get(exe(*args));duration+=time.perf_counter()-start
        for i,p in enumerate(batch):
            count=int(summaries['steps'][i]);length=min(m['acquisition_steps'],count+1);name=f'{start_index+i:03d}.npz'
            np.savez_compressed(out/name,**{k:v[i,:length] for k,v in traces.items()})
            rows.append(dict(id=p['id'],file=name,sha256=sha256(out/name),family=p['family'],partition=p['partition'],
                status=int(summaries['status'][i]),steps=count,final_state=summaries['final_state'][i].tolist()))
        progress=dict(stage='acquired_batch',parents_complete=len(rows),parents_total=len(pp),physical_steps=sum(r['steps'] for r in rows),execute_seconds=duration)
        write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    write_json(out/'index.json',rows);assert fn._cache_size()==0
    write_json(out/'manifest.json',dict(source=str(Path(src).resolve()),source_sha256=sha256(Path(src)/'manifest.json'),config=m['config'],
        steps=m['acquisition_steps'],transition_guard=True,adaptive_gain_trace=m.get('adaptive_gain_trace',False),index_sha256=sha256(out/'index.json'),source_files=m['source_files'],
        gain_bank=m['gain_bank'],parent_ids=[p['id'] for p in pp],compile_seconds=cold,execute_seconds=duration,implicit_jit_cache_entries=0,device=str(jax.devices()[0])))
    print(json.dumps(dict(stage='acquired',parents=len(rows),steps=sum(r['steps'] for r in rows),seconds=duration)),flush=True)

def acquisition_benchmark(src,output,slot):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);m,pp=source(src,slot);c=control_config(m['config'])
    if m['schema']!=WIDE_SCHEMA:raise ValueError('Wide-gain history benchmark required')
    batch=pp[:12];host=observation_arrays(batch,220)+(np.zeros(12),np.zeros((12,12)),np.array([p['gain_schedule'] for p in batch]))
    args=to_device(host);fn,exe,cold=compile_fn(jax.vmap(make_observed_rollout(220,c,True)),args)
    result,timing=measure(exe,args,repeats=3);s,t=jax.device_get(result)
    np.savez_compressed(out/'numerics.npz',**s,control=t['control'],gains=t['controller_gain'],nominal_bias=t['nominal_bias_estimate'])
    assert fn._cache_size()==0
    write_json(out/'report.json',dict(source_sha256=sha256(Path(src)/'manifest.json'),compile_seconds=cold,**timing,
        numerics_sha256=sha256(out/'numerics.npz'),implicit_jit_cache_entries=0,device=str(jax.devices()[0])))

def audit_acquisition(output,workers=12):
    from .quad3d_observation_audit import audit_parent
    out=Path(output);m=read(out/'manifest.json');sm=read(Path(m['source'])/'manifest.json');pp=read(Path(m['source'])/'parents.json');by={p['id']:p for p in pp};rows=read(out/'index.json')
    assert m['source_sha256']==sha256(Path(m['source'])/'manifest.json') and m['index_sha256']==sha256(out/'index.json')
    assert m['source_files']==sm['source_files'];verify_runtime(sm)
    from .quad3d_data import audit_exploration
    auditor=audit_exploration if m.get('adaptive_gain_trace',False) else audit_parent
    audited=[]
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(auditor,[(by[r['id']],r,str(out),m) for r in rows]):
            audited.append(result);write_json(out/'audit_progress.json',dict(parents_audited=len(audited),parents_total=len(rows),physical_steps=sum(r['steps'] for r in audited)))
    write_json(out/'independent_audit.json',sanitize(dict(audit_passed=True,manifest_sha256=sha256(out/'manifest.json'),index_sha256=sha256(out/'index.json'),
        physical_steps=sum(r['steps'] for r in audited),rows=audited)))
    print(json.dumps(dict(stage='acquisition_audited',parents=len(rows))),flush=True)

def acquisition_inputs(src,acquisition,slot):
    m,parents=source(src,slot);ac=Path(acquisition);am=read(ac/'manifest.json');aa=read(ac/'independent_audit.json');index=read(ac/'index.json')
    assert aa['audit_passed'] and aa['manifest_sha256']==sha256(ac/'manifest.json') and aa['index_sha256']==am['index_sha256']==sha256(ac/'index.json')
    assert am['source_sha256']==sha256(Path(src)/'manifest.json') and [p['id'] for p in parents]==[r['id'] for r in index]
    for r,a in zip(index,aa['rows'],strict=True):
        assert all(a[k]==r[k] for k in ('id','file','sha256','steps','status','final_state'))
    index=[dict(r,first_physical_stop_step=a['first_physical_stop_step'],audited_status=a['audited_status']) for r,a in zip(index,aa['rows'],strict=True)]
    return m,parents,index

def benchmark(src,acquisition,output,slot):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);m,pp,index=acquisition_inputs(src,acquisition,slot);c=control_config(m['config'])
    i=next(i for i,p in enumerate(pp) if p['noise_level']==1);p=pp[i];row=index[i]
    assert row['sha256']==sha256(Path(acquisition)/row['file'])
    with np.load(Path(acquisition)/row['file']) as z:ctx=query_context(p,dict(z),0,c,ticks=m['snapshot_ticks'])
    _,host=branch_arrays(p,ctx,40,bank=np.asarray(m['gain_bank']));args=to_device(host);records=[];compiled=[]
    f,exe,cold=compile_fn(jax.vmap(make_observed_rollout(40,c,True)),args);(s,t),timing=measure(exe,args,repeats=3);compiled.append(f)
    records.append(dict(kind='candidate_40',batch=64,compile_seconds=cold,**timing))
    np.savez_compressed(out/'rollout.npz',**{k:np.asarray(v) for k,v in s.items()},first_control=np.asarray(t['proposed'])[:,0])
    def problem(x,g,o,mask,k,p,rm,n,bx,bo,ix,io,cursor,bias=None):
        xx,oo=observe(x,o,mask,bx,bo,n,ix[0],io[0])
        return observed_problem(xx,g,oo,mask,k,p,rm,cursor,n,c,True,nominal_bias=bias)[:3]
    inp=tuple(a[::4] for a in args);f,ex,_=compile_fn(jax.vmap(problem),inp);compiled.append(f);refs,aa,bb=map(np.asarray,ex(*inp));oracles=[]
    controls=np.asarray(t['proposed'])[::4,0];accepted=np.asarray(t['feasible'])[::4,0]
    for i in range(16):
        lp=linprog(np.zeros(4),A_ub=aa[i],b_ub=bb[i],bounds=[(None,None)]*4,method='highs');r=dict(feasible=bool(lp.success),lp_status=int(lp.status))
        if lp.success:
            opt=minimize(lambda u:.5*np.sum((u-refs[i])**2),lp.x,jac=lambda u:u-refs[i],constraints={'type':'ineq','fun':lambda u:bb[i]-aa[i]@u,'jac':lambda u:-aa[i]},method='SLSQP',options={'ftol':1e-11,'maxiter':1000})
            r.update(oracle_converged=bool(opt.success and np.max(aa[i]@opt.x-bb[i])<1e-7),action_error=float(np.max(abs(opt.x-controls[i]))))
        else:assert lp.status==2
        oracles.append(r)
    eligible=all(bool(accepted[i])==r['feasible'] and (not r['feasible'] or r['oracle_converged'] and r['action_error']<3e-5) for i,r in enumerate(oracles))
    np.savez_compressed(out/'numerics.npz',control=controls,accepted=accepted,reference=refs,a=aa,b=bb)
    write_json(out/'report.json',sanitize(dict(source_sha256=sha256(Path(src)/'manifest.json'),acquisition_manifest_sha256=sha256(Path(acquisition)/'manifest.json'),
        parent=p['id'],query_tick=0,numerics_sha256=sha256(out/'numerics.npz'),rollout_sha256=sha256(out/'rollout.npz'),oracles=oracles,accuracy_eligible=eligible,
        records=records,implicit_jit_cache_entries=[f._cache_size() for f in compiled],device=str(jax.devices()[0]))))
    print(json.dumps(dict(stage='branch_benchmark',eligible=eligible,records=records)),flush=True)

def collect(src,acquisition,output,slot):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);m,pp,index=acquisition_inputs(src,acquisition,slot);c=control_config(m['config'])
    queries=[];missing=[];exe=graph_exe=None;total_steps=0;duration=0.;begin=time.perf_counter()
    for p,row in zip(pp,index,strict=True):
        if shutil.disk_usage(out).free/2**30<storage_floor(m):raise ValueError('Learning storage reserve reached')
        assert row['sha256']==sha256(Path(acquisition)/row['file'])
        with np.load(Path(acquisition)/row['file']) as z:acquired=dict(z)
        for tick in m['snapshot_ticks']:
            if (tick>=len(acquired['active']) or not acquired['active'][:tick].all()
                or row['first_physical_stop_step'] is not None and tick>=max(1,row['first_physical_stop_step'])):
                missing.append(dict(parent=p['id'],query_tick=tick,reason='Beyond recorded or independently valid physical acquisition prefix',
                    acquisition_status=row['status'],audited_status=row['audited_status'],first_physical_stop_step=row['first_physical_stop_step'],applied_steps=row['steps']));continue
            ctx=query_context(p,acquired,tick,c,ticks=m['snapshot_ticks']);branches,host=branch_arrays(p,ctx,bank=np.asarray(m['gain_bank']));args=to_device(host);gargs=to_device(graph_args(p,ctx)+((ctx['nominal_bias'],) if 'nominal_bias' in ctx else ()))
            if exe is None:
                f,exe,cold=compile_fn(jax.vmap(make_observed_rollout(HORIZON,c,True)),args)
                gf,graph_exe,gcold=compile_fn(lambda *a:history_graph(*a[:10],config=c,nominal_bias=a[10] if len(a)>10 else None),gargs)
                print(json.dumps(dict(stage='compiled_labels',seconds=cold+gcold,batch=64,horizon=HORIZON)),flush=True)
            start=time.perf_counter();summaries,traces=jax.device_get(exe(*args));elapsed=time.perf_counter()-start;duration+=elapsed
            features,node_mask=jax.device_get(graph_exe(*gargs));reference,rm=numpy_history_graph(*graph_args(p,ctx),config=c,nominal_bias=ctx.get('nominal_bias'))
            np.testing.assert_allclose(features,reference,atol=2e-10,rtol=1e-10);np.testing.assert_array_equal(node_mask,rm)
            qpath=out/f'query_{len(queries):03d}';qpath.mkdir();rows=[]
            for j,bp in enumerate(branches):
                count=int(summaries['steps'][j]);length=min(HORIZON,count+1);file=f'{j:03d}.npz'
                np.savez_compressed(qpath/file,**{k:v[j,:length] for k,v in traces.items()})
                rows.append(dict(id=bp['id'],file=file,sha256=sha256(qpath/file),candidate=j//REPLICAS,replica=j%REPLICAS,
                    status=int(summaries['status'][j]),steps=count,final_state=summaries['final_state'][j].tolist()))
            write_json(qpath/'index.json',rows)
            np.savez_compressed(qpath/'graph.npz',features=features.astype(np.float32),node_mask=node_mask)
            qm=dict(parent=p['id'],query_tick=tick,acquisition_file=row['file'],acquisition_sha256=row['sha256'],index_sha256=sha256(qpath/'index.json'),graph_sha256=sha256(qpath/'graph.npz'),
                execute_seconds=elapsed,branches=64,physical_steps=sum(r['steps'] for r in rows))
            write_json(qpath/'query.json',qm);queries.append(dict(directory=qpath.name,query_sha256=sha256(qpath/'query.json')));total_steps+=qm['physical_steps']
            write_json(out/'progress.json',dict(queries_complete=len(queries),queries_maximum=len(pp)*len(m['snapshot_ticks']),physical_steps=total_steps,execute_seconds=duration,elapsed_seconds=time.perf_counter()-begin))
            print(json.dumps(dict(stage='labels',parent=p['id'],tick=tick,queries=len(queries),steps=qm['physical_steps'],seconds=elapsed)),flush=True)
    assert f._cache_size()==gf._cache_size()==0
    write_json(out/'queries.json',queries);write_json(out/'unavailable_queries.json',missing)
    write_json(out/'manifest.json',dict(source=str(Path(src).resolve()),source_sha256=sha256(Path(src)/'manifest.json'),acquisition=str(Path(acquisition).resolve()),
        acquisition_manifest_sha256=sha256(Path(acquisition)/'manifest.json'),queries_sha256=sha256(out/'queries.json'),unavailable_queries_sha256=sha256(out/'unavailable_queries.json'),
        config=m['config'],gain_bank=m['gain_bank'],snapshot_ticks=m['snapshot_ticks'],steps=HORIZON,transition_guard=True,source_files=m['source_files'],compile_seconds=cold+gcold,execute_seconds=duration,
        implicit_jit_cache_entries=0,queries=len(queries),physical_steps=total_steps,device=str(jax.devices()[0])))

def audit_query(arguments):
    from .quad3d_observation_audit import audit_parent
    parent,entry,output,manifest=arguments;out=Path(output);qpath=out/entry['directory'];qm=read(qpath/'query.json');c=control_config(manifest['config'])
    assert entry['query_sha256']==sha256(qpath/'query.json') and qm['index_sha256']==sha256(qpath/'index.json') and qm['graph_sha256']==sha256(qpath/'graph.npz')
    ac=Path(manifest['acquisition'])/qm['acquisition_file'];assert sha256(ac)==qm['acquisition_sha256']
    acquisition_audit=read(Path(manifest['acquisition'])/'independent_audit.json')
    acquired_row=next(r for r in acquisition_audit['rows'] if r['id']==parent['id'])
    stop=acquired_row['first_physical_stop_step'];assert stop is None or qm['query_tick']<max(1,stop)
    bank=np.asarray(manifest.get('gain_bank',GAIN_BANK));ticks=manifest.get('snapshot_ticks')
    with np.load(ac) as z:ctx=query_context(parent,dict(z),qm['query_tick'],c,ticks=ticks)
    with np.load(qpath/'graph.npz') as z:features=z['features'];mask=z['node_mask']
    expected,em=numpy_history_graph(*graph_args(parent,ctx),config=c,nominal_bias=ctx.get('nominal_bias'));np.testing.assert_allclose(features,expected,atol=2e-7,rtol=2e-7);np.testing.assert_array_equal(mask,em)
    raw=read(qpath/'index.json');audited=[];traces=[]
    assert len(raw)==64
    for j,row in enumerate(raw):
        assert row['candidate']==j//REPLICAS and row['replica']==j%REPLICAS
        bp=branch_parent(parent,ctx,bank[row['candidate']],row['replica'])
        audited.append(audit_parent((bp,row,str(qpath),manifest)))
        with np.load(qpath/row['file']) as z:traces.append(dict(z))
    payload=labels(parent,ctx,audited,traces,c)
    payload.update(features=features,node_mask=mask,gains=np.repeat(bank,REPLICAS,axis=0).astype(np.float32),group_id=np.asarray(parent['group_id']),
        partition=np.asarray(parent['partition']),query_tick=np.asarray(qm['query_tick']),goal=np.asarray(parent['goal']),noise=ctx['noise'],initial_state=ctx['observed'],
        obstacles=ctx['observed_obstacles'],obstacle_mask=ctx['mask'],previous_control=ctx['previous_u'],previous_gain=ctx['previous_gain'])
    if 'nominal_bias' in ctx:payload['nominal_bias_estimate']=ctx['nominal_bias']
    np.savez_compressed(qpath/'data.npz',**{k:np.asarray(v)[None] for k,v in payload.items()})
    record=dict(audit_passed=True,query_sha256=sha256(qpath/'query.json'),data_sha256=sha256(qpath/'data.npz'),rows=audited,
        physical_steps=sum(r['steps'] for r in audited),all_features_independently_checked=True,all_branch_physics_checked=True,
        all_current_observations_shared=True,all_future_replicas_paired=True,
        all_observer_memories_inherited_and_replayed='nominal_bias' in ctx)
    write_json(qpath/'audit.json',sanitize(record))
    return dict(file=str(Path(entry['directory'])/'data.npz'),sha256=sha256(qpath/'data.npz'),groups=1,parent_id=parent['group_id'],partition=parent['partition'],
        query_sha256=sha256(qpath/'query.json'),audit_sha256=sha256(qpath/'audit.json'),branches=64,physical_steps=record['physical_steps'],
        outcomes=dict(Counter(str(r['audited_status']) for r in audited)))

def audit_labels(output,workers=12):
    out=Path(output);m=read(out/'manifest.json');sm=read(Path(m['source'])/'manifest.json');pp=read(Path(m['source'])/'parents.json');by={p['id']:p for p in pp}
    assert m['source_sha256']==sha256(Path(m['source'])/'manifest.json') and m['queries_sha256']==sha256(out/'queries.json')
    assert m['source_files']==sm['source_files'];verify_runtime(sm)
    assert m['acquisition_manifest_sha256']==sha256(Path(m['acquisition'])/'manifest.json')
    if sm['schema']==WIDE_SCHEMA:
        am=read(Path(m['acquisition'])/'manifest.json')
        assert m['gain_bank']==am['gain_bank']==sm['gain_bank']==candidate_bank(sm['schema']).tolist()
        assert m['snapshot_ticks']==sm['snapshot_ticks']==list(WIDE_QUERY_TICKS)
    args=[(by[read(out/e['directory']/'query.json')['parent']],e,str(out),m) for e in read(out/'queries.json')];index=[]
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for r in pool.map(audit_query,args):
            index.append(r);write_json(out/'audit_progress.json',dict(queries_audited=len(index),queries_total=len(args),physical_steps=sum(v['physical_steps'] for v in index)))
    write_json(out/'index.json',index)
    write_json(out/'independent_audit.json',dict(audit_passed=True,manifest_sha256=sha256(out/'manifest.json'),index_sha256=sha256(out/'index.json'),
        physical_steps=sum(r['physical_steps'] for r in index),branches=64*len(index),all_features_independently_checked=True,all_branch_physics_checked=True))
    print(json.dumps(dict(stage='all_labels_audited',queries=len(index),branches=64*len(index))),flush=True)



def schedule(seed,exploratory,initial_gain=2.,bank=GAIN_BANK):
    bank=np.asarray(bank,np.float64);gains=np.full((8,4),initial_gain,np.float64)
    if exploratory:
        rng=np.random.default_rng(np.random.SeedSequence([seed,9801]))
        gains[1:]=bank[rng.integers(0,len(bank),7)]
    return gains

def audit_exploration(arguments):
    from .quad3d_observation_audit import audit_parent
    from .quad3d_audit import independent_envelope_values
    from .quad3d_audit import independent_obstacle_values
    parent,row,directory,manifest=arguments
    result=audit_parent(arguments)
    c=control_config(manifest['config'])
    with np.load(Path(directory)/row['file']) as z:d=dict(z)
    assert len(set(parent['gains']))==1
    expected=schedule(parent['seed'],parent['exploratory'],parent['gains'][0],manifest.get('gain_bank',GAIN_BANK))
    np.testing.assert_array_equal(parent['gain_schedule'],expected)
    # Gain proposals depend only on the predeclared seed and elapsed tick.
    x=np.asarray(parent['x']);o=np.asarray(parent['obstacles']);mask=np.asarray(parent['mask'])
    noise=np.asarray(parent['noise']);gain=np.asarray(parent['gains']);u=np.zeros(4)
    bx,bo,ix,io=unit_tape(parent['sensor_seed'],manifest['steps'],len(mask))
    for k in range(len(d['active'])):
        np.testing.assert_array_equal(d['previous_gain'][k],gain)
        np.testing.assert_array_equal(d['previous_control'][k],u)
        proposal=expected[k//200] if k%200==0 else gain
        np.testing.assert_array_equal(d['scheduled_gain'][k],proposal)
        true_o=o.copy();true_o[:,:2]+=k*c.robot.dt*o[:,3:5]
        seen,so=numpy_observe(d['state'][k],true_o,mask,bx,bo,noise,ix[k],io[k])
        controlled=numpy_obstacles(so,mask,noise)
        psi,_=independent_obstacle_values(seen,np.zeros(4),controlled,mask,proposal,c)
        domain,_=independent_envelope_values(seen,np.zeros(4),c)
        np.testing.assert_allclose(d['attempted_psi'][k],psi,atol=1e-8,rtol=1e-10)
        np.testing.assert_allclose(d['attempted_domain'][k],domain,atol=1e-8,rtol=1e-10)
        valid=bool(d['attempted_feasible'][k] and psi>=-c.qp_tolerance and domain>=-c.qp_tolerance)
        retry=not valid and np.any(proposal!=gain)
        assert retry==d['qp_switch_fallback'][k]
        np.testing.assert_array_equal(d['controller_gain'][k],gain if retry else proposal)
        if not retry:np.testing.assert_array_equal(d['proposed'][k],d['attempted_control'][k])
        if d['active'][k]:u=d['control'][k];gain=d['controller_gain'][k]
    result.update(exploration_history_verified=True,
        applied_gain_changes=int(np.sum(d['active']&np.any(d['controller_gain']!=d['previous_gain'],axis=-1))),
        qp_switch_fallbacks=int(d['qp_switch_fallback'].sum()))
    return result


def assert_disjoint_parents(new,old):
    """Seeded suites and early unseeded probes both participate in overlap checks."""
    assert not {p['id'] for p in new}&{p['id'] for p in old}
    assert not {p['seed'] for p in new if 'seed' in p}&{p['seed'] for p in old if 'seed' in p}
    def physical(p):
        active=np.asarray(p['obstacles'],float)[np.asarray(p['mask'],bool)]
        return json.dumps([np.asarray(p['x'],float).tolist(),np.asarray(p['goal'],float).tolist(),active.tolist()],separators=(',',':'))
    assert not {physical(p) for p in new}&{physical(p) for p in old}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','acquire','acquisition_benchmark','audit_acquisition','benchmark','collect','audit_labels'])
    p.add_argument('--version',type=int,choices=[94,95,98,99,104,106],default=94);p.add_argument('--source');p.add_argument('--acquisition');p.add_argument('--output',required=True);p.add_argument('--slot',type=int,default=0);p.add_argument('--workers',type=int,default=12);a=p.parse_args()
    if a.action=='prepare':prepare(a.output,a.version)
    elif a.action=='acquire':acquire(a.source,a.output,a.slot)
    elif a.action=='acquisition_benchmark':acquisition_benchmark(a.source,a.output,a.slot)
    elif a.action=='audit_acquisition':audit_acquisition(a.output,a.workers)
    elif a.action=='benchmark':benchmark(a.source,a.acquisition,a.output,a.slot)
    elif a.action=='collect':collect(a.source,a.acquisition,a.output,a.slot)
    else:audit_labels(a.output,a.workers)
