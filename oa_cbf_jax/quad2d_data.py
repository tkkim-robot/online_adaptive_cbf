"""Grouped flight data from genuine nonlinear branches; initial-state pilot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc
from .quad2d_control import FlightConfig,flight_config_from_contract,normalize_flight_contract
from .quad2d_features import flight_graph,SCHEMA as GRAPH_SCHEMA
from .quad2d_rollout import flight_branch,NAMES,GOAL,TIMEOUT,COLLISION
from .multiscale_scenes import scene as geometry
from .scenes import DIVERSE_FAMILIES
from .routing import plan_route
from .dataset import sha256,source_fingerprint,load_dataset
from .io import write_json

SCHEMA='oa_cbf_quad2d_initial_flight_v1'
TARGETS=['failure_capped_clearance_cost','observed_prefix_route_progress_div_horizon_cruise_distance']
EVENTS=['collision_first','any_adverse_termination']


def augment_gain_bank(base,queries,seed,upper=8.):
    """Preserve the audited bank, cover its upper boundary, then add Sobol pairs.

    Deterministic design coverage, independent of outcomes. This expands label
    queries and learned deployment candidates; it is not an online oracle.
    """
    base=np.asarray(base,np.float32)
    if (base.ndim!=2 or base.shape[1]!=2 or not len(base) or not np.isfinite(base).all()
            or base.min()<.5 or base.max()>8 or len(np.unique(base,axis=0))!=len(base)):
        raise ValueError('Unique finite base gain pairs in [.5,8] required')
    if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:raise ValueError('Nonnegative integer augmentation seed required')
    if isinstance(upper,bool) or upper not in (8.,16.):raise ValueError('Only original8 or explicit16 gain ceiling supported')
    if isinstance(queries,bool) or not isinstance(queries,int) or queries<4 or queries&(queries-1):raise ValueError('Power-of-two query budget required')
    boundary=np.array([[8,8],[.5,8],[8,.5],[1,8],[8,1],[2,8],[8,2],[4,8],[8,4]],np.float32)
    if upper==16.:
        boundary=np.array([(16.,v) for v in (.5,1.,2.,4.,8.,16.)]+[(v,16.) for v in (.5,1.,2.,4.,8.)],np.float32)
    rows=[tuple(row) for row in base]
    for row in boundary:
        if tuple(row) not in rows:rows.append(tuple(row))
    if len(rows)>queries:raise ValueError('Query budget cannot fit preserved bank and boundary coverage')
    sobol=np.exp(np.log(.5)+qmc.Sobol(2,scramble=True,seed=seed).random_base2(int(np.log2(queries))+1)*np.log(upper/.5)).astype(np.float32)
    for row in sobol:
        if len(rows)==queries:break
        if upper==16. and row.max()<=8.:continue
        if tuple(row) not in rows:rows.append(tuple(row))
    if len(rows)!=queries:raise ValueError('Insufficient unique candidate coverage')
    bank=np.asarray(rows,np.float32)
    contract=dict(schema='preserved_bank_upper_boundary_log_sobol_v1',seed=seed,queries=queries,
        boundary_pairs=boundary.tolist(),candidates=bank.tolist())
    if upper==16.:contract.update(schema='preserved_bank_single_ceiling_expansion_v1',lower=.5,upper=16.,original_upper=8.,additional_pairs='Only outside original[.5,8]^2; original bank preserved exactly')
    return bank,contract


def load_gain_bank(dataset,config=None,queries=None):
    """Read a frozen, audited physical candidate bank with its provenance."""
    root=Path(dataset);m=json.loads((root/'manifest.json').read_text())
    complete=json.loads((root/'complete.json').read_text())
    if not complete.get('audit_passed') or complete.get('status')!='completed' or complete['manifest_sha256']!=sha256(root/'manifest.json') or complete['index_sha256']!=sha256(root/'index.json'):
        raise ValueError('Exactly audited gain-bank dataset required')
    if config is not None and normalize_flight_contract(m['config'])!=asdict(config):raise ValueError('Gain-bank physical contract mismatch')
    entry=json.loads((root/'index.json').read_text())[0]
    if sha256(root/entry['file'])!=entry['sha256']:raise ValueError('Changed gain-bank shard')
    with np.load(root/entry['file']) as f:
        bank=f['gains'][0,::m['replicas']].copy()
        np.testing.assert_array_equal(f['gains'],np.broadcast_to(np.repeat(bank,m['replicas'],axis=0),f['gains'].shape))
    if bank.shape!=(m['queries'],2) or (queries is not None and len(bank)!=queries) or not np.isfinite(bank).all() or bank.min()<m['gain_domain']['lower'] or bank.max()>m['gain_domain']['upper']:
        raise ValueError('Invalid frozen candidate bank')
    return bank,dict(dataset=str(root.resolve()),manifest_sha256=sha256(root/'manifest.json'),index_sha256=sha256(root/'index.json'),
        shard_file=entry['file'],shard_sha256=entry['sha256'],candidates=bank.tolist())


def prepare(output,groups=1024,seed=6301,workers=28,stationary_obstacles=False):
    if groups<64 or groups%8:raise ValueError('At least64 groups balanced over8families')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);config=FlightConfig(stationary_obstacles=stationary_obstacles);start=time.perf_counter()
    rng=np.random.default_rng(seed);assignments={}
    for family in DIVERSE_FAMILIES:
        order=rng.permutation(groups//8)
        for k,i in enumerate(order):assignments[family,int(i)]='train' if k<int(.7*len(order)) else 'validation' if k<int(.85*len(order)) else 'development_calibration'
    def create(i):
        family=DIVERSE_FAMILIES[i%8];index=i//8;local_seed=seed*100000+i;rng=np.random.default_rng(local_seed+103)
        scene=geometry(local_seed,family);x=np.r_[scene.initial_state[:2],rng.uniform(-.1,.1),rng.uniform(-.25,.25,2),rng.uniform(-.15,.15)].astype(np.float32)
        obs=scene.obstacles.astype(np.float32);goal=scene.goal.astype(np.float32)
        if config.stationary_obstacles:obs[:,3:5]=0.
        route=plan_route(x[:2],goal,obs,scene.obstacle_mask,config.robot,capacity=64,visibility_batch_nodes=32)
        noise_scale=float(rng.choice([0.,.5,1.,2.]));noise=noise_scale*np.array([.015,.01,.015,.015,.02,.02,.008],np.float32)
        return dict(group_id=f'quad2d_multiscale_v1:{family}:{local_seed}',family=family,seed=local_seed,partition=assignments[family,index],
            initial_state=x.tolist(),goal=goal.tolist(),obstacles=obs.tolist(),obstacle_mask=scene.obstacle_mask.tolist(),noise=noise.tolist(),
            route={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in asdict(route).items()},solvability='unknown')
    with ThreadPoolExecutor(max_workers=workers) as pool:records=list(pool.map(create,range(groups)))
    write_json(root/'scenes.json',records);manifest=dict(stage='quad2d_initial_flight_pilot',final_test=False,training_use=True,groups=groups,seed=seed,
        config=asdict(config),source_fingerprint=source_fingerprint(),scenes_sha256=sha256(root/'scenes.json'),elapsed_seconds=time.perf_counter()-start,
        schema=SCHEMA,capacity=64,route_capacity=64,stationary_physical_obstacles=config.stationary_obstacles,
        distribution='Fresh multiscale geometry across8families and<=16/32/48/64counts. Six-state flight initial pitch±.1,world velocities±.25,rate±.15. Gravity is always world vertical, never rotated with geometry. No outcome filtering or known hero coordinates.',
        split='70/15/15 parent split stratified byfamily before branches; all gains/replicas share parent.',
        limitations='Initial-observation pilot, not visited-state coverage, generalization proof, ground-contact model or final dataset. Static route is not a dynamically feasible witness.')
    write_json(root/'manifest.json',manifest);print(json.dumps(manifest),flush=True)


def collection_kernels(config,gains,queries,replicas,horizon,guidance):
    """The actual label, graph and trace kernels, shared by collection/timing."""
    if np.asarray(gains).shape != (queries*replicas,2):
        raise ValueError('Expected the complete gain/replica bank')
    def one(x,g,o,m,points,rmask,noise,seed,ready,cursor):
        keys=jax.random.split(jax.random.PRNGKey(seed),replicas);keys=jnp.tile(keys,(queries,1))
        return jax.vmap(lambda alpha,key:flight_branch(x,g,o,m,alpha,points,rmask,cursor,noise,key,ready,config,horizon,guidance)[0])(jnp.asarray(gains),keys)
    summaries=jax.jit(jax.vmap(one))
    graph=jax.jit(jax.vmap(lambda x,g,o,m,p,rm,n,cursor,u,gain:flight_graph(x,g,o,m,p,rm,cursor,u,gain,n,config)))
    trace=jax.jit(lambda *args:flight_branch(*args,config=config,steps=horizon,guidance=guidance))
    return summaries,graph,trace


def collect(source,output,queries=16,replicas=4,horizon=160,shard_groups=16,shard_index=0,shards=1,guidance_horizon=0,gain_dataset=None,noise_clearance_weight=0.,gain_augmentation_seed=None,terminal_transition_distance=0.,performance_target='route',gain_upper=8.):
    if (isinstance(gain_upper,bool) or gain_upper not in (8.,16.)
            or (gain_upper==16. and (gain_dataset is None or gain_augmentation_seed is None or queries!=64))):
        raise ValueError('Explicit64query preserved-bank augmentation required for ceiling16')
    source=Path(source);root=Path(output);root.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    sm=json.loads((source/'manifest.json').read_text());records=json.loads((source/'scenes.json').read_text())
    config=flight_config_from_contract(sm['config'])
    if sm['scenes_sha256']!=sha256(source/'scenes.json'):raise ValueError('Source contract changed')
    from .quad2d_guidance import GuidanceConfig,NoiseClearanceGuidanceConfig,TerminalGuidanceConfig
    from .quad2d_relabel import conditional_labels,TARGETS as CONDITIONAL_TARGETS
    guidance=GuidanceConfig(horizon=guidance_horizon) if guidance_horizon else None
    if noise_clearance_weight:
        if guidance is None:raise ValueError('Noise-aware score requires predictive guidance')
        guidance=NoiseClearanceGuidanceConfig(horizon=guidance_horizon,noise_clearance_weight=noise_clearance_weight)
    if not np.isfinite(terminal_transition_distance) or terminal_transition_distance<0:raise ValueError('Invalid terminal transition')
    if terminal_transition_distance:
        if not noise_clearance_weight:raise ValueError('Terminal guidance requires the noise-aware parent contract')
        guidance=TerminalGuidanceConfig(horizon=guidance_horizon,noise_clearance_weight=noise_clearance_weight,terminal_transition_distance=terminal_transition_distance)
    if performance_target not in ('route','terminal_task') or (performance_target!='route' and not terminal_transition_distance):raise ValueError('Terminal performance target requires matched terminal guidance')
    if sm.get('visited_observations') and guidance is None:raise ValueError('Visited guidance source requires its matched controller')
    if sm.get('visited_observations'):
        audit=json.loads((source/'independent_replay.json').read_text())
        if not audit['audit_passed'] or audit['manifest_sha256']!=sha256(source/'manifest.json') or audit['scenes_sha256']!=sha256(source/'scenes.json'):
            raise ValueError('Audited visited observation source required')
        if sm['predictive_guidance']!=json.loads(json.dumps(asdict(guidance))):raise ValueError('Guidance source controller changed')
    if len(records)%shard_groups or queries<4 or queries&(queries-1) or replicas<1 or not 0<=shard_index<shards:raise ValueError('Invalid fixed branch/shard budget')
    canonical=np.exp(np.log(.5)+qmc.Sobol(2,scramble=True,seed=sm['seed']+1).random_base2(int(np.log2(queries)))*np.log(8/.5)).astype(np.float32)
    # Same gains across parents and common physical replica keys across gains.
    canonical[:4]=np.array([[.5,.5],[1.,1.],[2.,2.],[4.,4.]],np.float32)
    bank_provenance=None;augmentation=None
    if gain_augmentation_seed is not None and gain_dataset is None:raise ValueError('Augmentation requires an audited source bank')
    if gain_dataset is not None:
        canonical,bank_provenance=load_gain_bank(gain_dataset,config,queries if gain_augmentation_seed is None else None)
        if gain_augmentation_seed is not None:canonical,augmentation=augment_gain_bank(canonical,queries,gain_augmentation_seed,gain_upper)
    gains=np.repeat(canonical,replicas,axis=0);Q=len(gains)
    summaries,graph,trace_fn=collection_kernels(config,gains,queries,replicas,horizon,guidance)
    manifest=dict(schema=SCHEMA,stage='quad2d_initial_flight_pilot',production_eligible=False,final_test=False,
        source=str(source.resolve()),source_manifest_sha256=sha256(source/'manifest.json'),source_fingerprint=source_fingerprint(),
        config=asdict(config),capacity=64,route_capacity=64,graph_schema=GRAPH_SCHEMA,graph_features=40,
        queries=queries,replicas=replicas,horizon_steps=horizon,gain_domain=dict(lower=.5,upper=gain_upper),targets=TARGETS,events=EVENTS,
        controller=dict(dynamics='Quad2D',config=asdict(config),nominal='bounded velocity/pitch feedback',solver='exact reduced8row hard QP with every original row rechecked',
            sensor='raw7range latent uniform+15%innovations',prediction='held gains across actual nonlinear8substepRK4 physical branches'),
        groups=[{k:r[k] for k in ['group_id','family','seed','partition']} for r in records],scene_distribution=sm['distribution'],
        censoring='Adverse solver/domain/physical bound/collision termination receives explicit task cost1. Otherwise risk cost is negative physical clearance capped at-2. Progress is actual recorded prefix, divided by full horizon cruise distance; no unobserved continuation. Collision negative labels masked on earlier censoring. Any-adverse event labels remain valid.',
        limitations='Initial-observation development pilot only; no flight learned policy, calibration or superiority claimed.')
    if bank_provenance is not None:manifest['frozen_gain_bank']=bank_provenance
    if sm.get('data_role') in ('pilot','training','comparison'):
        role=sm['data_role']
        if sm.get('weight_fit_authorized') is not (role=='training'):
            raise ValueError('Invalid static source training authorization')
        manifest.update(data_role=role,weight_fit_authorized=sm['weight_fit_authorized'],
            training_use=sm['training_use'])
    if sm.get('data_role')=='fresh_predictive_calibration':
        if sm.get('weight_fit_authorized') is not False or sm.get('training_use') is not False:
            raise ValueError('Invalid fresh-calibration reservation')
        manifest.update(data_role=sm['data_role'],weight_fit_authorized=False,training_use=False,
            calibration_reservation=sm['calibration_reservation'])
    if augmentation is not None:manifest['gain_bank_augmentation']=augmentation
    if guidance is not None:
        manifest.update(schema='oa_cbf_quad2d_guided_hurdle_v1',stage='quad2d_guided_hurdle_development',targets=CONDITIONAL_TARGETS,
            risk_target_mode='conditional_physical_clearance',observation_context='initial_or_visited_40',
            visited_observations=bool(sm.get('visited_observations')),
            censoring='Physical clearance through horizon/goal or actual collision only; mask risk at earlier policy/solver/domain/bound stops. All first-adverse events and observed prefix progress remain labeled. Conditional risk and calibrated adverse-event screens must both pass; not unconditional future safety.',
            limitations='Development observation-conditioned fixed-gain labels for the receding predictive nominal; not adaptive-policy calibration or superiority evidence.')
        manifest['controller'].update(nominal='receding observed-route velocity/pitch guidance',predictive_guidance=asdict(guidance),
            initial_gain=[4.,4.],graph_schema=GRAPH_SCHEMA,observation_context='initial_or_visited_40')
    if terminal_transition_distance:
        from .quad2d_task_targets import contract
        manifest['controller']['performance_target']=contract(performance_target);manifest['targets']=list(manifest['targets']);manifest['targets'][1]=contract(performance_target)['target']
    executable=None;graph_executable=None;trace_executable=None;compile_seconds=0.
    write_json(root/'manifest.json',manifest);index=[]
    for number,offset in enumerate(range(0,len(records),shard_groups)):
        if number%shards!=shard_index:continue
        tick=time.perf_counter();batch=records[offset:offset+shard_groups]
        x,g,o,m=(np.asarray([r[k] for r in batch],bool if k=='obstacle_mask' else np.float32) for k in ['initial_state','goal','obstacles','obstacle_mask'])
        points=np.asarray([r['route']['points'] for r in batch],np.float32);rm=np.asarray([r['route']['mask'] for r in batch],bool)
        n=np.asarray([r['noise'] for r in batch],np.float32);seeds=np.asarray([r['seed'] for r in batch],np.uint32);ready=np.asarray([r['route']['status']=='ready' for r in batch],bool)
        cursor=np.asarray([r.get('cursor',0.) for r in batch],np.float32)
        previous_control=np.asarray([r.get('previous_control',[config.robot.mass*config.robot.gravity/2]*2) for r in batch],np.float32)
        previous_gain=np.asarray([r.get('previous_gain',[2.,2.]) for r in batch],np.float32)
        args=tuple(jnp.asarray(a) for a in (x,g,o,m,points,rm,n,seeds,ready,cursor))
        graph_args=tuple(jnp.asarray(a) for a in (x,g,o,m,points,rm,n,cursor,previous_control,previous_gain))
        if executable is None:
            cold=time.perf_counter();executable=summaries.lower(*args).compile();graph_executable=graph.lower(*graph_args).compile()
            trace_args=tuple(jnp.asarray(a) for a in (x[0],g[0],o[0],m[0],gains[0],points[0],rm[0],cursor[0],n[0],jax.random.PRNGKey(0),ready[0]))
            trace_executable=trace_fn.lower(*trace_args).compile()
            compile_seconds=time.perf_counter()-cold
            print(json.dumps(dict(stage='compiled',seconds=compile_seconds,groups=shard_groups,branches=shard_groups*Q,device=str(jax.devices()[0]))),flush=True)
        run_start=time.perf_counter();summary=jax.device_get(executable(*args));execution_seconds=time.perf_counter()-run_start
        features,node_mask=jax.device_get(graph_executable(*graph_args))
        status=summary['status'];adverse=~np.isin(status,[GOAL,TIMEOUT]);collided=status==COLLISION
        target=np.stack((np.where(adverse,1.,-np.minimum(summary['min_clearance'],.6)/.3),summary['route_progress']/(horizon*config.robot.dt*config.cruise_speed)),axis=-1).astype(np.float32)
        target_mask=np.ones_like(target,bool)
        if guidance is not None:target,target_mask=conditional_labels(status,summary['min_clearance'],target[...,1])
        events=np.stack((collided,adverse),axis=-1).astype(np.float32)
        event_mask=np.stack((~adverse|collided,np.ones_like(adverse,bool)),axis=-1)
        payload=dict(features=features,node_mask=node_mask,gains=np.broadcast_to(gains,(len(batch),Q,2)),target=target,target_mask=target_mask,
            events=events,event_mask=event_mask,group_id=np.asarray([r['group_id'] for r in batch]),partition=np.asarray([r['partition'] for r in batch]),
            initial_state=x,goal=g,obstacles=o,obstacle_mask=m,points=points,route_mask=rm,noise=n,ready=ready,seeds=seeds,**summary)
        if guidance is not None:payload.update(cursor=cursor,previous_control=previous_control,previous_gain=previous_gain)
        if terminal_transition_distance:
            from .quad2d_task_targets import values
            payload['target'][...,1]=values(payload,manifest)
        if not np.isfinite(target).all():raise ValueError('Nonfinite physical target')
        path=root/f'shard_{number:05d}.npz';np.savez_compressed(path,**payload)
        def save_trace(parent,candidate,name):
            key=jax.random.split(jax.random.PRNGKey(seeds[parent]),replicas)[candidate%replicas]
            _,trace,truth=jax.device_get(trace_executable(jnp.asarray(x[parent]),jnp.asarray(g[parent]),jnp.asarray(o[parent]),jnp.asarray(m[parent]),jnp.asarray(gains[candidate]),
                jnp.asarray(points[parent]),jnp.asarray(rm[parent]),jnp.asarray(cursor[parent]),jnp.asarray(n[parent]),key,jnp.asarray(ready[parent])))
            steps=int(summary['steps'][parent,candidate]);terminal=int(status[parent,candidate]);length=min(horizon,steps+int(terminal not in (GOAL,COLLISION,8)))
            length=max(1,length);path=root/name
            np.savez_compressed(path,**{k:v[:length] for k,v in trace.items()},true_initial_state=truth['initial_state'],true_obstacles=truth['obstacles'],
                parent=parent,candidate=candidate,group_id=batch[parent]['group_id'],key=np.asarray(key),expected_steps=steps,final_status=terminal)
            return dict(file=path.name,sha256=sha256(path),parent=int(parent),candidate=int(candidate))
        # A predetermined branch plus EVERY physical collision/bound event.
        parent=number%shard_groups;candidate=number%Q
        selected=save_trace(parent,candidate,f'trace_{number:05d}.npz')
        event_traces=[save_trace(int(p),int(q),f'event_{number:05d}_{p:02d}_{q:03d}.npz') for p,q in np.argwhere(np.isin(status,[COLLISION,8])) if (p,q)!=(parent,candidate)]
        if guidance is not None:
            traced={(parent,candidate)}|{(e['parent'],e['candidate']) for e in event_traces}
            for p in range(len(batch)):
                q=(offset+p)%Q
                if (p,q) not in traced:event_traces.append(save_trace(p,q,f'context_{number:05d}_{p:02d}_{q:03d}.npz'))
        entry=dict(file=path.name,sha256=sha256(path),trace_file=selected['file'],trace_sha256=selected['sha256'],event_traces=event_traces,offset=offset,groups=len(batch),
            branches=int(status.size),steps=int(summary['steps'].sum()),outcomes={NAMES[int(k)]:int(v) for k,v in zip(*np.unique(status,return_counts=True))},seconds=time.perf_counter()-tick,execution_seconds=execution_seconds)
        index.append(entry);write_json(root/'index.json',index);print(json.dumps(entry),flush=True)
    write_json(root/'worker_complete.json',dict(completed=True,seconds=time.perf_counter()-start,summary_signatures=1,graph_signatures=1,ahead_of_time=True,compile_seconds=compile_seconds))


def merge(parts,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);parts=list(map(Path,parts))
    manifests=[json.loads((p/'manifest.json').read_text()) for p in parts]
    if any(m!=manifests[0] for m in manifests):raise ValueError('Mismatched flight worker contracts')
    index=[];workers=[]
    for p in parts:
        workers.append(json.loads((p/'worker_complete.json').read_text()))
        for e in json.loads((p/'index.json').read_text()):
            e=dict(e)
            for filekey,hashkey in [('file','sha256'),('trace_file','trace_sha256')]:
                original=p/e[filekey]
                if sha256(original)!=e[hashkey]:raise ValueError('Changed flight shard')
                destination=root/e[filekey]
                if destination.exists():raise ValueError('Duplicate flight shard')
                destination.hardlink_to(original)
            for extra in e.get('event_traces',[]):
                original=p/extra['file']
                if sha256(original)!=extra['sha256']:raise ValueError('Changed physical-event trace')
                (root/extra['file']).hardlink_to(original)
            index.append(e)
    index.sort(key=lambda e:e['offset']);manifest=manifests[0]
    ids=[]
    for e in index:
        with np.load(root/e['file']) as f:ids.extend(map(str,f['group_id']))
    if ids!=[g['group_id'] for g in manifest['groups']] or len(set(ids))!=len(ids):raise ValueError('Lost/duplicated parent')
    write_json(root/'manifest.json',manifest);write_json(root/'index.json',index)
    partitions={}
    for name in ['train','validation','development_calibration']:
        d=load_dataset(root,name);partitions[name]=dict(groups=len(d['group_id']),branches=d['status'].size,steps=int(d['steps'].sum()),
            target_std=[float(d['target'][...,i][d['target_mask'][...,i]].std()) for i in range(2)],mean_candidate_cost_spread=float(np.mean(np.ptp(d['target'][...,0],axis=1))),
            outcomes={NAMES[int(k)]:int(v) for k,v in zip(*np.unique(d['status'],return_counts=True))})
    report=dict(partitions=partitions,groups=len(ids),workers=workers,contract_valid=all(p['groups']>0 and p['steps']>0 and min(p['target_std'])>1e-6 for p in partitions.values()))
    write_json(root/'contract_audit.json',report);print(json.dumps(report),flush=True)
    if not report['contract_valid']:raise ValueError('Flight labels lack required integrity/variation; do not train')
    # independent physical audit writes complete.json only after its own gate.


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','collect','merge']);p.add_argument('--output',required=True)
    p.add_argument('--source');p.add_argument('--parts',nargs='+');p.add_argument('--groups',type=int,default=1024);p.add_argument('--seed',type=int,default=6301)
    p.add_argument('--stationary-obstacles',action='store_true',help='Prepare stationary physical obstacles while retaining noisy observations; collection reads the saved source flag')
    p.add_argument('--workers',type=int,default=28);p.add_argument('--queries',type=int,default=16);p.add_argument('--replicas',type=int,default=4);p.add_argument('--horizon',type=int,default=160)
    p.add_argument('--shard-groups',type=int,default=16);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--guidance-horizon',type=int,default=0);p.add_argument('--gain-dataset')
    p.add_argument('--noise-clearance-weight',type=float,default=0.);p.add_argument('--gain-augmentation-seed',type=int)
    p.add_argument('--gain-upper',type=float,default=8.,choices=[8.,16.])
    p.add_argument('--terminal-transition-distance',type=float,default=0.);p.add_argument('--performance-target',choices=['route','terminal_task'],default='route');a=p.parse_args()
    if a.stationary_obstacles and a.action!='prepare':p.error('--stationary-obstacles belongs to prepare; collect uses its saved source contract')
    if a.action=='prepare':prepare(a.output,a.groups,a.seed,a.workers,a.stationary_obstacles)
    elif a.action=='merge':merge(a.parts,a.output)
    else:collect(a.source,a.output,a.queries,a.replicas,a.horizon,a.shard_groups,a.shard_index,a.shards,a.guidance_horizon,a.gain_dataset,a.noise_clearance_weight,a.gain_augmentation_seed,a.terminal_transition_distance,a.performance_target,a.gain_upper)
