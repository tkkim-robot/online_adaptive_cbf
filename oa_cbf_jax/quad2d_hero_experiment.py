"""Audited ordered flight missions with frozen learned/default comparators."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import jax
import numpy as np
from .quad2d_control import FlightConfig
from .quad2d_policy import FlightPolicy,FlightPolicyConfig,SOURCE_NAMES
from .quad2d_guidance import GuidanceConfig,NoiseClearanceGuidanceConfig,TerminalGuidanceConfig
from .quad2d_data import load_gain_bank
from .quad2d_audit import check_trace,check_guidance_trace,check_gain_sources
from .quad2d_rollout import NAMES
from .quad2d_waypoints import CONTRACT,validate_parent,check_episode
from .dataset import sha256,source_fingerprint
from .io import write_json
from .cli import sanitize


def counts(rows):
    return dict(episodes=len(rows),goals=sum(r['status_code']==1 for r in rows),collisions=sum(r['status_code']==2 for r in rows),
        stops=sum(r['status_code'] in (3,5,6,7) for r in rows),state_bounds=sum(r['status_code']==8 for r in rows),timeouts=sum(r['status_code']==4 for r in rows),
        waypoint_visits=sum(r['waypoints_visited'] for r in rows),required_waypoint_visits=sum(r['required_waypoints'] for r in rows),applied_steps=sum(r['steps'] for r in rows))


def run(source,output,method='learned',dataset=None,bundle=None,calibration=None,noise_clearance_weight=0.,steps=1600,batch=8,shard_index=0,shards=1,terminal_transition_distance=0.,clearance_guard='none'):
    source=Path(source);root=Path(output);root.mkdir(parents=True,exist_ok=False);c=FlightConfig()
    sm=json.loads((source/'manifest.json').read_text());parents=json.loads((source/'scenes.json').read_text())
    if sm['scenes_sha256']!=sha256(source/'scenes.json') or sm['config']!=asdict(c) or sm['waypoint_contract']!=CONTRACT or sm['training_use'] is not False:raise ValueError('Changed/unsupported hero input contract')
    for row in parents:validate_parent(row)
    if not 0<=shard_index<shards or not parents[shard_index::shards] or steps<1:raise ValueError('Invalid full mission budget')
    chosen=parents[shard_index::shards];adaptive=method in ('learned','backup','fixed')
    if not np.isfinite(terminal_transition_distance) or terminal_transition_distance<0:raise ValueError('Invalid terminal transition')
    if clearance_guard not in ('none','one_step','predictive','inflated'):raise ValueError('Unknown clearance-guard mode')
    manifest=dict(schema='oa_cbf_quad2d_ordered_hero_episode_v1',source=str(source.resolve()),source_manifest_sha256=sha256(source/'manifest.json'),source_fingerprint=source_fingerprint(),
        config=asdict(c),waypoint_contract=CONTRACT,steps=steps,method=method,adaptive=adaptive,batch=batch if adaptive else 1,shard_index=shard_index,shards=shards,
        selected_parents=[p['group_id'] for p in chosen],final_test=False,scope='Known radius0.3 ordered hero variants; all original obstacles/XY goals retained. No final-generalization or adaptive-trajectory certificate.')
    index=[];start=time.perf_counter()
    if adaptive:
        if dataset is None:raise ValueError('Explicit audited gain bank required')
        bank,provenance=load_gain_bank(dataset,c);dm=json.loads((Path(dataset)/'manifest.json').read_text())
        if {p['group_id'] for p in parents}&{r['group_id'] for r in dm['groups']}:raise ValueError('Hero parents overlap training/calibration')
        g=NoiseClearanceGuidanceConfig(noise_clearance_weight=noise_clearance_weight) if noise_clearance_weight else GuidanceConfig()
        if terminal_transition_distance:
            if not noise_clearance_weight:raise ValueError('Terminal guidance requires the noise-aware parent contract')
            g=TerminalGuidanceConfig(noise_clearance_weight=noise_clearance_weight,terminal_transition_distance=terminal_transition_distance)
        if clearance_guard!='none':
            from .quad2d_guidance import GuardedGuidanceConfig,InflatedGuidanceConfig
            if not isinstance(g,TerminalGuidanceConfig):raise ValueError('Clearance guard requires the terminal guidance contract')
            cls=InflatedGuidanceConfig if clearance_guard=='inflated' else GuardedGuidanceConfig
            g=cls(**asdict(g),hard_prediction_clearance=clearance_guard in ('predictive','inflated'))
        pc=FlightPolicyConfig(mode=method,validation_horizon=32 if method=='fixed' else g.horizon)
        policy=FlightPolicy(bundle,calibration,c,pc,g)
        if method=='learned' and policy.metadata['dataset_manifest_sha256']!=provenance['manifest_sha256']:raise ValueError('Wrong neural candidate source')
        manifest.update(policy=asdict(pc),predictive_guidance=asdict(g),gain_bank=provenance,model_weights_sha256=policy.metadata.get('weights_sha256'),calibration_sha256=sha256(calibration) if calibration else None)
        write_json(root/'manifest.json',manifest);cold=policy.warm(bank,batch,64,64,steps,waypoint_capacity=3)
        print(json.dumps(dict(stage='warmed',compile_seconds=cold,device=str(jax.devices()[0]))),flush=True)
        for offset in range(0,len(chosen),batch):
            rows=chosen[offset:offset+batch];real=len(rows);padded=rows+[rows[-1]]*(batch-real)
            arrays=[np.asarray([r[k] for r in padded],bool if k=='obstacle_mask' else np.float32) for k in ['initial_state','waypoint_goals','obstacles','obstacle_mask']]
            arrays += [np.asarray([r['waypoint_routes'][k] for r in padded],np.float32 if k=='points' else bool) for k in ['points','mask']]
            noise=np.asarray([r['noise'] for r in padded],np.float32);keys=np.asarray([jax.random.PRNGKey(r['seed']+7193) for r in padded]);ready=np.asarray([r['waypoint_routes']['ready'] for r in padded],bool);totals=np.asarray([r['waypoint_count'] for r in padded],np.int32)
            before=time.perf_counter();summaries,traces,truth=jax.device_get(policy.run(*arrays,noise,keys,ready,steps,waypoint_count=totals));seconds=time.perf_counter()-before
            for i,parent in enumerate(rows):
                summary={k:v[i] for k,v in summaries.items()};count=int(summary['steps']);status=int(summary['status']);length=max(1,min(steps,count+int(status not in (1,2,8))))
                data={k:v[i,:length] for k,v in traces.items()};data.update(true_initial_state=truth['initial_state'][i],true_obstacles=truth['obstacles'][i],initial_observation=arrays[0][i],observed_obstacles_initial=arrays[2][i],obstacle_mask=arrays[3][i],noise=noise[i],goal=np.asarray(parent['goal'],np.float32),key=keys[i])
                row=dict(group_id=parent['group_id'],family=parent['family'],obstacles=int(arrays[3][i].sum()),noise_scale=parent['noise_scale'],variant_index=parent['variant_index'],original_kind=parent['original_kind'],
                    status=NAMES[status],status_code=status,steps=count,min_clearance=float(summary['min_clearance']),route_progress=float(summary['route_progress']),final_state=summary['final_state'].tolist(),
                    waypoint_index=int(summary['waypoint_index']),waypoints_visited=int(summary['waypoints_visited']),required_waypoints=parent['waypoint_count'],waypoint_handoffs=int(data['waypoint_handoff'].sum()),
                    source_counts={name:int(np.sum(data['requery']&(data['source']==j))) for j,name in enumerate(SOURCE_NAMES)})
                path=root/f'episode_{shard_index+(offset+i)*shards:05d}.npz';np.savez_compressed(path,**data);row.update(file=path.name,sha256=sha256(path));index.append(sanitize(row))
            write_json(root/'index.json',index);print(json.dumps(dict(completed=len(index),total=len(chosen),batch_seconds=seconds)),flush=True)
        manifest['compiled_signatures']=len(policy.compiled)
    else:
        if any(v is not None for v in [dataset,bundle,calibration]) or noise_clearance_weight or terminal_transition_distance or clearance_guard!='none':raise ValueError('Default comparator must not silently ignore neural/guidance overrides')
        from .quad2d_mpc import Quad2DMPC,DEFAULTS
        from .quad2d_mpc_experiment import FlightPhysicalKernels,episode
        if method not in DEFAULTS:raise ValueError('Unknown flight comparator')
        solver=Quad2DMPC(64,method,c);kernels=FlightPhysicalKernels(64,64,steps,c);manifest['discrete_mpc']=solver.contract()
        write_json(root/'manifest.json',manifest);print(json.dumps(dict(stage='warmed',physical_compile_seconds=kernels.compile_seconds,solver_setup_seconds=solver.setup_seconds)),flush=True)
        for i,parent in enumerate(chosen):
            row,data=episode(parent,solver,kernels,steps,ordered=True);row.update(variant_index=parent['variant_index'],original_kind=parent['original_kind'])
            path=root/f'episode_{shard_index+i*shards:05d}.npz';np.savez_compressed(path,**data);row.update(file=path.name,sha256=sha256(path));index.append(sanitize(row));write_json(root/'index.json',index)
            print(json.dumps(dict(completed=len(index),total=len(chosen),steps=row['steps'],status=row['status'])),flush=True)
    write_json(root/'manifest.json',manifest);write_json(root/'summary.json',dict(complete=True,physical_audit_pending=True,execution_seconds=time.perf_counter()-start,
        manifest_sha256=sha256(root/'manifest.json'),index_sha256=sha256(root/'index.json'),aggregate=counts(index),by_kind_noise={f'{kind}_noise{n}':counts([r for r in index if r['original_kind']==kind and r['noise_scale']==n]) for kind in ['narrow','wide'] for n in [0,1,2]}))


def audit(directory):
    root=Path(directory);m=json.loads((root/'manifest.json').read_text());c=FlightConfig();source=Path(m['source']);sm=json.loads((source/'manifest.json').read_text())
    if m['config']!=asdict(c) or m['waypoint_contract']!=CONTRACT or sha256(source/'manifest.json')!=m['source_manifest_sha256'] or sha256(source/'scenes.json')!=sm['scenes_sha256']:raise ValueError('Changed hero contract/source')
    parents=json.loads((source/'scenes.json').read_text())[m['shard_index']::m['shards']];index=json.loads((root/'index.json').read_text());audits=[]
    if [r['group_id'] for r in index]!=[p['group_id'] for p in parents]:raise ValueError('Lost/extra/reordered mission')
    for row,parent in zip(index,parents):
        if sha256(root/row['file'])!=row['sha256']:raise ValueError('Changed mission trace')
        with np.load(root/row['file']) as f:data=dict(f)
        for field,original in [('initial_observation','initial_state'),('observed_obstacles_initial','obstacles'),('obstacle_mask','obstacle_mask'),('noise','noise'),('goal','goal')]:np.testing.assert_array_equal(data[field],np.asarray(parent[original],data[field].dtype))
        np.testing.assert_array_equal(data['key'],np.asarray(jax.random.PRNGKey(parent['seed']+7193)))
        np.testing.assert_array_equal(data['state'][-1],np.asarray(row['final_state'],np.float32))
        if row['status_code']==4 and row['steps']!=m['steps']:raise ValueError('Censoring mislabeled timeout')
        if m['adaptive']:
            summary=dict(status=row['status_code'],steps=row['steps'],min_clearance=np.inf if row['min_clearance'] is None else row['min_clearance'])
            result=check_trace(data,summary,data['initial_observation'],data['observed_obstacles_initial'],data['obstacle_mask'],data['noise'],data['gain'],c)
            check_guidance_trace(data,m['predictive_guidance']);check_gain_sources(data,m['policy'],m['gain_bank']['candidates'])
            result.update(check_episode(parent,data,row,c))
        else:
            from .quad2d_mpc_experiment import audit_episode
            from .quad2d_mpc import DEFAULTS
            np.testing.assert_array_equal(m['discrete_mpc']['gains'],DEFAULTS[m['method']])
            result=audit_episode(data,row,c,np.asarray(DEFAULTS[m['method']]),waypoint_parent=parent)
        if not result['audit_passed']:raise ValueError('Physical mission audit failed: '+str(result))
        audits.append(dict(group_id=row['group_id'],**result))
    result=dict(audit_passed=bool(audits),episodes=len(audits),steps=sum(a['steps'] for a in audits),handoffs=sum(len(a['handoff_ticks']) for a in audits),rows=audits,
        manifest_sha256=sha256(root/'manifest.json'),index_sha256=sha256(root/'index.json'),auditor_source_fingerprint=source_fingerprint(),scope='Every physical prefix, sensor range, current command and ordered goal/handoff/memory transition. Every applied default MPC prediction audited. Censored future not certified.')
    write_json(root/'independent_replay.json',sanitize(result));print(json.dumps({k:v for k,v in result.items() if k not in ['rows','scope']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True);r=sub.add_parser('run')
    for name in ['source','output']:r.add_argument('--'+name,required=True)
    for name in ['dataset','bundle','calibration']:r.add_argument('--'+name)
    r.add_argument('--method',default='learned',choices=['learned','backup','fixed','fixed_low','fixed_high','optimal_decay']);r.add_argument('--noise-clearance-weight',type=float,default=0.)
    r.add_argument('--terminal-transition-distance',type=float,default=0.)
    r.add_argument('--clearance-guard',choices=['none','one_step','predictive','inflated'],default='none')
    r.add_argument('--steps',type=int,default=1600);r.add_argument('--batch',type=int,default=8);r.add_argument('--shard-index',type=int,default=0);r.add_argument('--shards',type=int,default=1)
    a=sub.add_parser('audit');a.add_argument('--directory',required=True);args=vars(p.parse_args());action=args.pop('action');run(**args) if action=='run' else audit(**args)
