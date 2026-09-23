"""Resumable deterministic contract pilot, with immutable scene-group splits.

This is an initial-state, fixed-QP pilot. It does not yet implement observation
noise, visited-state acquisition, route planning, or final calibration. Its
manifest explicitly prohibits promotion to the final claimed method.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import qmc

from .config import UnicycleConfig,config_hash
from .io import write_json
from .models import unicycle_graph
from .scenes import random_scene
from .simulation import rollout_fixed,COLLISION,INFEASIBLE,GOAL,STATUS_NAMES

SCHEMA='oa_cbf_initial_state_pilot_v1'
TARGETS=['negative_min_clearance_div_0.3_capped_below_minus_2','goal_progress_div_horizon_distance']
EVENTS=['collision_first','solver_failure_first']


def source_fingerprint():
    from .source_snapshot import snapshot_package
    return snapshot_package()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pilot_manifest(groups,capacity,gains,horizon_steps,seed,config):
    if groups<20 or groups%4 or gains<4 or gains&(gains-1) or horizon_steps<1:
        raise ValueError('Use >=20 groups divisible by four, a power-of-two gain count >=4, and a positive horizon')
    rng=np.random.default_rng(seed)
    # Stratify partitions by family, before querying gains or simulating.
    families=['scatter','corridor','cluster','moving']
    records=[]
    for family_index,family in enumerate(families):
        ids=rng.permutation(groups//4)
        partitions={int(v):('train' if i<int(.7*len(ids)) else 'validation' if i<int(.85*len(ids)) else 'development_calibration') for i,v in enumerate(ids)}
        for i in range(groups//4):
            scene_seed=seed*100000+family_index*(groups//4)+i
            records.append(dict(group_id=f'{family}:{scene_seed}',family=family,seed=scene_seed,partition=partitions[i]))
    return dict(schema=SCHEMA,stage='contract_pilot_only',production_eligible=False,groups=records,
                capacity=capacity,gains=gains,horizon_steps=horizon_steps,seed=seed,config=asdict(config),
                config_hash=config_hash(config),source_fingerprint=source_fingerprint(),targets=TARGETS,events=EVENTS,
                assumptions=dict(observation='exact_state',obstacle_prediction='constant_velocity',gain_hold='entire_label_horizon',
                                 controller='hard_instantaneous_unicycle_cbf_qp',nominal='shared_goal_feedback',replicas=1,
                                 risk='minimum until first collision, goal, or horizon; masked on earlier solver failure',
                                 progress='observed fixed-horizon/goal progress; masked on collision or solver failure',
                                 event_semantics='mutually exclusive first terminal events by fixed horizon; no assertion about unobserved later events',
                                 calibration='development only; final calibration must be fresh'))


def collect(output,groups=1024,capacity=16,gains=8,horizon_steps=160,seed=2718,shard_groups=64):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    config=UnicycleConfig()
    manifest=pilot_manifest(groups,capacity,gains,horizon_steps,seed,config)
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())!=manifest:
            raise ValueError('Dataset/source contract changed: choose a new output directory')
    else:write_json(manifest_path,manifest)
    if groups%shard_groups:raise ValueError('Fixed shard size must divide group count; no dropped remainder')
    start=time.perf_counter();index=[]
    values=qmc.Sobol(2,scramble=True,seed=seed).random_base2(int(np.log2(gains)))
    canonical=np.exp(np.log(.3)+values*np.log(4/.3)).astype(np.float32)
    canonical[:4]=np.array([[.5,.5],[1.,1.],[2.,2.],[3.,3.]])
    encode=jax.jit(jax.vmap(unicycle_graph))
    simulate=jax.jit(jax.vmap(jax.vmap(lambda x,g,o,m,a:rollout_fixed(x,g,o,m,a,config,horizon_steps)[0],
                                       in_axes=(None,None,None,None,0))))
    for number,start_group in enumerate(range(0,groups,shard_groups)):
        path=out/f'shard_{number:05d}.npz';meta=path.with_suffix('.json')
        if path.exists() and meta.exists():
            record=json.loads(meta.read_text())
            if sha256(path)!=record['sha256']:raise ValueError(f'Corrupt shard {path}')
            index.append(record);continue
        entries=manifest['groups'][start_group:start_group+shard_groups]
        scenes=[random_scene(e['seed'],e['family'],capacity,count=0 if (start_group+i)%29==0 else None) for i,e in enumerate(entries)]
        arrays=[np.stack([getattr(s,key) for s in scenes]) for key in ['initial_state','goal','obstacles','obstacle_mask']]
        x,g,o,m=(jnp.asarray(v,dtype=bool if i==3 else jnp.float32) for i,v in enumerate(arrays))
        alpha=np.broadcast_to(canonical,(shard_groups,gains,2)).copy()
        tick=time.perf_counter();result=simulate(x,g,o,m,jnp.asarray(alpha));jax.block_until_ready(result)
        features,node_mask=encode(x,g,o,m)
        status=np.asarray(result.status)
        risk=-np.minimum(np.asarray(result.min_clearance),.6)/.3
        progress=np.asarray(result.progress)/(horizon_steps*config.dt*config.v_max)
        target=np.stack((risk,progress),axis=-1).astype(np.float32)
        valid=np.stack((status!=INFEASIBLE,(status!=INFEASIBLE)&(status!=COLLISION)),axis=-1)
        target=np.where(valid,target,0.)
        events=np.stack((status==COLLISION,status==INFEASIBLE),axis=-1).astype(np.float32)
        if not np.isfinite(target).all():raise ValueError('Nonfinite unmasked target')
        payload=dict(features=np.asarray(features),node_mask=np.asarray(node_mask),gains=alpha,target=target,target_mask=valid,
                     events=events,event_mask=np.ones_like(events,bool),status=status,steps=np.asarray(result.steps),
                     min_clearance=np.asarray(result.min_clearance),min_psi1=np.asarray(result.min_psi1),
                     worst_qp_violation=np.asarray(result.worst_qp_violation),final_state=np.asarray(result.final_state),
                     group_id=np.asarray([e['group_id'] for e in entries]),partition=np.asarray([e['partition'] for e in entries]),
                     initial_state=arrays[0],goal=arrays[1],obstacles=arrays[2],obstacle_mask=arrays[3])
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as file:np.savez_compressed(file,**payload)
        tmp.replace(path)
        outcomes={STATUS_NAMES[int(k)]:int(v) for k,v in zip(*np.unique(status,return_counts=True))}
        record=dict(file=path.name,sha256=sha256(path),groups=shard_groups,branches=int(status.size),
                    observed_steps=int(payload['steps'].sum()),outcomes=outcomes,wall_seconds=time.perf_counter()-tick)
        write_json(meta,record);index.append(record)
        write_json(out/'progress.json',dict(completed_groups=start_group+shard_groups,total_groups=groups,
                                         elapsed_seconds=time.perf_counter()-start,shards=index))
        print(json.dumps(record),flush=True)
    write_json(out/'index.json',index)
    audit=audit_dataset(out)
    write_json(out/'audit.json',audit)
    write_json(out/'complete.json',dict(status='completed',elapsed_seconds=time.perf_counter()-start,
                                      manifest_sha256=sha256(manifest_path),audit_passed=audit['contract_valid']))
    print(json.dumps(audit),flush=True)
    if not audit['contract_valid']:raise ValueError('Pilot dataset failed contract audit')


def load_dataset(directory,partition,verify=True):
    path=Path(directory);parts=[]
    if (path/'INVALIDATED.json').exists():raise ValueError(f'Invalidated dataset: {path}; inspect INVALIDATED.json')
    for entry in json.loads((path/'index.json').read_text()):
        shard=path/entry['file']
        if verify and sha256(shard)!=entry['sha256']:raise ValueError(f'Corrupt {shard}')
        with np.load(shard,allow_pickle=False) as data:
            keep=data['partition']==partition
            selected={k:data[k][keep] for k in data.files}
            if 'query_origin' in entry:
                selected['query_origin']=np.full(int(keep.sum()),entry['query_origin'])
            parts.append(selected)
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}


def audit_dataset(directory):
    path=Path(directory);manifest=json.loads((path/'manifest.json').read_text())
    ids=[g['group_id'] for g in manifest['groups']]
    audit=dict(schema=SCHEMA,unique_groups=len(set(ids)),requested_groups=len(ids),partitions={})
    total_steps=0
    for partition in ['train','validation','development_calibration']:
        data=load_dataset(path,partition)
        valid=data['target_mask'];y=data['target']
        count=len(data['group_id']);total_steps+=int(data['steps'].sum())
        audit['partitions'][partition]=dict(groups=count,branches=int(data['status'].size),observed_steps=int(data['steps'].sum()),
              outcomes={STATUS_NAMES[int(k)]:int(v) for k,v in zip(*np.unique(data['status'],return_counts=True))},
              valid_target_std=[float(np.std(y[...,i][valid[...,i]])) for i in range(2)],
              mean_candidate_progress_spread=float(np.mean(np.ptp(np.where(valid[...,1],y[...,1],np.nan),axis=1)[np.all(valid[...,1],axis=1)])),
              zero_step_branches=int((data['steps']==0).sum()))
    audit['contract_valid']=bool(len(set(ids))==len(ids) and total_steps>0 and all(p['groups']>0 and min(p['valid_target_std'])>1e-6 for p in audit['partitions'].values()))
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--groups',type=int,default=1024)
    p.add_argument('--capacity',type=int,default=16);p.add_argument('--gains',type=int,default=8)
    p.add_argument('--horizon-steps',type=int,default=160);p.add_argument('--seed',type=int,default=2718)
    p.add_argument('--shard-groups',type=int,default=64)
    collect(**vars(p.parse_args()))
