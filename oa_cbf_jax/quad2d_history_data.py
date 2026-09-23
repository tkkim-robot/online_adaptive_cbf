"""Audited pilot labels from authentic visited flight histories."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
from . import quad2d_history as history
from .quad2d_control import flight_config_from_contract
from .quad2d_guidance import ObservedMotionGuidanceConfig
from .quad2d_guided_data import observation_tick
from .quad2d_data import load_gain_bank
from .quad2d_relabel import conditional_labels,TARGETS
from .quad2d_task_targets import contract,values
from .dataset import sha256,source_fingerprint,load_dataset
from .io import write_json
from .cli import sanitize


def collect(source,output,replicas=4,horizon=160,shard_groups=4,gain_dataset='artifacts/datasets/quad2d_terminal_task_v42'):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);source=Path(source);start=time.perf_counter()
    sm=json.loads((source/'manifest.json').read_text());audit=json.loads((source/'independent_replay.json').read_text());index=json.loads((source/'index.json').read_text())
    c=flight_config_from_contract(sm['config']);g=ObservedMotionGuidanceConfig(noise_clearance_weight=1.)
    raw_source=Path(sm['source']);raw_manifest=json.loads((raw_source/'manifest.json').read_text());parents=json.loads((raw_source/'scenes.json').read_text())
    if not audit['audit_passed'] or audit['manifest_sha256']!=sha256(source/'manifest.json') or audit['index_sha256']!=sha256(source/'index.json'):
        raise ValueError('Unaudited acquisition')
    if sm['source_manifest_sha256']!=sha256(raw_source/'manifest.json') or raw_manifest['scenes_sha256']!=sha256(raw_source/'scenes.json') or raw_manifest.get('training_use') is not True:
        raise ValueError('Changed/nontraining acquisition parents')
    if flight_config_from_contract(raw_manifest['config'])!=c or sm['predictive_guidance']!=json.loads(json.dumps(asdict(g))) or [r['group_id'] for r in parents]!=[r['group_id'] for r in index]:raise ValueError('Changed acquisition contract')
    if len(parents)%shard_groups or replicas<1 or horizon<1:raise ValueError('Invalid fixed label shape')
    bank,provenance=load_gain_bank(gain_dataset,c,32)
    gains=np.repeat(bank,replicas,axis=0);queries=len(bank);Q=len(gains);ticks=[0,40,120,240,400,640]
    manifest=dict(schema=history.SCHEMA,stage='history_conditioned_label_pilot',production_eligible=False,final_test=False,
        source=str(source.resolve()),source_manifest_sha256=sha256(source/'manifest.json'),source_index_sha256=sha256(source/'index.json'),source_audit_sha256=sha256(source/'independent_replay.json'),
        config=asdict(c),source_fingerprint=source_fingerprint(),capacity=64,route_capacity=64,graph_schema=history.GRAPH_SCHEMA,graph_features=50,
        groups=[{k:r[k] for k in ['group_id','family','seed','partition']} for r in parents],queries=queries,replicas=replicas,horizon_steps=horizon,observation_ticks=ticks,
        frozen_gain_bank=provenance,gain_domain=dict(lower=.5,upper=8.),targets=[TARGETS[0],contract('terminal_task')['target']],events=['collision_first','any_adverse_termination'],
        controller=dict(dynamics='Quad2D',config=asdict(c),predictive_guidance=asdict(g),graph_schema=history.GRAPH_SCHEMA,performance_target=contract('terminal_task'),
            prediction='Held candidate gains; causal observer updated every actual nonlinear physical tick. Authentic preaction physical state/bias/history snapshot, future innovations only.',
            observation_context='raw sensor plus two-anchor causal velocity history; no true state/bias in graph',initial_gain=[4.,4.]),
        replica_semantics='One sampled physical latent history per parent, shared across all gains and replicas. Replicas resample future innovations only, paired by key across gains. Parent bootstrap required; replicas are not independent latent/scene samples.',
        censoring='Conditional physical clearance observed through horizon/goal/collision only. Earlier solver/domain/policy/envelope stops mask future clearance/collision-negative targets; first-adverse event and actual prefix task progress remain observed.',
        limitations='Small development data-contract pilot, not adaptive-trajectory calibration, trained-policy superiority or production eligibility. Old50/40feature bundles incompatible.')
    write_json(root/'manifest.json',manifest)
    def one(ctx,goal,mask,points,rm,noise,seed,ready):
        keys=jnp.tile(jax.random.split(jax.random.PRNGKey(seed),replicas),(queries,1))
        return jax.vmap(lambda gain,key:history.branch(ctx,goal,mask,points,rm,noise,gain,key,ready,c,g,horizon)[0])(jnp.asarray(gains),keys)
    fn=jax.jit(jax.vmap(one));gf=jax.jit(jax.vmap(lambda ctx,goal,mask,p,rm,n:history.graph(ctx['observed'],goal,ctx['obstacles'],mask,p,rm,ctx['cursor'],ctx['previous_control'],ctx['previous_gain'],n,ctx['memory'],c,g)))
    tf=jax.jit(lambda ctx,goal,mask,p,rm,n,gain,key,ready:history.branch(ctx,goal,mask,p,rm,n,gain,key,ready,c,g,horizon))
    executable=graph_executable=trace_executable=None;out_index=[];compile_seconds=0.
    for offset in range(0,len(parents),shard_groups):
        batch=parents[offset:offset+shard_groups];contexts=[];selected=[];traces=[]
        for i,parent in enumerate(batch):
            row=index[offset+i];path=source/row['file']
            if sha256(path)!=row['sha256']:raise ValueError('Changed acquisition trace')
            with np.load(path) as f:data=dict(f)
            if c.stationary_obstacles:
                from .comparison_contracts import physical_obstacle_scope
                physical_obstacle_scope('quad2d',data['true_obstacles'],data['obstacle_mask'])
            q=min(observation_tick(parent['seed'],ticks),len(data['active'])-1)
            contexts.append(history.snapshot(data,q,g.motion_window));selected.append(q);traces.append(data)
        contexts=jax.tree.map(lambda *a:np.stack(a),*contexts)
        goal=np.asarray([r['goal'] for r in batch],np.float32);mask=np.asarray([r['obstacle_mask'] for r in batch],bool)
        points=np.asarray([r['route']['points'] for r in batch],np.float32);rm=np.asarray([r['route']['mask'] for r in batch],bool)
        noise=np.asarray([r['noise'] for r in batch],np.float32);seeds=np.asarray([r['seed']+7501 for r in batch],np.uint32);ready=np.asarray([r['route']['status']=='ready' for r in batch],bool)
        args=jax.tree.map(jnp.asarray,(contexts,goal,mask,points,rm,noise,seeds,ready));ga=(args[0],args[1],args[2],args[3],args[4],args[5])
        if executable is None:
            t=time.perf_counter();executable=fn.lower(*args).compile();graph_executable=gf.lower(*ga).compile()
            ta=(jax.tree.map(lambda a:a[0],args[0]),args[1][0],args[2][0],args[3][0],args[4][0],args[5][0],jnp.array(gains[0]),jax.random.PRNGKey(0),args[7][0])
            trace_executable=tf.lower(*ta).compile();compile_seconds=time.perf_counter()-t
            print(json.dumps(dict(stage='compiled',seconds=compile_seconds,groups=shard_groups,branches=shard_groups*Q)),flush=True)
        t=time.perf_counter();summaries=jax.device_get(executable(*args));features,node_mask=jax.device_get(graph_executable(*ga))
        targets,target_mask=conditional_labels(summaries['status'],summaries['min_clearance'],summaries['route_progress']/(horizon*c.robot.dt*c.cruise_speed))
        adverse=~np.isin(summaries['status'],[1,4]);collision=summaries['status']==2
        payload=dict(features=features,node_mask=node_mask,gains=np.broadcast_to(gains,(len(batch),Q,2)),target=targets,target_mask=target_mask,
            events=np.stack((collision,adverse),-1).astype(np.float32),event_mask=np.stack((~adverse|collision,np.ones_like(adverse)),-1),
            group_id=np.asarray([r['group_id'] for r in batch]),partition=np.asarray([r['partition'] for r in batch]),goal=goal,obstacle_mask=mask,points=points,route_mask=rm,noise=noise,seeds=seeds,ready=ready,
            selected_tick=np.asarray(selected),initial_state=contexts['observed'],obstacles=contexts['obstacles'],cursor=contexts['cursor'],previous_control=contexts['previous_control'],previous_gain=contexts['previous_gain'],**summaries)
        for name,value in contexts.items():
            if name!='memory':payload['context_'+name]=value
        for name,value in zip(contexts['memory']._fields,contexts['memory']):payload['memory_'+name]=value
        payload['target'][...,1]=values(payload,manifest)
        path=root/f'shard_{offset//shard_groups:05d}.npz';np.savez_compressed(path,**payload)
        saved=[]
        for parent in range(len(batch)):
            candidates={(offset+parent)*7%Q}|set(np.flatnonzero(np.isin(summaries['status'][parent],[2,8])).tolist())
            for candidate in sorted(candidates):
                key=jax.random.split(jax.random.PRNGKey(seeds[parent]),replicas)[candidate%replicas]
                ctx=jax.tree.map(lambda a:a[parent],args[0]);s,trace=jax.device_get(trace_executable(ctx,args[1][parent],args[2][parent],args[3][parent],args[4][parent],args[5][parent],jnp.array(gains[candidate]),key,args[7][parent]))
                for name in s:np.testing.assert_allclose(s[name],summaries[name][parent,candidate],atol=2e-5,rtol=1e-6)
                count=int(s['steps']);length=max(1,min(horizon,count+int(s['status'] not in (1,2,8))))
                name=f'trace_{offset+parent:05d}_{candidate:03d}.npz';np.savez_compressed(root/name,**{k:v[:length] for k,v in trace.items()})
                saved.append(dict(file=name,sha256=sha256(root/name),parent=parent,candidate=candidate,acquisition_file=index[offset+parent]['file'],acquisition_sha256=index[offset+parent]['sha256']))
        out_index.append(dict(file=path.name,sha256=sha256(path),groups=len(batch),traces=saved));write_json(root/'index.json',out_index)
        print(json.dumps(dict(stage='labels',parents=offset+len(batch),total=len(parents),seconds=time.perf_counter()-t,physical_steps=int(summaries['steps'].sum()),elapsed_seconds=time.perf_counter()-start)),flush=True)
    write_json(root/'summary.json',dict(complete=True,parents=len(parents),branches=len(parents)*Q,compile_seconds=compile_seconds,elapsed_seconds=time.perf_counter()-start,compiled_signatures=3,physical_audit_pending=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--replicas',type=int,default=4);p.add_argument('--horizon',type=int,default=160);p.add_argument('--shard-groups',type=int,default=4)
    p.add_argument('--gain-dataset',default='artifacts/datasets/quad2d_terminal_task_v42',help='Audited bank with exactly the source physical contract; static data requires a new matching bank')
    collect(**vars(p.parse_args()))
