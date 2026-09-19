"""Fresh grouped flight observations, native default expert and training.

All row selection precedes teacher solves. Invalid solves stay in the dataset;
their absent labels are masked. No evaluation/hero observations enter training.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from .quad2d_barriernet import features,nominal,constraints,bounded_qp,contract,require_x64
from .quad2d_control import FlightConfig
from .barriernet_training import benchmark,train
from .dataset import sha256,source_fingerprint
from .io import write_json


def prepare(source,output,rows=200000,seed=901):
    require_x64();source=Path(source);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    c=FlightConfig();sm=json.loads((source/'manifest.json').read_text())
    audit=json.loads((source/'independent_replay.json').read_text())
    if sm['training_use'] is not True or sm['final_test'] is not False or sm['config']!=asdict(c):raise ValueError('Training-only flight source required')
    if not audit['audit_passed'] or audit['manifest_sha256']!=sha256(source/'manifest.json') or audit['index_sha256']!=sha256(source/'index.json') or sm['scenes_sha256']!=sha256(source/'scenes.json'):raise ValueError('Changed/unaudited acquisition')
    entries=json.loads((source/'index.json').read_text());parents=json.loads((source/'scenes.json').read_text())
    if [e['group_id'] for e in entries]!=[p['group_id'] for p in parents]:raise ValueError('Lost/reordered training parents')
    start=time.perf_counter();pool=[]
    for i,e in enumerate(entries):
        count=max(1,e['steps']);group=e['group_id']
        group_seed=int.from_bytes(hashlib.sha256((str(seed)+group).encode()).digest()[:8],'little')
        ticks=np.sort(np.random.default_rng(group_seed).choice(count,min(256,count),replace=False))
        pool.extend((i,int(k)) for k in ticks)
    selected=np.sort(np.random.default_rng(seed).choice(len(pool),min(rows,len(pool)),replace=False))
    pairs=np.asarray(pool,np.int32)[selected];groups=np.asarray([entries[i]['group_id'] for i in pairs[:,0]])
    unique=sorted(set(groups));order=np.random.default_rng(seed+1).permutation(len(unique))
    assignment={unique[i]:0 if k<int(.8*len(unique)) else 1 if k<int(.9*len(unique)) else 2 for k,i in enumerate(order)}
    split=np.asarray([assignment[g] for g in groups],np.int8)
    # Exclusion binds every already-declared held cohort, regardless of results.
    held_registry=Path('reports/quad2d_v43_default_odqp_fresh_comparison.json')
    registry=json.loads(held_registry.read_text());exclusions=[]
    for cohort,source_info in registry['sources'].items():
        held=Path(source_info['path'] if isinstance(source_info,dict) else source_info)
        held_parents=json.loads((held/'scenes.json').read_text())
        if set(groups)&{p['group_id'] for p in held_parents}:raise ValueError('Training/evaluation parent leakage')
        exclusions.append(dict(cohort=cohort,source=str(held.resolve()),scenes_sha256=sha256(held/'scenes.json')))
    write_json(root/'sampling.json',dict(seed=seed,rows=len(pairs),groups=len(unique),source_manifest_sha256=sha256(source/'manifest.json'),held_exclusions=exclusions))
    def teacher(x,g,o,m):
        z,ctx=features(x,g,o,m,c.robot.radius)
        G,h=constraints(x,ctx[8:].reshape(5,7),jnp.full((5,2),1.5),c.robot.radius)
        ref=nominal(x,g,c);solved=bounded_qp(ref,G,h,c.robot.force_min,c.robot.force_max)
        candidate=jnp.clip(solved.control,c.robot.force_min,c.robot.force_max)
        violation=jnp.max(jnp.concatenate((G@candidate-h,c.robot.force_min-candidate,candidate-c.robot.force_max)))
        valid=solved.feasible&jnp.isfinite(candidate).all()&(violation<=c.robot.qp_tolerance)
        return z,ctx,jnp.where(valid,candidate,0.),valid,violation,ref,candidate,solved.control,solved.feasible,solved.max_violation
    fn=jax.jit(jax.vmap(teacher));execute=None;parts=[];pending=[];provenance=[]
    def flush(chunk,n):
        nonlocal execute
        chunk=chunk+[chunk[-1]]*(512-n)
        args=tuple(jnp.asarray(np.stack(v),bool if i==3 else jnp.float64) for i,v in enumerate(zip(*chunk)))
        if execute is None:execute=fn.lower(*args).compile()
        return tuple(v[:n] for v in jax.device_get(execute(*args)))
    for i in np.unique(pairs[:,0]):
        e=entries[i];path=source/e['file']
        if sha256(path)!=e['sha256']:raise ValueError('Changed training behavior trace')
        ticks=pairs[pairs[:,0]==i,1]
        with np.load(path) as trace:
            if str(trace['group_id'])!=e['group_id'] or int(trace['expected_steps'])!=e['steps']:raise ValueError('Training parent lineage mismatch')
            states=trace['observed_state'];obs=trace['observed_obstacles'];targets=trace['route_target'];mask=trace['obstacle_mask']
            for k in ticks:
                if k>=len(states):raise ValueError('Unexecuted future teacher observation')
                pending.append((states[k],targets[k],obs[k],mask))
                if len(pending)==512:parts.append(flush(pending,512));pending=[]
        provenance.append(dict(group_id=e['group_id'],source_trace=e['file'],trace_sha256=e['sha256'],available_steps=max(1,e['steps']),sampled_rows=len(ticks)))
        if len(provenance)%128==0:print(json.dumps(dict(stage='native_teacher',parents=len(provenance),rows=sum(len(p[0]) for p in parts),seconds=time.perf_counter()-start)),flush=True)
    if pending:parts.append(flush(pending,len(pending)))
    z,ctx,label,valid,violation,reference,candidate,raw,solver_valid,solver_violation=(np.concatenate(v) for v in zip(*parts))
    np.savez_compressed(root/'data.npz',z=z,ctx=ctx,u_ref=label,label=label,valid=valid,expert_violation=violation,original_nominal=reference,expert_candidate=candidate,
        solver_raw_control=raw,solver_feasible=solver_valid,solver_violation=solver_violation,group_id=groups,tick=pairs[:,1],split=split)
    manifest=dict(schema='barriernet_quad2d_training_v1',final_test=False,source=str(source.resolve()),source_manifest_sha256=sha256(source/'manifest.json'),source_index_sha256=sha256(source/'index.json'),
        data_sha256=sha256(root/'data.npz'),source_fingerprint=source_fingerprint(),task_contract=contract(c),radius=c.robot.radius,seed=seed,rows=len(z),groups=len(unique),provenance=provenance,held_exclusions=exclusions,
        partitions={name:dict(groups=len(set(groups[split==i])),rows=int(np.sum(split==i)),valid_labels=int(np.sum(valid&(split==i))),invalid_labels=int(np.sum(~valid&(split==i)))) for i,name in enumerate(['train','validation','development_audit'])},
        sampling='Up to256 genuine pre-action ticks per acquired training parent, including initial observation for zero-step parents; frozen200000row sample before teacher solves; grouped80/10/10.',
        expert='Native static-center five-obstacle CBFQP alpha1=alpha2=1.5, original nominal; actual shared actuator bounds, default exact-JAX tolerance. Rejected labels retained/masked.',
        training_reference='Original source convention u_ref=u_expert; inference exclusively original observed-target nominal. Small supervised error alone is not deployment evidence.',
        deviations='Unavailable legacy data replaced by independently audited fresh V42 training behaviors; correctly aligned pre-action observations, grouped splits. Common observed route target, task radius.3 and actual actuator limits2.5..5.5 explicitly replace original training setup.',
        runtime_compilation='One AOT512-row teacher signature, padded last batch discarded.',elapsed_seconds=time.perf_counter()-start)
    write_json(root/'manifest.json',manifest);print(json.dumps({k:manifest[k] for k in ['rows','groups','partitions','elapsed_seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','benchmark','train']);p.add_argument('--source');p.add_argument('--dataset');p.add_argument('--output',required=True);p.add_argument('--rows',type=int,default=200000);a=p.parse_args()
    if a.action=='prepare':prepare(a.source,a.output,a.rows)
    elif a.action=='benchmark':benchmark(a.dataset,a.output,flight=True)
    else:train(a.dataset,a.output,flight=True)
