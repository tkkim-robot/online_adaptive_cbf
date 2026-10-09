"""Untuned single-model BarrierNet training on authentic grouped observations.

Source method architecture/loss/50epoch/64batch/Adam1e-3/patience10 are fixed.
Invalid expert solves remain recorded with no fabricated action label. This
development retraining changes the data source, not the baseline architecture.
"""

import argparse

import hashlib

import json

from pathlib import Path

import time

import numpy as np

import jax

import jax.numpy as jnp

import optax

from flax import serialization

from .barriernet import BarrierNet, features, nominal, constraints, hard_deployment_qp, train_prediction, require_x64

from .io import sha256, source_fingerprint

from .io import write_json

def prepare(source,output,rows=200000,seed=901):
    require_x64();source=Path(source);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    source_manifest=json.loads((source/'manifest.json').read_text());radius=float(source_manifest['config']['radius'])
    complete=json.loads((source/'complete.json').read_text())
    if not complete['audit_passed'] or complete['manifest_sha256']!=sha256(source/'manifest.json'):raise ValueError('Invalid source dataset')
    for name in ['independent_visitation_replay.json','independent_replay.json','independent_continuation_audit.json']:
        report=json.loads((source/name).read_text())
        if not report['audit_passed'] or report['manifest_sha256']!=complete['manifest_sha256']:raise ValueError('Missing source audit')
    inputs=[];groups=[];ticks=[];provenance=[];start=time.perf_counter()
    for entry in json.loads((source/'index.json').read_text()):
        trace_path=source/entry['visitation_file'];shard_path=source/entry['file']
        if sha256(trace_path)!=entry['visitation_sha256'] or sha256(shard_path)!=entry['sha256']:raise ValueError('Changed source shard/trace')
        with np.load(shard_path) as shard, np.load(trace_path) as trace:
            parent=int(trace['parent']);group=str(trace['group_id'])
            if str(shard['group_id'][parent])!=group:raise ValueError('Parent lineage mismatch')
            # Use each actual pre-action observation; zero-action parents also
            # retain their initial observation. No goal/success-only filtering.
            observed_states=trace['observed_state'];observed_obstacles=trace['raw_observed_obstacles']
            obstacle_mask=trace['obstacle_mask'];goal=shard['goal'][parent]
            count=max(1,int(trace['expected_steps']));count=min(count,len(observed_states))
            group_seed=int.from_bytes(hashlib.sha256((str(seed)+group).encode()).digest()[:8],'little')
            chosen=np.sort(np.random.default_rng(group_seed).choice(count,min(256,count),replace=False))
            for tick in chosen:
                inputs.append((observed_states[tick],goal,observed_obstacles[tick],obstacle_mask))
                groups.append(group);ticks.append(int(tick))
            provenance.append(dict(group_id=group,source_trace=entry['visitation_file'],trace_sha256=entry['visitation_sha256'],available_steps=count,sampled_rows=len(chosen)))
    # Freeze sampling before any expert solve. No replacement of failures.
    indices=np.sort(np.random.default_rng(seed).choice(len(inputs),min(rows,len(inputs)),replace=False))
    inputs=[inputs[i] for i in indices];groups=np.asarray(groups)[indices];ticks=np.asarray(ticks)[indices]
    unique=sorted(set(groups));order=np.random.default_rng(seed+1).permutation(len(unique))
    assignment={unique[i]:0 if k<int(.8*len(unique)) else 1 if k<int(.9*len(unique)) else 2 for k,i in enumerate(order)}
    split=np.array([assignment[g] for g in groups],np.int8)
    def teacher(x,g,o,m):
        z,ctx=features(x,g,o,m,radius);G,h=constraints(x,ctx[6:].reshape(5,7),jnp.full((5,2),1.5),radius)
        ref=nominal(x,g);u,valid,violation,raw,solver_valid,solver_violation=hard_deployment_qp(ref*(1+1e-6),G,h,return_diagnostics=True)
        return z,ctx,jnp.where(valid,u,0.),valid,violation,ref,u,raw,solver_valid,solver_violation
    fn=jax.jit(jax.vmap(teacher));parts=[];batch=512
    for offset in range(0,len(inputs),batch):
        chunk=inputs[offset:offset+batch];n=len(chunk);chunk += [chunk[-1]]*(batch-n)
        values=tuple(jnp.asarray(np.stack(v),bool if i==3 else jnp.float64) for i,v in enumerate(zip(*chunk)))
        result=jax.device_get(fn(*values));parts.append(tuple(a[:n] for a in result))
    z,ctx,label,valid,violation,reference,candidate,raw,solver_valid,solver_violation=(np.concatenate(a) for a in zip(*parts))
    np.savez_compressed(root/'data.npz',z=z,ctx=ctx,u_ref=label,label=label,valid=valid,expert_violation=violation,
                        original_nominal=reference,expert_candidate=candidate,solver_raw_control=raw,solver_feasible=solver_valid,
                        solver_violation=solver_violation,group_id=groups,tick=ticks,split=split)
    # Match the local method's training u_ref=u_expert convention explicitly.
    # Deployment will use only the original observed-goal nominal, never labels.
    manifest=dict(schema='barriernet_unicycle_training_v1',final_test=False,source=str(source.resolve()),
        source_manifest_sha256=sha256(source/'manifest.json'),data_sha256=sha256(root/'data.npz'),source_fingerprint=source_fingerprint(),
        radius=radius,seed=seed,requested_rows=rows,rows=len(z),groups=len(unique),expert='Original static-center five-obstacle CBF-QP, gains1.5/1.5, nominal default feedback, bounded inputs; default exact-JAX tolerance.',
        partitions={name:dict(groups=len(set(groups[split==i])),rows=int(np.sum(split==i)),valid_labels=int(np.sum(valid&(split==i))),invalid_labels=int(np.sum(~valid&(split==i)))) for i,name in enumerate(['train','validation','development_audit'])},
        provenance=provenance,jit_signatures=fn._cache_size(),elapsed_seconds=time.perf_counter()-start,
        lineage='Frozen row sampling before expert labels, at most256 actual pre-action observations per independent recorded parent. Parent-disjoint80/10/10 split before fitting. No hero/evaluation parent IDs, future sensor innovations or latent features.',
        deviations='Fresh broad acquired-state demonstrations replace unavailable legacy dataset. Correct pre-action alignment and grouped splits replace legacy post-step rows/rowwise split. All invalid teacher solves retained but excluded from regression because no expert action exists. Native five-nearest architecture retained, not all-obstacle OA replacement.',
        training_reference='u_ref equals bounded expert action during training, exactly as local source. Deployment must use observed nominal reference only; labels cannot enter inference.',
        solver_censoring='Invalid means no accepted teacher label, not a proof of mathematical infeasibility. Raw solve and clipped candidate retained with both residuals. Independent feasibility classification records numerical/post-clip rejections separately.',
        limitations='Development supervised data only, not baseline closed-loop results or safety evidence. Raw observations; no OA observer/gates/routing nominal. Native five-nearest controller still evaluated physically against every obstacle.')
    write_json(root/'manifest.json',manifest);print(json.dumps({k:manifest[k] for k in ['rows','groups','partitions','elapsed_seconds']}),flush=True)

def load_data(dataset):
    root=Path(dataset);manifest=json.loads((root/'manifest.json').read_text())
    if sha256(root/'data.npz')!=manifest['data_sha256']:raise ValueError('Changed BarrierNet data')
    with np.load(root/'data.npz') as f:data={k:f[k] for k in ['z','ctx','u_ref','label','valid','split']}
    train=(data['split']==0)&data['valid']
    if not train.any():raise ValueError('No valid training labels')
    mean=data['z'][train].mean(0);std=data['z'][train].std(0);std=np.where(std==0,1.,std)
    return manifest,data,mean,std

def training_kernels(mean,std,radius,seed=901,*,state_dim=4,goal_dim=2,control_dim=2,model=None,prediction_fn=train_prediction):
    require_x64();model=BarrierNet() if model is None else model;optimizer=optax.adam(1e-3)
    params=model.init(jax.random.PRNGKey(seed),jnp.zeros(25),jnp.zeros(state_dim),jnp.zeros(goal_dim),jnp.zeros(control_dim))['params']
    def losses(params,batch):
        z,ctx,ref,label,mask=batch
        pred,p=jax.vmap(lambda z,c,u:prediction_fn(model,params,z,c,u,mean,std,radius))(z,ctx,ref)
        mse=jnp.sum(jnp.mean((pred-label)**2,axis=-1)*mask)/jnp.maximum(mask.sum(),1.)
        regularizer=jnp.sum(jnp.mean(jnp.maximum(p-3.5,0.),axis=(1,2))*mask)/jnp.maximum(mask.sum(),1.)
        return mse+.01*regularizer,mse
    def step(carry,batch):
        params,opt_state=carry
        (loss,mse),gradient=jax.value_and_grad(losses,has_aux=True)(params,batch)
        updates,opt_state=optimizer.update(gradient,opt_state,params)
        return (optax.apply_updates(params,updates),opt_state),(loss,mse)
    def epoch(carry,batches):return jax.lax.scan(step,carry,batches)
    def validation(params,batches):
        # Weighted aggregate over actual valid observations, including remainder.
        return jax.lax.map(lambda batch:losses(params,batch)[1],batches)
    return (params,optimizer.init(params)),jax.jit(step),jax.jit(epoch),jax.jit(validation)

def batches(data,indices):
    indices=np.asarray(indices);count=len(indices);pad=(-count)%64
    padded=np.pad(indices,(0,pad),mode='edge');valid=data['valid'][padded].astype(float);valid[count:]=0.
    arrays=[data[k][padded] for k in ['z','ctx','u_ref','label']]+[valid]
    return tuple(jnp.asarray(a.reshape((-1,64)+a.shape[1:]),jnp.float64) for a in arrays)

def method_details(manifest,*,flight=False,variant=None):
    """Only dimensions/architecture change; all native training settings stay fixed."""
    options={}
    name='unicycle';state_dim=4
    if variant is not None:
        if flight:raise ValueError('Choose one native robot model')
        from .barriernet_variants import BarrierNetVariant, train_prediction as variant_prediction
        if variant not in ('Quad3D','KinematicBicycle2D_DPCBF'):raise ValueError('Unknown native variant')
        name='quad3d' if variant=='Quad3D' else 'bicycle'
        options=dict(model=BarrierNetVariant(variant),state_dim=6 if name=='quad3d' else 4,
            goal_dim=3 if name=='quad3d' else 2,control_dim=4 if name=='quad3d' else 2,prediction_fn=variant_prediction)
        architecture=('5x[5->512tanh->128tanh->128tanh], per-obstacle2sigmoid*4, mean pooling, 141->128tanh->128tanh->4 residual control'
            if name=='quad3d' else '5x[5->256ReLU->64ReLU], per-obstacle1sigmoid*4, mean pooling, 72->64ReLU->2 residual control')
    elif flight:
        from .quad2d_barriernet import train_prediction as flight_prediction
        options=dict(state_dim=6,prediction_fn=flight_prediction)
        name='quad2d';state_dim=6
    if variant is None:
        architecture=f'5x[5->256ReLU->64ReLU], per-obstacle2sigmoid*4, mean pooling, [64+{state_dim}+2+2]->64ReLU->2 residual control'
    if manifest['schema']!=f'barriernet_{name}_training_v1':raise ValueError('Wrong native training data')
    return options,f'barriernet_{name}_jax_v1',architecture

def benchmark(dataset,output,samples=128,*,flight=False,variant=None):
    require_x64();manifest,data,mean,std=load_data(dataset)
    options,_,_=method_details(manifest,flight=flight,variant=variant)
    carry,step,_,_=training_kernels(jnp.asarray(mean),jnp.asarray(std),manifest['radius'],**options)
    batch=tuple(x[0] for x in batches(data,np.flatnonzero((data['split']==0)&data['valid'])[:64]))
    begin=time.perf_counter();exe=step.lower(carry,batch).compile();carry,value=exe(carry,batch);jax.block_until_ready((carry,value));cold=time.perf_counter()-begin
    times=[]
    for _ in range(samples):
        t=time.perf_counter();carry,value=exe(carry,batch);jax.block_until_ready((carry,value));times.append(time.perf_counter()-t)
    if not all(np.isfinite(a).all() for a in jax.tree.leaves(jax.device_get((carry,value)))):raise ValueError('Nonfinite default training benchmark')
    report=dict(device=str(jax.devices()[0]),samples=samples,cold_seconds=cold,
        milliseconds=np.quantile(times,[.5,.95,.99,1]).tolist(),runtime_compilation=0,final_loss=np.asarray(value).tolist(),
        scope='Synchronized warm default64sample end-to-end FP64 BarrierNet differentiable-QP Adam steps. Single model; independent devices are not a larger ensemble or tuned baseline.')
    report['milliseconds']=[v*1000 for v in report['milliseconds']]
    write_json(output,report);print(json.dumps(report),flush=True)

def train(dataset,output,*,flight=False,variant=None):
    require_x64();root=Path(output);root.mkdir(parents=True,exist_ok=False)
    audit=json.loads((Path(dataset)/'independent_audit.json').read_text())
    if not audit['audit_passed'] or audit['manifest_sha256']!=sha256(Path(dataset)/'manifest.json'):raise ValueError('Independent dataset audit required')
    manifest,data,mean,std=load_data(dataset);radius=manifest['radius'];seed=901
    options,schema,architecture=method_details(manifest,flight=flight,variant=variant)
    carry,_,epoch,validation=training_kernels(jnp.asarray(mean),jnp.asarray(std),radius,seed,**options)
    train_indices=np.flatnonzero((data['split']==0)&data['valid']);valid_indices=np.flatnonzero((data['split']==1)&data['valid'])
    if not np.any(data['valid'][valid_indices]):raise ValueError('No valid validation labels')
    valid_batches=batches(data,valid_indices);train_batches=batches(data,train_indices)
    begin=time.perf_counter();execute=epoch.lower(carry,train_batches).compile();validate=validation.lower(carry[0],valid_batches).compile()
    print(json.dumps(dict(ready=True,device=str(jax.devices()[0]),compile_seconds=time.perf_counter()-begin,train_rows=len(train_indices),validation_rows=len(valid_indices))),flush=True)
    best=np.inf;stale=0;history=[];rng=np.random.default_rng(seed)
    for number in range(50):
        start=time.perf_counter();batch=batches(data,rng.permutation(train_indices));carry,loss=execute(carry,batch)
        val=np.asarray(validate(carry[0],valid_batches));weights=np.asarray(valid_batches[-1]).sum(1)
        value=float(np.sum(val*weights)/weights.sum());loss=np.asarray(jax.device_get(loss))
        if not np.isfinite(loss).all() or not np.isfinite(value):raise ValueError('Nonfinite baseline training; no fallback or result fabrication')
        row=dict(epoch=number+1,training_mse=float(loss[1].mean()),validation_mse=value,seconds=time.perf_counter()-start)
        history.append(row);print(json.dumps(row),flush=True)
        if value<best:
            best=value;stale=0
            (root/'weights.msgpack').write_bytes(serialization.to_bytes(jax.device_get(carry[0])))
            np.savez(root/'normalization.npz',mean=mean,std=std)
        else:stale+=1
        write_json(root/'history.json',history)
        if stale>=10:break
    write_json(root/'manifest.json',dict(schema=schema,final_test=False,production_eligible=False,
        dataset=str(Path(dataset).resolve()),dataset_manifest_sha256=sha256(Path(dataset)/'manifest.json'),
        weights_sha256=sha256(root/'weights.msgpack'),normalization_sha256=sha256(root/'normalization.npz'),
        source_fingerprint=source_fingerprint(),radius=radius,seed=seed,architecture=architecture,
        task_contract=manifest.get('task_contract'),
        settings=dict(epochs=50,batch_size=64,adam_lr=.001,patience=10,p_regularizer=.01,train_slack_rho=1e4,eps_q=1e-6),
        completed_epochs=len(history),best_validation_mse=best,elapsed_seconds=time.perf_counter()-begin,device=str(jax.devices()[0]),
        runtime_compilation='One compiled epoch and validation signature, no runtime tracing. Padded final batches carry explicit masks.',
        limitations='Untuned local-repository BarrierNet retraining with fresh broad grouped demonstrations. Validation loss is not closed-loop evidence. No baseline result or superiority claim until independent physical, constraint and matched-task evaluation.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','benchmark','train'])
    parser.add_argument('--source');parser.add_argument('--dataset');parser.add_argument('--output',required=True)
    parser.add_argument('--variant',choices=['Quad3D','KinematicBicycle2D_DPCBF'])
    parser.add_argument('--rows',type=int,default=200000);args=parser.parse_args()
    if args.action=='prepare':
        if args.variant is not None:parser.error('Variant demonstrations require their physical source adapter')
        prepare(args.source,args.output,args.rows)
    elif args.action=='benchmark':benchmark(args.dataset,args.output,variant=args.variant)
    else:train(args.dataset,args.output,variant=args.variant)
