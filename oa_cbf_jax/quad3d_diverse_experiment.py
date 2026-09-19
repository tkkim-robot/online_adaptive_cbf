"""Fresh, broader V88 controller diagnosis with unchanged V87 physics/control.

No learned model, noisy-sensor claim or baseline comparison. All predeclared
parents, including inadmissible starts and blocked goals, remain in denominators.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import shutil
import time
import numpy as np
import jax
from .scenes import DIVERSE_FAMILIES
from .multiscale_scenes import scene as diverse_scene,contract
from .generalization_scenes import scene as structural_scene,FAMILIES as STRUCTURAL
from .quad3d_control import Quad3DControlConfig,control_config
from .quad3d_foundation_experiment import arrays,to_device,compile_fn,make_rollout,source,read
from .io import write_json
from .dataset import sha256

FAMILIES=DIVERSE_FAMILIES+STRUCTURAL


def prepare(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);rows=[];seed=688100
    for i in range(384):
        family=FAMILIES[i%12];ss=seed+i
        scene=diverse_scene(ss,family) if family in DIVERSE_FAMILIES else structural_scene(ss,family,64)
        rng=np.random.default_rng(ss+11007);x=np.zeros(12);x[:2]=scene.initial_state[:2]
        x[2]=rng.uniform(.4,2.4);x[3:5]=rng.uniform(-.025,.025,2);x[5]=rng.uniform(-.2,.2)
        x[6:8]=rng.uniform(-.08,.08,2);x[8]=rng.uniform(-.05,.05);x[9:12]=rng.uniform(-.025,.025,3)
        goal=np.r_[scene.goal,rng.uniform(.2,2.8)]
        rows.append(dict(id=f'quad3d_v88:{family}:{ss}',index=i,seed=ss,capacity=64,family=family,
            x=x.tolist(),goal=goal.tolist(),obstacles=scene.obstacles.astype(np.float64).tolist(),mask=scene.obstacle_mask.tolist(),gains=[2.]*4,
            active_obstacles=int(scene.obstacle_mask.sum()),horizontal_extent=float(np.linalg.norm(scene.goal-x[:2])),
            solvability='unknown; no resampling/filtering by geometry, route, controller or outcome'))
    write_json(out/'parents.json',rows)
    write_json(out/'manifest.json',dict(schema='quad3d_fresh_diverse_v88',seed=seed,parents=384,capacity=64,families=FAMILIES,
        parents_sha256=sha256(out/'parents.json'),config=asdict(Quad3DControlConfig(hold_guard='bernstein_v87')),
        geometry_distribution=contract(),structural_families=list(STRUCTURAL),
        state_contract='Fresh 12-state linearized hover-neighborhood initialization with nonzero velocity/attitude/rates and independent3D goal; unicycle speed/heading not reused as flight state.',
        steps=1600,batch=16,perfect_current_observations=True,training_use=False,final_test=False,
        policy='Frozen V87 untrained fixed-gain controller, direct public goal; no scene identity/privileged future/route witness supplied.',
        note='Fresh development diagnosis. All384parents retained,32perfamily. Not a comparative or GAT result.'))


def indices(shard,shards=4):
    if shards!=4 or not 0<=shard<shards:raise ValueError('V88 requires four balanced fixed shards')
    return [i for i in range(384) if (i//12)%shards==shard]


def collect(src,output,shard,steps=1600):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);sm,all_parents=source(src,64)
    assert len(all_parents)==384 and steps==sm['steps']
    parents=[all_parents[i] for i in indices(shard)];c=control_config(sm['config']);batch=16
    args=to_device(arrays(parents[:batch]));f,exe,cold=compile_fn(jax.vmap(make_rollout(steps,c)),args)
    all_rows=[];execution=0.
    manifest=dict(source=str(Path(src).resolve()),source_sha256=sha256(Path(src)/'manifest.json'),capacity=64,
        parent_ids=[p['id'] for p in parents],shard=shard,shards=4,steps=steps,batch=batch,config=asdict(c),
        compile_seconds=cold,device=str(jax.devices()[0]),untrained_fixed_gain_diagnosis=True)
    write_json(out/'pending_manifest.json',manifest)
    for start in range(0,len(parents),batch):
        if shutil.disk_usage(out).free/2**30<150.35:raise ValueError('Retained disk buffer reached')
        ps=parents[start:start+batch];args=to_device(arrays(ps));begin=time.perf_counter()
        summary,trace=jax.device_get(exe(*args));elapsed=time.perf_counter()-begin;execution+=elapsed
        for i,p in enumerate(ps):
            file=f'{start+i:03d}.npz';np.savez_compressed(out/file,**{k:v[i] for k,v in trace.items()})
            all_rows.append(dict(id=p['id'],family=p['family'],active_obstacles=p['active_obstacles'],horizontal_extent=p['horizontal_extent'],
                file=file,sha256=sha256(out/file),status=int(summary['status'][i]),steps=int(summary['steps'][i]),final_state=summary['final_state'][i].tolist()))
        write_json(out/'progress.json',dict(parents_complete=len(all_rows),parents_total=len(parents),execution_seconds=execution,
            physical_steps=sum(r['steps'] for r in all_rows),implicit_jit_cache_entries=f._cache_size()))
        print(json.dumps(dict(stage='collected_batch',shard=shard,parents=len(all_rows),seconds=elapsed)),flush=True)
    write_json(out/'index.json',all_rows);manifest.update(index_sha256=sha256(out/'index.json'),execute_seconds=execution,
        physical_steps=sum(r['steps'] for r in all_rows),implicit_jit_cache_entries=f._cache_size())
    assert manifest['implicit_jit_cache_entries']==0
    write_json(out/'manifest.json',manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','collect']);p.add_argument('--source');p.add_argument('--output',required=True);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare(a.output)
    else:collect(a.source,a.output,a.shard)
