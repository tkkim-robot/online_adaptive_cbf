"""Matched development comparison of fixed gains and explicit predictive search."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np

from .dataset import source_fingerprint
from .io import write_json
from .route_control import rollout_route, INADMISSIBLE
from .predictive import rollout_search, SearchConfig, PREDICTIVE_REJECTED
from .simulation import STATUS_NAMES

CANDIDATES = np.array([[.3,.3],[.5,.5],[.75,.75],[1.,1.],[1.5,1.5],[2.,2.],
                       [3.,3.],[4.,4.],[.5,2.],[2.,.5],[1.,3.],[3.,1.]], np.float32)


def run(source, directory, horizon=40, interval=4, steps=800, batch=8, limit=None):
    root=Path(directory);root.mkdir(parents=True,exist_ok=False)
    source=Path(source);records=json.loads((source/'scenes.json').read_text())
    if limit is not None: records=records[:limit]+records[-4:]
    search=SearchConfig(horizon=horizon,interval=interval)
    manifest=dict(stage='development_predictive_search',final_test=False,source=source_fingerprint(),
                  input_scenes=str(source.resolve()), search=asdict(search), steps=steps,
                  candidates=CANDIDATES.tolist(),rows=len(records),all_methods_same_route=True,
                  initial_gain=[2.,2.],fallback='explicit termination if all candidate horizons rejected')
    write_json(root/'manifest.json',manifest)
    gains=jnp.asarray(CANDIDATES)
    fixed=jax.jit(jax.vmap(lambda x,g,o,m,p,r:jax.vmap(lambda a:rollout_route(x,g,o,m,a,p,r,steps=steps))(gains)))
    adaptive=jax.jit(jax.vmap(lambda x,g,o,m,p,r:rollout_search(x,g,o,m,gains,p,r,search=search,steps=steps)))
    names={**STATUS_NAMES,INADMISSIBLE:'hocbf_inadmissible',PREDICTIVE_REJECTED:'predictive_rejected'}
    rows=[];timings=[]
    start=time.perf_counter()
    for offset in range(0,len(records),batch):
        chosen=records[offset:offset+batch];valid=len(chosen);chosen=chosen+[chosen[-1]]*(batch-valid)
        inputs=[np.stack([r['scene'][key] for r in chosen]) for key in ('initial_state','goal','obstacles','obstacle_mask')]
        inputs.extend([np.stack([r['route'][key] for r in chosen]) for key in ('points','mask')])
        inputs=[jnp.asarray(a,dtype=bool if a.dtype==bool else jnp.float32) for a in inputs]
        for method,runner in (('fixed',fixed),('search',adaptive)):
            begin=time.perf_counter();summary,trace=runner(*inputs);jax.block_until_ready(summary)
            timings.append(dict(method=method,offset=offset,seconds=time.perf_counter()-begin,includes_compile=offset==0))
            summary=jax.device_get(summary)
            if method=='search':
                host=jax.device_get(trace)
                np.savez_compressed(root/f'traces_{offset:05d}.npz',**{k:v[:valid] for k,v in host.items()})
            for local in range(valid):
                for index in range(len(CANDIDATES) if method=='fixed' else 1):
                    item=jax.tree.map(lambda a:a[local,index] if method=='fixed' else a[local],summary)
                    route_ok=chosen[local]['route']['status']=='ready'
                    rows.append(dict(method=method,scene_id=chosen[local]['scene']['scene_id'],family=chosen[local]['scene']['family'],
                        gains=CANDIDATES[index].tolist() if method=='fixed' else None,
                        status=names[int(item.status)] if route_ok else 'planner_failure',
                        steps=int(item.steps) if route_ok else 0,progress=float(item.progress) if route_ok else 0.,
                        min_clearance=float(item.min_clearance) if np.isfinite(item.min_clearance) else None))
            print(json.dumps(dict(stage='batch',**timings[-1])),flush=True)
    write_json(root/'results.json',rows)
    aggregate=[]
    for method in ('fixed','search'):
        for gain in CANDIDATES.tolist() if method=='fixed' else [None]:
            subset=[r for r in rows if r['method']==method and r['gains']==gain and not r['scene_id'].startswith('fixture:')]
            aggregate.append(dict(method=method,gains=gain,count=len(subset),outcomes={s:sum(r['status']==s for r in subset) for s in sorted({r['status'] for r in subset})}))
    write_json(root/'summary.json',dict(aggregate=aggregate,timings=timings,elapsed_seconds=time.perf_counter()-start,
                                      jit_signatures=dict(fixed=fixed._cache_size(),search=adaptive._cache_size())))
    print(json.dumps(dict(stage='completed',aggregate=aggregate,elapsed_seconds=time.perf_counter()-start)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--directory',required=True)
    parser.add_argument('--horizon',type=int,default=40);parser.add_argument('--interval',type=int,default=4)
    parser.add_argument('--steps',type=int,default=800);parser.add_argument('--batch',type=int,default=8);parser.add_argument('--limit',type=int)
    run(**vars(parser.parse_args()))
