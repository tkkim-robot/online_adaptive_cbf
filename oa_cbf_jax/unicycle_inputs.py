"""Frozen stationary unicycle layouts and shared physical task contracts."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .config import UnicycleConfig
from .dataset import sha256,source_fingerprint
from .io import write_json
from .scenes import DIVERSE_FAMILIES,stationary_scene
from .multiscale_scenes import scene
from .routing import plan_routes
from .comparison_contracts import physical_obstacle_scope

SCHEMA='oa_cbf_unicycle_static_inputs'
PHYSICAL_FIELDS=tuple(k for k in asdict(UnicycleConfig()) if not k.startswith('guidance_'))


def source_robot(directory,requested=None):
    """Bind new evaluations to the frozen task; preserve historical sources.

    OA guidance belongs to the method, not the shared physical task. It may
    differ from a native baseline; the full plant, sensor and arrival contract
    must match exactly. A learned model's supplied config is never overwritten.
    """
    root=Path(directory);path=root/'manifest.json'
    if not path.exists():
        if requested is not None and requested.stationary_obstacles:
            raise ValueError('Stationary unicycle inputs require the explicit source schema')
        return requested if requested is not None else UnicycleConfig()
    m=json.loads(path.read_text())
    if m.get('schema')!=SCHEMA:
        if m.get('stationary_physical_obstacles') is True or (requested is not None and requested.stationary_obstacles):
            raise ValueError('Stationary unicycle inputs require the explicit source schema')
        return requested if requested is not None else UnicycleConfig()
    saved=m['robot']
    if set(saved)!=set(asdict(UnicycleConfig())):raise ValueError('Incomplete unicycle physical source contract')
    robot=UnicycleConfig(**saved)
    if not robot.stationary_obstacles or m.get('stationary_physical_obstacles') is not True:
        raise ValueError('Static unicycle source must bind stationary physical truth')
    if sha256(root/'scenes.json')!=m['scenes_sha256']:raise ValueError('Changed unicycle scenes')
    rows=json.loads((root/'scenes.json').read_text())
    ids=[r['scene']['scene_id'] for r in rows]
    if len(rows)!=m['groups'] or len(set(ids))!=len(ids):raise ValueError('Missing or duplicate unicycle parents')
    for r in rows:physical_obstacle_scope('unicycle',r['scene']['obstacles'],r['scene']['obstacle_mask'])
    if requested is not None:
        mismatched=[k for k in PHYSICAL_FIELDS if getattr(requested,k)!=getattr(robot,k)]
        if mismatched:raise ValueError('Unicycle source physical mismatch: '+', '.join(mismatched))
        return requested
    return robot


def prepare(output,groups=256,seed=930100,workers=4):
    if groups<8 or groups%8 or seed<0 or workers<1:raise ValueError('Balanced family counts and valid seed/workers required')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    robot=UnicycleConfig(stationary_obstacles=True)
    scenes=[stationary_scene(scene(seed+i,DIVERSE_FAMILIES[i%8],64)) for i in range(groups)]
    routes,_=plan_routes(scenes,robot,workers=workers,capacity=64,visibility_batch_nodes=32)
    records=[dict(scene=s.json_record(),route={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in asdict(r).items()}) for s,r in zip(scenes,routes)]
    write_json(root/'scenes.json',records)
    write_json(root/'manifest.json',dict(schema=SCHEMA,robot=asdict(robot),groups=groups,seed=seed,
        stationary_physical_obstacles=True,capacity=64,route_capacity=64,training_use=False,final_test=False,
        scenes_sha256=sha256(root/'scenes.json'),source_fingerprint=source_fingerprint(),
        layout_generators=list(DIVERSE_FAMILIES),route_failures=sum(r.status!='ready' for r in routes),
        selection='Consecutive seed range with balanced layout generators, zero physical velocities before routing, no outcome filtering. Names describe geometry generators, not actual motion.',
        noise='Evaluation chooses a recorded common noise scale; static sensor contract never moves true obstacles.'))
    source_robot(root)
    print(json.dumps(dict(source=str(root),parents=groups,stationary=True,route_failures=sum(r.status!='ready' for r in routes))),flush=True)


def verify_reservation(directory, artifacts='artifacts'):
    """Verify seeded physical inputs and disjointness before attaching models.

    Includes prior scene files and grouped training manifests. This is a
    development reservation, not an assertion of untouched final-test status.
    """
    root=Path(directory).resolve();source_robot(root)
    manifest=json.loads((root/'manifest.json').read_text())
    rows=json.loads((root/'scenes.json').read_text())
    expected=[stationary_scene(scene(manifest['seed']+i,DIVERSE_FAMILIES[i%8],64)).json_record()
              for i in range(manifest['groups'])]
    if [row['scene'] for row in rows]!=expected:
        raise ValueError('Reserved scene geometry differs from the predeclared seed rule')
    ids={row['scene_id'] for row in expected};checked=[]
    artifacts=Path(artifacts)
    for path in sorted((artifacts/'experiments').glob('*/scenes.json')):
        if path.resolve()==root/'scenes.json':continue
        values=json.loads(path.read_text())
        if not isinstance(values,list):continue
        previous={row['scene'].get('scene_id') for row in values
                  if isinstance(row,dict) and isinstance(row.get('scene'),dict)}
        previous.discard(None)
        if ids&previous:raise ValueError('Evaluation overlaps prior scene inputs: '+str(path))
        if previous:checked.append(dict(path=str(path.resolve()),sha256=sha256(path),parents=len(previous)))
    for path in sorted((artifacts/'datasets').glob('*/manifest.json')):
        values=json.loads(path.read_text()).get('groups')
        if not isinstance(values,list):continue
        previous={row.get('group_id') for row in values if isinstance(row,dict)}
        previous.discard(None)
        if ids&previous:raise ValueError('Evaluation overlaps grouped training data: '+str(path))
        if previous:checked.append(dict(path=str(path.resolve()),sha256=sha256(path),parents=len(previous)))
    return dict(source=str(root),manifest_sha256=sha256(root/'manifest.json'),
                scenes_sha256=sha256(root/'scenes.json'),parents=len(rows),
                seeded_geometry_and_stationary_truth_verified=True,prior_sources=checked,
                final_test=False,policy_outcomes_used_for_selection=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--groups',type=int,default=256)
    p.add_argument('--seed',type=int,default=930100);p.add_argument('--workers',type=int,default=4)
    prepare(**vars(p.parse_args()))
