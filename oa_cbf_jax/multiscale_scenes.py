"""Count/extent expansion with an unchanged small-scene acquisition fraction.

Scene geometry never depends on a controller, solver outcome or hero coordinates.
The profile is versioned because it changes parent identities and split lineage.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
from .scenes import Scene,DIVERSE_FAMILIES,diverse_scene

PROFILE='multiscale_v1'


def scene(seed,family,capacity=64):
    if family not in DIVERSE_FAMILIES or capacity!=64:
        raise ValueError('Multiscale v1 requires a declared family and capacity64')
    rng=np.random.default_rng(seed+731103)
    if rng.random()<.5:
        value=diverse_scene(seed,family,16)
        value.obstacles=np.pad(value.obstacles,((0,capacity-16),(0,0)))
        value.obstacle_mask=np.pad(value.obstacle_mask,(0,capacity-16))
        value.scene_id=f'{PROFILE}:{family}:{seed}'
        return value
    count=int(rng.choice([32,48,64]));length=rng.uniform(14.,28.)
    x=np.array([0.,0.,rng.uniform(-.65,.65),rng.uniform(0.,.8)])
    goal=np.array([length,rng.uniform(-.4,.4)])
    radius=rng.uniform(.15,.36,count);velocity=np.zeros((count,2))
    if family in ('scatter','moving','head_on','overtaking'):
        position=rng.uniform([1.5,-4.],[length-1.2,4.],(count,2))
        if family=='moving':velocity=rng.uniform(-.35,.35,(count,2))
        if family in ('head_on','overtaking'):
            velocity=np.column_stack((rng.uniform(.12,.65,count)*(1 if family=='overtaking' else -1),rng.uniform(-.12,.12,count)))
            x[3]=rng.uniform(.6,1.) if family=='overtaking' else rng.uniform(.1,.65)
    elif family=='corridor':
        along=np.repeat(np.linspace(1.3,length-1.3,count//2),2)
        # Vary width and slight local offset; do not carve a controller witness.
        center=np.repeat(rng.uniform(-.25,.25,count//2),2)
        sides=np.tile([-1.,1.],count//2)
        lateral=center+sides*(radius+.3+rng.uniform(.12,.7,count))
        position=np.column_stack((along,lateral))
    elif family=='cluster':
        clusters=count//8
        centers=np.column_stack((np.linspace(2.,length-2.,clusters),rng.uniform(-1.8,1.8,clusters)))
        position=np.repeat(centers,8,axis=0)+rng.uniform(-.65,.65,(count,2))
    elif family=='bottleneck':
        positions=[];radii=[]
        for along in np.linspace(2.,length-2.,count//8):
            r=rng.uniform(.23,.36);center=rng.uniform(-.8,.8)
            half_gap=r+.3+rng.uniform(.1,.5)
            for side in (-1.,1.):
                for j in range(4):
                    positions.append([along,center+side*(half_gap+1.9*r*j)]);radii.append(r)
        position=np.asarray(positions);radius=np.asarray(radii)
    else:
        # Preserve a concave start enclosure, then add an extended obstacle field.
        wall=np.arange(-1.4,1.61,.6)
        trap=np.concatenate((np.column_stack((wall,np.full(len(wall),1.3))),
             np.column_stack((wall,np.full(len(wall),-1.3))),[[2.2,0.],[2.2,.65],[2.2,-.65]]))
        position=np.concatenate((trap,rng.uniform([4.,-4.],[length-1.2,4.],(count-len(trap),2))))
        radius[:len(trap)]=rng.uniform(.28,.36);x[3]=rng.uniform(0.,.2)
    obstacles=np.zeros((capacity,5));obstacles[:count]=np.column_stack((position,radius,velocity))
    mask=np.arange(capacity)<count
    yaw=rng.uniform(-np.pi,np.pi);offset=rng.uniform(-10.,10.,2)
    rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    x[:2]=x[:2]@rotation.T+offset
    x[2]=np.arctan2(np.sin(x[2]+yaw),np.cos(x[2]+yaw))
    goal=goal@rotation.T+offset
    obstacles[mask,:2]=obstacles[mask,:2]@rotation.T+offset
    obstacles[mask,3:5]=obstacles[mask,3:5]@rotation.T
    return Scene(f'{PROFILE}:{family}:{seed}',family,seed,x,goal,obstacles,mask)


def contract():
    return dict(name=PROFILE,capacity=64,
        sampling='Per-parent seeded coin:50% original diverse_scene at capacity16, padded without changing active geometry;50% expanded family with32/48/64 active obstacles chosen uniformly.',
        extent='Expanded14..28m start-goal extent, varied density/clearance/relative motion, random global yaw/translation. Small fraction preserves original distribution.',
        lineage='Profile-prefixed parent identity. One actual acquired observation per independent scene; all descendant queries/replicas retain its partition.',
        selection='All generated parents retained, no outcome rejection sampling or feasible-path carving. Static planning is not a dynamically feasible witness.',
        exclusions='No original or modified hero coordinates, no inspected structural/final-test parent IDs. Mandatory waypoint training is a separate pending extension.')


def make(directory,groups=256,seed=932000,route_capacity=32,route_workers=1):
    from .config import UnicycleConfig
    from .dataset import sha256,source_fingerprint
    from .io import write_json
    from .routing import plan_routes
    from .scenes import fixture
    if groups<8 or groups%8:raise ValueError('Require positive equal allocation across eight families')
    root=Path(directory);root.mkdir(parents=True,exist_ok=False)
    values=[scene(seed+i,DIVERSE_FAMILIES[i%8]) for i in range(groups)]
    values.extend(fixture(name,64) for name in ('open','offset','blocking','crossing'))
    start=time.perf_counter()
    routes,times=plan_routes(values,workers=route_workers,capacity=route_capacity,visibility_batch_nodes=32)
    planning_wall_seconds=time.perf_counter()-start
    records=[]
    for value,route,elapsed in zip(values,routes,times):
        records.append(dict(scene=value.json_record(),planning_seconds=elapsed,
            route={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in asdict(route).items()},
            solvability='unknown',reason='No independently replayed dynamic witness; all prespecified seeds kept'))
    write_json(root/'scenes.json',records)
    write_json(root/'manifest.json',dict(stage='multiscale_count_development',final_test=False,training_use=False,
        groups=groups,seed=seed,capacity=64,robot=asdict(UnicycleConfig()),families=list(DIVERSE_FAMILIES),
        scene_distribution=contract(),source_fingerprint=source_fingerprint(),
        input_scene_sha256=sha256(root/'scenes.json'),visibility_batch_nodes=32,
        route_capacity=route_capacity,route_workers=route_workers,planning_wall_seconds=planning_wall_seconds,
        interpretation='Fresh profile-matched development parents, no outcome resampling. Same records/limits/noise/step budget across methods. Small and32/48/64 counts must also be reported separately. Not a locked test.'))
    print(str(root),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True)
    parser.add_argument('--groups',type=int,default=256);parser.add_argument('--seed',type=int,default=932000)
    parser.add_argument('--route-capacity',type=int,default=32);parser.add_argument('--route-workers',type=int,default=1)
    make(**vars(parser.parse_args()))
