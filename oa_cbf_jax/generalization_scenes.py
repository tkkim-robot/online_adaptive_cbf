"""Unseen layout compositions for development, kept separate from training.

These are new geometry/motion families, not an untouched final test. No scene is
selected by a controller outcome, and geometric routing is not a dynamic
feasibility certificate. The same immutable records go to every method.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
from .config import UnicycleConfig
from .dataset import source_fingerprint,sha256
from .io import write_json
from .scenes import Scene,fixture
from .routing import plan_route

FAMILIES=('alternating_chicane','offset_door_sequence','crossing_streams','mixed_slalom')


def scene(seed,family,capacity=16):
    if family not in FAMILIES:raise ValueError('Unknown generalization family')
    rng=np.random.default_rng(seed)
    x=np.array([0.,0.,rng.uniform(-.45,.45),rng.uniform(0.,.8)])
    length=rng.uniform(7.5,10.)
    goal=np.array([length,rng.uniform(-.35,.35)])
    rows=[]
    if family=='alternating_chicane':
        # Three alternating finite bars; the route must change direction twice.
        sign=rng.choice([-1.,1.])
        for i,fraction in enumerate((.27,.5,.73)):
            radius=rng.uniform(.23,.32)
            edge=rng.uniform(-.2,.15)
            for j in range(4):
                rows.append([length*fraction,sign*(-1)**i*(edge+j*1.9*radius),radius,0.,0.])
    elif family=='offset_door_sequence':
        # Two offset doorways, each formed by six disks. Doors are finite, so
        # going around a wall is a legitimate shared-route alternative.
        shift=rng.uniform(.45,.85)*rng.choice([-1.,1.])
        for fraction,center in ((.32,shift),(.66,-shift)):
            radius=rng.uniform(.25,.35);half_gap=radius+.3+rng.uniform(.1,.35)
            for side in (-1.,1.):
                for j in range(3):
                    rows.append([length*fraction,center+side*(half_gap+j*1.9*radius),radius,0.,0.])
    elif family=='crossing_streams':
        # Several independent crossing times and opposing transverse streams.
        count=int(rng.integers(6,13))
        for i in range(count):
            along=rng.uniform(2.,length-1.);speed=rng.uniform(.18,.55);direction=(-1)**i
            crossing_time=along/rng.uniform(.65,1.)+rng.uniform(-1.5,1.5)
            rows.append([along,-direction*speed*crossing_time,rng.uniform(.18,.34),rng.uniform(-.08,.08),direction*speed])
    else:
        # Static slalom geometry composed with crossing and longitudinal motion.
        for i,fraction in enumerate((.25,.43,.61,.79)):
            rows.append([length*fraction,(-1)**i*rng.uniform(.15,.5),rng.uniform(.25,.4),0.,0.])
        for i in range(4):
            along=rng.uniform(2.,length-1.);direction=(-1)**i;speed=rng.uniform(.18,.4)
            rows.append([along,-direction*speed*along,rng.uniform(.18,.3),rng.uniform(-.15,.15),direction*speed])
    if len(rows)>capacity:raise ValueError('Obstacle capacity exceeded; truncation is forbidden')
    obstacles=np.zeros((capacity,5));obstacles[:len(rows)]=rows
    mask=np.arange(capacity)<len(rows)
    yaw=rng.uniform(-np.pi,np.pi);translation=rng.uniform(-5.,5.,2)
    rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    x[:2]=x[:2]@rotation.T+translation
    x[2]=np.arctan2(np.sin(x[2]+yaw),np.cos(x[2]+yaw))
    goal=goal@rotation.T+translation
    obstacles[mask,:2]=obstacles[mask,:2]@rotation.T+translation
    obstacles[mask,3:5]=obstacles[mask,3:5]@rotation.T
    return Scene(f'composition:{family}:{seed}',family,seed,x,goal,obstacles,mask)


def make(directory,groups=256,seed=823000,capacity=16):
    if groups<4 or groups%4:raise ValueError('Require a positive equal allocation over four families')
    root=Path(directory);root.mkdir(parents=True,exist_ok=False)
    values=[scene(seed+i,FAMILIES[i%4],capacity) for i in range(groups)]
    values.extend(fixture(n,capacity) for n in ('open','offset','blocking','crossing'))
    records=[]
    for value in values:
        start=time.perf_counter();route=plan_route(value.initial_state[:2],value.goal,value.obstacles,value.obstacle_mask)
        records.append(dict(scene=value.json_record(),planning_seconds=time.perf_counter()-start,
            route={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in asdict(route).items()},
            solvability='unknown',reason='No independently replayed dynamically feasible witness; no controller-outcome filtering'))
    write_json(root/'scenes.json',records)
    write_json(root/'manifest.json',dict(stage='structural_generalization_development',final_test=False,groups=groups,
        seed=seed,capacity=capacity,robot=asdict(UnicycleConfig()),source=source_fingerprint(),families=list(FAMILIES),
        input_scene_sha256=sha256(root/'scenes.json'),training_use=False,selection='All specified seeds, no resampling by planner/controller outcomes',
        novelty='Three alternating turns, consecutive offset doors, opposing crossing streams, static/moving slalom compositions absent from the eight training families.',
        final_test_policy='May guide development once inspected; must never subsequently be called an untouched final test.',
        known_fixtures='Four original numerical regression fixtures appended separately; not included in generalization denominator'))
    print(str(root),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--groups',type=int,default=256)
    p.add_argument('--seed',type=int,default=823000);p.add_argument('--capacity',type=int,default=16)
    make(**vars(p.parse_args()))
