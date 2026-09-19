"""Known hero variants with every obstacle and mandatory waypoint retained.

These are development tasks under the trained radius0.3 unicycle contract, not
the unsupported original radius0.24/0.20 runs or a final-test dataset.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .config import UnicycleConfig
from .dataset import sha256,source_fingerprint
from .io import write_json
from .routing import plan_route


CONTRACT=dict(
    name='ordered_stop_and_go_v1',capacity=3,arrival_tolerance=.25,arrival_speed=.2,
    handoff='Only the next waypoint may advance. Pre-action sensed distance plus sqrt(2)*1.15*position noise bound <=0.25m and absolute sensed speed plus1.15*speed bound <=0.2m/s.',
    continuation='One uninterrupted physical episode and sensor stream. Preserve velocity/heading, observer history, biases, gains and previous action. Reset only the next route leg cursor and require a fresh policy decision.',
    final='Physical arrival at the last waypoint within the same tolerance/speed after all intermediate handoffs. Collision and bound violations take priority over success.',
    route='One visibility route per required leg from original static observations. All active obstacles remain in physical simulation and safety constraints. Routes are guidance, never a feasibility certificate.',
    interpretation='Declared stop-and-go task variant; no original or final hero-comparison claim. Route progress is local to the current leg; mission completion requires all waypoints.')


def make(directory,original='configs/eval/hero_scenes/original',variants=32):
    if isinstance(variants,bool) or not isinstance(variants,int) or variants<0:raise ValueError('Invalid variant count')
    root=Path(directory);root.mkdir(parents=True,exist_ok=False)
    records=[];provenance={}
    for kind,seed in [('narrow',930200),('wide',930400)]:
        path=Path(original)/f'dynamic_unicycle_{kind}.json';saved=json.loads(path.read_text())
        initial=np.asarray(saved['initial_state'],float);raw_obs=np.asarray(saved['obstacles'],float)
        waypoints=np.asarray(saved['waypoints'],float)
        if len(raw_obs)>64 or len(waypoints)-1>3 or not np.array_equal(waypoints[0,:2],initial[:2]):
            raise ValueError('Unsupported original shape/start anchor; no truncation or silent waypoint omission')
        goals=waypoints[1:,:2];count=len(goals)
        if not 1<=count<=3:raise ValueError('Empty or excessive waypoint mission')
        provenance[kind]=dict(path=str(path.resolve()),sha256=sha256(path),original_waypoints=waypoints.tolist(),
            original_radius=saved['media_robot_spec']['radius'],variant_radius=.3,
            original_obstacle_columns=saved['obstacle_columns'],original_obstacles=raw_obs.tolist(),
            changes='Radius0.3 shared by all methods, declared0.25m/0.2m/s ordered stop-and-go arrival. Original start anchor and every subsequent XY waypoint retained; original heading fields recorded without imposing a new heading-arrival constraint.',
            original_status='not_evaluated_under_original_radius')
        for index in range(variants+1):
            x=initial.copy();obs=raw_obs[:,:5].copy();draw_seed=None
            if index:
                draw_seed=seed+index-1;rng=np.random.default_rng(draw_seed)
                x[:2]+=rng.uniform(-.05,.05,2);x[2]+=rng.uniform(-.04,.04);x[3]=rng.uniform(0.,.05)
                obs[:,:2]+=rng.uniform(-.04,.04,(len(obs),2))
            padded_obs=np.zeros((64,5));padded_obs[:len(obs)]=obs;mask=np.arange(64)<len(obs)
            padded_goals=np.broadcast_to(goals[-1],(3,2)).copy();padded_goals[:count]=goals
            points=np.zeros((3,64,2));masks=np.zeros((3,64),bool);ready=np.zeros(3,bool);statuses=[];lengths=[]
            starts=np.vstack((x[:2],goals[:-1]))
            for leg,(start,goal) in enumerate(zip(starts,goals)):
                route=plan_route(start,goal,padded_obs,mask,capacity=64,visibility_batch_nodes=32)
                points[leg]=route.points;masks[leg]=route.mask;ready[leg]=route.status=='ready'
                statuses.append(route.status);lengths.append(route.length)
            points[count:]=points[count-1];masks[count:]=masks[count-1]
            scene_id=f'hero_radius03:{kind}:{index:03d}'
            records.append(dict(scene=dict(scene_id=scene_id,family=f'hero_{kind}',seed=draw_seed,
                initial_state=x.tolist(),goal=goals[-1].tolist(),obstacles=padded_obs.tolist(),obstacle_mask=mask.tolist()),
                waypoint_goals=padded_goals.tolist(),waypoint_count=count,
                waypoint_routes=dict(points=points.tolist(),mask=masks.tolist(),ready=ready.tolist(),statuses=statuses,lengths=lengths),
                changes=dict(initial_state_delta=(x-initial).tolist(),obstacle_xy_delta=(obs[:,:2]-raw_obs[:,:2]).tolist()),
                original_kind=kind,variant_index=index,solvability='unknown',reason='Every prespecified variant retained without outcome resampling'))
    write_json(root/'scenes.json',records)
    write_json(root/'manifest.json',dict(stage='known_ordered_hero_development_inputs',final_test=False,training_use=False,
        source_fingerprint=source_fingerprint(),groups=len(records),variants_per_kind=variants,
        robot=asdict(UnicycleConfig()),waypoint_contract=CONTRACT,provenance=provenance,
        input_scene_sha256=sha256(root/'scenes.json'),noise_scales=[0,1,2],
        interpretation='Geometry-only generation. No simulated outcomes or comparison claim. Broadly trained models must remain unchanged across diverse and hero evaluations.'))
    print(root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True)
    p.add_argument('--original',default='configs/eval/hero_scenes/original');p.add_argument('--variants',type=int,default=32)
    make(**vars(p.parse_args()))
