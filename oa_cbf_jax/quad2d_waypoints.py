"""Prespecified flight hero inputs and independent mandatory-waypoint audit."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .quad2d_control import FlightConfig
from .routing import plan_route
from .route_audit import check_transition
from .dataset import sha256,source_fingerprint
from .io import write_json

CONTRACT=dict(name='quad2d_ordered_stop_and_go_v1',capacity=3,
    handoff='Only the next waypoint advances. Pre-action sensed position/speed norms plus sqrt(2)*1.15*their declared ranges and sensed absolute pitch/rate plus1.15*their ranges must satisfy the existing flight arrival limits.',
    continuation='One continuous physical episode and sensor stream. Preserve all six physical states, sensor biases/innovations, gains, previous action and solver history. Reset only the route cursor on handoff and force a new adaptive decision.',
    final='After all intermediate handoffs, actual physical final-goal arrival under the existing position/speed/pitch/rate thresholds. Collision and physical-bound failures take priority.',
    routes='One static observed visibility route per leg. No geometric feasibility certificate. Every original obstacle and mandatory XY destination retained.',
    radius='All methods share trained radius0.3m and arm0.3m. Explicit variant of media radii0.24/0.20m; original-radius evaluations remain open.',
    calibration='Single-current-goal prediction/calibration reused under the unchanged local controller. This does not certify whole adaptive missions or future waypoint switches.')


def prepare(output,variants=31,original='configs/eval/hero_scenes/original'):
    if isinstance(variants,bool) or not isinstance(variants,int) or variants<0:raise ValueError('Invalid variant count')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);config=FlightConfig();rows=[];provenance={}
    for kind,base_seed in [('narrow',641100),('wide',641400)]:
        p=Path(original)/f'quad2d_{kind}.json';m=json.loads(p.read_text());initial=np.asarray(m['initial_state'],np.float32)
        original_obs=np.asarray(m['obstacles'],np.float32)[:,:5];waypoints=np.asarray(m['waypoints'],np.float32)
        if len(original_obs)>64 or not np.array_equal(waypoints[0,:2],initial[:2]) or not 1<=len(waypoints)-1<=3:raise ValueError('Unsupported original task; never omit obstacles or goals')
        goals=waypoints[1:,:2];count=len(goals)
        provenance[kind]=dict(path=str(p.resolve()),sha256=sha256(p),original_radius=m['media_robot_spec']['radius'],evaluated_radius=config.robot.radius,
            original_waypoints=m['waypoints'],obstacles=m['obstacle_count'],environment=m['environment'],
            modification='Shared radius0.3 and ordered stop-and-go arrival; original canonical XY geometry/start/waypoints retained. Environment dimensions describe the original drawing; no new physical walls introduced.')
        for variant in range(variants+1):
            seed=base_seed+variant;rng=np.random.default_rng(seed);x=initial.copy();obs=original_obs.copy()
            if variant:
                x[:2]+=rng.uniform(-.05,.05,2);x[2]+=rng.uniform(-.04,.04);x[3:5]=rng.uniform(-.05,.05,2);x[5]=rng.uniform(-.03,.03)
                obs[:,:2]+=rng.uniform(-.04,.04,(len(obs),2))
            padded=np.zeros((64,5),np.float32);padded[:len(obs)]=obs;mask=np.arange(64)<len(obs)
            padded_goals=np.broadcast_to(goals[-1],(3,2)).copy();padded_goals[:count]=goals
            points=np.zeros((3,64,2),np.float32);rm=np.zeros((3,64),bool);ready=np.zeros(3,bool);statuses=[]
            starts=np.vstack((x[:2],goals[:-1]))
            for leg,(start,goal) in enumerate(zip(starts,goals)):
                route=plan_route(start,goal,padded,mask,config.robot,capacity=64,visibility_batch_nodes=32)
                points[leg]=route.points;rm[leg]=route.mask;ready[leg]=route.status=='ready';statuses.append(route.status)
            points[count:]=points[count-1];rm[count:]=rm[count-1]
            for noise_scale in [0,1,2]:
                noise=noise_scale*np.array([.015,.01,.015,.015,.02,.02,.008],np.float32)
                rows.append(dict(group_id=f'quad2d_hero_radius03:{kind}:{variant:03d}:noise{noise_scale}',family='hero_'+kind,seed=seed,partition='showcase_only',
                    initial_state=x.tolist(),goal=goals[-1].tolist(),obstacles=padded.tolist(),obstacle_mask=mask.tolist(),noise=noise.tolist(),
                    waypoint_goals=padded_goals.tolist(),waypoint_count=count,waypoint_routes=dict(points=points.tolist(),mask=rm.tolist(),ready=ready.tolist(),statuses=statuses),
                    original_kind=kind,variant_index=variant,noise_scale=noise_scale,solvability='unknown',
                    changes=dict(initial_state_delta=(x-initial).tolist(),obstacle_xy_delta=(obs[:,:2]-original_obs[:,:2]).tolist()),
                    reason='Every prespecified variant retained; no outcome-based resampling or training use.'))
    write_json(root/'scenes.json',rows);write_json(root/'manifest.json',dict(schema='oa_cbf_quad2d_ordered_hero_inputs_v1',stage='known_flight_hero_development',training_use=False,final_test=False,
        config=asdict(config),source_fingerprint=source_fingerprint(),groups=len(rows),variants_per_kind=variants,noise_scales=[0,1,2],scenes_sha256=sha256(root/'scenes.json'),
        waypoint_contract=CONTRACT,provenance=provenance,limitations='Known hero neighborhoods, not independent diverse or final tests. Shared trained radius variant, original geometry retained for canonical cases; original media radius not evaluated.'))
    print(json.dumps(dict(stage='hero_inputs_prepared',parents=len(rows))),flush=True)


def validate_parent(parent):
    total=parent['waypoint_count'];goals=np.asarray(parent['waypoint_goals']);r=parent['waypoint_routes']
    if isinstance(total,bool) or not isinstance(total,int) or not 1<=total<=3 or goals.shape!=(3,2) or np.shape(r['points'])!=(3,64,2) or np.shape(r['mask'])!=(3,64) or np.shape(r['ready'])!=(3,):raise ValueError('Invalid ordered flight task')
    if not np.array_equal(np.asarray(parent['goal'],np.float32),goals[total-1].astype(np.float32)):raise ValueError('Final waypoint mismatch')


def numpy_arrived(x,goal,config,noise=None):
    e=np.zeros(4) if noise is None else 1.15*np.asarray(noise,float)[:4]
    return bool(np.linalg.norm(x[:2]-goal)+np.sqrt(2)*e[0]<=config.goal_tolerance
        and np.linalg.norm(x[3:5])+np.sqrt(2)*e[2]<=config.terminal_speed
        and abs(x[2])+e[1]<=config.terminal_pitch and abs(x[5])+e[3]<=config.terminal_pitch_rate)


def check_episode(parent,data,row,config=FlightConfig(),adaptive=True):
    """Independent sensor-only handoffs, physical visits and memory continuity."""
    validate_parent(parent);goals=np.asarray(parent['waypoint_goals'],float);total=parent['waypoint_count'];ready=np.asarray(parent['waypoint_routes']['ready'],bool)
    routes=np.asarray(parent['waypoint_routes']['points'],np.float32);rm=np.asarray(parent['waypoint_routes']['mask'],bool)
    initial=data['true_initial_state'].astype(float);truth=data['true_obstacles'].astype(float);mask=np.asarray(parent['obstacle_mask'],bool);noise=np.asarray(parent['noise'],float)
    bound=max(np.max(np.abs(initial[3:5]))-config.velocity_limit,abs(initial[2])-config.pitch_limit,abs(initial[5])-config.pitch_rate_limit)>config.robot.qp_tolerance
    collision=np.min(np.linalg.norm(initial[:2]-truth[mask,:2],axis=1)-config.robot.radius-truth[mask,2],initial=np.inf)<=0
    status=7 if not ready[0] else 2 if collision else 8 if bound else 1 if total==1 and numpy_arrived(initial,goals[0],config) else 0
    leg=0;cursor=np.float32(0);handoffs=[];ambiguous=0
    previous=np.full(2,config.robot.mass*config.robot.gravity/2,np.float32) if adaptive else np.zeros(2,np.float32);gain=np.array([4.,4.],np.float32)
    for k,active in enumerate(data['active']):
        sensed=np.asarray(data['observed_state'][k],float);expected=status==0 and leg<total-1 and numpy_arrived(sensed,goals[leg],config,noise)
        if bool(data['waypoint_handoff'][k])!=expected:raise ValueError('Incorrect observed waypoint handoff')
        if expected:
            before=initial if k==0 else data['state'][k-1].astype(float)
            if not numpy_arrived(before,goals[leg],config):raise ValueError('False physical intermediate arrival')
            leg+=1;cursor=np.float32(0);handoffs.append(k)
            if ready[leg] and adaptive and not data['requery'][k]:raise ValueError('Missing mandatory handoff requery')
        if int(data['waypoint_index'][k])!=leg:raise ValueError('Skipped or reordered waypoint')
        np.testing.assert_array_equal(data['mission_goal'][k],goals[leg].astype(np.float32))
        np.testing.assert_array_equal(data['mission_previous_control'][k],previous)
        np.testing.assert_allclose(data['mission_route_cursor_before'][k],cursor,atol=1e-7,rtol=0)
        if adaptive:np.testing.assert_array_equal(data['mission_previous_gain'][k],gain)
        if active and (status!=0 or not ready[leg]):raise ValueError('Applied action after terminal/unplanned leg')
        if not ready[leg] and status==0 and int(data['status'][k])!=7:raise ValueError('Missing next-leg planner failure')
        ambiguous+=int(check_transition(sensed,routes[leg],rm[leg],cursor,data['route_target'][k],data['route_progress'][k],active))
        status=int(data['status'][k])
        if int(data['waypoints_visited'][k])!=leg+int(status==1):raise ValueError('Wrong waypoint count')
        if status==1 and (leg!=total-1 or not numpy_arrived(np.asarray(data['state'][k],float),goals[leg],config)):raise ValueError('False final waypoint arrival')
        if active:
            previous=data['control'][k];cursor=data['route_progress'][k]
            if adaptive:gain=data['gain'][k]
    if (row['waypoint_index']!=leg or row['waypoints_visited']!=leg+int(row['status_code']==1) or row['required_waypoints']!=total or row['waypoint_handoffs']!=len(handoffs)):raise ValueError('Wrong mission result')
    if row['status_code']==1 and (leg!=total-1 or status!=1):raise ValueError('Bypassed intermediate goals')
    return dict(waypoint_audit_passed=True,required_waypoints=total,waypoints_visited=row['waypoints_visited'],handoff_ticks=handoffs,roundoff_ambiguous_route_decisions=ambiguous)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--variants',type=int,default=31);p.add_argument('--original',default='configs/eval/hero_scenes/original');prepare(**vars(p.parse_args()))
