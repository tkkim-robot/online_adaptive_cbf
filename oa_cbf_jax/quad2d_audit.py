"""Independent NumPy/SciPy flight trajectory and observation audit."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from .quad2d import Quad2DConfig
from .quad2d_control import FlightConfig,flight_config_from_contract
from .quad2d_rollout import NAMES,COLLISION,GOAL,TIMEOUT
from .dataset import sha256,source_fingerprint
from .io import write_json
from .cli import sanitize


def physical_flow(x,u,c):
    total=u.sum()/c.mass
    return np.array([x[3],x[4],x[5],-np.sin(x[2])*total,np.cos(x[2])*total-c.gravity,c.arm/c.inertia*(u[0]-u[1])])


def numpy_graph(x,goal,obs,mask,points,route_mask,noise,config,cursor,previous_control,previous_gain,*,projection_index=None):
    """Independent observed graph, including route memory at visited states."""
    c=config.robot;n=len(obs)+2;x=np.asarray(x,float);goal=np.asarray(goal,float);obs=np.asarray(obs,float);points=np.asarray(points,float)
    valid=route_mask[:-1]&route_mask[1:];vectors=points[1:]-points[:-1];length=np.where(valid,np.linalg.norm(vectors,axis=1),0.)
    cumulative=np.r_[0.,np.cumsum(length)]
    fractions=np.clip(np.sum((x[:2]-points[:-1])*vectors,axis=1)/np.maximum(length**2,1e-12),0.,1.)
    projected=cumulative[:-1]+fractions*length
    eligible=valid&(projected>=cursor-.05)&(projected<=cursor+1.)
    nearest=points[:-1]+fractions[:,None]*vectors
    index=np.argmin(np.where(eligible,np.sum((nearest-x[:2])**2,axis=1),np.inf))
    if projection_index is not None:
        if not eligible[projection_index]:raise ValueError('Ineligible graph route segment')
        index=projection_index
    updated=max(cursor,projected[index]) if eligible.any() else cursor
    desired=min(updated+np.clip(.45+.65*np.linalg.norm(x[3:5]),.35,1.1),cumulative[-1])
    segment=int(np.argmax(valid&(cumulative[1:]>=desired-1e-6)))
    target=points[segment]+np.clip((desired-cumulative[segment])/max(length[segment],1e-12),0,1)*vectors[segment]
    position=np.vstack((x[:2],goal,obs[:,:2]))-x[:2];velocity=np.vstack((x[3:5],np.zeros(2),obs[:,3:5]))-x[3:5]
    radii=np.r_[c.radius,0.,obs[:,2]];clear=np.sqrt(np.sum(position**2,axis=1)+1e-12)-c.radius-radii
    types=np.vstack(([1,0,0],[0,0,1],np.tile([0,1,0],(len(obs),1))))
    ego=np.array([np.sin(x[2]),np.cos(x[2]),x[3]/config.velocity_limit,x[4]/config.velocity_limit,x[5]/config.pitch_rate_limit,
        c.mass,c.inertia/.05,c.arm,c.force_min/10,c.force_max/10,c.gravity/9.81,c.dt,config.pitch_limit,config.pitch_rate_limit,config.velocity_limit])
    previous=np.asarray(previous_control)
    context=np.r_[(goal-x[:2])/5,(target-x[:2])/5,max(cumulative[-1]-updated,0.)/10,(2*previous-c.force_max-c.force_min)/(c.force_max-c.force_min),np.log(previous_gain),noise]
    features=np.column_stack((types,position/5,velocity/config.velocity_limit,radii,clear/3,np.tile(ego,(n,1)),np.tile(context,(n,1))))
    node_mask=np.r_[True,True,mask];features[~node_mask]=0.
    return features,node_mask


def check_observed_graph(features,node_mask,x,goal,obs,mask,points,route_mask,noise,config,cursor,previous_control,previous_gain):
    """Check every feature, accounting only for a near-tied route argmin.

    A fused FP32 distance can choose a different neighboring segment than the
    double-precision reference. Use independent outward operation bounds, then
    require one admissible segment to explain ALL nodes/features together at
    the original tolerance. No generic tolerance relaxation or graph rewrite.
    """
    args=(x,goal,obs,mask,points,route_mask,noise,config,cursor,previous_control,previous_gain)
    expected,expected_mask=numpy_graph(*args)
    np.testing.assert_array_equal(node_mask,expected_mask)
    if np.allclose(features,expected,atol=3e-6,rtol=2e-6):return None
    # A route tie cannot explain any unrelated or masked-node feature change.
    other=np.ones(expected.shape[1],bool);other[26:29]=False
    np.testing.assert_allclose(features[:,other],expected[:,other],atol=3e-6,rtol=2e-6)
    from .route_audit import distance_intervals
    points=np.asarray(points,float);x=np.asarray(x,float)
    vectors=points[1:]-points[:-1];valid=route_mask[:-1]&route_mask[1:]
    length=np.where(valid,np.linalg.norm(vectors,axis=1),0.);cumulative=np.r_[0.,np.cumsum(length)]
    fractions=np.clip(np.sum((x[:2]-points[:-1])*vectors,axis=1)/np.maximum(length**2,1e-12),0.,1.)
    projected=cumulative[:-1]+fractions*length
    eligible=valid&(projected>=cursor-.05)&(projected<=cursor+1.)
    low,high=distance_intervals(x[:2],points)
    candidates=np.flatnonzero(eligible&(low<=np.min(high[eligible],initial=np.inf)))
    for index in candidates:
        candidate,_=numpy_graph(*args,projection_index=int(index))
        if np.allclose(features,candidate,atol=3e-6,rtol=2e-6):return int(index)
    raise ValueError('Graph inconsistent with every numerically admissible route segment')


def initial_numpy_graph(x,goal,obs,mask,points,route_mask,noise,config):
    if not np.allclose(points[0],x[:2],atol=1e-7,rtol=0):raise ValueError('Initial route/observation mismatch')
    return numpy_graph(x,goal,obs,mask,points,route_mask,noise,config,0.,np.full(2,config.robot.mass*config.robot.gravity/2),np.array([2.,2.]))


def check_guidance_trace(data,guidance,config=FlightConfig(),noise=None,goal=None,mask=None,gains=None):
    """Every applied command must have an approved declared guidance profile."""
    import math
    bank=np.asarray([(math.radians(a),s) for s in guidance['speed_scales'] for a in guidance['angles_degrees']]+[(0.,0.)])
    active=data['active'];indices=data['guidance_index'][active]
    if not (data['guidance_approved'][active].all() and (data['guidance_valid_candidates'][active]>0).all() and
            np.isfinite(data['guidance_score'][active]).all() and ((indices>=0)&(indices<len(bank))).all()):
        raise ValueError('Unapproved predictive flight action')
    np.testing.assert_allclose(data['guidance_angle'][active],bank[indices,0],atol=2e-7,rtol=0)
    np.testing.assert_allclose(data['guidance_speed_scale'][active],bank[indices,1],atol=1e-7,rtol=0)
    if 'noise_clearance_weight' in guidance:
        noise=np.asarray(data.get('noise') if noise is None else noise,float)
        if noise.shape!=(7,) or not np.isfinite(noise).all() or np.any(noise<0):raise ValueError('Missing/invalid declared noise for guidance audit')
        target=config.robot.clearance_buffer+1.15*(np.sqrt(2)*(noise[0]+noise[4])+noise[6])
        clearance=data['guidance_predicted_clearance'][active];progress=data['guidance_predicted_progress'][active]
        declared_mask=data.get('obstacle_mask') if mask is None else mask
        empty_scene=declared_mask is not None and not np.asarray(declared_mask,bool).any()
        finite_clearance=np.isfinite(clearance) | (np.isposinf(clearance) & empty_scene)
        if not (finite_clearance.all() and np.isfinite(progress).all() and (clearance>0).all()):raise ValueError('Invalid chosen predictive witness summary')
        if 'terminal_transition_distance' in guidance:
            mission_goal=data.get('mission_goal')
            if mission_goal is None:
                task_goal=data.get('goal') if goal is None else goal
                if task_goal is None:raise ValueError('Terminal guidance audit requires the actual task goal')
                mission_goal=np.broadcast_to(task_goal,(len(active),2))
            chosen_goal=np.asarray(mission_goal,float)[active]
            np.testing.assert_allclose(data['guidance_terminal_goal'][active],chosen_goal,atol=1e-7,rtol=0)
            predicted=np.asarray(data['guidance_terminal_predicted_state'][active],float)
            if not np.isfinite(predicted).all():raise ValueError('Invalid terminal prediction')
            blend=np.clip(1-np.asarray(data['route_remaining'][active],float)/guidance['terminal_transition_distance'],0.,1.)
            np.testing.assert_allclose(data['guidance_terminal_blend'][active],blend,atol=1e-7,rtol=1e-6)
            from .fp32_audit import check_goal_progress
            # The difference of two rounded norms can cancel. Bound those
            # operations independently, then use the validated FP32 intermediate
            # to reconstruct its downstream mix/score at the original tolerance.
            goal_progress=np.asarray(data['guidance_terminal_goal_progress'][active],float)
            check_goal_progress(goal_progress,data['observed_state'][active,:2],predicted[:,:2],np.asarray(chosen_goal,np.float32))
            progress=(1-blend)*progress+blend*goal_progress
            np.testing.assert_allclose(data['guidance_effective_progress'][active],progress,atol=3e-6,rtol=3e-6)
        penalty=guidance['noise_clearance_weight']*(np.maximum(target-clearance,0)/target)**2 if np.any(noise>0) else np.zeros_like(clearance)
        score=progress/(guidance['horizon']*config.robot.dt*config.cruise_speed)
        score+=guidance['clearance_reward']*np.minimum(clearance,.5)/.5
        score-=guidance['deviation_penalty']*(np.abs(bank[indices,0])/np.pi+1-bank[indices,1])+penalty
        np.testing.assert_allclose(data['guidance_clearance_target'][active],target,atol=1e-7,rtol=1e-6)
        np.testing.assert_allclose(data['guidance_clearance_penalty'][active],penalty,atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(data['guidance_score'][active],score,atol=3e-6,rtol=3e-6)
    if 'clearance_guard' in guidance:
        from .quad2d_clearance_guard import check_guard_trace
        check_guard_trace(data,guidance,config,noise,mask,gains)


def check_gain_sources(data,policy,candidates,initial_gain=(4.,4.)):
    """Check declared learned/fallback/held gains without inventing acceptance."""
    if policy['mode']=='fixed':
        np.testing.assert_array_equal(data['gain'],np.broadcast_to(policy['fixed_gain'],data['gain'].shape))
        expected=np.where(data['requery'],2,4)  # Fixed selection or held fixed gain.
        if np.any(data['source'][data['active']]!=expected[data['active']]):raise ValueError('Wrong fixed behavior source')
        return
    previous=np.asarray(initial_gain);candidates=np.asarray(candidates);backups=np.asarray(policy['backup_gains']).reshape(-1,2)
    for k,gain in enumerate(data['gain']):
        if not data['active'][k] and not data['requery'][k]:continue
        source=int(data['source'][k]);stages=data['stages'][k]
        if source==0:
            if policy['mode']!='learned' or not data['requery'][k] or min(stages[:4])<1 or stages[-1]!=0 or np.any(np.diff(stages[:4])>0):raise ValueError('Invalid learned acceptance record')
            pool=np.vstack((candidates,previous))
        elif source==1:
            if not data['requery'][k] or stages[-1]<1:raise ValueError('Invalid fallback record')
            pool=np.vstack((backups,previous))
        elif source==4:
            if data['requery'][k]:raise ValueError('Requery incorrectly labeled held')
            pool=previous[None]
        elif source==3:
            if data['active'][k]:raise ValueError('Applied rejected flight action')
            continue
        else:raise ValueError('Unknown guided adaptive source')
        if not np.any(np.all(np.abs(pool-gain)<1e-6,axis=1)):raise ValueError('Gain outside declared candidate/fallback source')
        if data['active'][k]:previous=gain


def independent_residual(x,u,obs,mask,gains,config):
    c=config.robot;derivative=physical_flow(x,u,c);delta=x[:2]-obs[:,:2];velocity=x[3:5]-obs[:,3:5]
    h=np.sum(delta*delta,axis=1)-(c.radius+c.clearance_buffer+obs[:,2])**2
    hd=2*np.sum(delta*velocity,axis=1)
    hdd=2*np.sum(velocity*velocity,axis=1)+2*np.sum(delta*derivative[3:5],axis=1)
    residual=list((hdd+gains.sum()*hd+gains.prod()*h-c.cbf_margin)[mask]);k=config.envelope_gain
    for sign in [1,-1]:residual.append(-sign*derivative[5]-2*k*sign*x[5]+k*k*(config.pitch_limit-sign*x[2]))
    for sign in [1,-1]:residual.append(-sign*derivative[5]+k*(config.pitch_rate_limit-sign*x[5]))
    for axis in [0,1]:
        for sign in [1,-1]:residual.append(-sign*derivative[3+axis]+k/2*(config.velocity_limit-sign*x[3+axis]))
    residual.extend(u-c.force_min);residual.extend(c.force_max-u)
    return np.asarray(residual),min(h[mask],default=np.inf),min((hd+gains[0]*h)[mask],default=np.inf)


def check_trace(data,summary,observation,obstacles,mask,noise,gains,config,*,command_residual=independent_residual,command_obstacles=None):
    """Replay shared physical/sensor contract with the declared controller rows.

    Discrete MPC supplies its independent discrete-barrier/input validator;
    its dimensionless gains must never be substituted into continuous HOCBFs.
    The default remains the existing continuous OA command/domain audit.
    An explicit command_obstacles sequence must first pass its independent
    observer reconstruction at the caller. Raw sensors are always audited here.
    """
    c=config.robot;x0=data['true_initial_state'].astype(float);truth=data['true_obstacles'].astype(float)
    if config.stationary_obstacles and np.any(truth[mask,3:5]!=0.):
        raise ValueError('Static experiment contains moving physical obstacles')
    if command_obstacles is not None:
        command_obstacles=np.asarray(command_obstacles)
        if command_obstacles.shape!=data['observed_obstacles'].shape or not np.isfinite(command_obstacles).all():
            raise ValueError('Invalid explicit controller observation sequence')
        # This adapter changes velocity only. It cannot silently move obstacles
        # or shrink their radii to make physical constraints appear satisfied.
        np.testing.assert_array_equal(command_obstacles[:,mask,:3],data['observed_obstacles'][:,mask,:3])
    active=np.flatnonzero(data['active']);expected_steps=int(summary['steps'])
    if not np.array_equal(active,np.arange(expected_steps)):raise ValueError('Noncontiguous flight action prefix')
    xs=np.array([noise[0],noise[0],noise[1],noise[2],noise[2],noise[3]])
    os=np.array([noise[4],noise[4],noise[6],noise[5],noise[5]])
    latent=max(float(np.max(np.abs(x0-observation)-xs)),float(np.max((np.abs(truth-obstacles)-os)[mask],initial=-np.inf)))
    if latent>3e-6:raise ValueError('Physical latent error outside declared prior')
    state_error=clear_error=bound_error=violation=sensor_excess=0.
    minimum=float(np.min(np.linalg.norm(x0[:2]-truth[mask,:2],axis=1)-c.radius-truth[mask,2],initial=np.inf))
    for k in range(len(data['active'])):
        before=x0 if k==0 else data['state'][k-1].astype(float)
        seen=data['observed_obstacles'][k].astype(float);sensed=data['observed_state'][k].astype(float)
        physical_obs=truth.copy();physical_obs[:,:2]+=k*c.dt*truth[:,3:5]
        sensor_excess=max(sensor_excess,float(np.max(np.abs(sensed-before)-1.15*xs)),
            float(np.max((np.abs(seen-physical_obs)-1.15*os)[mask],initial=-np.inf)))
        if not data['active'][k]:continue
        u=data['control'][k].astype(float)
        solution=solve_ivp(lambda t,x:physical_flow(x,u,c),(0,c.dt),before,rtol=1e-11,atol=1e-12,dense_output=True)
        if not solution.success:raise ValueError('Independent flight integrator failed')
        times=np.linspace(0,c.dt,101);states=solution.sol(times).T
        difference=states[-1]-data['state'][k];difference[2]=np.arctan2(np.sin(difference[2]),np.cos(difference[2]))
        state_error=max(state_error,float(np.max(np.abs(difference))))
        if mask.any():
            centers=truth[None,:,:2]+(k*c.dt+times[:,None,None])*truth[None,:,3:5]
            clear=float(np.min((np.linalg.norm(states[:,None,:2]-centers,axis=-1)-c.radius-truth[None,:,2])[:,mask]))
            clear_error=max(clear_error,abs(clear-float(data['clearance'][k])));minimum=min(minimum,clear)
        bounds=np.maximum(np.max(np.abs(states[:,3:5]),axis=1)-config.velocity_limit,
            np.maximum(np.abs(states[:,2])-config.pitch_limit,np.abs(states[:,5])-config.pitch_rate_limit))
        bound_error=max(bound_error,abs(float(bounds.max()-data['state_bound_violation'][k])))
        recorded_gains=np.asarray(gains,float)
        control_seen=seen if command_obstacles is None else command_obstacles[k].astype(float)
        residual,h,psi=command_residual(sensed,u,control_seen,mask,recorded_gains[k] if recorded_gains.ndim==2 else recorded_gains,config)
        violation=max(violation,float(-residual.min()),-h,-psi)
    saved=float(summary['min_clearance']);minimum_error=abs(saved-minimum) if np.isfinite(saved) or np.isfinite(minimum) else 0.
    # The plant terminates at clearance <= 0. A numerical comparison tolerance
    # is not an additional physical collision radius: a positive near miss must
    # not become a collision, nor may a negative clearance pass as noncollision.
    status=int(summary['status']);collision_consistent=(minimum<=0.)==(status==COLLISION)
    passed=bool(state_error<1e-5 and clear_error<1e-4 and minimum_error<1e-4 and bound_error<1e-4 and violation<=2e-5 and sensor_excess<2e-5 and collision_consistent)
    return dict(audit_passed=passed,steps=expected_steps,status=NAMES[status],state_error=state_error,clearance_error=clear_error,
        minimum_clearance_error=minimum_error,replayed_minimum_clearance=minimum,
        bound_recording_error=bound_error,max_cbf_input_violation=violation,sensor_bound_excess=sensor_excess,collision_consistent=collision_consistent)


def audit(dataset):
    root=Path(dataset);manifest=json.loads((root/'manifest.json').read_text());config=flight_config_from_contract(manifest['config'])
    source=Path(manifest['source'])
    if sha256(source/'manifest.json')!=manifest['source_manifest_sha256']:raise ValueError('Source contract changed')
    source_manifest=json.loads((source/'manifest.json').read_text())
    if flight_config_from_contract(source_manifest['config'])!=config:raise ValueError('Source physical contract changed')
    if sha256(source/'scenes.json')!=source_manifest['scenes_sha256']:raise ValueError('Source observations changed')
    relabeled=manifest['schema']=='oa_cbf_quad2d_initial_hurdle_v2'
    guided=manifest['schema']=='oa_cbf_quad2d_guided_hurdle_v1'
    conditional=relabeled or guided
    terminal=guided and 'terminal_transition_distance' in manifest['controller'].get('predictive_guidance',{})
    if terminal:
        from .quad2d_task_targets import contract,check_physical_target_inputs,values
        target_contract=manifest['controller'].get('performance_target',{})
        if target_contract.get('kind') not in ('route','terminal_task') or target_contract!=contract(target_contract['kind']) or manifest['targets'][1]!=target_contract['target']:raise ValueError('Unbound terminal performance target')
    retarget=manifest.get('performance_retarget_source')
    if retarget:
        retarget=Path(retarget)
        if not terminal or sha256(retarget/'manifest.json')!=manifest['performance_retarget_manifest_sha256'] or sha256(retarget/'index.json')!=manifest['performance_retarget_index_sha256']:raise ValueError('Changed performance retarget source')
        retarget_index={e['file']:e for e in json.loads((retarget/'index.json').read_text())}
    frozen_gains=None
    if 'gain_bank_augmentation' in manifest and 'frozen_gain_bank' not in manifest:raise ValueError('Missing augmented gain-bank source')
    if 'frozen_gain_bank' in manifest:
        from .quad2d_data import load_gain_bank,augment_gain_bank
        augmented=manifest.get('gain_bank_augmentation')
        bank,provenance=load_gain_bank(manifest['frozen_gain_bank']['dataset'],config,None if augmented is not None else manifest['queries'])
        if provenance!=manifest['frozen_gain_bank']:raise ValueError('Frozen label gain-bank provenance changed')
        if augmented is not None:
            bank,expected=augment_gain_bank(bank,manifest['queries'],augmented['seed'],augmented.get('upper',8.))
            if augmented!=expected:raise ValueError('Augmented label gain-bank provenance changed')
            if manifest['gain_domain']!=dict(lower=.5,upper=augmented.get('upper',8.)):
                raise ValueError('Label gain domain differs from explicit augmentation')
        frozen_gains=np.repeat(bank,manifest['replicas'],axis=0)
    source_rows={r['group_id']:r for r in json.loads((source/'scenes.json').read_text())} if guided else {}
    if relabeled:
        original=Path(manifest['relabel_source'])
        if sha256(original/'manifest.json')!=manifest['relabel_manifest_sha256'] or sha256(original/'index.json')!=manifest['relabel_index_sha256']:
            raise ValueError('Relabel provenance changed')
        original_index={e['file']:e for e in json.loads((original/'index.json').read_text())}
    rows=[];graph_route_rounding=[]
    for entry in json.loads((root/'index.json').read_text()):
        if sha256(root/entry['file'])!=entry['sha256'] or sha256(root/entry['trace_file'])!=entry['trace_sha256']:raise ValueError('Flight trace/shard hash mismatch')
        with np.load(root/entry['file']) as f:shard={k:f[k] for k in f.files}
        if terminal:check_physical_target_inputs(shard,manifest)
        if retarget:
            old=retarget_index[entry['file']]
            if sha256(retarget/entry['file'])!=old['sha256']:raise ValueError('Changed retarget raw shard')
            with np.load(retarget/entry['file']) as raw:
                if set(raw.files)!=set(shard):raise ValueError('Changed retarget fields')
                for key in set(shard)-{'target'}:np.testing.assert_array_equal(shard[key],raw[key])
                np.testing.assert_array_equal(shard['target'][...,0],raw['target'][...,0])
        if frozen_gains is not None:np.testing.assert_array_equal(shard['gains'],np.broadcast_to(frozen_gains,shard['gains'].shape))
        if relabeled:
            old=original_index[entry['file']]
            if sha256(original/entry['file'])!=old['sha256']:raise ValueError('Raw relabel source changed')
            with np.load(original/entry['file']) as raw:
                if set(raw.files)!=set(shard):raise ValueError('Changed raw branch fields')
                for key in set(shard)-{'target','target_mask'}:
                    np.testing.assert_array_equal(shard[key],raw[key])
        for i in range(len(shard['initial_state'])):
            args=(shard['initial_state'][i],shard['goal'][i],shard['obstacles'][i],shard['obstacle_mask'][i],shard['points'][i],shard['route_mask'][i],shard['noise'][i],config)
            if guided:
                r=source_rows[str(shard['group_id'][i])]
                for key in ['initial_state','goal','obstacles','obstacle_mask','noise']:
                    np.testing.assert_array_equal(shard[key][i],np.asarray(r[key],dtype=shard[key].dtype))
                np.testing.assert_array_equal(shard['points'][i],np.asarray(r['route']['points'],np.float32))
                np.testing.assert_array_equal(shard['route_mask'][i],r['route']['mask'])
                if str(shard['partition'][i])!=r['partition'] or int(shard['seeds'][i])!=r['seed'] or bool(shard['ready'][i])!=(r['route']['status']=='ready'):
                    raise ValueError('Observation parent binding changed')
                for key,default in [('cursor',0.),('previous_control',[config.robot.mass*config.robot.gravity/2]*2),('previous_gain',[2.,2.])]:
                    np.testing.assert_array_equal(shard[key][i],np.asarray(r.get(key,default),np.float32))
                rounded=check_observed_graph(shard['features'][i],shard['node_mask'][i],*args,shard['cursor'][i],shard['previous_control'][i],shard['previous_gain'][i])
                if rounded is not None:graph_route_rounding.append(dict(group_id=str(shard['group_id'][i]),projection_index=rounded))
            else:
                features,node_mask=initial_numpy_graph(*args)
                np.testing.assert_allclose(shard['features'][i],features,atol=3e-6,rtol=2e-6)
                np.testing.assert_array_equal(shard['node_mask'][i],node_mask)
        traced=set()
        for trace_entry in [dict(file=entry['trace_file'],sha256=entry['trace_sha256'])]+entry.get('event_traces',[]):
            if sha256(root/trace_entry['file'])!=trace_entry['sha256']:raise ValueError('Event trace changed')
            with np.load(root/trace_entry['file']) as f:data={k:f[k] for k in f.files}
            parent=int(data['parent']);candidate=int(data['candidate']);traced.add((parent,candidate))
            if str(data['group_id'])!=str(shard['group_id'][parent]):raise ValueError('Flight trace lineage mismatch')
            summary={k:shard[k][parent,candidate] for k in ['status','steps','min_clearance']}
            if int(data['expected_steps'])!=int(summary['steps']) or int(data['final_status'])!=int(summary['status']):raise ValueError('Flight trace/label status mismatch')
            row=check_trace(data,summary,shard['initial_state'][parent],shard['obstacles'][parent],shard['obstacle_mask'][parent],shard['noise'][parent],shard['gains'][parent,candidate],config)
            if terminal:
                np.testing.assert_array_equal(shard['physical_initial_state'][parent,candidate],data['true_initial_state'])
                np.testing.assert_allclose(shard['final_state'][parent,candidate],data['state'][-1],atol=1e-5,rtol=0)
            if guided:check_guidance_trace(data,manifest['controller']['predictive_guidance'],config,shard['noise'][parent],shard['goal'][parent],shard['obstacle_mask'][parent],shard['gains'][parent,candidate])
            rows.append(dict(group_id=str(data['group_id']),candidate=candidate,**row))
        if not {tuple(pair) for pair in np.argwhere(np.isin(shard['status'],[COLLISION,8]))}<=traced:raise ValueError('Unaudited physical collision/bound branch')
        # Check target/event semantics on every row, not only the traced branch.
        adverse=~np.isin(shard['status'],[GOAL,TIMEOUT]);collision=shard['status']==COLLISION
        physical_risk=-np.minimum(shard['min_clearance'],.6)/.3
        risk=np.where(~adverse|collision,physical_risk,0.) if conditional else np.where(adverse,1.,physical_risk)
        target=np.stack((risk,
            shard['route_progress']/(manifest['horizon_steps']*config.robot.dt*config.cruise_speed)),axis=-1)
        if terminal:target[...,1]=values(shard,manifest)
        np.testing.assert_allclose(shard['target'],target,atol=1e-6)
        np.testing.assert_array_equal(shard['target_mask'],np.stack((~adverse|collision if conditional else np.ones_like(adverse),np.ones_like(adverse)),axis=-1))
        np.testing.assert_array_equal(shard['events'],np.stack((collision,adverse),axis=-1))
        np.testing.assert_array_equal(shard['event_mask'],np.stack((~adverse|collision,np.ones_like(adverse)),axis=-1))
    passed=bool(rows) and all(r['audit_passed'] for r in rows)
    report=dict(audit_passed=passed,manifest_sha256=sha256(root/'manifest.json'),index_sha256=sha256(root/'index.json'),
        auditor_source_fingerprint=source_fingerprint(),graph_route_rounding=graph_route_rounding,
        audited_parents=len({r['group_id'] for r in rows}),audited_branches=len(rows),replayed_steps=sum(r['steps'] for r in rows),rows=rows,
        all_collision_bound_branches_audited=True,all_initial_graph_features_independently_checked=not manifest.get('visited_observations',False),
        all_observed_graph_features_independently_checked=True,all_guidance_approvals_checked=guided,
        all_physical_task_target_inputs_checked=terminal,
        scope='Predetermined parent/gain traces plus EVERYcollision/bound branch, all nonlinear6state trajectories with SciPy, all physical moving obstacles, independent NumPy CBF/envelope/input rows and raw sensor bounds. All shard graph/context/target/event bindings checked.',
        limitation='Flight label implementation audit, not certified continuous-time safety, feasibility of unobserved continuation, learned policy performance or generalization evidence.')
    write_json(root/'independent_replay.json',sanitize(report));print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    if not passed:raise ValueError('Independent flight replay failed')
    contract=json.loads((root/'contract_audit.json').read_text())
    if not contract['contract_valid']:raise ValueError('Dataset contract invalid')
    write_json(root/'complete.json',dict(status='completed',audit_passed=True,manifest_sha256=sha256(root/'manifest.json'),index_sha256=sha256(root/'index.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);audit(p.parse_args().dataset)
