"""Independent NumPy native flight equations and SciPy optimality checks."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.special import expit
from .barriernet_audit import check_optimality
from .quad2d_odqp import nominal as numpy_nominal
from .quad2d_control import FlightConfig,flight_config_from_contract
from dataclasses import asdict
from .dataset import sha256
from .io import write_json


def numpy_features(x,goal,obs,mask,radius=.3):
    valid=obs[np.asarray(mask,bool)]
    order=np.argsort(np.linalg.norm(valid[:,:2]-x[:2],axis=1)-valid[:,2]-radius,kind='stable')[:5]
    selected=np.pad(valid[order],((0,0),(0,7-obs.shape[-1])))
    selected=np.vstack((selected,np.tile([100.,100.,0.,0.,0.,0.,0.],(5-len(selected),1))))
    delta=selected[:,:2]-x[:2]
    angle=(np.arctan2(delta[:,1],delta[:,0])-x[2]+np.pi)%(2*np.pi)-np.pi
    z=np.column_stack((delta,angle,np.linalg.norm(delta,axis=1)-selected[:,2]-radius,np.full(5,np.hypot(x[3],x[4])))).reshape(25)
    return z,np.r_[x,goal,selected.ravel()]


def numpy_constraints(ctx,p,radius=.3,lower=1.,upper=10.):
    x=ctx[:6];obs=ctx[8:].reshape(5,7);rows=[];rhs=[]
    for obstacle,gain in zip(obs,p):
        dx,dz=x[:2]-obstacle[:2]
        barrier=dx*dx+dz*dz-1.01*(obstacle[2]+radius)**2
        derivative=2*(dx*x[3]+dz*x[4])
        drift=2*(x[3]*x[3]+x[4]*x[4])-2*9.81*dz
        coefficient=2*(-dx*np.sin(x[2])+dz*np.cos(x[2]))
        rows.append([-coefficient,-coefficient])
        rhs.append(drift+(gain[0]+gain[1])*derivative+gain[0]*gain[1]*barrier)
    return np.vstack((rows,np.eye(2),-np.eye(2))),np.r_[rhs,[upper]*2,[-lower]*2]


def numpy_network(weights,mean,std,z,ctx,reference):
    def linear(x,name):return x@weights[name]['kernel']+weights[name]['bias']
    encoded=np.maximum(linear(((z-mean)/std).reshape(5,5),'obs_fc1'),0.)
    encoded=np.maximum(linear(encoded,'obs_fc2'),0.)
    gains=4*expit(linear(encoded,'fc_p'))
    hidden=np.maximum(linear(np.r_[encoded.mean(0),ctx[:8],reference],'u_fc1'),0.)
    return reference+linear(hidden,'u_out'),gains


def feasibility(A,b):
    result=linprog(np.zeros(2),A_ub=A,b_ub=b,bounds=[(None,None)]*2,method='highs')
    if result.status not in (0,2):raise ValueError('Independent native QP feasibility unresolved')
    return result.status==0


def check_deployment(result,ctx,p,u_nom,config=FlightConfig(),*,classify=False):
    """Raw native optimum, wrapper clipping and final FP32 acceptance separately."""
    c=config.robot;A,b=numpy_constraints(ctx,p,c.radius)
    shared_A,shared_b=numpy_constraints(ctx,p,c.radius,c.force_min,c.force_max)
    raw=np.asarray(result['raw_control']);candidate=np.asarray(result['candidate']);solver_valid=bool(result['solver_valid'])
    clipped=np.clip(raw,c.force_min,c.force_max)
    np.testing.assert_array_equal(candidate,clipped)
    stationarity=0.;raw_violation=np.inf
    if solver_valid:
        if not np.isfinite(raw).all():raise ValueError('Nonfinite successful native solve')
        raw_violation,stationarity=check_optimality(raw,u_nom,A,b,1+1e-6)
        if raw_violation>c.qp_tolerance+1e-9 or stationarity>2e-6:raise ValueError(f'Invalid native raw optimum {raw_violation}, {stationarity}')
        np.testing.assert_allclose(result['raw_violation'],raw_violation,atol=1e-9,rtol=1e-10)
    elif np.isfinite(raw).any():raise ValueError('Missing native action must retain NaNs')
    violation=float(np.max(shared_A@candidate-shared_b)) if np.isfinite(candidate).all() else np.inf
    valid=solver_valid and np.isfinite(candidate).all() and violation<=c.qp_tolerance
    if valid!=bool(result['valid']):raise ValueError('Native post-clip acceptance changed')
    if np.isfinite(violation):np.testing.assert_allclose(result['violation'],violation,atol=1e-9,rtol=1e-10)
    stored_violation=float(np.max(shared_A@candidate.astype(np.float32).astype(float)-shared_b)) if np.isfinite(candidate).all() else np.inf
    accepted=valid and stored_violation<=c.qp_tolerance
    return dict(accepted=bool(accepted),raw_violation=float(raw_violation),stored_violation=stored_violation,stationarity=stationarity,
        independently_feasible=feasibility(A,b) if classify and not solver_valid else bool(solver_valid),
        clipped=bool(solver_valid and not np.array_equal(raw,candidate)))


def audit(dataset,samples=512):
    root=Path(dataset);m=json.loads((root/'manifest.json').read_text());source=Path(m['source'])
    if m['schema']!='barriernet_quad2d_training_v1' or sha256(root/'data.npz')!=m['data_sha256']:raise ValueError('Changed flight teacher data')
    if sha256(source/'manifest.json')!=m['source_manifest_sha256'] or sha256(source/'index.json')!=m['source_index_sha256']:raise ValueError('Changed training lineage')
    sm=json.loads((source/'manifest.json').read_text());c=flight_config_from_contract(sm['config'])
    if 'source_config' in m and m['source_config']!=asdict(c):raise ValueError('Changed flight physical contract')
    with np.load(root/'data.npz') as f:d=dict(f)
    groups=d['group_id'];split=d['split'];valid=d['valid'];sets=[set(groups[split==i]) for i in range(3)]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('Parent leakage')
    if len(set(zip(groups.tolist(),d['tick'].tolist())))!=len(groups):raise ValueError('Duplicate physical parent/tick')
    for exclusion in m['held_exclusions']:
        path=Path(exclusion['source'])/'scenes.json'
        if sha256(path)!=exclusion['scenes_sha256'] or set(groups)&{p['group_id'] for p in json.loads(path.read_text())}:raise ValueError('Held parent leakage or changed registry')
    np.testing.assert_array_equal(d['u_ref'],d['label'])
    if not all(np.isfinite(d[k]).all() for k in ['z','ctx','label']) or np.any(d['label'][~valid]!=0.):raise ValueError('Invalid label storage')
    chosen=[];rng=np.random.default_rng(903)
    for s in range(3):
        for v in (False,True):
            pool=np.flatnonzero((split==s)&(valid==v));chosen.extend(rng.choice(pool,min(len(pool),samples//6),replace=False).tolist())
    pool=np.setdiff1d(np.arange(len(groups)),chosen);chosen=sorted(chosen+rng.choice(pool,min(len(pool),samples-len(chosen)),replace=False).tolist())
    by_group={};entries={e['group_id']:e for e in json.loads((source/'index.json').read_text())}
    for row in chosen:by_group.setdefault(str(groups[row]),[]).append(row)
    details=[];maximum_feature=maximum_primal=maximum_kkt=0.;rejected=feasible_rejections=0
    for group,rows in by_group.items():
        e=entries[group];path=source/e['file']
        if sha256(path)!=e['sha256']:raise ValueError('Changed raw pre-action observations')
        with np.load(path) as f:
            states=f['observed_state'];obs=f['observed_obstacles'];targets=f['route_target'];mask=f['obstacle_mask']
            if str(f['group_id'])!=group:raise ValueError('Mismatched acquired parent')
            for row in rows:
                tick=int(d['tick'][row])
                if not 0<=tick<max(1,e['steps']):raise ValueError('Unexecuted future feature')
                z,ctx=numpy_features(states[tick].astype(float),targets[tick].astype(float),obs[tick].astype(float),mask,m['radius'])
                error=max(np.max(np.abs(z-d['z'][row])),np.max(np.abs(ctx-d['ctx'][row])))
                if error>1e-10:raise ValueError('Raw feature/source mismatch')
                maximum_feature=max(maximum_feature,float(error));ref=numpy_nominal(ctx[:6],ctx[6:8],c)
                np.testing.assert_allclose(ref,d['original_nominal'][row],atol=1e-12,rtol=1e-12)
                A,b=numpy_constraints(ctx,np.full((5,2),1.5),m['radius'],c.robot.force_min,c.robot.force_max)
                raw=d['solver_raw_control'][row];candidate=d['expert_candidate'][row]
                np.testing.assert_array_equal(candidate,np.clip(raw,c.robot.force_min,c.robot.force_max))
                feasible=bool(valid[row])
                if d['solver_feasible'][row]:
                    primal,kkt=check_optimality(raw,ref,A,b)
                    if primal>1e-5+1e-9 or kkt>2e-6:raise ValueError('Incorrect teacher raw optimum')
                    maximum_primal=max(maximum_primal,primal);maximum_kkt=max(maximum_kkt,kkt)
                elif np.isfinite(raw).any():raise ValueError('Invalid raw teacher storage')
                residual=np.max(A@candidate-b) if np.isfinite(candidate).all() else np.inf
                accepted=bool(d['solver_feasible'][row] and residual<=1e-5)
                if accepted!=feasible:raise ValueError('Incorrect teacher censoring')
                if accepted:np.testing.assert_array_equal(candidate,d['label'][row])
                else:rejected+=1;feasible=feasibility(A,b);feasible_rejections+=int(feasible)
                details.append(dict(row=row,group_id=group,tick=tick,valid=bool(valid[row]),independently_feasible=bool(feasible)))
    report=dict(audit_passed=True,manifest_sha256=sha256(root/'manifest.json'),samples=len(chosen),groups=len(by_group),invalid_labels_checked=rejected,feasible_rejections=feasible_rejections,
        max_feature_error=maximum_feature,max_primal_violation=maximum_primal,max_stationarity_error=maximum_kkt,all_rows_group_disjoint=True,all_rows_unique_parent_tick=True,all_held_cohorts_excluded=True,
        independent='NumPy source reconstruction, native nominal/HOCBF, SciPy NNLS KKT and HiGHS feasibility. Sampled solver checks; complete dataset parent/split/hash checks.',samples_detail=details)
    write_json(root/'independent_audit.json',report);print(json.dumps({k:v for k,v in report.items() if k!='samples_detail'}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--samples',type=int,default=512);a=p.parse_args();audit(a.dataset,a.samples)
