"""Independent NumPy/SciPy lineage, feature and default teacher checks.

No JAX BarrierNet feature or constraint implementation is used by this audit.
Feasible labels require primal feasibility and nonnegative KKT multipliers.
Rejected labels are classified by independent feasibility and their recorded
residuals; solver rejection is never assumed to prove infeasibility.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog,nnls
from .dataset import sha256
from .io import write_json


def numpy_features(x,goal,obs,mask,radius):
    valid=obs[np.asarray(mask,bool)]
    order=np.argsort(np.linalg.norm(valid[:,:2]-x[:2],axis=1)-valid[:,2]-radius,kind='stable')[:5]
    selected=np.pad(valid[order],((0,0),(0,7-obs.shape[-1])))
    selected=np.vstack((selected,np.tile([100.,100.,0.,0.,0.,0.,0.],(5-len(selected),1))))
    delta=selected[:,:2]-x[:2]
    angle=(np.arctan2(delta[:,1],delta[:,0])-x[2]+np.pi)%(2*np.pi)-np.pi
    z=np.column_stack((delta,angle,np.linalg.norm(delta,axis=1)-selected[:,2]-radius,np.full(5,x[3]))).reshape(25)
    return z,np.r_[x,goal,selected.ravel()]


def numpy_constraints(ctx,p,radius):
    x=ctx[:4];obs=ctx[6:].reshape(5,7);rows=[];rhs=[]
    for obstacle,gain in zip(obs,p):
        dx,dy=x[:2]-obstacle[:2];c=np.cos(x[2]);s=np.sin(x[2]);v=x[3]
        barrier=dx*dx+dy*dy-1.01*(obstacle[2]+radius)**2
        along=dx*c+dy*s
        rows.append([-2*along,-2*v*(-dx*s+dy*c)])
        rhs.append(2*v*v+(gain[0]+gain[1])*2*v*along+gain[0]*gain[1]*barrier)
    return np.vstack((rows,np.eye(2),-np.eye(2))),np.r_[rhs,[.5]*4]


def numpy_nominal(x,goal):
    dx,dy=goal-x[:2];distance=max(np.hypot(dx,dy)-.05,0.)
    angle=(np.arctan2(dy,dx)-x[2]+np.pi)%(2*np.pi)-np.pi
    speed=0. if abs(angle)>np.pi/2 else min(distance*np.cos(angle),1.)
    return np.array([speed-x[3],2*angle])


def check_optimality(u,reference,A,b,diagonal=1.):
    residual=A@u-b
    active=residual>=-1e-6
    gradient=diagonal*u-reference
    if active.any():
        multipliers,_=nnls(A[active].T,-gradient,maxiter=1000)
        stationarity=np.linalg.norm(gradient+A[active].T@multipliers,ord=np.inf)
    else:stationarity=np.linalg.norm(gradient,ord=np.inf)
    return float(np.max(residual)),float(stationarity)


def audit(dataset,samples=256):
    root=Path(dataset);manifest=json.loads((root/'manifest.json').read_text());source=Path(manifest['source'])
    if sha256(root/'data.npz')!=manifest['data_sha256']:raise ValueError('Data hash mismatch')
    if sha256(source/'manifest.json')!=manifest['source_manifest_sha256']:raise ValueError('Source manifest mismatch')
    with np.load(root/'data.npz') as f:data={k:f[k] for k in f.files}
    groups=data['group_id'];splits=data['split'];valid=data['valid'];rng=np.random.default_rng(903)
    group_sets=[set(groups[splits==i]) for i in range(3)]
    if any(group_sets[i]&group_sets[j] for i in range(3) for j in range(i)):raise ValueError('Parent split leakage')
    if len(set(zip(groups.tolist(),data['tick'].tolist())))!=len(groups):raise ValueError('Duplicate physical parent/tick rows')
    if not np.array_equal(data['u_ref'],data['label']):raise ValueError('Training reference convention changed')
    if not np.isfinite(data['z']).all() or not np.isfinite(data['ctx']).all() or not np.isfinite(data['label']).all():raise ValueError('Nonfinite features/labels')
    if not np.all(data['label'][~valid]==0.):raise ValueError('Invalid label storage convention changed')
    chosen=[]
    for split in range(3):
        for feasible in (False,True):
            pool=np.flatnonzero((splits==split)&(valid==feasible))
            chosen.extend(rng.choice(pool,min(len(pool),max(1,samples//6)),replace=False).tolist())
    pool=np.setdiff1d(np.arange(len(groups)),chosen)
    chosen=sorted(chosen+rng.choice(pool,min(len(pool),max(0,samples-len(chosen))),replace=False).tolist())
    provenance={p['group_id']:p for p in manifest['provenance']};index=json.loads((source/'index.json').read_text())
    entries={p['visitation_file']:p for p in index};by_group={}
    for row in chosen:by_group.setdefault(str(groups[row]),[]).append(row)
    max_feature=max_violation=max_stationarity=0.;invalid=0;feasible_rejections=0;details=[]
    for group,rows in by_group.items():
        entry=entries[provenance[group]['source_trace']];trace_path=source/entry['visitation_file'];shard_path=source/entry['file']
        if sha256(trace_path)!=entry['visitation_sha256'] or sha256(shard_path)!=entry['sha256']:raise ValueError('Source sample changed')
        with np.load(trace_path) as trace,np.load(shard_path) as shard:
            parent=int(trace['parent']);x=trace['observed_state'];o=trace['raw_observed_obstacles'];mask=trace['obstacle_mask'];goal=shard['goal'][parent]
            if str(trace['group_id'])!=group or str(shard['group_id'][parent])!=group:raise ValueError('Source parent mismatch')
            count=max(1,int(trace['expected_steps']))
            for row in rows:
                tick=int(data['tick'][row])
                if tick>=count:raise ValueError('Feature from unexecuted future state')
                z,ctx=numpy_features(x[tick].astype(float),goal.astype(float),o[tick].astype(float),mask,manifest['radius'])
                error=max(np.max(np.abs(z-data['z'][row])),np.max(np.abs(ctx-data['ctx'][row])))
                if error>1e-10:raise ValueError(f'Pre-action raw feature mismatch {error}')
                max_feature=max(max_feature,float(error));reference=numpy_nominal(ctx[:4],ctx[4:6])
                np.testing.assert_allclose(reference,data['original_nominal'][row],atol=1e-12,rtol=1e-12)
                A,b=numpy_constraints(ctx,np.full((5,2),1.5),manifest['radius'])
                if valid[row]:
                    violation,stationarity=check_optimality(data['label'][row],reference,A,b)
                    if violation>1e-5 or stationarity>2e-6:raise ValueError(f'Invalid teacher optimum: {violation}, {stationarity}')
                    max_violation=max(max_violation,violation);max_stationarity=max(max_stationarity,stationarity)
                else:
                    result=linprog(np.zeros(2),A_ub=A,b_ub=b,bounds=[(None,None)]*2,method='highs')
                    if result.status not in (0,2):raise ValueError(f'Independent feasibility unresolved: {result.message}')
                    if result.status==0:
                        feasible_rejections+=1
                        if 'expert_candidate' not in data:raise ValueError(f'Feasible rejection needs raw solver diagnostics (row{row})')
                        raw=data['solver_raw_control'][row];candidate=data['expert_candidate'][row]
                        if data['solver_feasible'][row]:
                            if not np.isfinite(raw).all() or np.max(A@raw-b)>1e-5+1e-10:raise ValueError('False raw solver feasibility')
                            np.testing.assert_array_equal(candidate,np.clip(raw,-.5,.5))
                            if np.max(A@candidate-b)<=1e-5:raise ValueError('Post-clip rejection lacks a violated constraint')
                        elif np.isfinite(raw).any():raise ValueError('Unaccepted solver must preserve missing raw action')
                    invalid+=1
                details.append(dict(row=row,group_id=group,tick=tick,valid=bool(valid[row]),
                    independently_feasible=True if valid[row] else result.status==0))
    report=dict(audit_passed=True,manifest_sha256=sha256(root/'manifest.json'),samples=len(chosen),groups=len(by_group),
        invalid_labels_checked=invalid,feasible_solver_rejections_checked=feasible_rejections,independently_infeasible_labels_checked=invalid-feasible_rejections,
        maximum_feature_error=max_feature,maximum_primal_violation=max_violation,
        maximum_stationarity_error=max_stationarity,all_rows_group_disjoint=True,all_rows_unique_parent_tick=True,
        independent='NumPy source feature/nominal/HOCBF assembly, SciPy NNLS KKT multipliers, HiGHS invalid-label feasibility. No production feature/controller formula reused.',
        limitations='Checks recorded feature alignment and default teacher, not future policy performance or safety after teacher failure.',samples_detail=details)
    write_json(root/'independent_audit.json',report);print(json.dumps({k:v for k,v in report.items() if k!='samples_detail'}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--samples',type=int,default=256)
    args=p.parse_args();audit(args.dataset,args.samples)
