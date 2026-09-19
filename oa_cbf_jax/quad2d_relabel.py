"""Separate censored physical margin from failure events in audited flight data.

No rollout is added, changed or dropped. The Gaussian models physical clearance
when observed through the horizon/goal or an actual collision. Solver/domain/
bound stops do not supply a fictitious future clearance. Their separate adverse
head remains observed and is mandatory at selection. Prefix progress is unchanged.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .dataset import sha256,source_fingerprint,load_dataset
from .io import write_json
from .quad2d_rollout import GOAL,TIMEOUT,COLLISION,NAMES

SCHEMA='oa_cbf_quad2d_initial_hurdle_v2'
TARGETS=['conditional_negative_min_clearance_div_0.3_capped_below_minus_2','observed_prefix_route_progress_div_horizon_cruise_distance']


def conditional_labels(status,clearance,progress):
    observed=np.isin(status,[GOAL,TIMEOUT,COLLISION])
    risk=-np.minimum(clearance,.6)/.3
    targets=np.stack((np.where(observed,risk,0.),progress),axis=-1).astype(np.float32)
    masks=np.stack((observed,np.ones_like(observed)),axis=-1)
    return targets,masks


def relabel(source,output):
    source=Path(source);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text());audit=json.loads((source/'independent_replay.json').read_text())
    if manifest['schema']!='oa_cbf_quad2d_initial_flight_v1' or not audit['audit_passed'] or not audit['all_collision_bound_branches_audited'] or audit['manifest_sha256']!=sha256(source/'manifest.json') or audit['index_sha256']!=sha256(source/'index.json'):
        raise ValueError('Exact independently audited original flight data required')
    manifest.update(schema=SCHEMA,stage='quad2d_initial_hurdle_development',targets=TARGETS,
        source_fingerprint=source_fingerprint(),relabel_source=str(source.resolve()),relabel_manifest_sha256=sha256(source/'manifest.json'),relabel_index_sha256=sha256(source/'index.json'),
        risk_target_mode='conditional_physical_clearance',
        censoring='Physical clearance through horizon/goal or collision only; mask risk at earlier solver/domain/bound stops. All adverse events retained. Prefix progress unchanged. Conditional Gaussian risk AND separate calibrated first-adverse event screens are mandatory; no joint physical CVaR or unconditional future-collision guarantee.',
        limitations='Development initial-observation conditional Gaussian plus first-event heads; not adaptive-trajectory calibration or visited-state coverage.')
    index=[]
    for old in json.loads((source/'index.json').read_text()):
        if sha256(source/old['file'])!=old['sha256']:raise ValueError('Changed raw branch shard')
        with np.load(source/old['file']) as f:data={k:f[k] for k in f.files}
        data['target'],data['target_mask']=conditional_labels(data['status'],data['min_clearance'],data['target'][...,1])
        path=root/old['file'];np.savez_compressed(path,**data)
        entry=dict(old,sha256=sha256(path));index.append(entry)
        for name,digest in [(old['trace_file'],old['trace_sha256'])]+[(t['file'],t['sha256']) for t in old.get('event_traces',[])]:
            if sha256(source/name)!=digest:raise ValueError('Changed original physical trace')
            (root/name).hardlink_to(source/name)
    write_json(root/'manifest.json',manifest);write_json(root/'index.json',index)
    parts={}
    for partition in ['train','validation','development_calibration']:
        d=load_dataset(root,partition)
        parts[partition]=dict(groups=len(d['group_id']),branches=int(d['status'].size),steps=int(d['steps'].sum()),
            observed_risk=int(d['target_mask'][...,0].sum()),censored_risk=int((~d['target_mask'][...,0]).sum()),
            target_std=[float(d['target'][...,h][d['target_mask'][...,h]].std()) for h in range(2)],
            outcomes={NAMES[int(k)]:int(v) for k,v in zip(*np.unique(d['status'],return_counts=True))})
    report=dict(contract_valid=all(p['groups']>0 and p['observed_risk']>0 and min(p['target_std'])>1e-6 for p in parts.values()),partitions=parts,
        groups=len(manifest['groups']),preserved_all_branches=True,preserved_all_events=True,preserved_all_physical_traces=True)
    write_json(root/'contract_audit.json',report);print(json.dumps(report),flush=True)
    if not report['contract_valid']:raise ValueError('Conditional target contract failed')
    # The independent flight audit must run again before training.


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);relabel(**vars(p.parse_args()))
