"""Recover verified collection shards when increasing route storage capacity.

Old acquisition/branch outcomes remain actual recorded outcomes under their old
source and route shape. Only masked route padding changes in recovered files.
Uncollected parents are simulated normally with the larger static capacity.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .dataset import sha256
from .io import write_json


def inventory(source):
    source=Path(source)
    manifest=json.loads((source/'manifest.json').read_text())
    expected=len(manifest['groups'])//manifest['shard_groups'];records=[]
    for i in range(expected):
        path=source/f'shard_{i:05d}.json'
        if not path.exists():continue
        record=json.loads(path.read_text())
        if record['file']!=f'shard_{i:05d}.npz' or record['groups']!=manifest['shard_groups']:
            raise ValueError('Invalid source shard identity')
        records.append(dict(record=record,metadata_sha256=sha256(path)))
    return records


def origin_for(source,target_manifest):
    source=Path(source).resolve();original=json.loads((source/'manifest.json').read_text())
    if 'recovery_origin' in original:raise ValueError('Nested recovery needs a separately audited migration')
    old_capacity=original.get('route_capacity',32);new_capacity=target_manifest.get('route_capacity',32)
    if new_capacity<=old_capacity:raise ValueError('Recovery must increase route capacity')
    excluded={'source_fingerprint','route_capacity','recovery_origin'}
    if {k:v for k,v in original.items() if k not in excluded}!={k:v for k,v in target_manifest.items() if k not in excluded}:
        raise ValueError('Recovery may only change source implementation and route capacity')
    records=inventory(source)
    if not records:raise ValueError('No completed source shards to recover')
    digest=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    return dict(dataset=str(source),manifest_sha256=sha256(source/'manifest.json'),
        source_fingerprint=original['source_fingerprint'],inventory_sha256=digest,completed_shards=len(records),
        original_route_capacity=old_capacity,target_route_capacity=new_capacity,
        interpretation='Keep every existing parent, feature, physical/sensor/context/query/key and recorded outcome. Only append inactive route storage. Recorded trajectories retain their original source and numerical route shape; all previously uncollected parents are simulated with the new capacity. No physical/noise reset or outcome resampling.')


def pad_payload(data,capacity):
    old=data['route_points'];mask=data['route_mask']
    if old.ndim!=3 or old.shape[2]!=2 or mask.shape!=old.shape[:2] or old.shape[1]>capacity:
        raise ValueError('Invalid route shape or attempted truncation')
    if mask.dtype!=bool or np.any(mask.sum(axis=1)<2) or np.any(np.diff(mask.astype(int),axis=1)>0):
        raise ValueError('Invalid route mask')
    extra=capacity-old.shape[1]
    points=np.concatenate((old,np.broadcast_to(data['goal'][:,None,:],(len(old),extra,2)).astype(old.dtype)),axis=1)
    padded_mask=np.pad(mask,((0,0),(0,extra)))
    return {**data,'route_points':points,'route_mask':padded_mask}


def verify_padding(original,changed,capacity):
    if set(original)!=set(changed):return False
    expected=pad_payload(original,capacity)
    return all(expected[k].dtype==changed[k].dtype and expected[k].shape==changed[k].shape and
               (np.array_equal(expected[k],changed[k],equal_nan=True) if expected[k].dtype.kind in 'fc'
                else np.array_equal(expected[k],changed[k])) for k in expected)


def reuse_completed(source,output,workers=8):
    source=Path(source).resolve();output=Path(output).resolve()
    if source==output:raise ValueError('Preserve original dataset; recovery needs a new directory')
    manifest=json.loads((output/'manifest.json').read_text());origin=origin_for(source,manifest)
    if manifest.get('recovery_origin')!=origin:raise ValueError('Recovery origin changed')
    records=inventory(source);capacity=manifest['route_capacity'];start=time.perf_counter()
    def one(item):
        old=item['record'];path=output/old['file'];meta=path.with_suffix('.json')
        for field,digest in [('file','sha256'),('audit_file','audit_sha256'),('visitation_file','visitation_sha256')]:
            if sha256(source/old[field])!=old[digest]:raise ValueError('Corrupt source data or physical trace')
        if meta.exists():
            saved=json.loads(meta.read_text())
            if saved.get('recovery_source_sha256')!=old['sha256'] or sha256(path)!=saved['sha256']:
                raise ValueError('Changed recovered shard')
            for field,digest in [('audit_file','audit_sha256'),('visitation_file','visitation_sha256')]:
                if sha256(output/saved[field])!=saved[digest]:raise ValueError('Changed recovered physical trace')
            return old['file']
        with np.load(source/old['file']) as z:data={k:z[k] for k in z.files}
        padded=pad_payload(data,capacity)
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,**padded)
        # Independently read serialized output; check every unchanged array and
        # every newly masked route slot, including NaNs in recorded diagnostics.
        with np.load(tmp) as z:restored={k:z[k] for k in z.files}
        if not verify_padding(data,restored,capacity):raise ValueError('Recovery changed more than inactive route storage')
        tmp.replace(path)
        for field in ('audit_file','visitation_file'):shutil.copyfile(source/old[field],output/old[field])
        saved={**old,'sha256':sha256(path),'recovery_source_sha256':old['sha256'],
            'recovery_source_metadata_sha256':item['metadata_sha256'],
            'collection_source_fingerprint':origin['source_fingerprint'],
            'collection_route_capacity':origin['original_route_capacity'],'stored_route_capacity':capacity}
        write_json(meta,saved)
        return old['file']
    with ThreadPoolExecutor(workers) as pool:done=list(pool.map(one,records))
    write_json(output/'recovery_audit.json',dict(audit_passed=True,origin=origin,shards=len(done),
        groups=len(done)*manifest['shard_groups'],elapsed_seconds=time.perf_counter()-start,
        scope='Every copied shard and both physical traces hashed; every output array read back and compared to the source, permitting only inactive route padding. Complements later physical/causal replay.'))
    print(json.dumps(dict(stage='recovery_complete',shards=len(done),elapsed_seconds=time.perf_counter()-start)),flush=True)
