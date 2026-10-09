"""Io functions and shared contracts."""

import json

from pathlib import Path

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sanitize(value):
    import numpy as np
    if hasattr(value,"tolist"): return sanitize(value.tolist())
    if isinstance(value,dict): return {k:sanitize(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [sanitize(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


import hashlib


import numpy as np

def source_fingerprint():
    from .io import snapshot_package
    return snapshot_package()

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_dataset(directory,partition,verify=True):
    path=Path(directory);parts=[]
    if (path/'INVALIDATED.json').exists():raise ValueError(f'Invalidated dataset: {path}; inspect INVALIDATED.json')
    for entry in json.loads((path/'index.json').read_text()):
        shard=path/entry['file']
        if verify and sha256(shard)!=entry['sha256']:raise ValueError(f'Corrupt {shard}')
        with np.load(shard,allow_pickle=False) as data:
            keep=data['partition']==partition
            selected={k:data[k][keep] for k in data.files}
            if 'query_origin' in entry:
                selected['query_origin']=np.full(int(keep.sum()),entry['query_origin'])
            parts.append(selected)
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}


def parent_mean(values, ids, mask=None):
    values = np.asarray(values)
    ids = np.asarray(ids)
    mask = np.ones_like(values, bool) if mask is None else np.asarray(mask, bool)
    results = []
    for group in np.unique(ids):
        keep = ids == group
        valid = mask[keep]
        if valid.any():
            results.append(float(values[keep][valid].mean()))
    return float(np.mean(results)) if results else None


import io


import os


import tarfile

import uuid

def snapshot_package():
    package=Path(__file__).resolve().parent
    files={p.name:p.read_bytes() for p in sorted(package.glob('*.py'))}
    digest=hashlib.sha256(b''.join(name.encode()+data for name,data in files.items())).hexdigest()
    root=Path(os.environ.get('OA_CBF_SOURCE_ARCHIVE_ROOT',package.parent/'artifacts'/'source_snapshots'));root.mkdir(parents=True,exist_ok=True)
    destination=root/(digest+'.tar.gz')
    if not destination.exists():
        temporary=root/(digest+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with tarfile.open(temporary,'w:gz') as archive:
                for name,data in files.items():
                    member=tarfile.TarInfo('oa_cbf_jax/'+name);member.size=len(data);member.mode=0o644
                    archive.addfile(member,io.BytesIO(data))
                metadata=json.dumps(dict(package_sha256=digest,files={name:hashlib.sha256(data).hexdigest() for name,data in files.items()}),sort_keys=True).encode()
                member=tarfile.TarInfo('source_manifest.json');member.size=len(metadata);member.mode=0o644
                archive.addfile(member,io.BytesIO(metadata))
            try:os.link(temporary,destination)
            except FileExistsError:pass
        finally:temporary.unlink(missing_ok=True)
    return digest
