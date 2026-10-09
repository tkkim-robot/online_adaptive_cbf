"""Shared bicycle trace storage implementation."""

from contextlib import contextmanager

import hashlib

import io

import json

from pathlib import Path

import numpy as np

MARKER='__oa_cbf_shared_prefix_v1__'

FIELDS=('observed_obstacles','raw_observed_obstacles','innovation_o','innovation_x')

STORAGE_SCHEMA='byte_exact_shared_prefix_v1'

def _digest(data):return hashlib.sha256(data).hexdigest()

def _neighbor(root,name):
    if not isinstance(name,str) or Path(name).name!=name or not name.endswith('.npz'):
        raise ValueError('Shared trace names must be adjacent NPZ files')
    path=root/name
    if path.resolve().parent!=root.resolve():raise ValueError('Shared trace dependency leaves its directory')
    return path

def _arrays(data):
    with np.load(io.BytesIO(data),allow_pickle=False) as archive:
        return {k:archive[k] for k in archive.files}

def verify_index_dependencies(directory,entries):
    """Recheck audited records and every shared file before consuming labels.

    The index cannot silently omit a dependency: compare it to the metadata
    embedded in the hash-bound branch record. Hash each shared file once.
    """
    root=Path(directory);checked={}
    for entry in entries:
        path=root/entry['file'];raw=path.read_bytes()
        if _digest(raw)!=entry['sha256']:raise ValueError('Changed trace record')
        expected=entry.get('shared_dependencies',{});actual={}
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:
            if MARKER in z:
                metadata=json.loads(str(z[MARKER]))
                if metadata['schema']!=1:raise ValueError('Unsupported shared trace schema')
                for ref in metadata['fields'].values():
                    name=ref['file'];digest=ref['sha256']
                    if name in actual and actual[name]!=digest:raise ValueError('Conflicting shared dependency')
                    actual[name]=digest
        if actual!=expected:raise ValueError('Shared dependency index differs from record')
        for name,digest in actual.items():
            neighbor=_neighbor(path.parent,name)
            if neighbor not in checked:checked[neighbor]=_digest(neighbor.read_bytes())
            if checked[neighbor]!=digest:raise ValueError('Changed shared trace dependency')
    return dict(records=len(entries),shared_files=len(checked),all_shared_dependencies_verified=True)

@contextmanager
def open_trace(path,expected_sha256=None):
    """Load original arrays, verifying the exact bytes of every dependency."""
    path=Path(path);raw=path.read_bytes()
    if expected_sha256 is not None and _digest(raw)!=expected_sha256:raise ValueError('Changed trace record')
    values=_arrays(raw)
    if MARKER in values:
        reference=values.pop(MARKER)
        if reference.shape!=() or reference.dtype.kind!='U':raise ValueError('Invalid shared trace metadata')
        metadata=json.loads(str(reference))
        if set(metadata)!={'schema','fields'} or metadata['schema']!=1 or not isinstance(metadata['fields'],dict):
            raise ValueError('Unsupported shared trace schema')
        cache={}
        for key,ref in metadata['fields'].items():
            if key not in FIELDS or key in values or not isinstance(ref,dict) or set(ref)!={'file','sha256','shape','dtype'}:
                raise ValueError('Invalid shared field reference')
            dependency=_neighbor(path.parent,ref['file'])
            binding=(dependency,ref['sha256'])
            if binding not in cache:
                raw=dependency.read_bytes()
                if _digest(raw)!=ref['sha256']:raise ValueError('Changed shared trace dependency')
                cache[binding]=_arrays(raw)
            shared=cache[binding]
            if MARKER in shared or key not in shared:raise ValueError('Invalid or recursive shared dependency')
            array=shared[key];shape=ref['shape']
            if not isinstance(shape,list) or not shape or any(type(n) is not int or n<0 for n in shape):raise ValueError('Invalid prefix shape')
            if array.dtype.str!=ref['dtype'] or array.ndim!=len(shape) or tuple(shape[1:])!=array.shape[1:] or shape[0]>len(array):
                raise ValueError('Shared trace dtype/shape/length mismatch')
            values[key]=array[:shape[0]]
    yield values

def write_query_traces(directory,names,payloads,replicas,stem):
    """Write one complete gain/replica query without overwriting old evidence.

    Payload order is gain-major, replica-minor, matching the existing collector.
    Any field without exact prefix equality stays in its individual trace.
    """
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    if type(replicas) is not int or replicas<1 or not payloads or len(payloads)%replicas or len(names)!=len(payloads) or len(set(names))!=len(names):
        raise ValueError('Invalid complete gain/replica query')
    shared_names=[f'{stem}_shared_r{r}.npz' for r in range(replicas)]
    paths=[_neighbor(root,name) for name in [*names,*shared_names]]
    if len(set(paths))!=len(paths) or any(p.exists() for p in paths):raise ValueError('Trace output already exists or names collide')
    payloads=[{k:np.asarray(v) for k,v in p.items()} for p in payloads]
    if any(MARKER in p or any(a.dtype.hasobject for a in p.values()) for p in payloads):raise ValueError('Object arrays and nested shared traces are unsupported')
    references=[{} for _ in payloads]
    for replica in range(replicas):
        ids=list(range(replica,len(payloads),replicas));shared={}
        for key in FIELDS:
            if not all(key in payloads[i] and payloads[i][key].ndim>=1 for i in ids):continue
            longest=max(ids,key=lambda i:len(payloads[i][key]));full=payloads[longest][key]
            if all(payloads[i][key].dtype==full.dtype and payloads[i][key].shape[1:]==full.shape[1:]
                   and payloads[i][key].tobytes()==full[:len(payloads[i][key])].tobytes() for i in ids):shared[key]=full
        if not shared:continue
        file=shared_names[replica];path=root/file;np.savez_compressed(path,**shared);digest=_digest(path.read_bytes())
        for i in ids:
            for key in shared:
                a=payloads[i][key];references[i][key]=dict(file=file,sha256=digest,shape=list(a.shape),dtype=a.dtype.str)
    records=[]
    for name,payload,refs in zip(names,payloads,references,strict=True):
        local={k:v for k,v in payload.items() if k not in refs}
        if refs:local[MARKER]=np.array(json.dumps(dict(schema=1,fields=refs),sort_keys=True))
        path=root/name;np.savez_compressed(path,**local)
        records.append(dict(file=name,sha256=_digest(path.read_bytes()),shared_dependencies={r['file']:r['sha256'] for r in refs.values()}))
    return records
