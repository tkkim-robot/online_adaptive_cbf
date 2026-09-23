"""Resolve recorded observation histories without loading physical targets."""
from pathlib import Path
from .bicycle_experiment import read
from .dataset import sha256


def source_map(dataset):
    root=Path(dataset);m=read(root/'manifest.json');union=m['shared_observation_union'];result={};bindings={}
    for base in [Path(union['base']),*[Path(p['directory']) for p in union['parts']]]:
        index=base/'index.json';bindings[str(index)]=sha256(index)
        for e in read(index):
            file=str((base/e['file']).resolve());t=e['traces'][0]
            path=(base/t['acquisition_file']).resolve()
            result[file]=dict(path=str(path),sha256=t['acquisition_sha256'],query_sha256=e['sha256'],
                group_id=e['group_id'],query_tick=e['query_tick'],query_origin=e.get('acquisition_encoder','fixed'))
    return result,bindings
