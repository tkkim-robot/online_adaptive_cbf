"""Local source archives for reproducible experiments without git commits."""

import hashlib
import io
import json
import os
from pathlib import Path
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
