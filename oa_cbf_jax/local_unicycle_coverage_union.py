"""Retain multiple audited policy generations on the same physical parents.

The index references immutable existing shards instead of copying labels. Every
observation is retained, including identical fixed-reference views in multiple
generations; these are correlated visits, never additional independent parents.
"""


import json
from pathlib import Path

from .dataset import sha256


SCHEMA = 'local_unicycle_observed_coverage_union_v1'


def read(path):
    return json.loads(Path(path).read_text())


def members(dataset, sources):
    from .local_unicycle_expansion import validate as validate_single
    from .local_unicycle_observed_coverage import SCHEMA as OBSERVED_SCHEMA
    sources = [Path(p).resolve() for p in sources]
    if len(sources) < 2 or len(set(sources)) != len(sources):
        raise ValueError('At least two distinct coverage generations required')
    proofs, index, records = [], [], None
    navigations = set()
    for generation, source in enumerate(sources):
        manifest = read(source/'manifest.json')
        if manifest['schema'] != OBSERVED_SCHEMA:
            raise ValueError('Only original observed coverage may enter a union; no nested unions')
        identity = manifest['navigation_protocol_sha256']
        if identity in navigations:
            raise ValueError('Duplicate acquisition policy generation')
        navigations.add(identity)
        proof, rows = validate_single(dataset, source)
        if records is not None and records != rows:
            raise ValueError('Coverage generations use different physical parents/roles')
        if proofs:
            for key in ('primary_manifest_sha256', 'physical_parent_count',
                        'shared_acquisition_modes', 'position_observer',
                        'horizon_seconds', 'adaptation_seconds'):
                if proof[key] != proofs[0][key]:
                    raise ValueError('Coverage generations changed '+key)
        records = rows
        proofs.append(proof)
        for entry in read(source/'index.json'):
            entry = dict(entry, generation=generation, source_dataset=str(source))
            for field in ('file', 'audit_file', 'acquisition_file'):
                entry[field] = str((source/entry[field]).resolve())
            index.append(entry)
    return proofs, records, index


def validate(dataset, additional):
    root = Path(additional).resolve()
    manifest = read(root/'manifest.json')
    if (manifest['schema'] != SCHEMA or manifest['primary_dataset'] != str(Path(dataset).resolve())
            or manifest['weight_fit_authorized'] is not True
            or manifest['validation_weight_fitting'] or manifest['forward_parents_used']):
        raise ValueError('Changed union or parent-role authorization')
    proofs, records, expected = members(dataset, manifest['sources'])
    if (proofs != manifest['member_proofs'] or read(root/'index.json') != expected
            or sha256(root/'index.json') != manifest['index_sha256']
            or manifest['generations'] != len(proofs)
            or manifest['physical_parent_count'] != len({r['group_id'] for r in records})
            or manifest['added_observations'] != sum(p['added_observations'] for p in proofs)
            or manifest['added_branches'] != sum(p['added_branches'] for p in proofs)):
        raise ValueError('Changed or partial coverage union')
    bound = {str(root/name): sha256(root/name) for name in ('manifest.json', 'index.json')}
    for proof in proofs:
        bound.update(proof['bound_files'])
    return dict(schema=SCHEMA, primary_dataset=str(Path(dataset).resolve()),
        additional_dataset=str(root), primary_manifest_sha256=proofs[0]['primary_manifest_sha256'],
        bound_files=bound, physical_parent_count=manifest['physical_parent_count'],
        added_observations=manifest['added_observations'], added_branches=manifest['added_branches'],
        shared_acquisition_modes=proofs[0]['shared_acquisition_modes'],
        coverage_generations=len(proofs), member_proofs=proofs,
        position_observer=proofs[0]['position_observer'], horizon_seconds=8., adaptation_seconds=.2,
        validation_weight_fitting=False, forward_parents_used=False,
        weighting=manifest['weighting'], retention=manifest['retention'],
        normalization='TRAIN only, base plus every retained shared observation; equal total weight per physical parent.'), records
