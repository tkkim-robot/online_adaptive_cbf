"""Relabel frozen shared-policy visits with a new, ordered candidate bank.

Acquisition continues to refer to the original models AND original gains.
Only eight-second branches at saved, live contexts use the new gain bank.
The original source collector and its replay contract are never changed.
"""


import json
from pathlib import Path


from .dataset import sha256


from . import local_unicycle_observed_coverage as original
from .local_unicycle_observer import Memory, contract
from .local_unicycle_observer_data import branch_inputs, graph_inputs, label_kernel, validate as validate_base
from .local_unicycle_ordered_data import selected_indices


SCHEMA = 'unicycle_local_ordered_shared_coverage_v1'


def read(path):
    return json.loads(Path(path).read_text())


def validate_source(dataset):
    root = Path(dataset)
    m = read(root/'manifest.json')
    if m['schema'] != SCHEMA or m['collector_sha256'] != sha256(__file__):
        raise ValueError('Unregistered ordered coverage or changed collector')
    base, source = Path(m['base_dataset']), Path(m['source_coverage'])
    bm = validate_base(base)
    old, acquisition_base, records = original.validate_source(source)
    for directory, bindings in [(base, m['bound_base_files']), (source, m['bound_source_files'])]:
        for name, digest in bindings.items():
            if sha256(directory/name) != digest:
                raise ValueError('Changed relabel source: '+str(directory/name))
    if (m['acquisition_base'] != str(acquisition_base.resolve())
        or m['acquisition_base_manifest_sha256'] != sha256(acquisition_base/'manifest.json')
        or bm['base_dataset'] != str(acquisition_base.resolve())
        or m['candidate_bank_contract'] != bm['candidate_bank_contract']
        or m['original_collector_sha256'] != old['collector_sha256']
        or m['models'] != old['models'] or m['position_observer'] != contract()
        or m['indices'] != selected_indices(records, m['pilot'])
        or m['forward_parents_used'] or m['calibration_parents_used']
        or not m['acquisition_policy_and_bank_unchanged'] or not m['label_bank_only_changed']):
        raise ValueError('Changed acquisition/label/split contract')
    for key in ['modes', 'snapshot_fractions', 'horizon_steps', 'held_gain_seconds',
                'adaptation_interval_seconds', 'replicas', 'queries', 'numpy_version']:
        if m[key] != old[key]:
            raise ValueError('Changed shared visitation contract: '+key)
    return m, base, records
