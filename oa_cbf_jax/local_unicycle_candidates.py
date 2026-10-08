"""Explicit candidate-domain admission; legacy bundles retain their contract."""
import json
from pathlib import Path
import numpy as np
from .dataset import sha256


def read(path):
    return json.loads(Path(path).read_text())


def candidate_contract(manifest):
    value = manifest.get('candidate_bank_contract')
    if value is None:
        return None
    from .local_unicycle_ordered_data import candidate_bank, bank_contract
    if value.get('schema') != 'unicycle_ordered_unique_candidates_v1':
        raise ValueError('Unregistered unicycle candidate contract')
    original = np.asarray(value['original_bank'], np.float32)
    bank = candidate_bank(original, value['seed'])
    if value != bank_contract(original, bank, value['seed']):
        raise ValueError('Changed ordered candidate construction')
    if manifest['gain_domain'] != {'lower': .5, 'upper': 8.}:
        raise ValueError('Changed candidate bounds')
    if manifest['held_gain_seconds'] != 8 or manifest['adaptation_interval_seconds'] != .2:
        raise ValueError('Changed held-label or deployment cadence')
    return value


def validate_bank(dataset, manifest=None):
    root = Path(dataset)
    m = read(root/'manifest.json') if manifest is None else manifest
    value = candidate_contract(m)
    if value is not None:
        if sha256(root/'source.npz') != m['source_sha256']:
            raise ValueError('Changed candidate source')
        with np.load(root/'source.npz') as source:
            if source['gains'].dtype != np.float32:
                raise ValueError('Ordered gain source must retain FP32 candidates')
            np.testing.assert_array_equal(source['gains'], np.asarray(value['bank'], np.float32))
    return value
