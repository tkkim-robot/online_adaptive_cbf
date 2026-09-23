"""Outcome-independent parent selection and learned-observation source validation."""
import copy
from pathlib import Path
import numpy as np
from .bicycle_experiment import read
from .dataset import sha256
from .scenes import DIVERSE_FAMILIES
PHASE = 'training_acquisition'


def select_training_parents(parents):
    """One lowest-seed TRAIN parent in each prespecified alternating gain cell."""
    train = [p for p in parents if p['partition'] == 'train']
    noises = sorted({p['noise'][0] for p in train})
    bank = np.geomspace(.5, 8, 8).astype(np.float32)
    if len(noises) != 4 or len({p['group_id'] for p in parents}) != len(parents):
        raise ValueError('Unique parents and all four training noise levels required')
    selected = []
    for fi, family in enumerate(DIVERSE_FAMILIES):
        for ni, noise in enumerate(noises):
            for gi, gain in enumerate(bank):
                if (fi+ni+gi) % 2:
                    continue
                cell = [p for p in train if p['family'] == family and p['noise'][0] == noise
                        and np.float32(p['acquisition_gain']) == gain]
                if not cell:
                    raise ValueError('Missing prespecified training cell')
                selected.append(copy.deepcopy(min(cell, key=lambda p: p['seed'])))
    if len(selected) != 128 or any(p['calibration_role'] != 'none' for p in selected):
        raise ValueError('128 train-only parents with no reserved calibration role required')
    return selected


def select_parents(parents):
    selected = select_training_parents(parents)
    validation = [p for p in parents if p['partition']=='validation']
    noises = sorted({p['noise'][0] for p in validation})
    if len(noises)!=4: raise ValueError('Four validation noise levels required')
    for family in DIVERSE_FAMILIES:
        for noise in noises:
            cell=[p for p in validation if p['family']==family and p['noise'][0]==noise]
            if not cell: raise ValueError('Missing validation family/noise cell')
            selected.append(copy.deepcopy(min(cell,key=lambda p:p['seed'])))
    if len(selected)!=160 or any(p['calibration_role']!='none' for p in selected):
        raise ValueError('Use128TRAIN/32validation parents, no calibration roles')
    return selected


def validate_source(directory):
    directory=Path(directory);m=read(directory/'manifest.json')
    if (m.get('data_role')!='reserved_bicycle_learned_policy_acquisition'
            or m.get('training_use') is not True or m.get('weight_fit_authorized') is not True
            or m.get('final_test') is not False or m.get('groups')!=160
            or sha256(directory/'scenes.json')!=m['scenes_sha256']):
        raise ValueError('Reserved acquisition source required')
    source=Path(m['acquisition_parent_source']);original=read(source/'manifest.json')
    if (sha256(source/'manifest.json')!=m['acquisition_parent_manifest_sha256']
            or sha256(source/'scenes.json')!=original['scenes_sha256']
            or original.get('training_use') is not True or original.get('weight_fit_authorized') is not True
            or original.get('final_test') is not False):
        raise ValueError('Changed or unauthorized original acquisition parents')
    parents=read(directory/'scenes.json')
    if parents!=select_parents(read(source/'scenes.json')):
        raise ValueError('Changed original physical parents, roles or selection')
    if m['config']!=original['config'] or m['routing_contract']!=original['routing_contract']:
        raise ValueError('Acquisition physics/routing changed')
    if m['phase_order']!={PHASE:list(range(160))}:
        raise ValueError('Missing or reordered acquisition parents')
    return m,parents
