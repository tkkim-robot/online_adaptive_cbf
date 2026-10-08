"""Fresh held labels for an ordered, unique unicycle gain bank.

The original collector, observations, parent reservations, QP and stop checks
are reused verbatim. This module changes only the candidate bank. Qualification
data cannot train; deployment requires new training and reserved calibration.
"""


import numpy as np
from scipy.stats import qmc


from .local_unicycle_study import GAINS


def candidate_bank(original, seed):
    """Canonicalize the original bank, then fill duplicates without outcomes."""
    stream = np.exp(np.log(.5) + qmc.Sobol(2, scramble=True, seed=seed)
                    .random_base2(6) * np.log(16)).astype(np.float32)
    expected = np.concatenate((GAINS, stream[:16]))
    if original.dtype != np.float32 or not np.array_equal(original, expected):
        raise ValueError('Expected the original registered 32-gain Sobol bank')
    bank, seen = [], set()
    for pair in np.concatenate((original, stream[16:])):
        ordered = tuple(sorted(map(float, pair), reverse=True))
        if ordered not in seen:
            seen.add(ordered)
            bank.append(ordered)
        if len(bank) == 32:
            break
    if len(bank) != 32:
        raise ValueError('Insufficient unique ordered candidates')
    return np.asarray(bank, np.float32)


def bank_contract(original, bank, seed):
    return dict(schema='unicycle_ordered_unique_candidates_v1', seed=seed,
        rule='Stable unique descending original pairs, then next unique descending pairs from the same scrambled log-Sobol stream starting at index16.',
        original_candidates=32, canonical_original_candidates=27, candidates=32,
        original_bank=original.tolist(), bank=bank.tolist(),
        outcome_dependent_candidate_selection=False, gain_bounds=[.5, 8.],
        held_gain_seconds=8, adaptation_interval_seconds=.2,
        original_collector_unchanged=True, physical_controller_unchanged=True,
        requires_new_labels_training_and_reserved_calibration=True)


def selected_indices(records, pilot):
    if not pilot:
        return list(range(len(records)))
    parents = set()
    for family in sorted({r['family'] for r in records}):
        eligible = sorted({r['group_id'] for r in records
                           if r['family'] == family and r['partition'] == 'train'})
        if len(eligible) < 2:
            raise ValueError('Two TRAIN parents per family required')
        parents.update(eligible[:2])
    return [i for i, r in enumerate(records) if r['group_id'] in parents]
