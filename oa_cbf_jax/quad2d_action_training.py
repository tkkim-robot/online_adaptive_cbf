"""Append audited action-context TRAIN pairs to the retained shared union."""
from collections import Counter
import hashlib
from pathlib import Path


from .dataset import load_dataset, sha256
from .quad2d_static_inputs import read
from .quad2d_training_expansion import disjoint_reservations, concatenate_train
from .quad2d_reflection_training import load_pair, check_training_rows


def append_contexts(dataset, reflection, raw, mirrored, previous):
    from .quad2d_static_learning import validate_training_dataset
    dataset, reflection = map(Path, (dataset, reflection))
    primary, additional = [Path(previous[k]) for k in ('primary_dataset','additional_dataset')]
    manifests = [read(p/'manifest.json') for p in (primary, additional, dataset)]
    for i in range(3):
        for j in range(i+1, 3):
            disjoint_reservations(manifests[i], manifests[j])
    if 'action_context' in previous:
        raise ValueError('Action-context expansion cannot be appended twice')
    for path, digest in previous['bindings'].items():
        if sha256(path) != digest:
            raise ValueError('Changed existing TRAIN union')
    validate_training_dataset(dataset, dataset.parent/'data_review.json',
        dict(manifests[0], dataset=str(primary.resolve())))
    report = read(dataset.parent/'review.json')
    proof = read(dataset.parent/'independent_completion_review.json')
    protocol = read(dataset.parent/'protocol.json')
    if (protocol['schema'] != 'quad2d_observed_action_training_collection_v1'
            or not report['every_failed_parent_retained']
            or not report['all_selection_rules_independently_reconstructed']
            or report['data_review_sha256'] != sha256(dataset.parent/'data_review.json')
            or proof['status'] != 'passed' or proof['review_sha256'] != sha256(dataset.parent/'review.json')):
        raise ValueError('Reviewed, outcome-independent action observations required')
    reflected_review = read(reflection.parent/'independent_completion_review.json')
    if (reflected_review['status'] != 'passed'
            or reflected_review['review_sha256'] != sha256(reflection.parent/'review.json')):
        raise ValueError('Independent reflected-label completion audit required')
    old_ids = [r['group_id'] for m in manifests[:2] for r in m['groups'] if r['partition']=='train']
    if (len(old_ids) != previous['independent_train_parents']
            or hashlib.sha256('\n'.join(old_ids).encode()).hexdigest() != previous['training_parent_ids_sha256']):
        raise ValueError('Changed retained TRAIN identities')
    new = load_dataset(dataset, 'train'); reflected, pair = load_pair(dataset, reflection, new)
    new_ids = [r['group_id'] for r in manifests[2]['groups'] if r['partition']=='train']
    joined = concatenate_train(raw, new, old_ids, new_ids)
    joined_mirror = concatenate_train(mirrored, reflected, old_ids, new_ids)
    ids = old_ids+new_ids
    check_training_rows(joined, joined_mirror, ids)
    paths = [p/f for p in (dataset, reflection) for f in ('manifest.json','index.json','complete.json','independent_replay.json')]
    paths += [dataset.parent/f for f in ('review.json','protocol.json','data_review.json','independent_completion_review.json')]
    paths += [reflection.parent/f for f in ('review.json','protocol.json','qualification.json','independent_completion_review.json')]
    addition = dict(dataset=str(dataset.resolve()),manifest_sha256=sha256(dataset/'manifest.json'),
        reflection=str(reflection.resolve()),pair=pair,train_parents=len(new_ids),
        partitions=dict(Counter(r['partition'] for r in manifests[2]['groups'])))
    expanded = dict(previous, action_context=addition, previous_expansion=previous,
        independent_train_parents=len(ids),
        training_parent_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        bindings={**previous['bindings'], **{str(p.resolve()):sha256(p) for p in paths}},
        weighting='Same gain-opportunity weights recomputed on the complete shared TRAIN union; full-parent ensemble sampling.',
        calibration='All three original calibration reservations excluded from weight fitting.')
    return joined, joined_mirror, expanded
