"""Unicycle training functions and shared contracts."""

import numpy as np

def contract():
    return dict(schema='local_unicycle_prediction_checkpoint_selection_v1',
        score='mean(normalized_mae) + mean(event_brier)',
        targets=['normalized_clearance','normalized_conditional_progress'],
        events=['collision','controller_stop'],
        population='Original live-query and target masks, with equal physical-parent validation weights.',
        optimization='Unchanged Gaussian NLL and event BCE; no new objective or gradient.',
        selection='Minimum validation score with original 1e-4 improvement tolerance and patience.',
        calibration='Refit every model-dependent calibration after selection; likelihood is still recorded.',
        forward_or_navigation_selection=False)

def score(metrics):
    mae=np.asarray(metrics['normalized_mae'],dtype=float)
    brier=np.asarray(metrics['brier'],dtype=float)
    if (mae.shape!=(2,) or brier.shape!=(2,) or not np.isfinite(mae).all()
            or not np.isfinite(brier).all() or (mae<0).any() or (brier<0).any() or (brier>1).any()):
        raise ValueError('Two finite nonnegative mean errors and binary-event Brier scores required')
    return float(mae.mean()+brier.mean())


from collections import Counter

import hashlib

import json

from pathlib import Path


from .io import sha256

SCHEMA = 'local_unicycle_actual_paired_reflection_v1'

KEYS = ('features', 'node_mask', 'gains', 'target', 'target_mask', 'events', 'event_mask')

OUTCOMES = ('final_state', 'min_clearance', 'progress', 'status', 'steps')

PREFIX = 'reflected_'

def read(path):
    return json.loads(Path(path).read_text())

def check_summary(spec, done, review, proof, observations):
    if (done['status'] != 'completed' or not spec['full_collection']
            or spec['partition'] != 'train_and_validation'
            or spec['physical_parents'] != 768 or spec['physical_variants'] != 3072
            or spec['observations'] != observations or spec['branches_per_orientation'] != observations * 64
            or spec['held_seconds'] != 8 or spec['adaptation_seconds'] != .2
            or spec['weight_fitting'] or spec['forward_parents_used']):
        raise ValueError('Complete original TRAIN/validation reflection population required')
    if (proof['status'] != 'passed' or not proof['independently_reproduced']
            or not proof['denominator_and_targets_reproduced'] or proof['physical_parents'] != 768
            or proof['observations'] != observations
            or proof['branches'] != {o: observations * 64 for o in ('original', 'reflected')}
            or proof['forward_parents_used'] or proof['weight_fitting']
            or not review['full_collection'] or review['forward_parents_used'] or review['weight_fitting']):
        raise ValueError('Independent complete reflection review required')
    summary = review['summary']
    if (proof['summary'] != summary or not summary['original_exact_replay']
            or not summary['all_reflected_trace_fields_qualified']
            or any(summary[k] != 0 for k in ('status_changes', 'step_changes', 'target_mask_changes'))):
        raise ValueError('Reflection numerical changes require separate qualification')

def reviewed_job(job, donor, observations):
    job, donor = Path(job).resolve(), Path(donor).resolve()
    spec = read(job / 'protocol.json'); done = read(job / 'complete.json')
    root = Path(spec['output']); review = read(root / 'review.json')
    proof = read(root / 'independent_completion_review.json')
    if (read(job / 'job.json')['status'] != 'completed'
            or Path(spec.get('source_coverage', spec['dataset'])).resolve() != donor):
        raise ValueError('Wrong completed reflection donor')
    check_summary(spec, done, review, proof, observations)
    if (done['review_sha256'] != sha256(root / 'review.json')
            or proof['review_sha256'] != done['review_sha256']
            or review['protocol_sha256'] != sha256(job / 'protocol.json')):
        raise ValueError('Changed reflection evidence')
    bindings = dict(spec['bound_files'])
    for file, digest in bindings.items():
        if sha256(file) != digest:
            raise ValueError('Changed reflection source: ' + file)
    files = [job / n for n in ('protocol.json', 'complete.json', 'job.json')]
    files += [root / n for n in ('review.json', 'independent_completion_review.json')]
    files += [donor / n for n in ('manifest.json', 'index.json', 'audit.json')]
    mapping = {}
    for slot in range(spec['workers']):
        index = root / f'index_{slot}.json'; complete = root / f'complete_{slot}.json'
        lane = read(complete)
        if (lane['status'] != 'completed' or lane['backend'] != 'cpu'
                or lane['implicit_signatures'] != 0 or lane['index_sha256'] != sha256(index)):
            raise ValueError('Incomplete or changed reflection worker')
        files.extend((index, complete))
        for row in read(index):
            src = row['source']; name = src['file']
            if name in mapping or not row['original_exact_replay']:
                raise ValueError('Duplicate or unverified reflected source')
            if not all(v.get('within_tolerance', v.get('exact'))
                       for v in row['reflection_differences'].values()):
                raise ValueError('Unqualified reflected trajectory')
            mapping[name] = (root / row['file'], row)
    entries = read(donor / 'index.json')
    if set(mapping) != {e['file'] for e in entries} or len(entries) != len(mapping):
        raise ValueError('Missing or repeated original observation shard')
    bindings.update({str(p): sha256(p) for p in files})
    return spec, entries, mapping, bindings

def check_pair(original, pair, row):
    """Check actual outcomes before applying the live-query supervision view."""
    for key in ('group_id', 'partition', 'gains'):
        np.testing.assert_array_equal(pair[key], original[key])
    np.testing.assert_array_equal(pair['indices'], row['source']['indices'])
    if not np.isin(original['partition'], ('train', 'validation')).all():
        raise ValueError('Held-out role in reflected training pair')
    if original['features'].shape[1:] != (12, 18) or original['gains'].shape[1:] != (64, 2):
        raise ValueError('Expected original graph18 and 32 paired gains')
    if not original['event_mask'].all():
        raise ValueError('An original factual event was censored')
    for key in (*OUTCOMES, 'features', 'node_mask', 'target', 'target_mask', 'events'):
        suffix = 'event' if key == 'events' else key
        np.testing.assert_array_equal(pair['original_' + suffix], original[key])
    signs = np.ones(18, np.float32); signs[[4, 6, 9, 17]] = -1
    np.testing.assert_allclose(pair['reflected_features'], original['features'] * signs, atol=2e-6, rtol=0)
    np.testing.assert_array_equal(pair['reflected_node_mask'], original['node_mask'])
    for orientation in ('original', 'reflected'):
        s = pair[orientation + '_status']
        mask = np.stack((np.isin(s, (1, 2, 4)), np.isin(s, (1, 4))), -1)
        target = np.where(mask, np.stack((-np.minimum(pair[orientation + '_min_clearance'], .6) / .3,
                            pair[orientation + '_progress'] / 12), -1), 0.).astype(np.float32)
        event = np.stack((s == 2, np.isin(s, (3, 5, 8))), -1).astype(np.float32)
        for name, expected in (('target_mask', mask), ('target', target), ('event', event)):
            np.testing.assert_array_equal(pair[orientation + '_' + name], expected)
        audit = row['audits'][orientation]
        if (not audit['passed'] or audit['branches'] != s.size
                or audit['applied_steps'] != int(pair[orientation + '_steps'].sum())):
            raise ValueError('Incomplete physical outcome audit')

def load_pairs(dataset, additional, base_job, shared_job, train, validation):
    """Return mirrors in EXACT load_dataset + expansion order for each role."""
    data, extra = Path(dataset).resolve(), Path(additional).resolve()
    if (data / 'INVALIDATED.json').exists() or (extra / 'INVALIDATED.json').exists():
        raise ValueError('Invalidated original dataset')
    manifest = read(data / 'manifest.json')
    if (manifest['schema'] != 'unicycle_local_observation_learning_v1'
            or manifest['pilot'] or not manifest['weight_fit_authorized']
            or manifest['graph_features'] != 18 or manifest['queries'] != 32 or manifest['replicas'] != 2
            or 'candidate_bank_contract' in manifest):
        raise ValueError('Only the original complete local unicycle gain bank is qualified')
    supplied = dict(train=train, validation=validation)
    reflected = {role: {k: np.empty_like(v) for k, v in raw.items()} for role, raw in supplied.items()}
    offsets = dict(train=0, validation=0); bindings = {}; populations = []
    for donor, job, observations in ((data, base_job, 24576), (extra, shared_job, 36864)):
        spec, entries, mapping, bound = reviewed_job(job, donor, observations)
        if Path(spec['dataset']).resolve() != data:
            raise ValueError('Different primary reflection population')
        if donor == extra and Path(spec['reflected_base_job']).resolve() != Path(base_job).resolve():
            raise ValueError('Shared reflection has a different base qualification')
        bindings.update(bound); count = 0
        # Use source index order, never lane order. Source rows must match every
        # caller array before the freshly simulated reflected outcomes attach.
        for entry in entries:
            path, row = mapping[entry['file']]
            if row['source'] != entry or sha256(path) != row['sha256'] or sha256(donor / entry['file']) != entry['sha256']:
                raise ValueError('Changed or mismatched observation pair')
            with np.load(donor / entry['file'], allow_pickle=False) as z:
                original = {k: z[k] for k in z.files}
            with np.load(path, allow_pickle=False) as z:
                pair = {k: z[k] for k in z.files if not any('_' + model + '_' in k for model in ('gat', 'nearest_fc'))}
            check_pair(original, pair, row)
            mirror = dict(original)
            for key in (*OUTCOMES, 'features', 'node_mask', 'target', 'target_mask', 'events'):
                mirror[key] = pair['reflected_' + ('event' if key == 'events' else key)]
            mirror['initial_state'] = original['initial_state'] * np.array([1, -1, -1, 1], np.float32)
            for role, raw in supplied.items():
                keep = original['partition'] == role; n = int(keep.sum()); sl = slice(offsets[role], offsets[role] + n)
                if set(original) != set(raw):
                    raise ValueError('Unexpected original data fields')
                for key in raw:
                    np.testing.assert_array_equal(raw[key][sl], original[key][keep])
                    if original[key].dtype != mirror[key].dtype or original[key].shape != mirror[key].shape:
                        raise ValueError('Changed reflected layout: ' + key)
                    reflected[role][key][sl] = mirror[key][keep]
                offsets[role] += n
            count += len(original['group_id'])
        if count != observations:
            raise ValueError('Incomplete paired observation population')
        populations.append(dict(donor=str(donor), reflection_job=str(Path(job).resolve()), observations=count))
    for role, parents in (('train', 576), ('validation', 192)):
        raw = supplied[role]
        if (offsets[role] != len(raw['group_id']) or offsets[role] != parents * 80
                or len(set(raw['group_id'])) != parents or not np.all(raw['partition'] == role)
                or set(Counter(raw['group_id']).values()) != {80}):
            raise ValueError('Changed parent/role denominator')
    if set(train['group_id']) & set(validation['group_id']):
        raise ValueError('Parent split leakage')
    proof = dict(schema=SCHEMA, original_dataset=str(data), additional_dataset=str(extra), bindings=bindings,
        populations=populations, independent_training_parents=576, independent_validation_parents=192,
        training_observations=46080, validation_observations=15360, orientations_per_observation=2,
        additional_independent_parents=0, copied_targets=False, forward_parents_used=False,
        sampling='One Bernoulli(0.5) orientation per sampled observation and optimizer step, shared across all 64 gain/future branches.',
        seed_rule='fold_in(PRNGKey(member_seed+47001), optimizer_step)',
        normalization='Unchanged original base+shared TRAIN-only target normalization.',
        weighting='Unchanged original live-parent bootstrap; both views share one parent and observation weight.',
        validation='Original validation only for checkpoint selection; reflected validation is secondary paired evidence.',
        original_training_order_sha256=hashlib.sha256('\n'.join(train['group_id']).encode()).hexdigest())
    return reflected['train'], reflected['validation'], proof

def pack(original, reflected):
    if set(original) != set(KEYS) or set(reflected) != set(KEYS):
        raise ValueError('Paired local augmentation requires the original prediction heads')
    if any(original[k].shape != reflected[k].shape or original[k].dtype != reflected[k].dtype for k in KEYS):
        raise ValueError('Paired batch layout differs')
    return dict(original, **{PREFIX + k: reflected[k] for k in KEYS})

def select_orientation(batch, seed, step):
    import jax
    import jax.numpy as jnp
    choice = jax.random.bernoulli(jax.random.fold_in(jax.random.PRNGKey(seed), step), .5,
                                (batch['features'].shape[0],))
    return {key: jnp.where(choice.reshape((-1,) + (1,) * (batch[key].ndim - 1)),
                          batch[PREFIX + key], batch[key]) for key in KEYS}

def check_proof(proof):
    if (proof['schema'] != SCHEMA or proof['copied_targets'] or proof['forward_parents_used']
            or proof['additional_independent_parents'] != 0
            or proof['independent_training_parents'] != 576 or proof['independent_validation_parents'] != 192):
        raise ValueError('Unknown or leaky reflection training evidence')
    for path, digest in proof['bindings'].items():
        if sha256(path) != digest:
            raise ValueError('Changed augmentation evidence: ' + path)

def bind_export(members, bundle):
    from .io import write_json
    settings = [read(Path(p) / 'settings.json') for p in members]
    proofs = [s['local_unicycle_reflection'] for s in settings]
    if any(p != proofs[0] for p in proofs):
        raise ValueError('Mixed reflection treatments in an ensemble')
    check_proof(proofs[0])
    seeds = [s['local_unicycle_reflection_seed'] for s in settings]
    if seeds != [s['seed'] + 47001 for s in settings]:
        raise ValueError('Changed paired orientation seed rule')
    root = Path(bundle); manifest = read(root / 'manifest.json')
    if manifest['dataset_manifest_sha256'] != settings[0]['dataset_manifest_sha256']:
        raise ValueError('Different primary dataset during reflection export')
    manifest.update(local_unicycle_reflection=proofs[0], local_unicycle_reflection_member_seeds=seeds,
                    training_settings_sha256=[sha256(Path(p) / 'settings.json') for p in members])
    write_json(root / 'manifest.json', manifest)


import math


import jax.numpy as jnp

def stop_contrast_contract(weight=1.):
    if not math.isfinite(weight) or weight != 1.:
        raise ValueError('The registered stop-contrast experiment uses weight one')
    return dict(schema='local_unicycle_stop_probability_contrast_v1', weight=weight,
        target='controller_stop', event_index=1, replicas=2,
        objective='MSE of gain-centered predicted probability and paired-replica event frequency.',
        query_weight='Original live-observation parent-bootstrap weights; all flat and nonflat queries retained.',
        masks='Every gain/replica event must be observed; already absorbing queries remain masked.',
        primary_losses='Original Gaussian NLL and collision/controller-stop BCE unchanged.',
        checkpoint_selection='Original validation loss plus declared contrast MSE.',
        horizon_seconds=8., adaptation_seconds=.2, runtime_changed=False)

def validate_population(data, replicas):
    from .training import validate_contrast_layout
    validate_contrast_layout(data, replicas)
    if replicas != 2 or data['events'].shape[-1] != 2:
        raise ValueError('Expected two paired futures and original two event heads')
    live = np.asarray(data['prior_status']) == 0
    mask = np.asarray(data['event_mask'])[..., 1]
    if not np.array_equal(mask, np.broadcast_to(live[:, None], mask.shape)):
        raise ValueError('Every live gain/replica stop outcome must remain supervised')
    values = np.asarray(data['events'])[live, :, 1]
    if not np.isin(values, (0., 1.)).all():
        raise ValueError('Stop outcomes must be factual binary events')

def loss(probability, events, observed, parent_weight, replicas=2):
    """Equivalent to half the mean squared error of all pairwise contrasts.

    Average replicas before centering so they do not become independent scene
    weight. All-gain masks are justified by validate_population, never by which
    gain succeeds. False variation on truly flat targets is also penalized.
    """
    if (probability.ndim != 2 or probability.shape != events.shape
            or probability.shape != observed.shape
            or parent_weight.shape != probability.shape[:1]
            or replicas < 1 or probability.shape[1] % replicas):
        raise ValueError('Invalid stop-contrast tensor layout')
    valid = jnp.all(observed, axis=1)
    # Mask before arithmetic, including for wholly unobserved absorbing rows.
    p = jnp.where(observed, probability, 0.).reshape(len(probability), -1, replicas).mean(-1)
    y = jnp.where(observed, events, 0.).reshape(len(probability), -1, replicas).mean(-1)
    residual = (p-p.mean(-1, keepdims=True))-(y-y.mean(-1, keepdims=True))
    weight = parent_weight*valid
    return jnp.sum(jnp.mean(residual**2, axis=1)*weight)/jnp.maximum(jnp.sum(weight), 1.)


CONTRACT = dict(schema='local_unicycle_live_query_supervision_v1',
    population='Acquisition is live before the prospective gain query (prior_status == 0).',
    terminal_observations='Retained, with zero supervised target weight; never revived.',
    live_failure_targets='All observed collision, QP, admissibility and bound failures retained.',
    horizon_seconds=8., adaptation_seconds=.2,
    weighting='Equal physical-parent weight shared over live observations; parent bootstrap for training.')

def live_query_view(data):
    prior = np.asarray(data['prior_status'])
    if prior.shape != (len(data['group_id']),) or prior.dtype.kind not in 'iu':
        raise ValueError('One integer prior status per observation is required')
    if not np.isin(prior, (0, 1, 2, 3, 4, 5, 8)).all():
        raise ValueError('Unknown acquisition status')
    live = prior == 0
    result = dict(data)
    for key, values in (('target_mask', 'target'), ('event_mask', 'events')):
        mask = np.asarray(data[key])
        if mask.dtype != bool or mask.ndim != 3 or mask.shape != data[values].shape:
            raise ValueError('Invalid prospective target mask')
        result[key] = mask & live[:, None, None]
    proof = dict(contract=CONTRACT, observations=len(prior), live_observations=int(live.sum()),
        already_terminal_observations=int((~live).sum()),
        excluded_absorbing_stop_labels=int(data['event_mask'][~live, :, 1].sum()),
        live_stop_targets_before=int((data['events'][live, :, 1] * data['event_mask'][live, :, 1]).sum()),
        live_stop_targets_after=int((result['events'][live, :, 1] * result['event_mask'][live, :, 1]).sum()),
        physical_parent_count=len(set(data['group_id'])), outcome_rows_removed=0)
    if proof['live_stop_targets_before'] != proof['live_stop_targets_after']:
        raise ValueError('A genuine live-query failure was removed')
    # Validate that no parent silently disappears from the supervised population.
    parent_live_weights(data['group_id'], live)
    return result, proof

def parent_live_weights(groups, live, multiplicity=None):
    """Parent weight does not depend on how many absorbing visits it produced."""
    groups = np.asarray(groups); live = np.asarray(live, bool)
    if groups.ndim != 1 or live.shape != groups.shape:
        raise ValueError('Invalid live-parent mask')
    ids, inverse = np.unique(groups, return_inverse=True)
    count = np.bincount(inverse, weights=live, minlength=len(ids))
    if np.any(count == 0):
        raise ValueError('Every reserved parent must retain at least one live observation')
    multiplier = np.ones(len(ids)) if multiplicity is None else np.asarray(multiplicity)
    if multiplier.shape != ids.shape or not np.isfinite(multiplier).all() or np.any(multiplier < 0):
        raise ValueError('Invalid parent bootstrap multiplicities')
    return np.where(live, multiplier[inverse] / count[inverse], 0.).astype(np.float32)
