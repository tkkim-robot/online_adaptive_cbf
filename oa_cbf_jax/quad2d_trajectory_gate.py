"""Frozen-policy trajectory disagreement calibration, with no safety claim.

Scores include every queried bank candidate and the previous gain, including
rejected decisions. Observation reconstruction never reads the latent state.
The new gate changes the policy: coverage on its trajectories is an independent
empirical audit, not inherited exchangeability with the calibration policy.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from .dataset import sha256
from .io import write_json

SCHEMA = 'oa_cbf_frozen_trajectory_gate_v1'
SCORE_SCHEMA = 'quad2d_query_pool_trajectory_cs_v1'
FRESH_SCHEMA = 'oa_cbf_fresh_flight_predictive_calibration_v61'
BASE_SCHEMAS = ('oa_cbf_development_calibration_v1', FRESH_SCHEMA)
FRESH_BINDINGS = ('fresh_dataset', 'fresh_dataset_manifest_sha256',
    'fresh_dataset_index_sha256', 'fresh_dataset_audit_sha256', 'training_dataset',
    'event_statistic', 'delayed_calibration_valid', 'candidates', 'label_replicas')


def read(path):
    return json.loads(Path(path).read_text())


def query_contexts(data, config):
    """Recover pre-query memories by replaying only accepted command records."""
    if 'waypoint_index' in data or 'forecast_obstacles' in data:
        raise ValueError('This contract covers raw graph40 single-goal flight only')
    cursor = np.float32(0)
    previous_u = np.full(2, config.robot.mass*config.robot.gravity/2, np.float32)
    previous_gain = np.full(2, 4., np.float32)
    for k in range(len(data['active'])):
        if data['requery'][k]:
            yield k, (data['observed_state'][k], data['goal'], data['observed_obstacles'][k],
                data['obstacle_mask'], data['points'], data['route_mask'], cursor,
                previous_u.copy(), previous_gain.copy(), data['noise'])
        if data['active'][k]:
            cursor = data['route_progress'][k]
            previous_u = data['control'][k]
            previous_gain = data['gain'][k]


def numpy_cs(mean, variance):
    """Independent scalar Gaussian pair formula; ensemble is the first axis."""
    m, v = np.moveaxis(np.asarray(mean, float), 0, -1), np.moveaxis(np.asarray(variance, float), 0, -1)
    a, b = v[..., :, None], v[..., None, :]
    d = m[..., :, None] - m[..., None, :]
    return np.maximum((.25*np.log((a+b)**2/(4*a*b))+.5*d*d/(a+b)).mean(axis=(-2, -1)), 0.)


def check_live_statistics(live, stages, calibration, policy):
    """Audit the saved values consumed by the decision, including boundaries."""
    mean, variance, cs, risk, event = (np.asarray(live[k]) for k in ('mean','variance','cs','risk','event'))
    if (mean.ndim!=3 or mean.shape[1:]!=(33,2) or variance.shape!=mean.shape
            or cs.shape!=(33,) or risk.shape!=(33,) or event.shape!=(33,2)
            or not all(np.isfinite(a).all() for a in (mean,variance,cs,risk,event))
            or np.any(variance<=0) or np.any((event<0)|(event>1))):
        raise ValueError('Invalid actual query statistics')
    independent = numpy_cs(mean[...,0],variance[...,0])
    np.testing.assert_allclose(cs, independent, atol=2e-6, rtol=2e-5)
    from scipy.stats import norm
    coefficient = norm.pdf(norm.isf(policy.tail_mass))/policy.tail_mass
    reference_risk = np.max(mean[...,0].astype(float)+coefficient*np.sqrt(variance[...,0].astype(float)),axis=0)
    np.testing.assert_allclose(risk, reference_risk, atol=2e-5, rtol=2e-5)
    finite=np.ones(33,bool)
    ep=finite & (cs<=np.float32(calibration['cs_threshold']))
    risk_ok=ep & (risk<=np.float32(policy.risk_threshold))
    admitted=risk_ok & (event[:,0]<=np.float32(policy.collision_probability_limit)) & (event[:,1]<=np.float32(policy.failure_probability_limit))
    actual=np.array([x.sum() for x in (finite,ep,risk_ok,admitted)])
    if not np.array_equal(actual,np.asarray(stages)[:4]):
        raise ValueError('Saved live statistics disagree with the actual query stages')
    return float(np.max(np.abs(cs-independent)))


def score(directory, bundle, calibration, output, batch=128):
    """AOT replay of all recorded query graphs; CPU independently checks graphs/CS."""
    import jax
    import jax.numpy as jnp
    from .models import predict_ensemble
    from .quad2d_control import FlightConfig
    from .quad2d_features import flight_graph
    from .quad2d_audit import check_observed_graph
    from .quad2d_policy import FlightPolicy, FlightPolicyConfig
    from .quad2d_guidance import TerminalGuidanceConfig
    from .uncertainty import cs_disagreement, worst_member_cvar

    root, dest = Path(directory), Path(output)
    dest.mkdir(parents=True, exist_ok=False)
    manifest, index = read(root/'manifest.json'), read(root/'index.json')
    live_statistics = manifest.get('query_statistics_schema') == 'quad2d_live_query_statistics_v1'
    audit = read(root/'independent_replay.json')
    config = FlightConfig()
    if (not audit['audit_passed'] or audit['manifest_sha256'] != sha256(root/'manifest.json')
            or audit['index_sha256'] != sha256(root/'index.json') or manifest['config'] != asdict(config)
            or manifest['calibration_sha256'] != sha256(calibration) or manifest['steps'] != 1600
            or manifest['predictive_guidance'] != json.loads(json.dumps(asdict(TerminalGuidanceConfig(noise_clearance_weight=1.))))):
        raise ValueError('Matched audited terminal flight experiment required')
    policy_config = FlightPolicyConfig(**manifest['policy'])
    policy = FlightPolicy(bundle, calibration, config, policy_config, TerminalGuidanceConfig(noise_clearance_weight=1.))
    if policy.metadata['graph_features'] != 40 or policy.metadata['weights_sha256'] != manifest['model_weights_sha256']:
        raise ValueError('Matched graph40 weights required')
    candidates = np.asarray(manifest['candidates'], np.float32)
    if candidates.shape != (32, 2):
        raise ValueError('This frozen score contract requires the V42 32-gain pool')
    norm = policy.norm
    target_scale, target_mean = (jnp.asarray(norm[k], jnp.float32) for k in ('target_scale', 'target_mean'))
    def evaluate(params, cal, bank, *context):
        features, mask = jax.vmap(lambda *x: flight_graph(*x, config))(*context)
        previous = context[8]
        pools = jnp.concatenate((jnp.broadcast_to(bank, (batch, *bank.shape)), previous[:, None]), axis=1)
        out = predict_ensemble(policy.model, params, features, mask, pools)
        mean = out['mean']*target_scale+target_mean
        variance = jnp.exp(out['log_variance'])*target_scale**2*cal['variance_scale']
        cs = cs_disagreement(jnp.transpose(mean[..., :1], (1, 2, 0, 3)), jnp.transpose(variance[..., :1], (1, 2, 0, 3)))
        risk = worst_member_cvar(jnp.transpose(mean[..., 0], (1, 2, 0)), jnp.transpose(variance[..., 0], (1, 2, 0)), policy_config.tail_mass)
        event = jnp.max(jax.nn.sigmoid(out['event_logits']/cal['temperature']+cal['bias']), axis=0)
        finite = jnp.all(jnp.isfinite(mean)&jnp.isfinite(variance), axis=(0, 3)) & jnp.all(jnp.isfinite(event), axis=-1) & jnp.isfinite(cs) & jnp.isfinite(risk)
        ep = finite & (cs <= cal['cs_threshold'])
        risk_ok = ep & (risk <= policy_config.risk_threshold)
        admitted = risk_ok & (event[..., 0] <= policy_config.collision_probability_limit) & (event[..., 1] <= policy_config.failure_probability_limit)
        stages = jnp.stack([a.sum(axis=1) for a in (finite, ep, risk_ok, admitted)], axis=1)
        return dict(features=features, node_mask=mask, mean=mean[..., 0], variance=variance[..., 0], cs=cs, stages=stages,
            all_mean=mean, all_variance=variance, risk=risk, event=event)
    dummy = (np.zeros((batch, 6), np.float32), np.ones((batch, 2), np.float32), np.zeros((batch, 64, 5), np.float32),
        np.zeros((batch, 64), bool), np.zeros((batch, 64, 2), np.float32), np.ones((batch, 64), bool),
        np.zeros(batch, np.float32), np.full((batch, 2), 4.905, np.float32), np.full((batch, 2), 4., np.float32), np.zeros((batch, 7), np.float32))
    args = (policy.params, policy.calibration, jnp.asarray(candidates), *[jnp.asarray(a) for a in dummy])
    t = time.perf_counter()
    executable = jax.jit(evaluate).lower(*args).compile()
    jax.block_until_ready(executable(*args))
    compile_seconds = time.perf_counter()-t
    records, query_ids, query_ticks, query_scores, pending = [], [], [], [], []
    stage_differences, max_cs_error, route_ties = 0, 0., 0
    replay_differences=[]; max_live_replay_cs_error=0.
    def flush():
        nonlocal stage_differences, max_cs_error, route_ties, max_live_replay_cs_error
        if not pending:
            return
        padded = pending+[pending[-1]]*(batch-len(pending))
        contexts = tuple(np.stack([p[2][i] for p in padded]) for i in range(10))
        result = jax.device_get(executable(policy.params, policy.calibration, jnp.asarray(candidates), *map(jnp.asarray, contexts)))
        independent = numpy_cs(result['mean'], result['variance'])
        np.testing.assert_allclose(result['cs'], independent, atol=2e-6, rtol=2e-5)
        max_cs_error = max(max_cs_error, float(np.max(np.abs(result['cs']-independent))))
        if not np.isfinite(result['cs']).all():
            raise ValueError('Nonfinite candidate CS; cannot produce a usable calibrated gate')
        for i, (parent, tick, context, stages, live) in enumerate(pending):
            x, goal, obs, mask, points, rm, cursor, u, gain, noise = context
            tie = check_observed_graph(result['features'][i], result['node_mask'][i], x, goal, obs, mask, points, rm, noise, config, cursor, u, gain)
            route_ties += tie is not None
            mismatch=not np.array_equal(result['stages'][i],stages[:4])
            if mismatch:
                replay_differences.append(dict(group_id=parent,tick=int(tick),recorded=stages[:4].tolist(),replayed=result['stages'][i].tolist()))
            if live is None:
                # Old traces lack the live values. Boundary disagreement remains
                # fatal; do not silently relabel them with replayed statistics.
                stage_differences += int(mismatch)
                values=result['cs'][i]
            else:
                max_cs_error=max(max_cs_error,check_live_statistics(live,stages,policy.calibration,policy_config))
                for key,replayed in [('mean',result['all_mean'][:,i]),('variance',result['all_variance'][:,i]),
                        ('risk',result['risk'][i]),('event',result['event'][i])]:
                    np.testing.assert_allclose(live[key],replayed,atol=2e-5,rtol=2e-5)
                # CS is nonlinear in the model statistics: two nearby network
                # outputs need not differ by the scalar formula's roundoff
                # tolerance. Both CS calculations are checked independently
                # against their OWN mean/variance inputs above. Mean/variance
                # replay remains checked; never replace the actual CS here.
                max_live_replay_cs_error=max(max_live_replay_cs_error,float(np.max(np.abs(live['cs']-result['cs'][i]))))
                values=live['cs']  # Exactly the values used during this flight.
            query_ids.append(parent); query_ticks.append(tick); query_scores.append(values.copy())
        pending.clear()
    t = time.perf_counter()
    for row in index:
        if row['sha256'] != sha256(root/row['file']):
            raise ValueError('Changed source trajectory')
        with np.load(root/row['file']) as f:
            fields = ('active', 'requery', 'observed_state', 'goal', 'observed_obstacles', 'obstacle_mask', 'points', 'route_mask', 'route_progress', 'control', 'gain', 'noise', 'stages')
            if 'waypoint_index' in f or 'forecast_obstacles' in f:
                raise ValueError('Unsupported trajectory memory contract')
            data = {k: f[k] for k in fields}
            stats = {k:f['guidance_query_'+k] for k in ('mean','variance','cs','risk','event')} if live_statistics else None
        n = 0
        for tick, context in query_contexts(data, config):
            live={k:v[tick] for k,v in stats.items()} if stats is not None else None
            pending.append((row['group_id'], tick, context, data['stages'][tick],live)); n += 1
            if len(pending) == batch:
                flush()
        records.append(dict(group_id=row['group_id'], family=row['family'], queries=n, status_code=row['status_code']))
    flush()
    arrays = dict(group_id=np.asarray(query_ids), tick=np.asarray(query_ticks), candidate_cs=np.asarray(query_scores, np.float32).reshape(-1, 33))
    np.savez_compressed(dest/'query_scores.npz', **arrays)
    maxima = {r['group_id']: 0. for r in records}
    for parent, values in zip(query_ids, query_scores):
        maxima[parent] = max(maxima[parent], float(np.max(values)))
    for row in records:
        row['maximum_cs'] = maxima[row['group_id']]
    result = dict(schema=SCORE_SCHEMA, source=str(root.resolve()), manifest_sha256=sha256(root/'manifest.json'), index_sha256=sha256(root/'index.json'),
        physical_audit_sha256=sha256(root/'independent_replay.json'), weights_sha256=manifest['model_weights_sha256'],
        calibration_sha256=sha256(calibration), policy=manifest['policy'], predictive_guidance=manifest['predictive_guidance'],
        config=manifest['config'], candidates=manifest['candidates'], steps=1600, records=records,
        scores_sha256=sha256(dest/'query_scores.npz'), compile_seconds=compile_seconds, scoring_seconds=time.perf_counter()-t,
        compiled_signatures=1, all_query_graphs_independently_checked=True, all_query_cs_independently_checked=True,
        maximum_independent_cs_error=max_cs_error, route_roundoff_ties=route_ties, query_stage_differences=stage_differences,
        statistic_source='actual_policy_values' if live_statistics else 'exact_stage_replay',
        maximum_live_replay_cs_error=max_live_replay_cs_error, replay_stage_differences=replay_differences,
        audit_passed=stage_differences == 0, interpretation='Maximum over all33 candidate scores at every actual requery until first terminal event. Rejected queries included; empty-query episode score0. No post-stop future or safety claim.')
    write_json(dest/'scores.json', result)
    if stage_differences:
        raise ValueError(f'{stage_differences} replayed query stage counts differ from live policy; inspect before calibration')
    print(json.dumps({k: result[k] for k in ('compile_seconds', 'scoring_seconds', 'query_stage_differences', 'audit_passed')}), flush=True)


def checked_scores(directory):
    root = Path(directory); report = read(root/'scores.json')
    if report['schema'] != SCORE_SCHEMA or not report['audit_passed'] or report['scores_sha256'] != sha256(root/'query_scores.npz'):
        raise ValueError('Unaudited or changed trajectory scores')
    source = Path(report['source'])
    for name, key in [('manifest.json', 'manifest_sha256'), ('index.json', 'index_sha256'), ('independent_replay.json', 'physical_audit_sha256')]:
        if sha256(source/name) != report[key]:
            raise ValueError('Changed trajectory score provenance')
    with np.load(root/'query_scores.npz') as f:
        ids, values = f['group_id'], f['candidate_cs']
    if values.shape != (len(ids), 33) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Invalid complete candidate pool scores')
    if len({r['group_id'] for r in report['records']}) != len(report['records']):
        raise ValueError('Duplicate trajectory parent')
    if set(ids)-{r['group_id'] for r in report['records']}:
        raise ValueError('Unaccounted query parent')
    for r in report['records']:
        selected = values[ids == r['group_id']]
        if len(selected) != r['queries'] or float(np.max(selected, initial=0.)) != r['maximum_cs']:
            raise ValueError('Trajectory maximum or query count changed')
    return report


def family_thresholds(records, coverage=.95):
    from .uncertainty import conformal_threshold
    from .scenes import DIVERSE_FAMILIES
    if set(r['family'] for r in records) != set(DIVERSE_FAMILIES) or len({r['group_id'] for r in records}) != len(records):
        raise ValueError('All eight unique-parent family strata required')
    thresholds = {}
    for family in DIVERSE_FAMILIES:
        scores = [r['maximum_cs'] for r in records if r['family'] == family]
        if len(scores) < 50:
            raise ValueError('At least50 independent trajectory parents per family required')
        thresholds[family] = conformal_threshold(scores, coverage)
        if thresholds[family]['status'] != 'calibrated':
            raise ValueError('Insufficient finite-sample trajectory calibration')
    return thresholds


def check_fresh_lineage(base, reports):
    """Reserve full parent lineages, including predictive fit/gate/audit roles."""
    if base['schema'] != FRESH_SCHEMA:
        return
    if base.get('event_statistic') != 'maximum_member_probability' or base.get('delayed_calibration_valid') is not False:
        raise ValueError('Changed fresh instantaneous calibration semantics')
    training, fresh = Path(base['training_dataset']), Path(base['fresh_dataset'])
    for path, key in [(training/'manifest.json', 'dataset_manifest_sha256'),
            (fresh/'manifest.json', 'fresh_dataset_manifest_sha256'),
            (fresh/'index.json', 'fresh_dataset_index_sha256'),
            (fresh/'independent_replay.json', 'fresh_dataset_audit_sha256')]:
        if sha256(path) != base[key]:
            raise ValueError('Changed fresh predictive calibration lineage')
    excluded = read(training/'manifest.json')['groups'] + read(fresh/'manifest.json')['groups']
    forbidden_ids = {r['group_id'] for r in excluded}
    forbidden_seeds = {r['seed'] for r in excluded}
    seen_ids, seen_seeds = set(), set()
    for report in reports:
        if report.get('statistic_source') != 'actual_policy_values':
            raise ValueError('Fresh trajectory calibration requires actual policy query statistics')
        episode = read(Path(report['source'])/'manifest.json')
        source = Path(episode['source']); manifest = read(source/'manifest.json')
        if (sha256(source/'manifest.json') != episode['source_manifest_sha256']
                or manifest.get('trajectory_gate_role') != 'calibration'
                or manifest.get('training_use') is not False
                or manifest.get('weight_fit_authorized') is not False
                or manifest.get('frozen_predictive_calibration_sha256') != report['calibration_sha256']
                or manifest['scenes_sha256'] != sha256(source/'scenes.json')):
            raise ValueError('Fresh independent trajectory reservation required')
        parents = read(source/'scenes.json')
        ids = [r['group_id'] for r in parents]; seeds = [r['seed'] for r in parents]
        if (ids != [r['group_id'] for r in report['records']]
                or len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds)
                or (set(ids) & (forbidden_ids | seen_ids))
                or (set(seeds) & (forbidden_seeds | seen_seeds))):
            raise ValueError('Predictive/training/trajectory parent-lineage overlap or mismatch')
        seen_ids.update(ids); seen_seeds.update(seeds)
        if report['candidates'] != base['candidates']:
            raise ValueError('Fresh trajectory candidate bank changed')


def fit(scores, base_calibration, output, coverage=.95):
    base = read(base_calibration)
    if base['schema'] not in BASE_SCHEMAS:
        raise ValueError('A frozen original predictive calibration is required')
    reports = [checked_scores(path) for path in scores]
    reference = reports[0]
    for report in reports:
        episode_manifest = read(Path(report['source'])/'manifest.json')
        source = Path(episode_manifest['source']); source_manifest = read(source/'manifest.json')
        if (sha256(source/'manifest.json') != episode_manifest['source_manifest_sha256']
                or source_manifest.get('trajectory_gate_role') != 'calibration' or source_manifest.get('training_use') is not False):
            raise ValueError('Prospectively reserved trajectory calibration parents required')
        for key in ('weights_sha256', 'calibration_sha256', 'policy', 'predictive_guidance', 'config', 'candidates', 'steps'):
            if report[key] != reference[key]:
                raise ValueError('Calibration trajectories used different frozen policies')
    if reference['calibration_sha256'] != sha256(base_calibration) or reference['weights_sha256'] != base['weights_sha256']:
        raise ValueError('Wrong frozen predictive transform')
    check_fresh_lineage(base, reports)
    records = [row for report in reports for row in report['records']]
    thresholds = family_thresholds(records, coverage)
    result = deepcopy(base)
    result.update(schema=SCHEMA, stage='frozen_policy_trajectory_disagreement_pilot', production_eligible=False,
        predictive_calibration_schema=base['schema'],
        cs_gate=dict(threshold=max(t['threshold'] for t in thresholds.values()), status='calibrated', coverage=coverage,
            n_groups=len(records), aggregation='maximum_of_eight_within_family_split_conformal_thresholds', by_family=thresholds),
        trajectory_gate=dict(base_calibration=str(Path(base_calibration).resolve()), base_calibration_sha256=sha256(base_calibration),
            score_reports=[dict(directory=str(Path(p).resolve()), sha256=sha256(Path(p)/'scores.json')) for p in scores],
            policy=reference['policy'], predictive_guidance=reference['predictive_guidance'], candidates=reference['candidates'], steps=reference['steps'],
            statistic_source=reference.get('statistic_source','exact_stage_replay'),
            group_ids=[r['group_id'] for r in records]),
        interpretation='One scalar gate is the maximum of family-specific finite-sample thresholds for whole-trajectory maximum CS over all queried candidates. Under exchangeable independent parents within a family, the frozen original rollout policy has marginal disagreement coverage. Changing the gate changes the trajectory distribution: new-policy coverage must be audited independently. No OOD, event-probability, physical-tail or collision safety guarantee.',
        required_before_final_claim='Predictive/event adequacy (including rare positive events), independent changed-policy trajectory audit, locked ID/OOD and live-delay evaluation remain required.')
    # Preserve old diagnostics with an explicit name; never mislabel them as
    # measurements of this new trajectory gate.
    result['frozen_observation_calibration_diagnostics'] = result.pop('diagnostics')
    result['frozen_observation_calibration_group_ids'] = result.pop('group_ids')
    path = Path(output); path.mkdir(parents=True, exist_ok=False)
    write_json(path/'calibration.json', result)
    print(json.dumps(result['cs_gate']), flush=True)


def validate_runtime(info, policy, guidance):
    """Bind changed gate to its frozen predictive transform and scored policy."""
    contract = info['trajectory_gate']; base = read(contract['base_calibration'])
    if sha256(contract['base_calibration']) != contract['base_calibration_sha256'] or base['schema'] not in BASE_SCHEMAS:
        raise ValueError('Changed trajectory gate base calibration')
    if info.get('predictive_calibration_schema', 'oa_cbf_development_calibration_v1') != base['schema']:
        raise ValueError('Changed trajectory predictive calibration schema')
    for key in ('weights_sha256', 'dataset_manifest_sha256', 'targets', 'events', 'robot', 'controller', 'variance_scale', 'event_calibration', 'gain_domain', 'horizon_steps'):
        if info[key] != base[key]:
            raise ValueError('Trajectory gate changed the frozen predictive contract')
    if base['schema'] == FRESH_SCHEMA:
        if any(info.get(key) != base.get(key) for key in FRESH_BINDINGS):
            raise ValueError('Trajectory gate changed the fresh predictive contract')
    if guidance is None or json.loads(json.dumps(asdict(policy))) != contract['policy'] or json.loads(json.dumps(asdict(guidance))) != contract['predictive_guidance']:
        raise ValueError('Trajectory gate policy changed')
    records, reports = [], []
    for item in contract['score_reports']:
        path = Path(item['directory'])
        if sha256(path/'scores.json') != item['sha256']:
            raise ValueError('Changed trajectory score report')
        report = checked_scores(path)
        if report['calibration_sha256'] != contract['base_calibration_sha256'] or report['weights_sha256'] != info['weights_sha256']:
            raise ValueError('Wrong trajectory calibration rollout')
        for key in ('policy', 'predictive_guidance', 'candidates', 'steps'):
            if report[key] != contract[key]:
                raise ValueError('Trajectory score policy contract changed')
        if report.get('statistic_source','exact_stage_replay') != contract.get('statistic_source','exact_stage_replay'):
            raise ValueError('Trajectory statistic source changed')
        records.extend(report['records'])
        reports.append(report)
    check_fresh_lineage(base, reports)
    thresholds = family_thresholds(records, info['cs_gate']['coverage'])
    if ([r['group_id'] for r in records] != contract['group_ids'] or info['cs_gate']['by_family'] != thresholds
            or info['cs_gate']['threshold'] != max(t['threshold'] for t in thresholds.values())):
        raise ValueError('Trajectory threshold or group lineage changed')
    return contract


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('score')
    for key in ('directory', 'bundle', 'calibration', 'output'):
        s.add_argument('--'+key, required=True)
    s.add_argument('--batch', type=int, default=128)
    f = sub.add_parser('fit'); f.add_argument('--scores', nargs='+', required=True)
    f.add_argument('--base-calibration', required=True); f.add_argument('--output', required=True); f.add_argument('--coverage', type=float, default=.95)
    args = vars(p.parse_args()); command = args.pop('command')
    (score if command == 'score' else fit)(**args)
