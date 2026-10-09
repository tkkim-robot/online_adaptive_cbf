"""Nearest fc functions and shared contracts."""

import math

import jax.numpy as jnp

SCHEMA = 'nearest_obstacle_fc_inputs_v2'

SHAPES = {'unicycle': (18, 31), 'bicycle': (35,), 'quad2d': (40,), 'quad3d': (50, 58)}

def from_manifest(manifest):
    columns = manifest['graph_features']
    matches = [name for name, widths in SHAPES.items() if columns in widths]
    if len(matches) != 1:
        raise ValueError('Unregistered nearest-FC observation schema')
    dynamics = matches[0]
    gain_dimension = manifest.get('gain_dimension', 2)
    if gain_dimension != {'unicycle': 2, 'bicycle': 1, 'quad2d': 2, 'quad3d': 4}[dynamics]:
        raise ValueError('Nearest-FC graph/gain dynamics mismatch')
    yaw_scale = manifest['config']['yaw_limit'] if dynamics == 'quad3d' else 1.
    return dynamics, float(yaw_scale)

def contract(dynamics, yaw_scale=1., constraint_features=False):
    if dynamics not in SHAPES or not math.isfinite(yaw_scale) or yaw_scale <= 0:
        raise ValueError('Invalid nearest-FC feature contract')
    velocities = {'unicycle': ['speed'], 'bicycle': ['speed'],
                  'quad2d': ['velocity_x', 'velocity_z'],
                  'quad3d': ['velocity_x', 'velocity_y', 'velocity_z']}[dynamics]
    result=dict(schema=SCHEMA, dynamics=dynamics, neural_obstacles=1,
        nearest='Minimum observed center distance computed in FP64 from observed positions; geometric tie breakers; masked obstacles excluded.',
        context=['signed_surface_clearance', *velocities, 'sin_relative_bearing', 'cos_relative_bearing'],
        candidate='Log of each queried positive class-K gain; same candidate domain as OA.',
        scaling='Dataset fixed physical units: clearance/3 and velocity/declared speed limit.',
        yaw_scale=yaw_scale,
        no_obstacle='Clearance 100/3, bearing sin=0 cos=1; robot velocity retained.',
        hidden_layers='ReLU W, 2W, 3W, W, as original FC/PENN.',
        controller_obstacles='All observed obstacles; restriction applies only to neural prediction.',
        excluded=['goal', 'route', 'history', 'noise_level', 'other_obstacles', 'obstacle_count'],
        quad3d_adaptation='Linearized 3D plant uses three translational velocities, actual yaw, and four queried gains; never altitude as heading.',
        comparison='Paper nearest-obstacle FC baseline; not a full-scene encoder-only ablation.')
    if constraint_features:
        if dynamics!='unicycle':raise ValueError('Coefficient variant requires unicycle')
        from .unicycle_features import contract as coefficient_contract
        result.update(schema='nearest_obstacle_fc_unicycle_coefficients_v1',
            observed_coefficient_features=coefficient_contract(),
            comparison='Augmented nearest-obstacle FC comparator; original FC results remain separate.',
            candidate='Log gains plus observed-coefficient gain basis, identical to the GAT treatment.')
    return result

def validate_metadata(metadata):
    arch = metadata['architecture']
    dynamics = arch.get('nearest_dynamics')
    expected = contract(dynamics, arch.get('nearest_yaw_scale', 1.),arch.get('unicycle_constraint_features',False))
    if (arch['encoder'] != 'nearest_fc' or metadata.get('nearest_fc_contract') != expected
            or metadata.get('graph_features') not in SHAPES[dynamics]
            or metadata.get('gain_dimension', 2) != {'unicycle': 2, 'bicycle': 1, 'quad2d': 2, 'quad3d': 4}[dynamics]):
        raise ValueError('Changed nearest-obstacle FC input contract')
    return expected

def encode(features, mask, dynamics, yaw_scale=1.):
    """Fixed-shape, JIT-compatible nearest selection; no pooled scene context."""
    if dynamics not in SHAPES or features.shape[-1] not in SHAPES[dynamics]:
        raise ValueError('Wrong dynamics-specific nearest-FC graph')
    clean = jnp.where(mask[..., None], features, 0.)
    ego, obstacles, valid = clean[:, 0], clean[:, 2:], mask[:, 2:]
    # A zero-capacity graph also has a well-defined finite empty observation.
    obstacles = jnp.pad(obstacles, ((0, 0), (0, 1), (0, 0)))
    valid = jnp.pad(valid, ((0, 0), (0, 1)))
    # A rounded FP32 sum can turn unequal distances into a tie on CPU, while
    # GPU multiply-add fusion preserves their order. That changes the selected
    # obstacle discontinuously. FP64 products of the FP32 observations retain
    # their exact mantissas; this key does not change the neural compute dtype.
    position = obstacles[..., 3:5].astype(jnp.float64)
    distance2 = jnp.sum(position ** 2, axis=-1)
    order = jnp.lexsort((obstacles[..., 7], obstacles[..., 4], obstacles[..., 3],
                        jnp.where(valid, distance2, jnp.inf)), axis=1)
    nearest = jnp.take_along_axis(obstacles, order[:, :1, None], axis=1)[:, 0]
    present = valid.any(axis=1)
    bearing = jnp.arctan2(nearest[:, 4], nearest[:, 3])
    if dynamics in ('unicycle', 'quad2d'):
        bearing -= jnp.arctan2(ego[:, 9], ego[:, 10])
        velocity = ego[:, 11:12] if dynamics == 'unicycle' else ego[:, 11:13]
    elif dynamics == 'bicycle':
        # Bicycle graph positions are already in the robot-heading frame.
        velocity = ego[:, 9:10]
    else:
        bearing -= ego[:, 20] * yaw_scale
        velocity = ego[:, 21:24]
    clearance = jnp.where(present, nearest[:, 8], 100. / 3.)[:, None]
    angles = jnp.stack((jnp.where(present, jnp.sin(bearing), 0.),
                        jnp.where(present, jnp.cos(bearing), 1.)), axis=-1)
    return jnp.concatenate((clearance, velocity, angles), axis=-1)


import argparse

import json

from pathlib import Path

import re

import time

from flax import serialization

import jax


import numpy as np

from .io import load_dataset, sha256, source_fingerprint

from .inference import ResearchPredictor

from .io import write_json

from .models import predict_ensemble

NEAREST_FC_QUALIFICATION_SCHEMA = 'nearest_fc_numerical_qualification_v1'

def read(path):
    return json.loads(Path(path).read_text())

def backend_platform(report):
    """Use JAX's platform property, with exact legacy display-name support."""
    platform = report.get('platform')
    if platform is not None:
        if platform not in ('cpu', 'gpu'):
            raise ValueError('Unrecognized qualification platform')
        return platform
    display = report['backend']
    if re.fullmatch(r'(TFRT_CPU_|cpu:)\d+', display):
        return 'cpu'
    if re.fullmatch(r'cuda:\d+', display):
        return 'gpu'
    raise ValueError('Unrecognized legacy qualification device')

def numpy_context(features, masks, dynamics, yaw_scale, constraint_features=False):
    result = []
    for f, mask in zip(np.asarray(features), np.asarray(masks), strict=True):
        # Select using only current observed centers, independently of JAX's
        # sorting/gather code and independently of predicted labels.
        indices = [i for i in range(2, len(mask)) if mask[i]]
        indices.sort(key=lambda i: (float(f[i, 3])**2+float(f[i, 4])**2,
                                    float(f[i, 3]), float(f[i, 4]), float(f[i, 7])))
        clearance, angle = 100./3., 0.
        if indices:
            obstacle = f[indices[0]]
            clearance = float(obstacle[8])
            angle = np.arctan2(float(obstacle[4]), float(obstacle[3]))
            if dynamics in ('unicycle', 'quad2d'):
                angle -= np.arctan2(float(f[0, 9]), float(f[0, 10]))
            elif dynamics == 'quad3d':
                angle -= float(f[0, 20])*yaw_scale
        if dynamics == 'unicycle': velocity = [f[0, 11]]
        elif dynamics == 'bicycle': velocity = [f[0, 9]]
        elif dynamics == 'quad2d': velocity = f[0, 11:13]
        else: velocity = f[0, 21:24]
        row=[clearance, *velocity, np.sin(angle), np.cos(angle)]
        if constraint_features:
            if dynamics!='unicycle' or f.shape[-1]!=18:raise ValueError('Wrong coefficient feature graph')
            extra=np.zeros(5)
            if indices:
                o=np.asarray(f[indices[0]],np.float64);ego=np.asarray(f[0],np.float64)
                p=o[3:5]*5.;v=o[5:7]*ego[14]
                direction=np.array([ego[10],ego[9]]);normal=np.array([-ego[9],ego[10]])
                extra=np.array([(p@p-(o[7]+ego[7]+.05)**2)/25.,
                    2.*(p@v)/10.,2.*(v@v)/8.,-2.*(p@direction)/10.,
                    -2.*ego[11]*ego[14]*(p@normal)/20.])
            row.extend(extra)
        result.append(row)
    return np.asarray(result, np.float64)

def check_export(bundle, metadata, params):
    bindings = {str(Path(bundle)/'manifest.json'): sha256(Path(bundle)/'manifest.json'),
                str(Path(bundle)/'weights.msgpack'): sha256(Path(bundle)/'weights.msgpack')}
    for i, member in enumerate(metadata['members']):
        directory = Path(member['directory']); best = read(directory/'best.json')
        settings = read(directory/'settings.json')
        state = directory/'checkpoints'/best['state_file']
        if (best['settings_sha256'] != sha256(directory/'settings.json')
                or best['state_sha256'] != member['weights_sha256'] or sha256(state) != best['state_sha256']
                or settings['seed'] != member['seed'] or best['epoch'] != member['best_epoch']):
            raise ValueError('Export is not bound to the validation-selected trained member')
        expected = serialization.msgpack_restore(state.read_bytes())['params']
        actual = jax.tree.map(lambda v: np.asarray(v)[i], params)
        if jax.tree.structure(expected) != jax.tree.structure(actual):
            raise ValueError('Changed exported parameter tree')
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(a, b)
        for path in (directory/'settings.json', directory/'best.json', directory/'complete.json', state):
            bindings[str(path)] = sha256(path)
    return bindings

def replay(bundle, dataset, output, clipped_mixture=False):
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    if clipped_mixture:
        from .clipped_inference import ClippedRiskPredictor
        model = ClippedRiskPredictor(bundle, allow_uncalibrated=True)
    else:
        model = ResearchPredictor(bundle, allow_uncalibrated=True)
    meta = model.metadata; info = validate_metadata(meta)
    if meta['dataset_manifest_sha256'] != sha256(Path(dataset)/'manifest.json'):
        raise ValueError('Wrong training/validation data source')
    data = load_dataset(dataset, 'validation')
    if not len(data['group_id']) or not np.all(data['partition'] == 'validation'):
        raise ValueError('Qualification requires validation-only observations')
    # Fixed outcome-independent coverage of the stored validation order.
    indices = np.linspace(0, len(data['group_id'])-1, min(128, len(data['group_id'])), dtype=int)
    f, mask, gains = (data[k][indices] for k in ('features', 'node_mask', 'gains'))
    coefficient_features=meta['architecture'].get('unicycle_constraint_features',False)
    expected_context = numpy_context(f, mask, info['dynamics'], info['yaw_scale'],coefficient_features)
    bindings = check_export(bundle, meta, model.params)
    for file in ('manifest.json', 'index.json'):
        path = Path(dataset)/file; bindings[str(path)] = sha256(path)
    context_fn = jax.jit(lambda features, masks: model.model.apply(
        {'params': jax.tree.map(lambda v: v[0], model.params)}, features, masks, method=model.model.encode))
    context_exe = context_fn.lower(jnp.asarray(f), jnp.asarray(mask)).compile()
    context = np.asarray(context_exe(jnp.asarray(f), jnp.asarray(mask)))
    np.testing.assert_allclose(context, expected_context, atol=3e-6, rtol=2e-6)
    expected = []
    gf=gains.astype(np.float64)
    gain_features=np.log(np.maximum(gf,1e-6))
    if coefficient_features:
        gain_features=np.concatenate((gain_features,gf[...,:1]/4.,
            gf.sum(-1,keepdims=True)/8.,np.prod(gf,axis=-1,keepdims=True)/16.),-1)
    for member in range(4):
        value = np.concatenate((np.broadcast_to(expected_context[:, None], (*gains.shape[:2], expected_context.shape[-1])),
                                gain_features), axis=-1)
        for hidden in range(4):
            layer = model.params[f'hidden_{hidden}']
            value = np.maximum(value @ np.asarray(layer['kernel'][member], np.float64)
                               + np.asarray(layer['bias'][member], np.float64), 0.)
        layer = model.params['output']
        expected.append(value @ np.asarray(layer['kernel'][member], np.float64)+np.asarray(layer['bias'][member], np.float64))
    expected = np.stack(expected)
    d = model.model.config.continuous_outputs
    reference = dict(mean=expected[..., :d], log_variance=np.clip(expected[..., d:2*d], -10., 3.),
                     event_logits=expected[..., 2*d:])
    if clipped_mixture:
        from scipy.special import softmax
        means=np.stack((expected[...,0],expected[...,6]),axis=-1)
        lv=np.clip(np.stack((expected[...,2],expected[...,7]),axis=-1),-10.,3.)
        logits=expected[...,8:10];weights=softmax(logits,axis=-1)
        center=(weights*means).sum(-1)
        var=(weights*(np.exp(lv)+(means-center[...,None])**2)).sum(-1)
        reference=dict(mean=np.stack((center,expected[...,1]),axis=-1),
            log_variance=np.stack((np.log(var),np.clip(expected[...,3],-10.,3.)),axis=-1),
            event_logits=expected[...,4:6],risk_component_mean=means,
            risk_component_log_variance=lv,risk_component_logits=logits)
    fn = jax.jit(lambda p, features, masks, candidates: predict_ensemble(model.model, p, features, masks, candidates))
    args = (model.params, jnp.asarray(f), jnp.asarray(mask), jnp.asarray(gains))
    start = time.monotonic(); executable = fn.lower(*args).compile()
    prediction = jax.block_until_ready(executable(*args)); cold = time.monotonic()-start
    errors = {}
    for key in reference:
        actual = np.asarray(prediction[key]); errors[key] = float(np.max(np.abs(actual-reference[key])))
        if not np.isfinite(actual).all() or not np.isfinite(reference[key]).all():
            raise ValueError('Nonfinite independently replayed FC output')
        np.testing.assert_allclose(actual, reference[key], atol=2e-5, rtol=2e-5)
    times = []
    for _ in range(10):
        start = time.monotonic(); jax.block_until_ready(executable(*args)); times.append(time.monotonic()-start)
    if fn._cache_size() or context_fn._cache_size():
        raise ValueError('Implicit numerical replay compilation')
    np.savez_compressed(root/'predictions.npz', **jax.tree.map(np.asarray, prediction),
                        context=context, indices=indices, group_id=data['group_id'][indices])
    result = dict(schema=NEAREST_FC_QUALIFICATION_SCHEMA, status='passed', backend=str(jax.devices()[0]), platform=jax.devices()[0].platform,
                  bundle=str(Path(bundle).resolve()), dataset=str(Path(dataset).resolve()),
                  bound_files=bindings, source_fingerprint=source_fingerprint(),
                  validation_queries=len(indices), validation_only=True, reserved_parents_used=False,
                  benchmark_parents_used=False, calibration_fitted=False, model_promoted=False,
                  independent_numpy_feature_max_error=float(np.max(np.abs(context-expected_context))),
                  independent_numpy_network_max_error=errors, exact_exported_members=True,
                  predictions_sha256=sha256(root/'predictions.npz'), compile_seconds=cold,
                  warmed_p50_seconds=float(np.median(times)), implicit_jit_cache_entries=0)
    if clipped_mixture:result['risk_distribution_contract']=meta['risk_distribution_contract']
    write_json(root/'report.json', result)
    print(json.dumps(result), flush=True)

def compare(cpu, gpu, output):
    roots = [Path(cpu), Path(gpu)]; reports = [read(p/'report.json') for p in roots]
    if reports[0].get('risk_distribution_contract')!=reports[1].get('risk_distribution_contract'):
        raise ValueError('Different qualified predictive distributions')
    for key in ('schema', 'status', 'bundle', 'dataset', 'bound_files', 'source_fingerprint',
                'validation_queries', 'validation_only', 'reserved_parents_used', 'benchmark_parents_used',
                'calibration_fitted', 'model_promoted', 'exact_exported_members', 'implicit_jit_cache_entries'):
        if reports[0][key] != reports[1][key]:
            raise ValueError('CPU/GPU qualification contract differs: '+key)
    if reports[0]['status'] != 'passed' or backend_platform(reports[0]) != 'cpu' or backend_platform(reports[1]) != 'gpu':
        raise ValueError('Both independent backend replays required')
    for path, digest in reports[0]['bound_files'].items():
        if sha256(path) != digest: raise ValueError('Changed qualification source '+path)
    with np.load(roots[0]/'predictions.npz') as a, np.load(roots[1]/'predictions.npz') as b:
        for key in a.files:
            if key in ('indices', 'group_id'): np.testing.assert_array_equal(a[key], b[key])
            else: np.testing.assert_allclose(a[key], b[key], atol=2e-5, rtol=2e-5)
    result = dict(schema=NEAREST_FC_QUALIFICATION_SCHEMA, status='passed', bundle=reports[0]['bundle'], dataset=reports[0]['dataset'],
        bound_files=reports[0]['bound_files'], validation_queries=reports[0]['validation_queries'],
        independent_numpy_features_and_network=True, exact_exported_members=True, cpu_gpu_parity=True,
        validation_only=True, calibration_fitted=False, reserved_parents_used=False, benchmark_parents_used=False,
        selected_backend='cpu' if reports[0]['warmed_p50_seconds'] <= reports[1]['warmed_p50_seconds'] else 'gpu',
        backend_reports={str(p/'report.json'):sha256(p/'report.json') for p in roots},
        whole_controller_qualification_complete=False, model_promoted=False, whole_goal_complete=False)
    if 'risk_distribution_contract' in reports[0]:
        result['risk_distribution_contract']=reports[0]['risk_distribution_contract']
    write_json(output, result); validate_qualification(output, result['bundle'], result['dataset'])
    return result

def validate_qualification(path, bundle, dataset):
    if path is None: raise ValueError('Nearest-FC numerical qualification required')
    proof = read(path)
    fields = ('independent_numpy_features_and_network', 'exact_exported_members', 'cpu_gpu_parity', 'validation_only')
    if (proof.get('schema') != NEAREST_FC_QUALIFICATION_SCHEMA or proof.get('status') != 'passed'
            or any(proof.get(k) is not True for k in fields)
            or proof.get('bundle') != str(Path(bundle).resolve()) or proof.get('dataset') != str(Path(dataset).resolve())
            or any(proof.get(k) is not False for k in ('calibration_fitted', 'reserved_parents_used', 'benchmark_parents_used', 'model_promoted'))):
        raise ValueError('Incomplete nearest-FC numerical qualification')
    for file, digest in {**proof['bound_files'], **proof['backend_reports']}.items():
        if sha256(file) != digest: raise ValueError('Changed qualified input '+file)
    for file in proof['backend_reports']:
        report = read(file); predictions = Path(file).parent/'predictions.npz'
        if sha256(predictions) != report['predictions_sha256']:
            raise ValueError('Changed replay outputs')
    metadata=read(Path(bundle)/'manifest.json')
    validate_metadata(metadata)
    if proof.get('risk_distribution_contract')!=metadata.get('risk_distribution_contract'):
        raise ValueError('Qualification has a different predictive distribution')
    return proof

def validate_fit(fit, bundle):
    path = fit.get('runtime_qualification')
    if path is None or fit.get('runtime_qualification_sha256') != sha256(path):
        raise ValueError('Nearest-FC calibration lacks its numerical qualification binding')
    dataset = fit.get('dataset', read(path)['dataset'])
    proof = validate_qualification(path, bundle, dataset)
    meta = read(Path(bundle)/'manifest.json')
    if (fit.get('nearest_fc_contract') != validate_metadata(meta)
            or fit['bundle_manifest_sha256'] != sha256(Path(bundle)/'manifest.json')
            or fit['weights_sha256'] != meta['weights_sha256']
            or fit['dataset_manifest_sha256'] != meta['dataset_manifest_sha256']
            or fit.get('diagnostic_identity_only')):
        raise ValueError('Nearest-FC prediction fit changed its qualified model/input contract')
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); commands = parser.add_subparsers(dest='action', required=True)
    p = commands.add_parser('replay')
    for name in ('bundle', 'dataset', 'output'): p.add_argument('--'+name, required=True)
    p.add_argument('--clipped-mixture',action='store_true')
    p = commands.add_parser('compare')
    for name in ('cpu', 'gpu', 'output'): p.add_argument('--'+name, required=True)
    args = vars(parser.parse_args()); action = args.pop('action')
    replay(**args) if action == 'replay' else compare(**args)
