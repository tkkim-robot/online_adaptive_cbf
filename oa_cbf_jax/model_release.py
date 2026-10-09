"""Portable, checksummed inference packages (no training-data dependencies)."""

import argparse

import hashlib

import json

from pathlib import Path

import shutil

import tempfile

from types import SimpleNamespace

import urllib.request

import zipfile

TAG = 'oa-cbf-models-2026-10'

ASSET = 'oa-cbf-models.zip'

URL = f'https://github.com/tkkim-robot/online_adaptive_cbf/releases/download/{TAG}/{ASSET}'

DEFAULT = Path(__file__).resolve().parents[1] / 'models' / 'paper'

SCHEMA = 'oa_cbf_deployment_package_v1'

def read(path):
    return json.loads(Path(path).read_text())

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify(root):
    root = Path(root).resolve()
    manifest = read(root / 'release.json')
    if manifest.get('schema') != SCHEMA:
        raise ValueError('Unknown model release format')
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
            raise ValueError('Missing or changed release file: ' + name)
    for dynamics, methods in manifest['models'].items():
        if manifest['heroes'][dynamics] not in manifest['files']:
            raise ValueError('Unbound navigation scenario')
        for method, relative in methods.items():
            prefix = Path(relative)
            for name in ('weights.msgpack', 'model.json', 'policy.json'):
                if (prefix / name).as_posix() not in manifest['files']:
                    raise ValueError('Unbound deployment model')
            model = read(root / prefix / 'model.json')
            if digest(root / prefix / 'weights.msgpack') != model['weights_sha256']:
                raise ValueError('Model/weights mismatch')
            if model['architecture']['encoder'] != method:
                raise ValueError('Wrong encoder in deployment package')
            policy = read(root / prefix / 'policy.json')
            if policy['dynamics'] != dynamics or policy['weights_sha256'] != model['weights_sha256']:
                raise ValueError('Policy/model mismatch')
        for baseline in manifest['baselines'][dynamics].values():
            if 'bundle' in baseline:
                prefix = Path(baseline['bundle'])
                for name in ('manifest.json', 'weights.msgpack', 'normalization.npz'):
                    if (prefix / name).as_posix() not in manifest['files']:
                        raise ValueError('Missing baseline model component')
    return manifest

def download(destination=DEFAULT, url=URL, sha256=None):
    """Install an immutable release in a new directory, rejecting unsafe ZIPs."""
    destination = Path(destination).resolve()
    if destination.exists():
        verify(destination)
        print(f'Existing verified models: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        temporary = Path(temporary)
        archive = temporary / ASSET
        urllib.request.urlretrieve(url, archive)
        if sha256 is None:
            with urllib.request.urlopen(url + '.sha256') as response:
                sha256 = response.read().decode().split()[0]
        if digest(archive) != sha256:
            raise ValueError('Downloaded archive checksum mismatch')
        unpack = temporary / 'unpack'
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                path = (unpack / entry.filename).resolve()
                if not path.is_relative_to(unpack.resolve()) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Unsafe archive entry')
            z.extractall(unpack)
        verify(unpack)
        shutil.move(str(unpack), destination)
    print(f'Installed verified models: {destination}')

def load(root, dynamics, method):
    """Load exported parameters, retaining the original inference arithmetic.

    Research constructors check training ancestry at export time. Deployment
    instead checks the complete portable package and its frozen policy receipt;
    it never follows genealogy paths or silently substitutes calibration.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from .models import GATConfig, make_model
    root = Path(root)
    manifest = verify(root)
    directory = root / manifest['models'][dynamics][method]
    metadata = read(directory / 'model.json')
    contract = read(directory / 'policy.json')
    if contract['backend'] == 'cpu' and jax.default_backend() != 'cpu':
        raise ValueError('This frozen bicycle package requires --device cpu')
    model = make_model(GATConfig(**metadata['architecture']))
    raw = serialization.msgpack_restore((directory / 'weights.msgpack').read_bytes())
    if model.config.compute_dtype == 'float64':
        # Preserve the evaluated global precision mode as well as the model
        # declaration (the final Quad2D experiment ran with x64 disabled).
        raw = jax.tree.map(lambda v: jnp.asarray(v, jnp.float64), raw)
    params = jax.device_put(raw)
    if not all(np.isfinite(v).all() for v in jax.tree.leaves(raw)):
        raise ValueError('Nonfinite weights')
    predictor = SimpleNamespace(model=model, metadata=metadata, params=params, device=jax.devices()[0])
    return predictor, contract

def calibrated_arrays(fit):
    import jax.numpy as jnp
    return {k: jnp.asarray(v, jnp.float32) for k, v in dict(
        variance_scale=fit['variance_scale'],
        temperature=[e['temperature'] for e in fit['event_calibration']],
        bias=[e['bias'] for e in fit['event_calibration']],
        cs_threshold=fit['cs_gate']['threshold']).items()}

def selector(root, dynamics, method):
    """Construct the same policy kernels used in the frozen comparisons."""
    import numpy as np
    predictor, spec = load(root, dynamics, method)
    if dynamics == 'unicycle':
        from .adaptive import PolicyConfig, make_selector
        from .config import UnicycleConfig
        config = PolicyConfig(**spec['policy'])
        robot = UnicycleConfig(**spec['config'])
        policy = SimpleNamespace(predictor=predictor, params=predictor.params,
            robot=robot, config=config, calibration=calibrated_arrays(spec['fit']))
        policy.selector = make_selector(predictor.model, predictor.metadata['normalization'], robot, config)
    elif dynamics == 'quad2d':
        from .quad2d_control import flight_config_from_contract
        from .quad2d_guidance import terminal_guidance_from_contract
        from .quad2d_policy import flight_policy_from_contract
        policy = SimpleNamespace(model=predictor.model, params=predictor.params,
            norm=predictor.metadata['normalization'], config=flight_config_from_contract(spec['config']),
            policy=flight_policy_from_contract(spec['policy']),
            guidance=terminal_guidance_from_contract(spec['guidance']), calibration=calibrated_arrays(spec['fit']))
    elif dynamics == 'quad3d':
        from .quad3d_policy import Quad3DSelector, Quad3DPolicyConfig
        from .quad3d_control import control_config
        from .quad3d_qp import InputSupportSelector
        policy = Quad3DSelector.__new__(Quad3DSelector)
        policy.predictor=predictor; policy.metadata=predictor.metadata
        policy.config=Quad3DPolicyConfig(**spec['policy']); policy.robot=control_config(spec['config'])
        policy.fit=spec['fit']; policy.bank=np.asarray(spec['candidates'], np.float32)
        policy.reference=False; policy.threshold=np.float32(spec['threshold'])
        policy._build_predictor()
        if spec['input_box_admission']:
            policy=InputSupportSelector(policy)
    elif dynamics == 'bicycle':
        from .bicycle_policy import BicycleSelector, BicyclePolicyConfig
        from .bicycle_control import control_config
        from .bicycle_guidance import BicycleGuidanceConfig
        policy = BicycleSelector.__new__(BicycleSelector)
        policy.predictor=predictor; policy.metadata=predictor.metadata
        policy.config=BicyclePolicyConfig(**spec['policy']); policy.robot=control_config(spec['config'])
        policy.fit=spec['fit']; policy.bank=np.asarray(spec['candidates'], np.float32)
        policy.reference_recording=False; policy.motion_history=predictor.model.config.bicycle_motion_history
        policy.guidance=BicycleGuidanceConfig(**spec['guidance']); policy.threshold=np.float32(spec['threshold'])
        policy._build_predictor()
    else:
        raise ValueError('Unsupported dynamics')
    return policy, spec


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=DEFAULT)
    parser.add_argument('--url', default=URL)
    parser.add_argument('--sha256')
    parser.add_argument('--verify', action='store_true')
    args=parser.parse_args()
    if args.verify:
        verify(args.destination)
        print(f'Verified: {args.destination}')
    else:
        download(args.destination, args.url, args.sha256)
