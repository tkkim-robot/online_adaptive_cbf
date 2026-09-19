"""Extract authentic legacy showcase inputs without importing legacy controllers.

Only explicitly listed geometry function definitions are executed. No simulation,
media trace, controller, learned weights, or outcome is imported. The source file
and exact selected function texts are hashed in every export for review.
"""

import argparse
import ast
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import runpy
from types import SimpleNamespace
import numpy as np
from .io import write_json
from .legacy_cases import CASES


FUNCTIONS = (
    'formatted_obstacles', 'make_init_state_from_xy', 'make_waypoints',
    'segment_blockers', 'append_obstacles', 'point_to_polyline_distance',
    'fill_obstacles_around_path', 'fill_dense_obstacle_field',
    'make_narrow_scene', 'make_wide_scene', 'make_dpcbf_dynamic_scene', 'make_scene',
)


def extract(repository,only_groups=None):
    repository = Path(repository)
    path = repository / 'plot/generate_paper_media.py'
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    nodes = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    selected = [nodes[name] for name in FUNCTIONS]  # Fail if a source helper moved.
    constants = {}
    geometry_settings = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], 'id', '')
            if name in ('NARROW_OBS_ID', 'WIDE_OBS_ID'):
                constants[name] = ast.literal_eval(node.value)
            if name=='ae' and isinstance(node.value, ast.Call) and getattr(node.value.func, 'id', '')=='SimpleNamespace':
                geometry_settings={k.arg:ast.literal_eval(k.value) for k in node.value.keywords}
    if geometry_settings is None: raise ValueError('Original geometry settings moved; inspect source before extraction')
    defaults_path = repository / 'online_cbf_config.py'
    defaults = runpy.run_path(str(defaults_path))['ALL_DEFAULTS']
    def spec(name):
        entry = defaults[name]
        return entry['robot_spec'], entry['default_obs']
    namespace = dict(np=np, math=math, ae=SimpleNamespace(**geometry_settings),
                     get_robot_spec_and_obs=spec, **constants)
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    # Execute the original spec-adjustment prefix, stopping before the first
    # controller construction/default lookup. Do not duplicate its overrides.
    prefix=[]
    for node in nodes['build_tracker'].body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and getattr(node.value.func, 'id', '')=='get_controller_defaults':
            break
        prefix.append(node)
    else: raise ValueError('Original tracker boundary moved; inspect before extraction')
    adjusted=ast.parse('def _media_spec(case, scene_kind): pass').body[0]
    adjusted.body=prefix+[ast.Return(value=ast.Name(id='robot_spec',ctx=ast.Load()))]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[adjusted],type_ignores=[])), str(path), 'exec'), namespace)
    digest = lambda data: hashlib.sha256(data).hexdigest()
    provenance = dict(
        geometry_file='plot/generate_paper_media.py', geometry_sha256=digest(path.read_bytes()),
        defaults_file='online_cbf_config.py', defaults_sha256=digest(defaults_path.read_bytes()),
        **geometry_settings,
        tracker_spec_prefix_sha256=digest('\n'.join(ast.get_source_segment(source,n) for n in prefix).encode()),
        geometry_functions={name: dict(line=nodes[name].lineno,
            sha256=digest(ast.get_source_segment(source, nodes[name]).encode())) for name in FUNCTIONS},
        extraction='Execute only named original geometry definitions, with original NumPy RandomState seeds. No controller or media rollout is executed.',
    )
    result = {}
    groups = {case.group: case.robot for case in CASES}
    if only_groups is not None:
        if set(only_groups)-set(groups):raise ValueError('Unknown hero group')
        groups={k:v for k,v in groups.items() if k in only_groups}
    for group, robot in groups.items():
        for kind in ('narrow', 'wide'):
            obs_id = constants[kind.upper()+'_OBS_ID'][group]
            environment, state, waypoints, obstacles = namespace['make_scene'](robot, obs_id, kind)
            base_spec = dict(spec(robot)[0])
            media_spec = namespace['_media_spec'](SimpleNamespace(robot=robot), kind)
            cases = []
            for case in CASES:
                if case.group!=group: continue
                cases.append(dict(**asdict(case), defaults=defaults[robot]['controller_params'].get(case.controller),
                                  evaluation_status='not_evaluated'))
            result[f'{group}_{kind}'] = dict(
                schema='oa_cbf_original_hero_v1', stage='known_showcase_input_only', final_test=False,
                group=group, robot=robot, kind=kind, observation_id=obs_id, provenance=provenance,
                environment=environment, initial_state=state.tolist(), waypoints=waypoints.tolist(),
                obstacles=obstacles.tolist(), obstacle_columns=['x','y','radius','vx','vy','legacy_column_5','legacy_column_6'],
                obstacle_count=len(obstacles), base_robot_spec=base_spec, media_robot_spec=media_spec,
                media_spec_changes={key:dict(before=base_spec.get(key),after=value) for key,value in media_spec.items() if base_spec.get(key)!=value},
                comparators=cases,
                evaluation_status='not_evaluated',
                interpretation='Original geometry and waypoint task preserved. This file contains no simulated trajectory or performance evidence. Original robot changes are recorded, not silently applied to a differently trained model.',
            )
    return result


def export(repository, output):
    output = Path(output)
    if output.exists(): raise ValueError('Use a new output directory; preserve existing original snapshots')
    records = extract(repository)
    for name, record in records.items(): write_json(output/(name+'.json'), record)
    write_json(output/'manifest.json', dict(schema='oa_cbf_original_hero_manifest_v1', final_test=False,
        scenes={name:hashlib.sha256((output/(name+'.json')).read_bytes()).hexdigest() for name in records},
        note='Frozen original inputs. Unicycle wide exceeds the current 16-obstacle learned contract. Preserve every obstacle; do not truncate to make a model run.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repository', default='.')
    parser.add_argument('--output', required=True)
    export(**vars(parser.parse_args()))
