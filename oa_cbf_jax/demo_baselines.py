"""Single-scenario adapters for the unchanged native comparison controllers."""
import json
from pathlib import Path
import tempfile
import time

import jax
import numpy as np


def run(root,dynamics,method,scenario,limit,spec):
    row=scenario['parent'];steps=scenario['steps'];root=Path(root);start=time.perf_counter()
    bundle=root/spec['bundle'] if method=='barriernet' else None
    if dynamics=='unicycle':
        from .config import UnicycleConfig
        from .physical_kernels import PhysicalKernels
        c=UnicycleConfig(**spec['robot']);scene=row['scene']
        if method=='optimal_decay_qp':
            from .optimal_decay_experiment import run as evaluate
            from .unicycle_inputs import SCHEMA
            from .dataset import sha256
            with tempfile.TemporaryDirectory() as temporary:
                source=Path(temporary)/'source';source.mkdir();out=Path(temporary)/'run'
                (source/'scenes.json').write_text(json.dumps([row]))
                (source/'manifest.json').write_text(json.dumps(dict(schema=SCHEMA,groups=1,robot=spec['robot'],stationary_physical_obstacles=True,scenes_sha256=sha256(source/'scenes.json'))))
                evaluate(source,out,noise_scale=scenario['noise_scale'],batch=1,steps=limit,**spec['policy'])
                report=json.loads((out/'results.json').read_text())[0]
                with np.load(out/'traces_00000.npz') as z:data={k:z[k][0] for k in z.files}
        else:
            kernels=PhysicalKernels(len(scene['obstacle_mask']),len(row['route']['mask']),steps,c)
            if method=='barriernet':
                from .barriernet_experiment import Controller,episode
                controller=Controller(bundle,len(scene['obstacle_mask']),c)
            else:
                from .discrete_mpc_experiment import episode
                from .discrete_mpc import DiscreteMPC
                controller=DiscreteMPC(len(scene['obstacle_mask']),method,c)
            report,data,truth=episode(row,controller,kernels,limit,scenario['noise_scale'])
            data.update(truth)
        initial=data['true_initial_state'];obs=data['true_obstacles'];mask=scene['obstacle_mask'];goal=scene['goal'];dt=c.dt;radius=c.radius
    elif dynamics=='quad2d':
        from .quad2d_control import flight_config_from_contract
        from .quad2d_mpc_experiment import FlightPhysicalKernels
        c=flight_config_from_contract(spec['config']);kernels=FlightPhysicalKernels(64,64,steps,c)
        if method=='optimal_decay_qp':
            from .quad2d_odqp_experiment import episode
            report,data=episode(row,kernels,limit,c)
        elif method=='barriernet':
            from .quad2d_barriernet_experiment import episode,Controller
            report,data=episode(row,kernels,Controller(bundle,config=c),limit,c)
        else:
            from .quad2d_mpc_experiment import episode
            from .quad2d_mpc import Quad2DMPC
            report,data=episode(row,Quad2DMPC(64,method,c),kernels,limit)
        initial=data['true_initial_state'];obs=data['true_obstacles'];mask=row['obstacle_mask'];goal=row['goal'];dt=c.robot.dt;radius=c.robot.radius
    elif dynamics=='quad3d':
        from .quad3d_control import control_config
        from .quad3d_mpc_experiment import PhysicalKernels
        c=control_config(spec['config']);kernels=PhysicalKernels(c)
        if method=='barriernet':
            from .demo_barriernet import quad3d_episode as episode, Quad3DController as Controller
            report,data=episode(row,kernels,Controller(bundle,c.robot),limit,c)
        else:
            from .quad3d_mpc_experiment import episode
            from .quad3d_mpc import Quad3DMPC
            report,data=episode(row,Quad3DMPC(64,method,c),kernels,limit)
        initial=row['x'];obs=row['obstacles'];mask=row['mask'];goal=row['goal'];dt=c.robot.dt;radius=c.robot.radius
    else:
        from .bicycle_experiment import control_config
        from .bicycle_native_experiment import PhysicalKernels
        c=control_config(spec['config']);kernels=PhysicalKernels(c)
        if method=='barriernet':
            from .demo_barriernet import bicycle_episode as episode, BicycleController as Controller
            report,data=episode(row,Controller(bundle,c.robot),kernels,limit)
        else:
            from .bicycle_native_experiment import episode
            report,data=episode(row,method,kernels,limit,acceptance=spec['acceptance'])
        initial=row['initial'];obs=row['obstacles'];mask=row['mask'];goal=row['goal'];dt=c.robot.dt;radius=c.robot.radius
    report.update(dynamics=dynamics,method=method,backend=jax.default_backend(),success=report['status'] in (1,'goal_reached'),
        full_scenario=limit==steps,reference=scenario['expected'][method],elapsed_with_setup_seconds=time.perf_counter()-start)
    state_key='next_state' if 'next_state' in data else 'state'
    data.update(states=np.vstack((initial,data[state_key])),obstacles=np.asarray(obs),mask=np.asarray(mask),goal=np.asarray(goal),dt=dt,radius=radius)
    return report,data
