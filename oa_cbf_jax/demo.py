"""Run navigation scenarios with released controller models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from .model_release import DEFAULT, read, verify

ALIASES = {'dynamic_unicycle':'unicycle', 'kinematic_bicycle_dpcbf':'bicycle'}
METHODS = {'ours_gat':'gat', 'ours_fc':'nearest_fc', 'fc':'nearest_fc',
           'od_cbf_qp':'optimal_decay_qp', 'od_cbf_mpc':'optimal_decay'}


def learned_stepper(root, dynamics, method, scenario):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .model_release import selector
    p, spec = selector(root, dynamics, method)
    row=scenario['parent']; steps=scenario['steps']; route=row['route']
    finish_status=lambda carry,elapsed: int(carry[1]) or 4
    f=lambda v:jnp.asarray(v,jnp.float32)
    b=lambda v:jnp.asarray(v,bool)
    if dynamics=='unicycle':
        from .closed_loop import make_closed_loop
        scene=row['scene']; seed=int.from_bytes(hashlib.sha256(scene['scene_id'].encode()).digest()[:4],'little')
        noise=f(scenario['noise_scale']*np.array([.02,.03,.02,.03,.025,.01]))
        setup=make_closed_loop(p,steps,return_stepper=True)
        tick,carry,innovations,truth=setup(p.params,p.calibration,f(scene['initial_state']),f(scene['goal']),
            f(scene['obstacles']),b(scene['obstacle_mask']),f(spec['candidates']),f(route['points']),b(route['mask']),
            noise,jax.random.PRNGKey(seed),route['status']=='ready')
        inputs=lambda k:(jnp.int32(k),innovations[k])
        dt=p.robot.dt; radius=p.robot.radius
    elif dynamics=='quad2d':
        from .quad2d_policy import make_episode
        setup=make_episode(p.model,p.norm,p.config,p.policy,steps,p.guidance,
            record_query_statistics=True,return_stepper=True)
        tick,carry,innovations,truth=setup(p.params,p.calibration,f(spec['candidates']),f(row['initial_state']),f(row['goal']),
            f(row['obstacles']),b(row['obstacle_mask']),f(route['points']),b(route['mask']),f(row['noise']),
            jax.random.PRNGKey(row['seed']+7193),b(route['status']=='ready'))
        inputs=lambda k:(jnp.int32(k),innovations[k])
        dt=p.config.robot.dt; radius=p.config.robot.radius
    elif dynamics=='quad3d':
        from .quad3d_policy_rollout import make_policy_rollout
        from .quad3d_observation import unit_tape
        f64=lambda v:jnp.asarray(v,jnp.float64)
        args=[f64(row[k]) if k!='mask' else b(row[k]) for k in ('x','goal','obstacles','mask')]
        args += [f64([p.config.initial_gain]*4),f64(route['points']),b(route['mask']),f64(row['noise'])]
        args += list(map(f64,unit_tape(row['sensor_seed'],steps,len(row['mask']))))
        tick,carry,finish_status,truth=make_policy_rollout(p,steps,failure_requery=spec['failure_requery'],return_stepper=True)(p.predictor.params,*args)
        finish_status=jax.jit(finish_status)
        original=tick
        def tick(c,k):
            c,(trace,_)=original(c,k)
            return c,trace
        inputs=lambda k:jnp.int32(k)
        dt=p.robot.robot.dt; radius=p.robot.robot.radius
    else:
        tick,carry,truth=bicycle_stepper(p,row)
        inputs=lambda k:jnp.int32(k)
        dt=p.robot.robot.dt; radius=p.robot.robot.radius
    return tick,carry,inputs,truth,dt,radius,finish_status


def bicycle_stepper(p,row):
    """Same observation -> selector -> one-tick plant sequence as evaluation."""
    import jax
    import jax.numpy as jnp
    from .bicycle_observed_rollout import make_observed_episode
    from .bicycle_control import constant
    from .bicycle_observation import observe,unit_errors
    f=lambda v:jnp.asarray(v,jnp.float32)
    f64=lambda v:jnp.asarray(v,jnp.float64)
    mask=jnp.asarray(row['mask'],bool); route=row['route']; rm=jnp.asarray(route['mask'],bool)
    o=f64(row['obstacles']); x=f64(row['initial']); goal=f(row['goal']); points=f(route['points'])
    bx=f(row['bias_x']);bo=f(row['bias_o']);noise=f(row['noise']);key=jax.random.PRNGKey(row['seed']+4)
    first_x=f(row['first_x']);first_o=f(row['first_o']);ready=jnp.asarray(route['status']=='ready')
    physical=make_observed_episode(p.robot,steps=1,guidance=p.guidance)
    def tick(carry,k):
        state,status,count,cursor,previous_u,gain=carry
        current=o.at[:,:2].add(k.astype(jnp.float64)*constant(p.robot.robot.dt,jnp.float64)*o[:,3:5])
        ix,io=unit_errors(jax.random.fold_in(key,k),64)
        ix=jnp.where(k==0,0.,ix);io=jnp.where(k==0,0.,io)
        sx,so=observe(state,current,mask,bx,bo,noise,ix,io)
        sx=jnp.where(k==0,first_x,sx);so=jnp.where(k==0,first_o,so)
        values=(sx,goal,so,mask,points,rm,cursor,previous_u,gain,noise)
        result=p._function(p.predictor.params,*[v[None] for v in values],jnp.asarray(p.threshold))
        selected=result['controller_gain'][0]
        summary,tr=physical(state,goal,current,mask,selected,points,rm,ready,cursor,sx,so,bx,bo,noise,key)
        trace=jax.tree.map(lambda v:v[0],tr)
        accepted=trace['active']
        next_carry=(summary['final_state'],summary['status'].astype(jnp.int32),count+accepted.astype(jnp.int32),
            summary['final_cursor'],jnp.where(accepted,trace['control'],previous_u),jnp.where(accepted,selected,gain))
        trace.update(controller_gain=selected,requery=jnp.bool_(True))
        return next_carry,trace
    # A one-tick bicycle episode reports TIMEOUT (4) when another tick may run.
    from .bicycle_rollout import TIMEOUT
    carry=(x,jnp.int32(TIMEOUT),jnp.int32(0),jnp.float32(0),jnp.zeros(2,jnp.float32),jnp.float32(p.config.initial_gain))
    return tick,carry,dict(initial_state=x,obstacles=o)


def run_learned(root,dynamics,method,scenario,limit):
    import jax
    import numpy as np
    start=time.perf_counter()
    tick,initial,inputs,truth,dt,radius,finish_status=learned_stepper(root,dynamics,method,scenario)
    fn=jax.jit(tick)
    compiled=fn.lower(initial,inputs(0)).compile()
    # Warm up without advancing the actual episode or including compilation.
    jax.block_until_ready(compiled(initial,inputs(0)))
    cold=time.perf_counter()-start
    print(json.dumps(dict(stage='ready',dynamics=dynamics,method=method,compile_and_warmup_seconds=cold)),flush=True)
    carry=initial;history=[];timings=[]
    for k in range(limit):
        inp=inputs(k)
        t=time.perf_counter();carry,trace=compiled(carry,inp)
        jax.block_until_ready((carry,trace))
        timings.append(time.perf_counter()-t)
        trace=jax.device_get(trace);history.append(trace)
        status=int(carry[1])
        if (dynamics=='bicycle' and status!=4) or (dynamics!='bicycle' and status!=0):break
    status=int(finish_status(carry,len(history)))
    data={key:np.asarray([t[key] for t in history]) for key in history[0]}
    state_key='next_state' if dynamics=='quad3d' else 'state'
    states=np.vstack((np.asarray(truth['initial_state']),data[state_key]))
    query_key=next((k for k in ('selection_tick','requery') if k in data),None)
    queries=data[query_key].astype(bool) if query_key else np.ones(len(timings),bool)
    times=np.asarray(timings)*1000
    # Every recorded time includes the full kernel: sensed features/observers,
    # scheduled selection, calibration, preview, QP, integration and checking.
    report=dict(dynamics=dynamics,method=method,backend=jax.default_backend(),device=str(jax.devices()[0]),
        steps=int(carry[2]),ticks=len(history),status=status,success=status==1,
        full_scenario=limit==scenario['steps'],reference=scenario['expected'][method],compile_and_warmup_seconds=cold,
        control_step_mean_ms=float(times.mean()),control_step_median_ms=float(np.median(times)),
        query_step_mean_ms=float(times[queries].mean()) if queries.any() else None,
        held_step_mean_ms=float(times[~queries].mean()) if (~queries).any() else None,
        first_20_step_mean_ms=float(times[:20].mean()),timing_scope='Synchronized full single-scene control/physics kernel; excludes JIT warmup, plotting and file IO. No batching amortization or padded inactive steps.')
    return report,dict(states=states,obstacles=np.asarray(truth['obstacles']),mask=np.asarray(scenario['parent'].get('mask',scenario['parent'].get('obstacle_mask',scenario['parent'].get('scene',{}).get('obstacle_mask')))),
        goal=np.asarray(scenario['parent'].get('goal',scenario['parent'].get('scene',{}).get('goal'))),dt=dt,radius=radius,**data)


def display(data,report,backend=None,video=None,hold=False,pause=.001):
    import matplotlib
    if backend:matplotlib.use(backend)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle
    import numpy as np
    states=data['states'];obs=data['obstacles'];mask=data['mask'].astype(bool)
    fig,ax=plt.subplots(figsize=(12,5));ax.set_aspect('equal')
    goal=data['goal'];radius=float(data['radius']);dt=float(data['dt'])
    ax.scatter(*goal[:2],marker='*',s=180,color='green',label='Goal')
    circles=[]
    for o in obs[mask]:
        c=Circle(o[:2],o[2],color='0.7');ax.add_patch(c);circles.append(c)
    robot=Circle(states[0,:2],radius,color='tab:blue');ax.add_patch(robot)
    path,=ax.plot([],[],color='tab:blue');ax.scatter(*states[0,:2],s=25,color='black')
    xy=np.vstack((states[:,:2],obs[mask,:2],goal[None,:2]));lo=xy.min(0)-1;hi=xy.max(0)+1
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='x (m)',ylabel='y (m)')
    label='GAT' if report['method']=='gat' else 'nearest-obstacle FC' if report['method']=='nearest_fc' else report['method']
    view=report['dynamics']+(' (top view)' if report['dynamics']=='quad3d' else '')
    title=ax.set_title(f"{view} | {label}")
    def frame(i):
        robot.center=states[i,:2];path.set_data(states[:i+1,0],states[:i+1,1])
        for c,o in zip(circles,obs[mask]):c.center=o[:2]+i*dt*o[3:5]
        title.set_text(f"{view} | {label} | t={i*dt:.2f} s")
        return robot,path,title,*circles
    animation=FuncAnimation(fig,frame,frames=range(0,len(states),2),interval=max(1,1000*dt*2),blit=False,repeat=False)
    if video:
        video=Path(video);video.parent.mkdir(parents=True,exist_ok=True)
        animation.save(str(video),fps=1/(2*dt),dpi=110)
    else:
        plt.show(block=True)
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dynamics',default='quad2d',choices=['unicycle','quad2d','quad3d','bicycle',*ALIASES])
    parser.add_argument('--method',default='ours_gat')
    parser.add_argument('--models',type=Path,default=DEFAULT)
    parser.add_argument('--device',choices=['cpu','gpu'],default='cpu')
    parser.add_argument('--max-t',type=float)
    parser.add_argument('--steps',type=int,help='Optional short smoke/timing run; default is the complete scenario')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--backend',help='Matplotlib backend')
    parser.add_argument('--video',type=Path)
    parser.add_argument('--output',type=Path,help='Optional JSON result (trace uses the same name with .npz)')
    parser.add_argument('--hold',action='store_true')
    parser.add_argument('--pause',type=float,default=.001)
    parser.add_argument('--list',action='store_true')
    args=parser.parse_args(argv)
    dynamics=ALIASES.get(args.dynamics,args.dynamics);method=METHODS.get(args.method,args.method)
    if args.method=='od_cbf_qp' and dynamics=='bicycle':
        method='optimal_decay'
    if args.method=='od_cbf_mpc' and dynamics=='bicycle':
        parser.error('The bicycle optimal-decay comparator uses QP; use --method optimal_decay')
    if args.list:
        for d in ('unicycle','quad2d','quad3d','bicycle'):
            extra=', optimal_decay_qp' if d in ('unicycle','quad2d') else ''
            print(f'{d}: ours_gat, ours_fc, fixed_low, fixed_high, optimal_decay, barriernet{extra}')
        return 0
    if not (args.models/'release.json').exists():
        parser.error('Download the final models first: python -m oa_cbf_jax.model_release')
    if dynamics=='bicycle' and args.device!='cpu':parser.error('The selected bicycle policy is CPU-qualified; use --device cpu')
    os.environ['JAX_PLATFORMS']='cuda' if args.device=='gpu' else 'cpu'
    os.environ['JAX_EXPLICIT_X64_DTYPES']='allow'
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    import jax
    jax.config.update('jax_enable_x64', dynamics in ('bicycle','quad3d') or method not in ('gat','nearest_fc'))
    manifest=verify(args.models)
    if method not in {*manifest['models'][dynamics],*manifest['baselines'][dynamics]}:parser.error('Unsupported method for this dynamics; use --list')
    scenario=read(args.models/manifest['heroes'][dynamics])
    spec=read(args.models/manifest['models'][dynamics]['gat']/'policy.json')
    dt=spec['config'].get('robot',spec['config'])['dt']
    limit=scenario['steps']
    if args.steps is not None:limit=min(limit,args.steps)
    if args.max_t is not None:limit=min(limit,int(args.max_t/dt))
    if limit<1:parser.error('Simulation must contain at least one step')
    if method in ('gat','nearest_fc'):
        report,data=run_learned(args.models,dynamics,method,scenario,limit)
    else:
        from .demo_baselines import run
        report,data=run(args.models,dynamics,method,scenario,limit,manifest['baselines'][dynamics][method])
    print(json.dumps(report,indent=2),flush=True)
    if args.output:
        import numpy as np
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        np.savez_compressed(args.output.with_suffix('.npz'),**data)
    if args.video or not args.headless:display(data,report,args.backend,args.video,args.hold,args.pause)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
