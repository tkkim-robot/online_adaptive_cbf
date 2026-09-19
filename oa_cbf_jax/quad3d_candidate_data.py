"""Authentic visited-state branches and censored labels for Quad3D learning."""
import itertools
import numpy as np
from .quad3d_observation import unit_tape,numpy_observe
from .quad3d_features import numpy_graph
from .quad3d_routing import numpy_flight_target
from .quad3d_observation import numpy_obstacles

SCHEMA='oa_cbf_quad3d_acquired_hurdle_v94'
GAIN_BANK=np.array([[a,a,b,b] for a,b in itertools.product((1.,2.,3.,4.),repeat=2)],np.float64)
REPLICAS=4
HORIZON=120
QUERY_TICKS=(0,80,240)
OBSERVER_SCHEMA='oa_cbf_quad3d_observer_history_hurdle_v98'
OBSERVER_QUERY_TICKS=(0,160,400,800,1200)
WIDE_SCHEMA='oa_cbf_quad3d_wide_gain_history_hurdle_v104'
WIDE_GAIN_BANK=np.array([[a,a,b,b] for a,b in itertools.product((2.,4.,6.,8.),repeat=2)],np.float64)
WIDE_QUERY_TICKS=(0,40,160,400,800)


def candidate_bank(schema):
    if schema==WIDE_SCHEMA:return WIDE_GAIN_BANK
    if schema in (SCHEMA,OBSERVER_SCHEMA):return GAIN_BANK
    raise ValueError('Unregistered Quad3D candidate bank')


def candidate_domain(schema):
    bank=candidate_bank(schema)
    return dict(lower=float(bank.min()),upper=float(bank.max()))

TARGETS=['conditional_negative_continuous_min_clearance_div_0.3_capped_below_minus_2',
         'physical_route_and_3d_terminal_blended_progress_plus_observed_early_arrival']
EVENTS=['collision_first','any_adverse_termination']


def query_context(parent,trace,tick,c,ticks=None):
    if ticks is None:ticks=OBSERVER_QUERY_TICKS if c.nominal_bias_observer=='innovation_ema_v97' else QUERY_TICKS
    if tick not in ticks or tick>=len(trace['active']) or not np.all(trace['active'][:tick]):
        raise ValueError('Query is outside the actually observed acquisition prefix')
    x=np.array(trace['state'][tick]);o=np.array(parent['obstacles']);o[:,:2]+=tick*c.robot.dt*o[:,3:5]
    noise=np.array(parent['noise']);mask=np.array(parent['mask']);bx,bo,ix,io=unit_tape(parent['sensor_seed'],tick,len(mask))
    seen,so=numpy_observe(x,o,mask,bx,bo,noise,ix[tick],io[tick])
    np.testing.assert_allclose(seen,trace['observed'][tick],atol=2e-12,rtol=1e-12)
    context=dict(x=x,obstacles=o,observed=seen,observed_obstacles=so,noise=noise,mask=mask,
        cursor=float(trace['route_cursor_before'][tick]),previous_u=np.array(trace['control'][tick-1]) if tick else np.zeros(4),
        previous_gain=np.array(trace['previous_gain'][tick] if 'previous_gain' in trace else parent['gains']),
        bias_x=bx,bias_o=bo,innovation_x=ix[tick],innovation_o=io[tick],tick=tick)
    if c.nominal_bias_observer=='innovation_ema_v97':context['nominal_bias']=np.array(trace['nominal_bias_estimate'][tick])
    return context


def future_seed(parent,tick,replica):
    # Disjoint deterministic query/replica seeds, paired across ALL gains.
    return int(parent['seed'])*10000+tick*REPLICAS+replica+9400000000


def branch_parent(parent,context,gain,replica):
    q=context['tick']
    branch=dict(parent,id=f"{parent['id']}:q{q}:gain"+','.join(map(str,gain))+f':rep{replica}',
        x=context['x'].tolist(),obstacles=context['obstacles'].tolist(),gains=np.asarray(gain).tolist(),
        initial_cursor=context['cursor'],branch=dict(query_tick=q,future_seed=future_seed(parent,q,replica)))
    if 'nominal_bias' in context:branch['initial_nominal_bias']=context['nominal_bias'].tolist()
    return branch


def branch_arrays(parent,context,horizon=HORIZON,bank=GAIN_BANK,replicas=REPLICAS):
    if replicas!=REPLICAS:raise ValueError('Frozen replica seed/layout contract')
    branches=[branch_parent(parent,context,gain,r) for gain in bank for r in range(replicas)]
    tapes=[]
    for r in range(replicas):
        _,_,ix,io=unit_tape(future_seed(parent,context['tick'],r),horizon,len(context['mask']))
        ix[0]=context['innovation_x'];io[0]=context['innovation_o'];tapes.append((ix,io))
    size=len(branches);repeat=lambda a:np.repeat(np.asarray(a)[None],size,axis=0)
    args=(repeat(context['x']),repeat(parent['goal']),repeat(context['obstacles']),repeat(context['mask']),
        np.repeat(bank,replicas,axis=0),repeat(parent['route']['points']),repeat(np.array(parent['route']['mask'],bool)),
        repeat(context['noise']),repeat(context['bias_x']),repeat(context['bias_o']),
        np.stack([tapes[r%replicas][0] for r in range(size)]),np.stack([tapes[r%replicas][1] for r in range(size)]),
        np.full(size,context['cursor'],np.float64))
    if 'nominal_bias' in context:args+= (repeat(context['nominal_bias']),)
    return branches,args


def graph_args(parent,context):
    return (context['observed'],np.array(parent['goal']),context['observed_obstacles'],context['mask'],
        np.array(parent['route']['points']),np.array(parent['route']['mask'],bool),np.array(context['cursor']),
        context['previous_u'],context['previous_gain'],context['noise'])


def route_coordinate(position,points,mask,cursor):
    """Independent physical projection near the recorded causal route cursor."""
    vectors=np.diff(points,axis=0);valid=mask[:-1]&mask[1:];length=np.where(valid,np.linalg.norm(vectors,axis=-1),0.)
    cum=np.r_[0.,np.cumsum(length)];frac=np.clip(np.sum((position-points[:-1])*vectors,axis=-1)/np.maximum(length**2,1e-12),0.,1.)
    arc=cum[:-1]+frac*length;eligible=valid&(arc>=cursor-1.)&(arc<=cursor+1.)
    # Routes always contain a segment. If the hint is beyond local projection
    # support, use nearest full segment and record physical progress honestly.
    eligible=eligible if np.any(eligible) else valid
    i=np.argmin(np.where(eligible,np.sum((points[:-1]+frac[:,None]*vectors-position)**2,axis=-1),np.inf))
    return float(arc[i])


def labels(parent,context,rows,traces,c,horizon=HORIZON):
    """Use independent continuous-time outcomes/clearance; never invent a future."""
    goal=np.array(parent['goal']);points=np.array(parent['route']['points']);rm=np.array(parent['route']['mask'],bool)
    _,_,remaining,_=numpy_flight_target(context['observed'],goal,numpy_obstacles(context['observed_obstacles'],context['mask'],context['noise'],True),
        context['mask'],points,rm,context['cursor'],c)
    blend=float(np.clip(1-remaining/2.5,0.,1.))
    start=route_coordinate(context['x'][:2],points,rm,context['cursor'])
    status=np.array([r['audited_status'] for r in rows]);raw_steps=np.array([r['steps'] for r in rows])
    steps=np.array([r['steps'] if r.get('first_physical_stop_step') is None else min(r['steps'],r['first_physical_stop_step']) for r in rows])
    clearance=np.array([r['minimum_clearance'] if r.get('first_physical_stop_step') is None else r['first_physical_stop_clearance'] for r in rows])
    final=np.array([r['final_state'] if r.get('first_physical_stop_step') is None else t['state'][0] if n==0 else t['next_state'][n-1]
                    for r,t,n in zip(rows,traces,steps,strict=True)])
    route_progress=np.array([route_coordinate(x[:2],points,rm,context['cursor'] if n==0 else float(t['route_progress'][min(n-1,len(t['route_progress'])-1)]))-start
                             for x,t,n in zip(final,traces,steps,strict=True)])
    goal_progress=np.linalg.norm(context['x'][:3]-goal)-np.linalg.norm(final[:,:3]-goal,axis=-1)
    performance=((1-blend)*route_progress+blend*goal_progress)/(horizon*c.robot.dt*c.cruise_speed)+(status==1)*(1-steps/horizon)
    observed=np.isin(status,[1,4,6]);collision=status==4;adverse=~np.isin(status,[1,6])
    risk=-np.minimum(clearance,.6)/.3
    target=np.stack((np.where(observed,risk,0.),performance),-1).astype(np.float32)
    target_mask=np.stack((observed,np.ones_like(observed)),-1)
    events=np.stack((collision,adverse),-1).astype(np.float32)
    event_mask=np.stack((observed,np.ones_like(observed)),-1)
    assert np.isfinite(target).all()
    return dict(target=target,target_mask=target_mask,events=events,event_mask=event_mask,status=status,steps=steps,recorded_physical_steps=raw_steps,
        min_clearance=clearance,final_state=final,physical_initial_state=np.repeat(context['x'][None],len(rows),axis=0),
        route_progress=route_progress,goal_progress=goal_progress,initial_terminal_blend=np.full(len(rows),blend))
