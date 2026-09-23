"""Causal observed-position memory and the shared training/runtime feature path."""
import math
import numpy as np
from .bicycle_motion_observer import WINDOW_TICKS
from .bicycle_motion_features import append_history, numpy_features, contract, SCHEMA


def validate_metadata(metadata):
    if (metadata.get('graph_features')!=39
            or metadata.get('architecture',{}).get('bicycle_motion_history') is not True
            or metadata.get('bicycle_motion_history_contract')!=contract()
            or metadata.get('bicycle_contract',{}).get('graph_schema')!=SCHEMA):
        raise ValueError('Changed observed motion-history model contract')
    from .bicycle_features import SCHEMA as ORIGINAL_SCHEMA
    if metadata['bicycle_contract'].get('source_graph_schema')!=ORIGINAL_SCHEMA:
        raise ValueError('Changed original graph source schema')


class ObservationHistory:
    """One fixed-shape batch, consecutive ticks, stable obstacle indices.

    Store only rounded observed positions; tick20 reads tick0 before replacing
    that slot. No state, true velocity, label or future observation is accepted.
    """
    def __init__(self,batch,capacity,dt):
        if batch<1 or capacity<1 or not math.isfinite(dt) or dt<=0:raise ValueError('Invalid history dimensions/time step')
        self.shape=(batch,capacity,2);self.dt=float(dt);self.next_tick=0
        self.positions=np.zeros((WINDOW_TICKS,*self.shape),np.float32)

    def observe(self,obstacles,tick):
        if type(tick) is not int or tick!=self.next_tick:raise ValueError('History requires consecutive ticks starting at zero')
        current=np.asarray(obstacles,np.float32)
        if current.shape!=(*self.shape[:-1],5):raise ValueError('Changed history batch or obstacle capacity')
        current=current[...,:2]
        past=current.copy() if tick==0 else self.positions[max(0,tick-WINDOW_TICKS)%WINDOW_TICKS].copy()
        self.positions[tick%WINDOW_TICKS]=current;self.next_tick+=1
        return past,np.full(self.shape[0],min(tick,WINDOW_TICKS)*self.dt,np.float64)


def graph(x,goal,obstacles,mask,points,route_mask,cursor,previous_control,previous_gain,noise,
          past_positions,elapsed,*,config,compute_dtype):
    from .bicycle_features import bicycle_inference_graph
    features,node_mask=bicycle_inference_graph(x,goal,obstacles,mask,points,route_mask,cursor,
        previous_control,previous_gain,noise,config=config,compute_dtype=compute_dtype)
    return append_history(features,node_mask,x,obstacles,past_positions,noise,elapsed),node_mask


def audit_trace_history(data,dt):
    """Reconstruct directly from recorded past observations, without the ring."""
    observed=data['observed_obstacles'];n=len(observed)
    if not np.array_equal(data['global_tick'],np.arange(n)):raise ValueError('Nonconsecutive observed history')
    for tick in range(n):
        past=observed[max(0,tick-WINDOW_TICKS),:,:2].astype(np.float32)
        np.testing.assert_array_equal(data['history_past_positions'][tick],past)
        np.testing.assert_array_equal(data['history_elapsed'][tick],min(tick,WINDOW_TICKS)*dt)
        features=data['features'][tick];mask=data['node_mask'][tick]
        expected=numpy_features(features[:,:35],mask,data['observed_state'][tick],observed[tick],past,
            data['noise'],min(tick,WINDOW_TICKS)*dt)
        np.testing.assert_allclose(features,expected,atol=3e-6,rtol=2e-6)
    return dict(causal_history_queries=n,all_past_observations_verified=True,
        independent_history_features_verified=True,physical_truth_used=False)
