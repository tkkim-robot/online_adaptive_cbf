"""Shared fixed-FP32 physical/sensor/waypoint kernels for baseline evaluation."""
import time
import jax
import jax.numpy as jnp
from .config import UnicycleConfig
from .dynamics import integrate_unicycle,signed_clearance,swept_disk_clearance
from .routing import route_target,physical_route_coordinate
from .stochastic import conditioned_sensor_model


class PhysicalKernels:
    def __init__(self,capacity,route_capacity,steps,robot=UnicycleConfig()):
        # Keep the common physical prior, RNG and plant identical when an FP64
        # controller is hosted in this process. Only AOT compilation uses this
        # context; runtime executes the fixed FP32 signatures.
        with jax.enable_x64(False):
            self._compile(capacity,route_capacity,steps,robot)

    def _compile(self,capacity,route_capacity,steps,robot):
        x=jnp.zeros(4);obs=jnp.zeros((capacity,5));mask=jnp.zeros(capacity,bool)
        noise=jnp.zeros(6);key=jax.random.PRNGKey(0);points=jnp.zeros((route_capacity,2));rmask=jnp.ones(route_capacity,bool)
        begin=time.perf_counter()
        def prepare(x,obs,mask,noise,key):
            return conditioned_sensor_model(x,obs,mask,noise,key,robot,steps)
        def sense(x,truth_obs,xb,ob,xs,os,innovation,k):
            physical_obs=truth_obs.at[:,:2].set(truth_obs[:,:2]+k*robot.dt*truth_obs[:,3:5])
            return x-xb+.15*xs*innovation[:4],physical_obs-ob+.15*os*innovation[4:].reshape(obs.shape)
        def advance(x,u,truth_obs,mask,k):
            y,sub=integrate_unicycle(x,u,robot.dt,robot.integration_substeps)
            starts=jnp.concatenate((x[None],sub[:-1]));times=k*robot.dt+jnp.arange(robot.integration_substeps)*robot.dt/robot.integration_substeps
            clear=jnp.min(jax.vmap(lambda a,b,t:swept_disk_clearance(a,b,truth_obs,mask,robot.radius,t,t+robot.dt/robot.integration_substeps))(starts,sub,times))
            return y,clear,jnp.maximum(-y[3],y[3]-robot.v_max)
        def arrived(x,goal,noise):
            return (jnp.linalg.norm(x[:2]-goal)+jnp.sqrt(2.)*1.15*noise[0]<=robot.goal_tolerance)&(jnp.abs(x[3])+1.15*noise[2]<=.2)
        self.prepare=jax.jit(prepare).lower(x,obs,mask,noise,key).compile()
        self.sense=jax.jit(sense).lower(x,obs,x,obs,x,jnp.zeros(5),jnp.zeros(4+capacity*5),jnp.int32(0)).compile()
        self.advance=jax.jit(advance).lower(x,jnp.zeros(2),obs,mask,jnp.int32(0)).compile()
        self.target=jax.jit(lambda x,p,m,c:route_target(x,p,m,c,robot)).lower(x,points,rmask,jnp.float32(0)).compile()
        self.coordinate=jax.jit(physical_route_coordinate).lower(x[:2],points,rmask,jnp.float32(0)).compile()
        self.clearance=jax.jit(lambda x,o,m:jnp.min(signed_clearance(x[:2],o,m,robot.radius))).lower(x,obs,mask).compile()
        self.arrived=jax.jit(arrived).lower(x,jnp.zeros(2),noise).compile()
        self.compile_seconds=time.perf_counter()-begin
