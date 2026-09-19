"""Unicycle optimal-decay baselines with all obstacle rows and live coefficients.

repo: the pinned repository's two freely optimized decay coefficients, retaining
its objective and inequality. hocbf: optimize the decay of a fixed first HOCBF
psi=hdot+alpha1*h, with omega>=1, following the relative-degree-two construction
in https://arxiv.org/html/2507.12717v1, IV-A. These are separate comparators.
Neither implementation claims continuous-time safety from sampled noisy checks.
"""

from dataclasses import dataclass
import math
import jax.numpy as jnp
from .config import UnicycleConfig
from .route_control import route_problem


@dataclass(frozen=True)
class OptimalDecayConfig:
    formulation: str = 'repo'
    alpha1: float = .5
    alpha2: float = .5
    penalty: float = 1e4

    def __post_init__(self):
        if self.formulation not in ('repo','hocbf'):
            raise ValueError('Unknown optimal-decay formulation')
        if any(not math.isfinite(v) or v<=0 for v in (self.alpha1,self.alpha2,self.penalty)):
            raise ValueError('Gains and decay penalty must be finite and positive')

    @property
    def weights(self):
        # A common factor of 1/2 leaves the repository minimizer unchanged.
        return (1.,1.)+(self.penalty,)*(2 if self.formulation=='repo' else 1)


def optimal_decay_problem(x,goal,obstacles,mask,points,route_mask,progress,
                          robot=UnicycleConfig(),config=OptimalDecayConfig(),speed_uncertainty=0.):
    reference,a,b,h,hdot,cursor,remaining,target=route_problem(
        x,goal,obstacles,mask,jnp.zeros(2,x.dtype),points,route_mask,progress,robot,speed_uncertainty)
    n=obstacles.shape[0]
    psi=hdot+config.alpha1*h
    if config.formulation=='repo':
        # hddot + (a1+a2)*omega1*hdot + a1*a2*omega2*h >= margin.
        coefficients=jnp.stack(((config.alpha1+config.alpha2)*hdot,config.alpha1*config.alpha2*h),axis=-1)
        extra=jnp.concatenate((-jnp.where(mask[:,None],coefficients,0.),jnp.zeros((4,2),x.dtype)))
        matrix=jnp.concatenate((a,extra),axis=-1)
        ref=jnp.concatenate((reference,jnp.ones(2,x.dtype)))
    else:
        # psidot + omega*a2*psi >= margin, omega>=1. alpha1 remains fixed.
        extra=jnp.concatenate((-jnp.where(mask,config.alpha2*psi,0.),jnp.zeros(4,x.dtype)))[:,None]
        matrix=jnp.concatenate((a,extra),axis=-1)
        b=b.at[:n].add(jnp.where(mask,config.alpha1*hdot,0.))
        matrix=jnp.concatenate((matrix,jnp.array([[0.,0.,-1.]],x.dtype)))
        b=jnp.concatenate((b,jnp.array([-1.],x.dtype)))
        ref=jnp.concatenate((reference,jnp.ones(1,x.dtype)))
    return ref,matrix,b,h,psi,cursor,remaining,target
