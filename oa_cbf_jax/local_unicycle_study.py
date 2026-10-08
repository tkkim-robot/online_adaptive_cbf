"""Prospective density/speed/neighborhood study with eight-second held gains.

No learned model is deployed or fitted. Gain enumeration diagnoses physical
decision capacity only; its best outcome is never reported as OA performance.
All physical obstacles are retained in collision measurement, even with local
graph/QP observations. Nominal goal tracking is common to every treatment.
"""


import numpy as np


from .config import UnicycleConfig


from .obstacle_selection import nearest_obstacles, nearest_numpy, contract


FAMILIES=('alternating_fronts','paired_gates','offset_clusters','wavy_channel')


GAINS=np.asarray([(v,v) for v in (.5,1.,2.,4.,8.)]+[(.5,8.),(8.,.5),(1.,4.),
    (4.,1.),(1.,8.),(8.,1.),(2.,4.),(4.,2.),(2.,8.),(8.,2.),(4.,8.)],np.float32)
ROBOT=UnicycleConfig(a_max=.5,w_max=.5,v_max=2.,stationary_obstacles=True)


def make_scene(seed, family, density, side):
    """Paired hidden surroundings, identical robot and nearest-obstacle input.

    Density changes longitudinal spacing, not robot/obstacle radii. Every
    generated parent is retained; no rejection based on controller outcomes.
    """
    rng=np.random.default_rng(seed)
    spacing=1. if density=='dense' else 1.6
    lead=np.array([3.,.85,.3,0.,0.])
    rows=[lead]
    radii=rng.uniform(.22,.30,31)
    jitter=rng.uniform(-.035,.035,(31,2))
    for i in range(31):
        if family=='alternating_fronts':
            px=4.+spacing*.68*i
            py=(-1 if i%2 else 1)*(radii[i]+ROBOT.radius+.10+.08*rng.random())
        elif family=='paired_gates':
            level=i//2;px=4.+spacing*1.35*level+.10*(i%2)
            center=.25*np.sin(level*.7)
            py=center+(1 if i%2 else -1)*(radii[i]+ROBOT.radius+.19)
        elif family=='offset_clusters':
            level=i//5;local=i%5
            px=4.+spacing*(2.9*level+.40*local)
            py=(1 if level%2 else -1)*(.67+.40*(local%3))
        elif family=='wavy_channel':
            level=i//2;px=4.+spacing*.75*level
            center=.20*np.sin(level*.45)
            py=center+(1 if i%2 else -1)*(radii[i]+ROBOT.radius+.17)
        else:raise ValueError('Unknown local study family')
        rows.append([px+jitter[i,0],side*(py+jitter[i,1]),radii[i],0.,0.])
    obs=np.asarray(rows,np.float32)
    # Both members of each alias pair have exactly the same leading obstacle,
    # initial robot state, goal, density and speed. Only farther obstacles differ.
    goal=np.array([float(obs[:,0].max()+3.),0.],np.float32)
    if not np.array_equal(nearest_numpy(np.zeros(2),obs,np.ones(32,bool),1)[0][0],lead.astype(np.float32)):
        raise ValueError('A secondary obstacle replaced the shared initial nearest')
    return obs,goal
