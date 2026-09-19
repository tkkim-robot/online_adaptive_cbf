"""Deterministic scene records; known demos and random families stay distinct."""

from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Scene:
    scene_id: str
    family: str
    seed: int
    initial_state: np.ndarray
    goal: np.ndarray
    obstacles: np.ndarray
    obstacle_mask: np.ndarray

    def json_record(self):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(self).items()}


def fixture(name, capacity=8):
    x = np.array([0., 0., .02, .4]); goal = np.array([5., 0.])
    rows = []
    if name == "open":
        x[3] = 0.
    elif name == "offset":
        rows = [[2.5, .6, .3, 0., 0.]]
    elif name == "blocking":
        rows = [[1.2, 0., .4, 0., 0.]]
    elif name == "collision":
        rows = [[0., 0., .3, 0., 0.]]
    elif name == "crossing":
        rows = [[2.5, -1., .3, 0., .3]]
    else:
        raise ValueError(f"Unknown fixture {name}")
    obs = np.zeros((capacity, 5)); obs[:len(rows)] = np.array(rows).reshape(-1,5)
    return Scene(f"fixture:{name}", name, 0, x, goal, obs, np.arange(capacity)<len(rows))


def random_scene(seed, family="scatter", capacity=8, count=None):
    rng = np.random.default_rng(seed)
    count = int(rng.integers(1, capacity+1)) if count is None else count
    if not 0 <= count <= capacity:
        raise ValueError("Obstacle capacity exceeded; truncation is forbidden")
    x = np.array([0., 0., rng.uniform(-.7,.7), rng.uniform(0.,1.)])
    goal = np.array([rng.uniform(4.,8.), rng.uniform(-1.5,1.5)])
    rows = []
    for _ in range(count):
        for attempt in range(1000):
            r = rng.uniform(.15,.45)
            p = rng.uniform([.8,-2.8],[goal[0]-.3,2.8])
            if family == "corridor":
                p[1] = rng.choice([-1.,1.]) * rng.uniform(.7,1.1)
            elif family == "cluster":
                p[0] = goal[0]*.5 + rng.uniform(-.6,.6)
            elif family not in ("scatter", "moving"):
                raise ValueError(f"Unknown family {family}")
            if min(np.linalg.norm(p-x[:2]),np.linalg.norm(p-goal)) > r+.4:
                break
        else:
            raise ValueError("Invalid initial scene construction")
        vel = rng.uniform(-.3,.3,2) if family == "moving" else np.zeros(2)
        rows.append([*p, r, *vel])
    obs = np.zeros((capacity,5)); obs[:count] = np.asarray(rows).reshape(-1,5)
    return Scene(f"{family}:{seed}",family,seed,x,goal,obs,np.arange(capacity)<count)


DIVERSE_FAMILIES=('scatter','corridor','cluster','moving','head_on','overtaking','bottleneck','u_trap')


def diverse_scene(seed, family, capacity=16):
    """Development family expansion, with planar yaw/translation randomization.

    Geometric routing does not certify these scenes dynamically solvable. All
    are initially labelled unknown until a genuine feasible witness is replayed.
    """
    if family not in DIVERSE_FAMILIES:raise ValueError('Unknown diverse family')
    rng=np.random.default_rng(seed+190019)
    if family in DIVERSE_FAMILIES[:4]:
        scene=random_scene(seed,family,capacity)
    else:
        x=np.array([0.,0.,rng.uniform(-.5,.5),rng.uniform(.1,.8)])
        goal=np.array([rng.uniform(5.5,8.),rng.uniform(-.5,.5)])
        if family in ('head_on','overtaking'):
            count=int(rng.integers(2,min(capacity,10)+1))
            p=rng.uniform([1.5,-2.], [goal[0]-.5,2.], (count,2))
            velocity=np.column_stack((rng.uniform(.15,.7,count)*(1 if family=='overtaking' else -1),rng.uniform(-.12,.12,count)))
            rows=np.column_stack((p,rng.uniform(.18,.4,count),velocity))
            x[3]=rng.uniform(.6,1.) if family=='overtaking' else rng.uniform(.1,.7)
        elif family=='bottleneck':
            radius=rng.uniform(.25,.4);half_gap=.3+radius+rng.uniform(.06,.4)
            y=half_gap+np.arange(4)*2*radius*.95
            positions=np.column_stack((np.full(8,rng.uniform(2.5,3.5)),np.r_[y,-y]))
            rows=np.column_stack((positions,np.full(8,radius),np.zeros((8,2))))
        else:
            x[3]=rng.uniform(0.,.2);wall_x=np.arange(-1.4,1.61,.6)
            positions=np.concatenate((np.column_stack((wall_x,np.full(len(wall_x),1.3))),
                                      np.column_stack((wall_x,np.full(len(wall_x),-1.3))),
                                      np.array([[2.2,0.],[2.2,.65],[2.2,-.65]])))
            rows=np.column_stack((positions,np.full(len(positions),rng.uniform(.28,.36)),np.zeros((len(positions),2))))
        if len(rows)>capacity:raise ValueError(f'Obstacle capacity exceeded: {len(rows)} > {capacity}')
        obs=np.zeros((capacity,5));obs[:len(rows)]=rows
        scene=Scene(f'{family}:{seed}',family,seed,x,goal,obs,np.arange(capacity)<len(rows))
    yaw=rng.uniform(-np.pi,np.pi);offset=rng.uniform(-5.,5.,2)
    rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    scene.initial_state[:2]=scene.initial_state[:2]@rotation.T+offset
    scene.initial_state[2]=np.arctan2(np.sin(scene.initial_state[2]+yaw),np.cos(scene.initial_state[2]+yaw))
    scene.goal=scene.goal@rotation.T+offset
    scene.obstacles[:,:2]=scene.obstacles[:,:2]@rotation.T+offset
    scene.obstacles[:,3:5]=scene.obstacles[:,3:5]@rotation.T
    scene.obstacles[~scene.obstacle_mask]=0.
    return scene
