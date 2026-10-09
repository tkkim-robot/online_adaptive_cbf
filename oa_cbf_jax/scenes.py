"""Scenes functions and shared contracts."""

from dataclasses import dataclass, asdict, replace

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

def stationary_scene(scene):
    """Keep the generated layout/identity and freeze its physical velocities."""
    obstacles=scene.obstacles.copy()
    obstacles[:,3:5]=0.
    return replace(scene,obstacles=obstacles)

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


FAMILIES=('alternating_chicane','offset_door_sequence','crossing_streams','mixed_slalom')

def scene(seed,family,capacity=16):
    if family not in FAMILIES:raise ValueError('Unknown generalization family')
    rng=np.random.default_rng(seed)
    x=np.array([0.,0.,rng.uniform(-.45,.45),rng.uniform(0.,.8)])
    length=rng.uniform(7.5,10.)
    goal=np.array([length,rng.uniform(-.35,.35)])
    rows=[]
    if family=='alternating_chicane':
        # Three alternating finite bars; the route must change direction twice.
        sign=rng.choice([-1.,1.])
        for i,fraction in enumerate((.27,.5,.73)):
            radius=rng.uniform(.23,.32)
            edge=rng.uniform(-.2,.15)
            for j in range(4):
                rows.append([length*fraction,sign*(-1)**i*(edge+j*1.9*radius),radius,0.,0.])
    elif family=='offset_door_sequence':
        # Two offset doorways, each formed by six disks. Doors are finite, so
        # going around a wall is a legitimate shared-route alternative.
        shift=rng.uniform(.45,.85)*rng.choice([-1.,1.])
        for fraction,center in ((.32,shift),(.66,-shift)):
            radius=rng.uniform(.25,.35);half_gap=radius+.3+rng.uniform(.1,.35)
            for side in (-1.,1.):
                for j in range(3):
                    rows.append([length*fraction,center+side*(half_gap+j*1.9*radius),radius,0.,0.])
    elif family=='crossing_streams':
        # Several independent crossing times and opposing transverse streams.
        count=int(rng.integers(6,13))
        for i in range(count):
            along=rng.uniform(2.,length-1.);speed=rng.uniform(.18,.55);direction=(-1)**i
            crossing_time=along/rng.uniform(.65,1.)+rng.uniform(-1.5,1.5)
            rows.append([along,-direction*speed*crossing_time,rng.uniform(.18,.34),rng.uniform(-.08,.08),direction*speed])
    else:
        # Static slalom geometry composed with crossing and longitudinal motion.
        for i,fraction in enumerate((.25,.43,.61,.79)):
            rows.append([length*fraction,(-1)**i*rng.uniform(.15,.5),rng.uniform(.25,.4),0.,0.])
        for i in range(4):
            along=rng.uniform(2.,length-1.);direction=(-1)**i;speed=rng.uniform(.18,.4)
            rows.append([along,-direction*speed*along,rng.uniform(.18,.3),rng.uniform(-.15,.15),direction*speed])
    if len(rows)>capacity:raise ValueError('Obstacle capacity exceeded; truncation is forbidden')
    obstacles=np.zeros((capacity,5));obstacles[:len(rows)]=rows
    mask=np.arange(capacity)<len(rows)
    yaw=rng.uniform(-np.pi,np.pi);translation=rng.uniform(-5.,5.,2)
    rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    x[:2]=x[:2]@rotation.T+translation
    x[2]=np.arctan2(np.sin(x[2]+yaw),np.cos(x[2]+yaw))
    goal=goal@rotation.T+translation
    obstacles[mask,:2]=obstacles[mask,:2]@rotation.T+translation
    obstacles[mask,3:5]=obstacles[mask,3:5]@rotation.T
    return Scene(f'composition:{family}:{seed}',family,seed,x,goal,obstacles,mask)


PROFILE='multiscale_v1'

def multiscale_scenes_scene(seed,family,capacity=64):
    if family not in DIVERSE_FAMILIES or capacity!=64:
        raise ValueError('Multiscale v1 requires a declared family and capacity64')
    rng=np.random.default_rng(seed+731103)
    if rng.random()<.5:
        value=diverse_scene(seed,family,16)
        value.obstacles=np.pad(value.obstacles,((0,capacity-16),(0,0)))
        value.obstacle_mask=np.pad(value.obstacle_mask,(0,capacity-16))
        value.scene_id=f'{PROFILE}:{family}:{seed}'
        return value
    count=int(rng.choice([32,48,64]));length=rng.uniform(14.,28.)
    x=np.array([0.,0.,rng.uniform(-.65,.65),rng.uniform(0.,.8)])
    goal=np.array([length,rng.uniform(-.4,.4)])
    radius=rng.uniform(.15,.36,count);velocity=np.zeros((count,2))
    if family in ('scatter','moving','head_on','overtaking'):
        position=rng.uniform([1.5,-4.],[length-1.2,4.],(count,2))
        if family=='moving':velocity=rng.uniform(-.35,.35,(count,2))
        if family in ('head_on','overtaking'):
            velocity=np.column_stack((rng.uniform(.12,.65,count)*(1 if family=='overtaking' else -1),rng.uniform(-.12,.12,count)))
            x[3]=rng.uniform(.6,1.) if family=='overtaking' else rng.uniform(.1,.65)
    elif family=='corridor':
        along=np.repeat(np.linspace(1.3,length-1.3,count//2),2)
        # Vary width and slight local offset; do not carve a controller witness.
        center=np.repeat(rng.uniform(-.25,.25,count//2),2)
        sides=np.tile([-1.,1.],count//2)
        lateral=center+sides*(radius+.3+rng.uniform(.12,.7,count))
        position=np.column_stack((along,lateral))
    elif family=='cluster':
        clusters=count//8
        centers=np.column_stack((np.linspace(2.,length-2.,clusters),rng.uniform(-1.8,1.8,clusters)))
        position=np.repeat(centers,8,axis=0)+rng.uniform(-.65,.65,(count,2))
    elif family=='bottleneck':
        positions=[];radii=[]
        for along in np.linspace(2.,length-2.,count//8):
            r=rng.uniform(.23,.36);center=rng.uniform(-.8,.8)
            half_gap=r+.3+rng.uniform(.1,.5)
            for side in (-1.,1.):
                for j in range(4):
                    positions.append([along,center+side*(half_gap+1.9*r*j)]);radii.append(r)
        position=np.asarray(positions);radius=np.asarray(radii)
    else:
        # Preserve a concave start enclosure, then add an extended obstacle field.
        wall=np.arange(-1.4,1.61,.6)
        trap=np.concatenate((np.column_stack((wall,np.full(len(wall),1.3))),
             np.column_stack((wall,np.full(len(wall),-1.3))),[[2.2,0.],[2.2,.65],[2.2,-.65]]))
        position=np.concatenate((trap,rng.uniform([4.,-4.],[length-1.2,4.],(count-len(trap),2))))
        radius[:len(trap)]=rng.uniform(.28,.36);x[3]=rng.uniform(0.,.2)
    obstacles=np.zeros((capacity,5));obstacles[:count]=np.column_stack((position,radius,velocity))
    mask=np.arange(capacity)<count
    yaw=rng.uniform(-np.pi,np.pi);offset=rng.uniform(-10.,10.,2)
    rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    x[:2]=x[:2]@rotation.T+offset
    x[2]=np.arctan2(np.sin(x[2]+yaw),np.cos(x[2]+yaw))
    goal=goal@rotation.T+offset
    obstacles[mask,:2]=obstacles[mask,:2]@rotation.T+offset
    obstacles[mask,3:5]=obstacles[mask,3:5]@rotation.T
    return Scene(f'{PROFILE}:{family}:{seed}',family,seed,x,goal,obstacles,mask)

def contract():
    return dict(name=PROFILE,capacity=64,
        sampling='Per-parent seeded coin:50% original diverse_scene at capacity16, padded without changing active geometry;50% expanded family with32/48/64 active obstacles chosen uniformly.',
        extent='Expanded14..28m start-goal extent, varied density/clearance/relative motion, random global yaw/translation. Small fraction preserves original distribution.',
        lineage='Profile-prefixed parent identity. One actual acquired observation per independent scene; all descendant queries/replicas retain its partition.',
        selection='All generated parents retained, no outcome rejection sampling or feasible-path carving. Static planning is not a dynamically feasible witness.',
        exclusions='No original or modified hero coordinates, no inspected structural/final-test parent IDs. Mandatory waypoint training is a separate pending extension.')
