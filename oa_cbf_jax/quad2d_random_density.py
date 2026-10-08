"""Prospectively seeded random static obstacle-density study, without outcomes.

Rejection sampling checks disk geometry only. It never queries a policy or
retries a failed route. The four counts share the same world extent.
"""


from pathlib import Path
import numpy as np

from .dataset import sha256


from .quad2d_static_inputs import read

FAMILIES = ('random_field', 'paired_clusters', 'staggered_layers', 'random_pockets')
COUNTS = (24, 36, 48, 60)
NOISE = (0., .5, 1., 2.)
REPLICAS = 32
GROUPS = len(FAMILIES)*len(COUNTS)*len(NOISE)*REPLICAS
SEED = 2609285100
SCHEMA = 'quad2d_random_density_development_v1'


def fields(index, *, seed=SEED, schema=SCHEMA):
    if type(index) is not int or not 0 <= index < GROUPS:
        raise ValueError('Index outside the complete prospective reservation')
    family, rest = divmod(index, len(COUNTS)*len(NOISE)*REPLICAS)
    density, rest = divmod(rest, len(NOISE)*REPLICAS)
    noise_index, replica = divmod(rest, REPLICAS)
    if type(seed) is not int or seed < 0 or not isinstance(schema,str) or not schema:
        raise ValueError('Require an explicit nonnegative seed and scene namespace')
    rng = np.random.default_rng(seed+index)
    count = COUNTS[density]
    disks = []
    # Uniform proposal box, all counts use the same physical extent. Structured
    # proposals add local pairs/occlusions, not a policy-dependent difficulty.
    pocket_centers = np.array([(x, z) for x in (4., 8., 12., 16.) for z in (-2., 2.)])
    pocket_centers += rng.uniform(-.5, .5, pocket_centers.shape)
    cell_order = rng.permutation(60)
    attempts = 0
    while len(disks) < count:
        attempts += 1
        if attempts > 20000:
            raise ValueError('Geometric sampler exhausted; preserve seed, never substitute')
        radius = rng.uniform(.19, .36)
        if family == 1 and len(disks)%2:
            anchor = np.asarray(disks[-1])
            theta = rng.uniform(-np.pi, np.pi)
            distance = anchor[2]+radius+rng.uniform(.10, .38)
            position = anchor[:2]+distance*np.array([np.cos(theta), np.sin(theta)])
        elif family == 2:
            cell = int(cell_order[len(disks)])
            column, row = divmod(cell, 6)
            position = np.array([2.8+1.4*column, -3.5+1.35*row])
            position += np.array([rng.uniform(-.18,.18), .22*(-1)**column+rng.uniform(-.18,.18)])
        elif family == 3:
            position = pocket_centers[rng.integers(len(pocket_centers))]+rng.normal(0., 1.0, 2)
        else:
            position = rng.uniform([2.5,-4.3],[16.5,4.3])
        if not (2.3 <= position[0] <= 17. and -4.5 <= position[1] <= 4.5):
            continue
        if disks:
            previous = np.asarray(disks)
            if np.min(np.linalg.norm(previous[:,:2]-position,axis=1)-previous[:,2]-radius) < .08:
                continue
        disks.append([*position, radius, 0., 0.])
    obstacles = np.zeros((64,5), np.float32)
    obstacles[:count] = np.asarray(disks, np.float32)[rng.permutation(count)]
    # Shared plant gravity stays vertical; only the world geometry is rotated.
    angle = rng.uniform(-np.pi/12, np.pi/12)
    rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    shift = rng.uniform(-3.,3.,2)
    obstacles[:count,:2] = obstacles[:count,:2]@rotation.T+shift
    goal = np.array([19.5, rng.uniform(-.5,.5)])@rotation.T+shift
    state = np.r_[shift, rng.uniform(-.08,.08), rng.uniform(-.15,.15,2), rng.uniform(-.1,.1)].astype(np.float32)
    noise = np.float32(NOISE[noise_index])*np.array([.015,.01,.015,.015,.02,.02,.008],np.float32)
    return dict(group_id=f'{schema}:{seed+index}', seed=seed+index,
        family=FAMILIES[family], density=density, obstacle_count=count,
        declared_noise_scale=NOISE[noise_index], replica=replica,
        partition='random_dense_development', initial_state=state.tolist(),
        goal=goal.astype(np.float32).tolist(), obstacles=obstacles.tolist(),
        obstacle_mask=(np.arange(64)<count).tolist(), noise=noise.tolist(),
        geometry_proposals=attempts, solvability='unknown', route_retry_authorized=False)


def geometry(row, config):
    o = np.asarray(row['obstacles'],float)[row['obstacle_mask']]
    distances = np.linalg.norm(o[:,None,:2]-o[None,:,:2],axis=-1)
    separation = distances-o[:,None,2]-o[None,:,2]
    np.fill_diagonal(separation,np.inf)
    c = config['robot']; inflation = c['radius']+c['clearance_buffer']
    clearance = [float(np.min(np.linalg.norm(o[:,:2]-np.asarray(p)[:2],axis=1)-o[:,2]-inflation))
                 for p in (row['initial_state'],row['goal'])]
    if len(o)!=row['obstacle_count'] or np.any(o[:,3:]!=0) or min(clearance)<=.2 or separation.min()<.07999:
        raise ValueError('Invalid static disk/endpoint geometry')
    return dict(minimum_surface_gap=float(separation.min()), endpoint_clearance=clearance,
        mean_neighbors_3m=float(((distances<3.)&(distances>0.)).sum(1).mean()),
        disk_area=float(np.sum(np.pi*o[:,2]**2)), geometry_only=True)


def verify(root):
    root=Path(root);m=read(root/'manifest.json');rows=read(root/'scenes.json')
    if m['schema']!=SCHEMA or m['groups']!=GROUPS or len(rows)!=GROUPS or m['training_use'] or m['weight_fit_authorized'] or m['final_test']:
        raise ValueError('Changed full random-density reservation')
    for filename,key in (('scenes.json','scenes_sha256'),('geometry_inventory.json','geometry_inventory_sha256'),
                         ('reservation.json','reservation_sha256'),('unrouted_scenes.json','unrouted_scenes_sha256')):
        if sha256(root/filename)!=m[key]:raise ValueError('Changed scene evidence')
    for i,row in enumerate(rows):
        if any(row[k]!=v for k,v in fields(i).items()):raise ValueError('Changed randomized parent')
    if [geometry(r,m['config']) for r in rows]!=read(root/'geometry_inventory.json'):
        raise ValueError('Changed density inventory')
    return rows
