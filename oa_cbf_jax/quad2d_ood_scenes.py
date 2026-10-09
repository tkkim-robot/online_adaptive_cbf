"""Shared quad2d ood scenes implementation."""

import numpy as np

FAMILIES=('alternating_gates','zigzag_channel','offset_rooms','nested_open_boxes',
          'crossing_streams','counterflow_lanes','closing_gate','large_disks')

def geometry(seed,family):
    if family not in FAMILIES:
        raise ValueError('Unknown held topology family')
    rng=np.random.default_rng(seed);rows=[];start=np.array([-6.,0.]);goal=np.array([6.,0.])
    def disk(x,z,r=.23,vx=0.,vz=0.):rows.append([x,z,r,vx,vz])
    if family=='alternating_gates':
        for j,x in enumerate((-3.6,-1.2,1.2,3.6)):
            opening=(-1 if j%2 else 1)*rng.uniform(.7,1.3)
            for z in np.linspace(-3.,3.,11):
                if abs(z-opening)>.85:disk(x,z)
    elif family=='zigzag_channel':
        phase=rng.uniform(-.4,.4)
        for x in np.linspace(-4.8,4.8,22):
            center=.9*np.sin(.8*x+phase)
            disk(x,center-1.05,.20);disk(x,center+1.05,.20)
    elif family=='offset_rooms':
        for x,gap in [(-2.5,rng.uniform(-2.,-1.2)),(0.,rng.uniform(1.2,2.)),(2.5,rng.uniform(-2.,-1.2))]:
            for z in np.linspace(-4.,4.,15):
                if abs(z-gap)>.9:disk(x,z,.24)
    elif family=='nested_open_boxes':
        start=np.array([0.,0.]);goal=np.array([6.5,0.])
        for extent,n,opening_side in [(1.8,5,1),(3.8,9,-1)]:
            coordinates=np.linspace(-extent,extent,n)
            for x in coordinates:
                disk(x,-extent,.22);disk(x,extent,.22)
            for z in coordinates[1:-1]:
                for side in (-1,1):
                    if side==opening_side and abs(z)<1.:continue
                    disk(side*extent,z,.22)
    elif family=='crossing_streams':
        for x in np.linspace(-3.5,3.5,6):
            for sign in (-1,1):
                disk(x,sign*rng.uniform(2.,3.8),rng.uniform(.16,.25),rng.uniform(-.1,.1),-sign*rng.uniform(.3,.75))
        for z in (-1.,1.):
            for x in np.linspace(-3.5,3.5,6):
                sign=1 if z<0 else -1
                disk(x,z,.18,sign*rng.uniform(.45,.85),0.)
    elif family=='counterflow_lanes':
        for lane,z in enumerate((-.8,0.,.8)):
            sign=-1 if lane%2 else 1
            for x in np.linspace(-4.5,4.5,10):
                disk(x,z+rng.uniform(-.08,.08),rng.uniform(.14,.20),sign*rng.uniform(.4,1.1),0.)
    elif family=='closing_gate':
        speed=rng.uniform(.12,.22)
        for sign in (-1,1):
            for z in np.linspace(1.25,4.6,7):
                disk(0.,sign*z,.24,0.,-sign*speed)
        # Stationary approach islands discourage a trivial straight side shift,
        # while preserving an open plane and the same initial route information.
        for x,z in [(-3.,-1.4),(-3.,1.4),(3.,-1.4),(3.,1.4)]:disk(x,z,.35)
    else:
        start=np.array([-7.,0.]);goal=np.array([7.,0.])
        for x,z in [(-3.,-1.3),(-1.5,1.5),(0.,-1.4),(1.5,1.5),(3.,-1.3)]:
            disk(x+rng.uniform(-.15,.15),z+rng.uniform(-.2,.2),rng.uniform(.8,1.15))
    obs=np.asarray(rows,np.float64)
    scale=rng.uniform(.9,1.1);angle=rng.uniform(-np.pi/8,np.pi/8);shift=rng.uniform(-2.,2.,2)
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    obs[:,:2]=scale*obs[:,:2]@rotation.T+shift;obs[:,2]*=scale;obs[:,3:5]=scale*obs[:,3:5]@rotation.T
    start=scale*start@rotation.T+shift;goal=scale*goal@rotation.T+shift
    if len(obs)>64 or len(obs)==0 or not np.isfinite(obs).all():raise ValueError('Invalid topology capacity')
    for point in (start,goal):
        if np.min(np.linalg.norm(obs[:,:2]-point,axis=1)-obs[:,2]-.3)<=.15:
            raise ValueError('Constructive template violates its initial/goal geometric separation')
    padded=np.zeros((64,5),np.float32);padded[:len(obs)]=obs
    state=np.r_[start,rng.uniform(-.1,.1),rng.uniform(-.2,.2,2),rng.uniform(-.12,.12)].astype(np.float32)
    return state,goal.astype(np.float32),padded,np.arange(64)<len(obs)
