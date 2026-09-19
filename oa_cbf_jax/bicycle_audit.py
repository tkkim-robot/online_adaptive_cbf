"""Independent NumPy bicycle barrier derivatives and physical references."""
import numpy as np


def reference_flow(_,state,control,config):
    theta,v=state[2:];a,b=control
    return np.array([v*np.cos(theta)-v*np.sin(theta)*b,
        v*np.sin(theta)+v*np.cos(theta)*b,v*b/config.robot.rear_axle_distance,a])


def reference_barrier(state,obstacles,config):
    """Complex-step-compatible direct scalar definition, without Lie formulas."""
    direction=np.stack((np.cos(state[...,2]),np.sin(state[...,2])),axis=-1)
    p=obstacles[...,:2]-state[...,:2];v=obstacles[...,3:5]-state[...,3,None]*direction
    distance=np.sqrt(np.sum(p*p,axis=-1));unit=p/distance[...,None]
    radial=np.sum(unit*v,axis=-1);lateral=unit[...,0]*v[...,1]-unit[...,1]*v[...,0]
    radius=(config.robot.radius+config.clearance_buffer+obstacles[...,2])*config.barrier_inflation
    root=np.sqrt(np.sum(p*p,axis=-1)-radius**2)
    speed=np.sqrt(np.sum(v*v,axis=-1)+config.relative_speed_epsilon**2)
    shape=np.sqrt(config.barrier_inflation**2-1)/radius
    return radial+.5*shape*root*lateral*lateral/speed+shape*root


def reference_rows(state,obstacles,mask,alpha,config):
    x=np.asarray(state,float);o=np.asarray(obstacles,float);mask=np.asarray(mask,bool)
    # Inactive disks may lie anywhere, including at the ego position. Put them
    # at a benign location only for derivative calculation; their rows are0<=1.
    o=o.copy();o[~mask,:2]=x[:2]+np.array([10.,10.])
    radius=(config.robot.radius+config.clearance_buffer+o[:,2])*config.barrier_inflation
    domain=np.sum((o[:,:2]-x[:2])**2,axis=1)-radius**2
    if np.any(domain[mask]<=0):raise ValueError('Barrier derivative outside its geometric domain')
    n=len(o);variables=np.concatenate((np.broadcast_to(x,(n,4)),o[:,:2]),axis=1)
    complex_variables=variables[:,None,:].astype(complex)+1j*1e-25*np.eye(6)[None]
    expanded=np.broadcast_to(o[:,None,:],(n,6,5)).astype(complex).copy()
    expanded[...,:2]=complex_variables[...,4:]
    grad=reference_barrier(complex_variables[...,:4],expanded,config).imag/1e-25
    drift=reference_flow(0,x,np.zeros(2),config)
    g=np.column_stack((reference_flow(0,x,np.array([1.,0.]),config)-drift,
        reference_flow(0,x,np.array([0.,1.]),config)-drift))
    h=reference_barrier(np.broadcast_to(x,(n,4)),o,config)
    a=-grad[:,:4]@g;b=grad[:,:4]@drift+np.sum(grad[:,4:]*o[:,3:5],axis=1)+alpha*h
    a=np.where(mask[:,None],a,0.);b=np.where(mask,b,1.)
    c=config.robot
    bounds=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    rhs=np.array([min(c.acceleration_max,(c.speed_max-x[3])/c.dt),
        -max(-c.acceleration_max,(c.speed_min-x[3])/c.dt),c.slip_max,c.slip_max])
    return np.vstack((a,bounds)),np.r_[b,rhs],h,domain


def polygon_qp(reference,a,b,weights,lower,upper):
    """Independent 2D polygon clipping, then projection onto every boundary edge."""
    reference=np.asarray(reference,float);a=np.asarray(a,float);b=np.asarray(b,float);weights=np.asarray(weights,float)
    x0,y0=lower;x1,y1=upper
    polygon=[np.array([x0,y0]),np.array([x1,y0]),np.array([x1,y1]),np.array([x0,y1])]
    for row,rhs in zip(a,b):
        clipped=[]
        for start,end in zip(polygon,polygon[1:]+polygon[:1]):
            fs,fe=row@start-rhs,row@end-rhs
            if fs<=1e-12:clipped.append(start)
            if (fs<0<fe) or (fe<0<fs):clipped.append(start+(end-start)*fs/(fs-fe))
        polygon=clipped
        if not polygon:return None
    if np.max(a@reference-b)<=1e-12:return reference.copy()
    candidates=[]
    for start,end in zip(polygon,polygon[1:]+polygon[:1]):
        delta=end-start;t=np.clip(np.dot(weights*(reference-start),delta)/max(np.dot(weights*delta,delta),1e-30),0.,1.)
        candidates.append(start+t*delta)
    objective=[.5*np.sum(weights*(u-reference)**2) for u in candidates]
    return candidates[int(np.argmin(objective))]
