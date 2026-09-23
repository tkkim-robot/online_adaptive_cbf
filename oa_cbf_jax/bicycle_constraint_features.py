"""Current DPCBF geometry as deterministic, gain-independent neural features.

Uses only the existing graph35. No latent physical state, future, candidate
rollout, QP solve, selected gain, or training label enters this transform.
"""
import numpy as np
import jax.numpy as jnp


def contract():
    return dict(schema='bicycle_graph35_current_constraint_transform',input_features=35,encoded_features=41,
        added_fields=['asinh_domain_over_inflated_radius_squared','positive_domain',
            'asinh_h_over_speed_max','asinh_drift_over_speed_max',
            'asinh_acceleration_authority_times_limit_over_speed_max',
            'asinh_slip_authority_times_limit_over_speed_max'],
        source='Existing observed ego-frame graph35 only; reconstruct its rounded geometry and declared dynamics.',
        precision='FP64 feature arithmetic, cast to original neural dtype; same transform for both encoders.',
        invalid_domain='Keep signed domain and explicit validity; zero undefined barrier derivatives.',
        invalid_or_padded_nodes='Padded nodes and ego/goal added fields are zero.',
        gain_dependent=False,controller_solve=False,training_labels_used=False,
        limitation='Describes instantaneous constraints at graph-rounded observations; not a future feasibility certificate.')


def append_constraints(features,mask):
    """B,N,35 -> B,N,41. Stateless and permutation equivariant."""
    if features.ndim!=3 or features.shape[-1]!=35 or mask.shape!=features.shape[:2]:
        raise ValueError('Current bicycle constraints require batched graph35 and its mask')
    f=jnp.where(mask[...,None],features,0.).astype(jnp.float64);ego=f[:,0,:];ob=f[:,2:,:];valid=mask[:,2:]
    vmax=ego[:,16,None];speed=ego[:,9,None]*vmax;rear=ego[:,12,None]
    radius=(ego[:,10,None]+ego[:,18,None]+ob[...,7])*ego[:,19,None]
    p=ob[...,3:5]*5.;relative=ob[...,5:7]*vmax[...,None]
    distance2=jnp.sum(p*p,-1);distance=jnp.sqrt(jnp.maximum(distance2,1e-24));unit=p/distance[...,None]
    normal=jnp.stack((-unit[...,1],unit[...,0]),axis=-1)
    radial=jnp.sum(unit*relative,-1);lateral=jnp.sum(normal*relative,-1)
    domain=distance2-radius**2;admissible=valid&(domain>0)
    # Benign guards are only arithmetic guards; validity is an explicit input.
    root=jnp.sqrt(jnp.maximum(domain,1e-12));r=jnp.maximum(radius,1e-12)
    speed_relative=jnp.sqrt(jnp.sum(relative*relative,-1)+ego[:,20,None]**2)
    speed_relative=jnp.maximum(speed_relative,1e-12)
    factor=jnp.sqrt(jnp.maximum(ego[:,19,None]**2-1,0))/r;lam=.5*factor
    h=radial+lam*root*lateral**2/speed_relative+factor*root
    dz=p/root[...,None];drad=lateral[...,None]*normal/distance[...,None];dlat=-radial[...,None]*normal/distance[...,None]
    dp=drad+(lam*lateral**2/speed_relative)[...,None]*dz+(2*lam*root*lateral/speed_relative)[...,None]*dlat+factor[...,None]*dz
    dv=unit+(lam*root)[...,None]*(2*lateral[...,None]*normal/speed_relative[...,None]-lateral[...,None]**2*relative/speed_relative[...,None]**3)
    drift=jnp.sum(dp*relative,-1);aa=-dv[...,0];ab=-speed*dp[...,1]-speed**2/jnp.maximum(rear,1e-12)*dv[...,1]
    derived=jnp.stack((h,drift,aa*ego[:,13,None],ab*ego[:,14,None]),axis=-1)/jnp.maximum(vmax[...,None],1e-12)
    extra=jnp.concatenate((jnp.arcsinh(domain/jnp.maximum(radius**2,1e-12))[...,None],admissible[...,None].astype(f.dtype),
        jnp.where(admissible[...,None],jnp.arcsinh(derived),0.)),axis=-1)
    extra=jnp.where(valid[...,None],extra,0.)
    extra=jnp.concatenate((jnp.zeros((len(f),2,6),f.dtype),extra),axis=1).astype(features.dtype)
    return jnp.concatenate((jnp.where(mask[...,None],features,0.),extra),axis=-1)


def numpy_constraints(features,mask):
    """Independent direct barrier + complex-step derivatives, vectorized."""
    f=np.asarray(features,dtype=float);mask=np.asarray(mask,bool)
    if f.ndim!=3 or f.shape[-1]!=35 or mask.shape!=f.shape[:2]:raise ValueError('Expected batched graph35')
    result=np.zeros((*f.shape[:2],6));ego=f[:,0,:];ob=f[:,2:,:]
    vmax=ego[:,16,None];speed=ego[:,9,None]*vmax
    p=ob[...,3:5]*5.;relative=ob[...,5:7]*vmax[...,None]
    radius=(ego[:,10,None]+ego[:,18,None]+ob[...,7])*ego[:,19,None]
    domain=np.sum(p*p,axis=-1)-radius**2;valid=mask[:,2:]&(domain>0)
    result[:,2:,0]=np.where(mask[:,2:],np.arcsinh(domain/np.maximum(radius**2,1e-12)),0.)
    result[:,2:,1]=valid
    rows,cols=np.nonzero(valid)
    if not len(rows):return result
    pp=p[rows,cols];vv=relative[rows,cols];rr=radius[rows,cols];inflation=ego[rows,19];eps=ego[rows,20]
    def barrier(position,velocity):
        distance=np.sqrt(np.sum(position*position,axis=-1));unit=position/distance[...,None]
        radial=np.sum(unit*velocity,axis=-1);lateral=unit[...,0]*velocity[...,1]-unit[...,1]*velocity[...,0]
        root=np.sqrt(np.sum(position*position,axis=-1)-rr**2)
        shape=np.sqrt(inflation**2-1)/rr
        return radial+.5*shape*root*lateral*lateral/np.sqrt(np.sum(velocity*velocity,axis=-1)+eps**2)+shape*root
    h=barrier(pp,vv);dp=np.zeros_like(pp);dv=np.zeros_like(vv)
    for k in range(2):
        changed=pp.astype(complex);changed[:,k]+=1e-25j;dp[:,k]=barrier(changed,vv).imag/1e-25
        changed=vv.astype(complex);changed[:,k]+=1e-25j;dv[:,k]=barrier(pp,changed).imag/1e-25
    drift=np.sum(dp*vv,axis=-1);v=speed[rows,0]
    authority_a=-dv[:,0];authority_b=-v*dp[:,1]-v*v/ego[rows,12]*dv[:,1]
    values=np.column_stack((h,drift,authority_a*ego[rows,13],authority_b*ego[rows,14]))/ego[rows,16,None]
    result[rows,cols+2,2:]=np.arcsinh(values)
    return result


def verify_features(features,mask,batch=256):
    """Every supplied observed graph, comparing independent derivative methods."""
    import hashlib
    import jax
    f=np.asarray(features);mask=np.asarray(mask,bool);fn=jax.jit(append_constraints);exe=None;maximum=0.;count=0
    digest=hashlib.sha256()
    for start in range(0,len(f),batch):
        n=min(batch,len(f)-start);a=np.pad(f[start:start+n],((0,batch-n),(0,0),(0,0)),mode='edge')
        m=np.pad(mask[start:start+n],((0,batch-n),(0,0)),mode='edge')
        # Explicit staging is necessary with global x64 disabled: passing a
        # NumPy float64 array to lower() otherwise silently canonicalizes FP32.
        a=jnp.asarray(a,dtype=jnp.float64 if f.dtype==np.float64 else jnp.float32)
        m=jnp.asarray(m,dtype=bool)
        if exe is None:exe=fn.lower(a,m).compile()
        actual=np.asarray(exe(a,m))[:n];reference=numpy_constraints(f[start:start+n],mask[start:start+n])
        np.testing.assert_array_equal(actual[...,:35],f[start:start+n])
        np.testing.assert_allclose(actual[...,35:],reference,atol=3e-6,rtol=2e-6)
        if not np.isfinite(actual).all() or np.any(actual[~mask[start:start+n]]!=0):raise ValueError('Nonfinite or nonzero padded features')
        maximum=max(maximum,float(np.max(abs(actual[...,35:]-reference))));count+=int((reference[...,1]>0).sum());digest.update(actual.tobytes())
    if fn._cache_size():raise ValueError('Implicit feature compilation')
    return dict(queries=len(f),positive_domain_obstacles=count,independent_max_error=maximum,
        encoded_features_sha256=digest.hexdigest(),source_features_sha256=hashlib.sha256(f.tobytes()).hexdigest(),
        feature_signature_count=1,implicit_jit_cache_entries=0,every_feature_independently_verified=True,contract=contract())
