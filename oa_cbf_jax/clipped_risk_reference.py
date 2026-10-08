"""Independent NumPy/SciPy references for observed clipped-mixture quantities."""
import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtr
from scipy.stats import norm

FLOOR=-2.


def parameters(prediction,scale=1.):
    m,v,w=(np.asarray(prediction[k],float) for k in
           ('risk_component_mean','risk_component_variance','risk_component_probability'))
    if (m.shape!=v.shape or m.shape!=w.shape or m.shape[-1]!=2
            or not all(np.isfinite(a).all() for a in (m,v,w)) or np.any(v<=0) or np.any(w<0)):
        raise ValueError('Invalid mixture arrays')
    np.testing.assert_allclose(w.sum(-1),1.,atol=2e-6,rtol=0)
    return m,v*scale,w/w.sum(-1,keepdims=True)


def tail(mean,variance,weights,mass=.01):
    sigma=np.sqrt(variance)
    bounds=mean+sigma*norm.isf(mass)
    lo,hi=bounds.min(-1),bounds.max(-1)
    for _ in range(64):
        mid=lo+(hi-lo)*.5
        above=(weights*ndtr((mean-mid[...,None])/sigma)).sum(-1)>mass
        lo,hi=np.where(above,mid,lo),np.where(above,hi,mid)
    q=np.maximum(FLOOR,lo+(hi-lo)*.5)
    z=(mean-q[...,None])/sigma
    excess=np.maximum(0.,(mean-q[...,None])*ndtr(z)+sigma*norm.pdf(z))
    return q,q+(weights*excess).sum(-1)/mass


def disagreement(mean,variance,weights):
    """Independent explicit member/component sums; member axis first."""
    e=mean.shape[0]
    overlap=np.empty((e,e,*mean.shape[1:-1]),float)
    # Work in log space so even well-separated, narrow distributions are valid.
    with np.errstate(divide='ignore'):
        atom_log=logsumexp(np.log(weights)+log_ndtr((FLOOR-mean)/np.sqrt(variance)),axis=-1)
        for i in range(e):
            for j in range(e):
                terms=[atom_log[i]+atom_log[j]]
                for a in range(mean.shape[-1]):
                    for b in range(mean.shape[-1]):
                        m,n=mean[i,...,a],mean[j,...,b]
                        v,w=variance[i,...,a],variance[j,...,b]
                        precision=1/v+1/w
                        product_mean=(m/v+n/w)/precision
                        terms.append(np.log(weights[i,...,a])+np.log(weights[j,...,b])
                            +norm.logpdf(m,n,np.sqrt(v+w))
                            +log_ndtr((product_mean-FLOOR)*np.sqrt(precision)))
                overlap[i,j]=logsumexp(np.stack(terms),axis=0)
    own=np.stack([overlap[i,i] for i in range(e)])
    return np.maximum((.5*(own[:,None]+own[None,:])-overlap).mean((0,1)),0.)
