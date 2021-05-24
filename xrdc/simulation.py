"""
Functions for simulating and augmenting 1D XRD data.
"""
import numpy as np
from . import xrdutils as xdu

def transition(start_scale, end_scale, start_pair, end_pair, N, sigma = .01, scale_type = 'scale'):
    """
    pair -> [qs, intensities]
    
    scale_type == 'scale' changes peak positions multiplicatively (in q)
    scale_type == 'shift' shifts peak positions equally (which is not what a lattice parameter change does,
        but it's simpler than doing the q -> log(q) substitution and pretty much equivalent for model testing
        purposes)
        
    TODO
    Lorentz factor
    peak height variation
    """
    scales = np.linspace(start_scale, end_scale, N)
    res = []
    q0, i0 = start_pair
    q1, i1 = end_pair
    
    qq, ii = np.hstack((q0, q1)), np.hstack((i0, i1))
    span = np.abs(start_scale - end_scale)
    for i, s in enumerate(scales):
        if scale_type == 'scale':
            q, i = qq * s, np.hstack((i0 * (N - i) / N, i1 * i / N)) #todo off by one
        elif scale_type == 'shift':
            q, i = qq + s, np.hstack((i0 * (N - i) / N, i1 * i / N)) #todo off by one
        else:
            raise Exception 
        pat = xdu.mk_pattern(i, q, sigma)
        res.append(pat)
    return np.vstack(res)

def sample_transition(X, n_classes = 24, per_class = 500, scale_min = -.5, scale_max = .5, scale_type = 'shift', **kwargs):
    X, _ = xdu.agg(X, n_classes = n_classes)
    if scale_type == 'scale':
        assert scale_min > 0
    X_samples, y, _ = xdu.mk_simdata(X, per_class, scale_min, scale_max,  xdu.q_grid, scale_type = scale_type, **kwargs)
    return X_samples, y
