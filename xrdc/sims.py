import numpy as np
from . import utils

def mk_simdata(patterns, n_per_basis, n_basis, scale_type = 'shift',\
        q_dim = 150):
    if scale_type == 'shift':
        rmin, rmax = -.5, .5
    elif scale_type == 'scale':
        rmin, rmax = .7, 1.
    print('scale type:', scale_type)
    sampled_patterns = np.vstack([np.vstack([utils.condense(
        utils.mutate_pattern(basis, np.random.uniform(rmin, rmax), scale_type = scale_type), q_dim)
                                         for _ in range(n_per_basis)])
                       for basis in patterns])
    y = np.hstack([np.repeat(i, n_per_basis) for i in range(n_basis)])[None, :].T
    return sampled_patterns, y
