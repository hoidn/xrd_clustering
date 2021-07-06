"""
Functions for simulating and augmenting 1D XRD data.
"""
import numpy as np
from . import xrdutils as xdu

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
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

# FCC Ni definition
lparam_Ni = 3.52
lat = Lattice(lparam_Ni * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
fcc_basis = np.array(((0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)))
struct = Structure(lat, ["Ni"] * len(fcc_basis), fcc_basis)

def xrd_from_geo(lattice, basis, species):
    gen = XRDCalculator("CuKa")
    lat = Lattice(lattice)
    struct = Structure(lat, species, basis)
    
    pat = gen.get_pattern(struct)
    return xdu.to_q(pat.x), pat.y

def distort_Ni(relative_c, common_scale = 1.):
    return xrd_from_geo((lparam_Ni * common_scale) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, relative_c]]), fcc_basis, ["Ni"] * 4)

def Ni_cubic_to_tetra(N, c_start, c_end, sigma = .01):
    """
    """
    cs =  np.linspace(c_start, c_end, N)
    res = []
    ordinal = []
    for c in cs:
        q, i = distort_Ni(c)
        qq, ii = xdu.mk_pattern(i, q, sigma, with_grid=True)
        ordinal.append(np.vstack([q, i]))
        res.append(ii)
    return ordinal, np.vstack(res)

def Ni_cubic_scale(N, s_start, s_end, sigma = .01):
    """
    """
    ss =  np.linspace(s_start, s_end, N)
    res = []
    ordinal = []
    for c in ss:
        q, i = distort_Ni(1, c)
        qq, ii = xdu.mk_pattern(i, q, sigma, with_grid=True)
        ordinal.append(np.vstack([q, i]))
        res.append(ii)
    return ordinal, np.vstack(res)

#TODO refactor

def Ni_tetra_random(N, cmin, cmax, scalemin, scalemax, sigma = .01):
    """
    """
    cs = cmin + np.random.uniform(size = N) * (cmax - cmin)
    scales = scalemin + np.random.uniform(size = N) * (scalemax - scalemin)
    res = []
    ordinal = []
    for c, s in zip(cs, scales):
        q, i = distort_Ni(c, common_scale=s)
        qq, ii = xdu.mk_pattern(i, q, sigma, with_grid=True)
        ordinal.append(np.vstack([q, i]))
        res.append(ii)
    return ordinal, np.vstack(res)

def Ni_cubic_random(N, scalemin, scalemax, sigma = .01):
    """
    """
    scales = scalemin + np.random.uniform(size = N) * (scalemax - scalemin)
    res = []
    ordinal = []
    for c in scales:
        q, i = distort_Ni(1, c)
        qq, ii = xdu.mk_pattern(i, q, sigma, with_grid=True)
        ordinal.append(np.vstack([q, i]))
        res.append(ii)
    return ordinal, np.vstack(res)
