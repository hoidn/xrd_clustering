import os
import scipy
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
#q_grid, I_ref = pd.read_csv("../TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/TiNiSn_500C_Y20190218_14x14_t60_0003_bkgdSub_1D.csv", header = None).values.T
q_grid = np.arange(1.5, 7.5, 6 / 800)

def extrap1d(interpolator, extend = 'zero'):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            #return ys[0]
            if extend == 'zero':
                return 0.
            else:
                raise exception
        elif x > xs[-1]:
            #return ys[-1]
            if extend == 'zero':
                return 0.
            else:
                raise exception
        else:
            return interpolator(x)

    def ufunclike(xs):
        try:
            iter(xs)
        except TypeError:
            xs = np.array([xs])
        return np.array(list(map(pointwise, np.array(xs))))

    return ufunclike

def rand_scale(scale_min, scale_max):
    return scale_min + np.random.uniform() * (scale_max - scale_min)

def lorenz_ang(twotheta):
    rad = twotheta * np.pi / 180
    return 1 / np.sin(rad)

def lorentz_q(q):
    twotheta = to_twotheta(q)
    return lorenz_ang(twotheta)

lamb = 1.54184

def to_q(twotheta):
    rad = twotheta * np.pi / 180
    return (4 * np.pi / lamb) * np.sin(rad / 2)

def to_twotheta(q):
    return 2 * np.arcsin(q * lamb / (4 * np.pi)) * 180 / np.pi


def random_qs(N, vmin, vmax):
    return vmin + ((vmax - vmin) * np.random.uniform(size = N))

def gauss(q, mu, I, sigma = .02):
    #print(I)
    return I * np.exp(-((q - mu)**2) / (2 * sigma**2))

from functools import reduce


def rand_pattern(npeaks = 5, sigma = .02):
    return mk_pattern(np.random.uniform(size = npeaks), random_qs(npeaks, q_grid.min(), q_grid.max()), sigma)

def mk_pattern(intensities, qs, sigma):
    return reduce(lambda a, b: a + b, [gauss(q_grid, mu, I, sigma) for I, mu in zip(intensities, qs)])

from scipy.ndimage.filters import gaussian_filter
def mutate_pattern(pattern, scale, sigma_width = None, scale_type = 'scale'):
    if scale_type == 'scale':
        interp = extrap1d(interp1d(q_grid * scale, pattern))
        relative_lorentz = lorentz_q(q_grid * scale) / lorentz_q(q_grid)
    elif scale_type == 'shift':
        interp = extrap1d(interp1d(q_grid - scale, pattern))
        # todo put lorentz factor back in
        relative_lorentz = 1 #lorentz_q(q_grid - scale) / lorentz_q(q_grid)

    resampled = interp(q_grid)
    resampled = resampled * relative_lorentz
    #print(relative_lorentz)

    noise = np.random.uniform(size = len(q_grid)) / 50
    return resampled + noise


def condense(arr, newsize):
    extra = newsize - (len(arr) % newsize)
    arr = np.hstack((arr, np.zeros(extra)))
    return arr.reshape(newsize, -1).sum(axis = 1)

def write_train_test(prefix, x_train, y_train, x_val, y_val):
    os.makedirs(prefix + '/train', exist_ok=True)
    os.makedirs(prefix + '/test', exist_ok=True)
    
    np.save("{}x.npy".format(prefix + '/train/'), x_train)
    np.save("{}y.npy".format(prefix + '/train/'), y_train)
    np.save("{}x.npy".format(prefix + '/test/'), x_val)
    np.save("{}y.npy".format(prefix + '/test/'), y_val)
