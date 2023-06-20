import os
import scipy
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

#import gpflow
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score
from sklearn.mixture import GaussianMixture as GMM


from sklearn.cluster import KMeans

#from gpflow.ci_utils import ci_niter

from scipy.interpolate import interp1d

from .utils.utils import utils
#q_grid, I_ref = pd.read_csv("../../TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/TiNiSn_500C_Y20190218_14x14_t60_0003_bkgdSub_1D.csv", header = None).values.T

#q_grid, I_ref = pd.read_csv("/Users/ollie/Documents/SLAC/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/TiNiSn_500C_Y20190218_14x14_t60_0003_bkgdSub_1D.csv", header = None).values.T
q_grid, I_ref = pd.read_csv(utils.resource_path("inputs/TiNiSn_500C_Y20190218_14x14_t60_0003_bkgdSub_1D.csv"), header = None).values.T
#q_grid = np.arange(1.5, 7.5, 6 / 800)

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

def to_q(twotheta, lamb = 1.54184):
    rad = twotheta * np.pi / 180
    return (4 * np.pi / lamb) * np.sin(rad / 2)

def to_twotheta(q, lamb = 1.54184):
    return 2 * np.arcsin(q * lamb / (4 * np.pi)) * 180 / np.pi


def random_qs(N, vmin, vmax):
    return vmin + ((vmax - vmin) * np.random.uniform(size = N))

def gauss(q, mu, I, sigma = .02):
    #print(I)
    return I * np.exp(-((q - mu)**2) / (2 * sigma**2))

from functools import reduce


def rand_pattern(npeaks = 5, sigma = .02):
    return mk_pattern(np.random.uniform(size = npeaks), random_qs(npeaks, q_grid.min(), q_grid.max()), sigma)

def mk_pattern(intensities, qs, sigma, norm = True, with_grid = False):
    res = reduce(lambda a, b: a + b, [gauss(q_grid, mu, I, sigma) for I, mu in zip(intensities, qs)])
    if norm:
        if with_grid:
            return q_grid, res / res.mean()
        else:
            return res / res.mean()
    else:
        if with_grid:
            return q_grid, res 
        else:
            return res 

def mk_generate_peak_scales(q_grid, sigma_peakvar = 20, scale = .15, clip = True):
    from gpflow.kernels import RBF
    from gpflow.likelihoods import Gaussian
    from gpflow.mean_functions import MeanFunction
    from gpflow.models import GPR
    from gpflow.base import Parameter
    # todo not in module scope
    dq = q_grid[1] - q_grid[0]
    sigma_peakvar = dq * sigma_peakvar
    jitter = 1e-6
    Xs = q_grid[:, None]
    kernel = RBF(lengthscales=sigma_peakvar)
    cov = kernel(Xs)
    L = np.linalg.cholesky(cov + np.eye(len(q_grid)) * jitter)
    def generate_peak_scales(N=800):
        """
        Model peak height variation as a guassian process.

        Returns:
            Tuple of np.arrays of size (N, 1) and (N, num_functions).
        """
        epsilon = np.random.randn(N) * scale #todo parameterize
        F = 1 + np.matmul(L, epsilon)
        if clip:
            mask = (F < 0)
            F[mask] = 0
        return F
    return generate_peak_scales


def poisson_additive(pattern, noise_scale = 1, poisson_lam = 30):
    """
    Add nonnegative noise to the pattern
    """
    return np.random.poisson(lam = poisson_lam, size = len(pattern)) / (poisson_lam / (noise_scale * pattern.mean()))

from scipy.ndimage.filters import gaussian_filter
def mutate_pattern(pattern, scale, q_grid, sigma_width = None, scale_type = 'scale', peak_height = True,
        q_jitter_magnitude = None, default_grid = True, noise_scale = 1, poisson_lam = 30,
        noise_type = 'uniform'):
    generate_peak_scales_default = mk_generate_peak_scales(q_grid)
    if scale_type == 'scale':
        interp = extrap1d(interp1d(q_grid * scale, pattern))
        relative_lorentz = lorentz_q(q_grid * scale) / lorentz_q(q_grid)
    elif scale_type == 'shift':
        interp = extrap1d(interp1d(q_grid - scale, pattern))
        # todo put lorentz factor back in
        relative_lorentz = 1 #lorentz_q(q_grid - scale) / lorentz_q(q_grid)
        

    if q_jitter_magnitude is not None:
        resampled = interp(q_grid * generate_peak_scales(scale = q_jitter_magnitude))
    else:
        resampled = interp(q_grid)
    resampled = resampled * relative_lorentz
    if peak_height:
        if default_grid:
            resampled = resampled  * generate_peak_scales_default()
        else:
            resampled = resampled  * mk_generate_peak_scales(q_grid)(q_grid.shape[0])#generate_peak_scales_default(q_grid)
    #print(relative_lorentz)

    if noise_type == 'uniform':
        noise = np.random.uniform(size = len(q_grid)) / 50
    elif noise_type == 'poisson':
        noise = poisson_additive(pattern, noise_scale = noise_scale, poisson_lam = poisson_lam)
    else:
        noise = 0
    return resampled + noise


def condense(arr, newsize, norm = True):
    extra = newsize - (len(arr) % newsize)
    arr = np.hstack((arr, np.zeros(extra)))
    arr = arr.reshape(newsize, -1).mean(axis = 1)
    if norm:
        arr = arr / arr.mean()
    return arr

def condense2d(arr, size):
    return np.vstack([condense(row, size) for row in arr])

def standardize_input(X, Y):
    X = xdu.condense2d(X, 150)
    Y = Y[:, None]
    return X, Y

def mk_simdata(patterns, n_per_basis, rmin, rmax, q_grid, y = None, scale_type = 'shift',\
        q_dim = 150, peak_height = True, q_jitter_magnitude = None, apply_condense = True, **kwargs):
#    if scale_type == 'shift':
#        rmin, rmax = -.5, .5
#    elif scale_type == 'scale':
#        rmin, rmax = .7, 1.
    n_basis = len(patterns)
    print('scale type:', scale_type)
    print('q jitter:', q_jitter_magnitude)
    print('peak height variation', peak_height)
    scale_vs = np.random.uniform(rmin, rmax, (n_basis, n_per_basis))
    def one_sample(i, j, basis):
        mutated = mutate_pattern(basis, scale_vs[i][j], q_grid, scale_type = scale_type,\
                        peak_height = peak_height, q_jitter_magnitude = q_jitter_magnitude, **kwargs)
        if apply_condense:
            return condense( mutated, q_dim)
        else:
            return mutated

    sampled_patterns = np.vstack(
                            [np.vstack(
                                [one_sample(i, j, basis)
                                    for j in range(n_per_basis)])
                                for (i, basis) in enumerate(patterns)])
    if y is None:
        ynew = np.hstack([np.repeat(i, n_per_basis) for i in range(n_basis)])[None, :].T
    else:
        ynew = np.hstack([np.repeat(y0, n_per_basis) for y0 in y])[None, :].T
    return sampled_patterns, ynew, scale_vs

def write_train_test(prefix, x_train, y_train, x_val, y_val):
    os.makedirs(prefix + '/train', exist_ok=True)
    os.makedirs(prefix + '/test', exist_ok=True)
    
    np.save("{}x.npy".format(prefix + '/train/'), x_train)
    np.save("{}y.npy".format(prefix + '/train/'), y_train)
    np.save("{}x.npy".format(prefix + '/test/'), x_val)
    np.save("{}y.npy".format(prefix + '/test/'), y_val)
    
    
def from_json(df):
    patterns = np.array(df.pattern.iloc[0], dtype = object)
    intensities, angles = patterns[:, 0].astype(float), patterns[:, 2].astype(float)
    q = to_q(angles)
    return q, intensities

def df_to_pat(df):
    q, i = from_json(df)
    pat = mk_pattern(i, q, .01)
    return q, i, pat

def plt_pat(df, **kwargs):
    q, i, pat = df_to_pat(df)
    plt.plot(q_grid, pat, **kwargs)
    return q_grid, pat, i, q\

def agg(X, n_classes = 25):
    CoNi_members = X.reshape(n_classes, 600 // n_classes, 800).sum(axis = 1)
    CoNi_y = np.hstack([np.repeat(i, 500) for i in range(n_classes)])[None, :].T
    return CoNi_members, CoNi_y

def sample_transition(*pieces, n_classes = 24, per_class = 500):
    X = np.vstack(pieces)
    X, y = agg(X, n_classes = n_classes)
    X_samples, y = mk_simdata(X, per_class, -.5, .5, scale_type = 'shift')
    return X_samples, y

def group_patterns(*pieces, n_classes = 200, rmin = -.2, rmax = .2, randomize = False, apply_condense = True):
    """
    Instead of averaging the rows and doing data augmentation, just group them.
    """
    if randomize == False:
        raise NotImplementedError
    def mutate_one(row):
        mutated = mutate_pattern(row, np.random.uniform(rmin, rmax), scale_type = 'shift', peak_height = True)
        if apply_condense:
            return condense(mutated, 150)
        else:
            return mutated

    per_class = 600 // n_classes
    X = np.vstack(pieces)
    y = np.hstack([np.repeat(i, per_class) for i in range(n_classes)])[None, :].T
    X = np.vstack([mutate_one(row) for row in X])
    return X, y


def gcluster(X, N = 6):
    gmm = GMM(n_components=N, covariance_type='full').fit(X)
    ypred = gmm.predict(X)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    Xpca = pca.fit_transform(X)

    return Xpca, ypred

from scipy.interpolate import interp1d
def logtransform(x, y):
    logx = np.log(x)
    interp = interp1d(logx, y)
    logx_new = np.linspace(logx.min(), logx.max(), len(logx))
    return logx_new, interp(logx_new)


def mk_basis_patterns(n_basis):
    return [xrdutils.rand_pattern() for _ in range(n_basis)]

def gen_train_and_test_with_scales(n_basis, n_per_basis, prefix, same_basis = True, scale_type = 'shift'):
    # TODO rename
    """
    Generate random patterns and write the patterns, labels, and random scale/shift values to disk
    """
    basis = mk_basis_patterns(n_basis)
    x_train, y_train, scale_vs_train = xrdutils.mk_simdata(basis, n_per_basis, -.6, .2, xrdutils.q_grid, scale_type = scale_type)
    if not same_basis:
        basis = mk_basis_patterns(n_basis)
    x_val, y_val, scale_vs_val = xrdutils.mk_simdata(basis, n_per_basis, -.6, .2, xrdutils.q_grid, scale_type = scale_type)
    os.makedirs(prefix + '/train', exist_ok=True)
    os.makedirs(prefix + '/test', exist_ok=True)
    
    scales_train = scale_vs_train.ravel()
    scales_val = scale_vs_val.ravel()
    
    np.save("{}x.npy".format(prefix + '/train/'), x_train)
    np.save("{}y.npy".format(prefix + '/train/'), y_train)
    np.save("{}x.npy".format(prefix + '/test/'), x_val)
    np.save("{}y.npy".format(prefix + '/test/'), y_val)
    
    np.save("{}scales.npy".format(prefix + '/train/'), scales_train)
    np.save("{}scales.npy".format(prefix + '/test/'), scales_val)
    return x_train, y_train, x_val, y_val, scales_train, scales_val

def standardize_input(X, Y):
    X = condense2d(X, 150)
    Y = Y[:, None]
    return X, Y

def augment(X, y, rmin, rmax, q, N = 10, scale_type = 'scale'):
    """
    Take N transformed samples of each row in X.
    
    Return new X, y (with y matching source pattern index)
    """
    y = np.hstack([np.repeat(v, N) for v in y])[None, :].T
    X = np.vstack([np.vstack([mutate_pattern(row, np.random.uniform(rmin, rmax), q  / 10,
                                           scale_type = scale_type, peak_height = True,
                                 default_grid = False) for _ in range(N)])
            for row in X])
    return X, y
