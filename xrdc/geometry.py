import ternary
#from k_means_constrained import KMeansConstrained as KMC
import numpy as np

epsilon = 1e-9

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

# Distribution to sample edges and corners more
def sample_1d(size = 1):
    rmax = 5
    yl = np.random.laplace(size = size)
    m1 = (yl < 0)
    yl[m1] = (rmax + yl[m1])
    #m2 = ((yl <= rmax) & (yl >= 0))
    yl = np.abs(yl) / rmax
    return yl

def get_simplex_sample(N, mode = 'uniform', tol = .001, kmeans = True):
    """
    Randomly sample points in the planar simplex.

    N: number of points to sampole
    mode: 'uniform' or 'biased'. 'biased' samples with highest marginal density at the edge of
        each dimension.
    """
    # todo 
    N0 = 3000000
    if mode == 'uniform':
        xx, yy, zz = np.random.uniform(size = N0), np.random.uniform(size = N0), np.random.uniform(size = N0)
    elif mode == 'biased':
        xx, yy, zz = sample_1d(N0), sample_1d(N0), sample_1d(N0)
    else:
        raise ValueError
    #print(xx.mean())
    mask = (((xx + yy + zz) < 1. + tol) & ((xx + yy + zz) > 1. - tol))

    xx, yy, zz = xx[mask] / 1., yy[mask] / 1., zz[mask] / 1.

    simplex_sample = np.vstack((xx, yy, zz)).T
    simplex_sample = simplex_sample / np.sum(simplex_sample, axis = 1)[:, None]
    simplex_sample = simplex_sample[:N]

    
    if kmeans:
        kmeans = KMC(n_clusters=10, size_min = 20).fit(simplex_sample)
        fig, tax = ternary.figure()

        for i in range(10):
            tax.scatter(simplex_sample[kmeans.labels_ == i], marker='s', s = 2)

        simplex_sample = simplex_sample[np.argsort(kmeans.labels_)]
    #print(simplex_sample.shape)
    return simplex_sample


# center of the simplex
x0, y0 = .5, np.sqrt(3/4) - 1/ np.sqrt(3)
r0 = np.sqrt(1/3)

def reflect(x, y):
    """
    Rotate simplex points into the 'first trirant' of the simplex (its lower third)
    """
    x, y = x.copy(), y.copy()
    mask1 = (y > ((2 * y0) * (1 - x))) & (x > .5)
    mask2 = (y > ((2 * y0) * x)) & (x < .5)
    
    x[mask1], y[mask1] = rotate((x0, y0), (x[mask1], y[mask1]), -2 * np.pi / 3)
    x[mask2], y[mask2] = rotate((x0, y0), (x[mask2], y[mask2]), 2 * np.pi / 3)
    
    print('this should be balanced:', mask1.sum(), mask2.sum(), (~(mask1 | mask2)).sum())
    return x, y

def center_to_edge(alpha, R = 1.):
    """todo"""
    offset = np.pi / 2 - np.arccos(.5)
    return R / (2 * np.cos(np.arccos(np.sin(3 * (alpha + offset))) / 3))


def simplex_pt_to_alpha(x, y):
    alpha = np.arctan(np.abs(x - x0) / (np.abs((y - y0)) + epsilon))
    return alpha

def planar_simplex_to_circle(x, y, R = 1.):
    """
    map planar simplex coords to interior of unit circle
    """
    xr, yr = reflect(x, y)
    scale = R / (r0 * center_to_edge(simplex_pt_to_alpha(xr, yr)))
    xdiff, ydiff = x - x0, y - y0
    return scale * xdiff, scale * ydiff

def circle_to_simplex(x, y):
    R = np.sqrt((x - x.mean())**2 + (y - y.mean())**2).max() + epsilon
    x, y = (x - x.mean()) / R, (y - y.mean()) / R

    xr, yr = reflect(((x - x.mean()) / R + x0), (y - y.mean()) / R + y0)

    #xr, yr = reflect(x + x0, y + y0)
    scale = (r0 * center_to_edge(simplex_pt_to_alpha(xr, yr)))

    return scale * x + x0, scale * y + y0

def csim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
