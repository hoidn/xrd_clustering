from scipy.signal import argrelextrema
from scipy.ndimage.filters import gaussian_filter as gf
import matplotlib.pyplot as plt
import numpy as np
import os

from .. import xrdutils as xdu
from .. import simulation as sim
from .. import simulation
from .. import datasets as xd

from ..waferutils.Clustering.data_grid import DataGrid

from ..utils.utils import utils
#from data_grid import DataGrid

"""
This module loads a compositional scan dataset and reshapes it into a 3d array with dimensions [wafer x, wafer y, q]

Also includes utilities for calculating 4-dimensional similarity cubes from 3-dimensional data.
"""

def get_coords(dgrid):
    coords = np.array([dgrid.coord(ii + 1) for ii in range(177)])
    return coords

def norm3d(arr, axis = 0, log_scale = False):
    """
    Log scale: scale the standard deviation along each feature dimension to the
    mean value of that feature
    
    For axis == 1, we scale features to mean and std, but only consider non-zero values.
    """
    xi, yi, zi = np.nonzero(arr)
    xxi, yyi, zzi = lambda i: xi[xi == i], lambda i: yi[xi == i], lambda i: zi[xi == i]
    arr = arr.copy()
    global_min = arr[xi, yi, zi].min()
    print(global_min)
    for i in range(arr.shape[0]):
        ai = arr[xxi(i), yyi(i), zzi(i)]
        arr[xxi(i), yyi(i), zzi(i)] = (ai - ai.min()) / ai.std()
        if log_scale:
            arr[xxi(i), yyi(i)] *= (np.log(ai.mean() - global_min + 1))
    return arr

swap3d = lambda arr: np.einsum("ijk->kij", arr)

def csim3d(activations_n1):
    #activations_n1 = swap3d(activations_n1)
    tmp = np.einsum("kij, klm->ijlm", activations_n1, activations_n1)

    norm = np.linalg.norm(activations_n1, axis = 0)

    norm4d = np.einsum("ij, kl->ijkl", norm, norm)

    tmp_csim = tmp / (norm4d + 1e-30)
    return tmp_csim

def gridstack(arr4d):
    return np.vstack(np.dstack(np.transpose(arr4d, (1, 0, 2, 3))))

def plot_clusters_grid(coords, clust, nclust):
    import matplotlib
    dim = coords.max()
    coordi = coords - 1
    hues = [float(float(x)/float(nclust)) for x in range(1,nclust+1)]
    cg = cluster_grid = np.zeros(shape = (dim,dim,3))
    for k, (x, y) in enumerate(coordi):
        cluster = clust[k]
        cluster_grid[x, y] = matplotlib.colors.hsv_to_rgb([hues[cluster],1,1])
#     if debug:
#         print(coords)

#     fig = plt.figure()
#     fig.tight_layout()
    start, end = nclust - 1, nclust
    Big_labels = []
    Big_labels.append(clust)
    plt.imshow(cg)
    

debug = False
nclust = 11# i = 2


path = os.path.dirname(
    utils.resource_path("inputs/")
    ) + '/'

#path = '/Users/ollie/Documents/SLAC/xrd_clustering/notebooks/../../TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/'
#path = "../../TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/"



dgrid = DataGrid(path, """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSub_1D.csv""",range=None)
#dgrid = DataGrid(path, """TiNiSn_500C_Y20190218_14x14_t60_0003_bkgdSub_1D.csv""",range=None)

q = dgrid.data[1][:,0]
X = dgrid.get_data_array()
X /= X.sum(axis = 1)[:, None]

coords = get_coords(dgrid)    
coords_to_idx = {tuple(c): i for i, c in enumerate(coords)}
coordi = coords - 1
dim = coordi.max() + 1

def to_3d(X):
    m3d = np.zeros((dim, dim, X.shape[1]))
    for k, (i, j) in enumerate(coordi):
        m3d[i, j, :] = X[k]
    return m3d

def to_2d(m3d):
    res = np.zeros_like(X)
    for k, (i, j) in enumerate(coordi):
        res[k] = m3d[i, j, :] 
    return res

m3d = to_3d(X)

def get_data():
    return m3d

def get_2d():
    return X

def oned_to_wafergrid(values):
    coordi = coords - 1
    dim = coords.max()
    cg = cluster_grid = np.zeros(shape = (dim,dim))
    for k, (x, y) in enumerate(coordi):
        cluster_grid[x][y] = values[k]
    return cluster_grid
