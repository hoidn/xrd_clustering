from ipywidgets import interactive
import nibabel as nib
import k3d
import math
import numpy as np
from k3d.helpers import download
from torch.utils.data import TensorDataset, DataLoader
import torch
from . import utils
import matplotlib.pyplot as plt

from .datasets import d3d

def do_pca(X):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    Xpca = pca.fit_transform(X)
    return pca, Xpca

def xrd_to_embedding(xface, yface, net, embedding_cb):
    #tensor_x = torch.Tensor(CoNiSn_x_shift_faces[:, None, :]) # transform to torch tensor
    tensor_x = torch.Tensor(xface[:, None, :]) # transform to torch tensor

    #tensor_y = torch.Tensor(CoNiSn_y_shift_faces)
    tensor_y = torch.Tensor(yface[:, None])

    my_dataset = TensorDataset(tensor_x, tensor_y) # create your datset
    dataloader = DataLoader(my_dataset) # create your dataloader
    
    emb_faces, y_faces = embedding_cb(net, dataloader)
    return emb_faces, y_faces

def xrd_to_pca(x, y, net, embedding_cb):
    emb_faces, y_faces = xrd_to_embedding(x, y, net, embedding_cb)
    pca, pca_faces = do_pca(emb_faces)
    return pca, pca_faces


def xrd_to_pca_original(net,  loader, embedding_cb):
    x, y = embedding_cb(net, loader)
    pca, pca_faces = do_pca(x)
    return pca, pca_faces

def heatmap3d(patterns, interpolation = True, color_range = [0.2,.9], scale = True, auto_color = True, **kwargs):
    """
    Plot a 3-dimensional heatmap.
    """
    img = patterns
    dx, dy, dz = 1, 1, 1#nii_source.header.get_zooms()

    if scale:
        img = img.astype(np.float32) / img.max()#np.swapaxes(img,0,2).astype(np.float32)
        img = np.log(1 + 10 * img)

    nz, ny, nx = img.shape

    if auto_color:
        volume = k3d.volume(img, interpolation = interpolation, bounds = np.array([0, 1, 0, .25, 0, .25]),
            color_range=color_range, color_map=np.array(k3d.basic_color_maps.Jet, dtype=np.float32), **kwargs)
    else:
        volume = k3d.volume(img, interpolation = interpolation, bounds = np.array([0, 1, 0, .25, 0, .25]),
            color_range=color_range, **kwargs)


    plot = k3d.plot()
    plot += volume
    plot.display()
    return plot, volume


def iplot_volume(patterns, log = False, offset = 0, height = '550px'):
    """
    Plot a 3D heatmap with adjustable truncating plane.
    """
    plot, volume = heatmap3d(patterns)
    def f(i):
        volume.samples = 512 * 2
        plot.clipping_planes = [[-1, 0, 0, 1 - i]]

    interactive_plot = interactive(f, i=(0, 1, .005))
    return interactive_plot

def overlay_mask(img, mask, ax = None, **kwargs):
    grayscale = 1 - mask
    alpha = mask
    rgb_img = np.dstack((grayscale, (grayscale), grayscale, (alpha)))
    if ax is not None:
        ax.imshow(np.log(1 + img), cmap = 'jet', **kwargs)
        ax.imshow(rgb_img, cmap = 'Greys', **kwargs)
    else:
        plt.imshow(np.log(1 + img), cmap = 'jet', **kwargs)
        plt.imshow(rgb_img, cmap = 'Greys', **kwargs)


imargs = {'cmap': 'jet', 'aspect': 8}
def heatmap(fig, arr, label = '', aspect = None, **kwargs):
    ax = fig

    stride = 100
    qticks = d3d.q[::stride]
    labels = ['{:0.1f}'.format(elt) for elt in qticks]
    
    ax.set_xticklabels(labels);
    ax.set_xticks(np.arange(len(qticks)) * stride);
    if kwargs is not None:
        (kwargs.update(imargs))
    else:
        kwargs = imargs
    if aspect:
        kwargs['aspect']  = aspect
    plt.imshow(arr, **kwargs)
    plt.title(label)
    annotate()

def annotate():
    plt.xlabel('$q~(1 / A)$')
    plt.ylabel('sample index')

def plot_clusters_grid(coords, clust, nclust, colors = None, **kwargs):
    import matplotlib
    dim = coords.max()
    coordi = coords - 1
    hues = [float(float(x)/float(nclust)) for x in range(1,nclust+1)]

    cg = cluster_grid = np.zeros(shape = (dim,dim,3))
    for k, (x, y) in enumerate(coordi):
        cluster = clust[k]
        if colors is None:
            colors = matplotlib.colors.hsv_to_rgb([hues[cluster],1,1])
        cluster_grid[x, y] = colors
#     if debug:
#         print(coords)
#     fig = plt.figure()
#     fig.tight_layout()
    start, end = nclust - 1, nclust
    Big_labels = []
    Big_labels.append(clust)
    plt.imshow(cg, **kwargs)
