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
import matplotlib

from .datasets import d3d
from . import featurization as feat

def do_pca(X, n = 2):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n)
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

def overlay_mask(img, mask, ax = None, color = 'black', **kwargs):
    if color == 'black':
        grayscale = 1 - mask
    else:
        grayscale =  mask
    alpha = mask
    rgb_img = np.dstack((grayscale, (grayscale), grayscale, (alpha)))
    if ax is not None:
        ax.imshow(img, cmap = 'jet', **kwargs)
        ax.imshow(rgb_img, cmap = 'Greys', **kwargs)
    else:
        plt.imshow(img, cmap = 'jet', **kwargs)
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


def norm(arr):
    return arr / arr.mean()

def plot_summary(patterns, slow_q, fast_q, slow_T, fast_T, activations_n1,
        straightened_heatmap = None, similarity = 'cos', **kwargs):
    """
    Make a figure summarizing source separation, peak shift correction,
    and similarity matrix for a 2D dataset.
    """
    if similarity == 'cos':
        simfn = feat.csim_pairs
        simlabel = 'Cos'
    elif similarity == 'l2':
        simfn = feat.l2_sim
        simlabel = 'L2'
    else:
        raise ValueError
    if kwargs is not None:
        (imargs.update(kwargs))
    matplotlib.rcParams.update({'font.size': 18})

    fig = plt.figure(figsize=(12, 23), constrained_layout=False)
    spec = fig.add_gridspec(7, 2)

    #imargs = {'cmap': 'jet', 'aspect': 1.5}

    ax0 = fig.add_subplot(spec[0, :])
    heatmap(ax0, np.log(1 + norm(patterns)), label = 'Log raw signal')
    # annotate_axes(ax0, 'ax0')

    ax1 = fig.add_subplot(spec[1, :])
    heatmap(ax1, np.log10(np.abs(fast_T / patterns) ), "Log noise magnitude (relative)")
    #annotate_axes(ax10, 'ax10')


    ax2 = fig.add_subplot(spec[2, :])
    heatmap(ax2, slow_q, 'Extracted background')
    # plt.title("Log signal")

    ax3 = fig.add_subplot(spec[3, :])
    #heatmap(ax3, np.log(1 + norm(fast_q - fast_q.min())))
    heatmap(ax3, np.log(1 + norm((slow_T - slow_q) - (slow_T - slow_q).min())), "Log extracted signal")
    # annotate_axes(ax11, 'ax11')

    if straightened_heatmap is not None:
        ax4 = fig.add_subplot(spec[4, :])
        heatmap(ax4, np.log(1 + straightened_heatmap), interpolation = 'none',
                label = 'Featurization (peak shift-corrected)')

    #imargs = {'cmap': 'jet', 'aspect': 1}

    ax50 = fig.add_subplot(spec[5:7, 0])
    img = ax50.imshow(feat.csim_pairs(patterns), cmap = 'jet')
    plt.title('Cos similarity (raw)')
    plt.colorbar(img, ax = ax50)

    ax51 = fig.add_subplot(spec[5:7, 1])
    img = ax51.imshow(simfn(activations_n1.T), cmap = 'jet')
    plt.title('{} similarity (featurized)'.format(simlabel))
    plt.colorbar(img, ax = ax51)

def iplot(f, arr, height = '550px'):
    """
    Plot a series of curves interactively.
    """
    plt.rcParams["figure.figsize"]=(12, 9)
    interactive_plot = interactive(f, i=(0, len(arr) - 1), step = 1)
    output = interactive_plot.children[-1]
    output.layout.height = height
    return interactive_plot
