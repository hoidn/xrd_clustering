from ipywidgets import interactive
import nibabel as nib
import k3d
import math
import numpy as np
from k3d.helpers import download
from torch.utils.data import TensorDataset, DataLoader
import torch
from . import utils

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
