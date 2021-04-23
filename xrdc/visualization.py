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
