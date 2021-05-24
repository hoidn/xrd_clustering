from . import visualization as vis

from sklearn.neighbors import KernelDensity
import numpy as np

def default_bw(a):
    """
    Rule of thumb BW selection
    """
    return a.std() * (a.shape[0]**(-.2))

def approx_mean_density(XX, YY, net, embedding_fn, bandwidth = None):
    a, b = vis.xrd_to_embedding(XX, YY, net, embedding_fn)
    a = a / a.std()
    if bandwidth is None:
        bandwidth = default_bw(a)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(a)

    return np.exp(kde.score_samples(a)).mean()

from sklearn.metrics import silhouette_score
def sil_score(XX, YY, net, embedding_fn, bandwidth = None):
    a, b = vis.xrd_to_embedding(XX, YY, net, embedding_fn)
    return silhouette_score(a, YY)
#    a = a / a.std()
#    if bandwidth is None:
#        bandwidth = default_bw(a)
#    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(a)
#
#    return np.exp(kde.score_samples(a)).mean()
