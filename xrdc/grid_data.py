from k_means_constrained import KMeansConstrained as KMC

def group_by_coord(cpairs, n_neighbor = 6, n_cluster = None):
    """
    Group by wafer coordinate using K Means with a cluster size constraint
    Returns group indices
    """
    N = cpairs.shape[0]
    if n_cluster is None:
        n_clusters = N // n_neighbor
        kmeans = KMC(n_clusters=n_clusters, size_min = n_neighbor).fit(cpairs)
    else:
        kmeans = KMC(n_clusters=n_cluster, size_min = 1).fit(cpairs)
    km_y = kmeans.labels_
    return km_y
