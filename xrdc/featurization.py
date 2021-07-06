from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.ndimage.filters import gaussian_filter as gf

# this one comes from head, might be different
def get_ridges(orig, axis = 1):
    # determine the indices of the local maxima
    max_ind = argrelextrema(orig, np.greater, axis = axis)
#     max_ind_2 = argrelextrema(orig, np.greater, axis = 0)
#     max_ind = np.hstack((max_ind[0], max_ind_2[0])), np.hstack((max_ind[1], max_ind_2[1]))
    
    edges = np.zeros_like(orig)
    edges[max_ind] = 1
    return edges

def shuffle(bin_img, size = 1):
    ret = np.zeros_like(bin_img)
    for s in range(-size, size + 1):
        ret += np.roll(bin_img, s, axis = 0)
        ret += np.roll(bin_img, s, axis = 1)
    return np.sign(ret)

def get_features_spans(labeled, i):
    indices = np.indices(labeled.shape)#.T[:,:,[1, 0]]
    if len(labeled.shape) == 2:
        vi, hi = indices[:, labeled == i]
    if len(labeled.shape) == 3:
        vi, hi, zi = indices[:, labeled == i]
        zspan = zi.max() - zi.min()
    vspan = vi.max() - vi.min()
    hspan = hi.max() - hi.min()
    if len(labeled.shape) == 2:
        return vspan, hspan
    return vspan, hspan, zspan

def get_feature_vspan(labeled, i):
    return get_features_spans(labeled, i)[0]

def get_size(labeled, i, mode = 'count'):
    if mode == 'count':
        return (labeled == i).sum()
    elif mode == 'vertical':
        return get_feature_vspan(labeled, i)
    raise Exception
    
def norm(arr, axis = 0, log_scale = False):
    """
    Log scale: scale the standard deviation along each feature dimension to the
    mean value of that feature
    
    For axis == 1, we scale features to mean and std, but only consider non-zero values.
    """
    # TODO background subtraction
    if axis == 0:
        return ((arr - arr.mean(axis = axis)) / arr.std(axis = axis))
    elif axis == 1:
        xi, yi = np.nonzero(arr)
        xxi, yyi = lambda i: xi[xi == i], lambda i: yi[xi == i]
#         means, stds = np.array([(arr[xxi(i), yyi(i)].mean(), arr[xxi(i), yyi(i)].std() )
#                                 for i in range(arr.shape[0])]).T
        #if log_scale:
        arr = arr.copy()
        global_min = arr[xi, yi].min()
        print(global_min)
        for i in range(arr.shape[0]):
            ai = arr[xxi(i), yyi(i)]
            arr[xxi(i), yyi(i)] = (ai - ai.min()) / ai.std()
            if log_scale:
                arr[xxi(i), yyi(i)] *= (np.log(ai.mean() - global_min + 1))
        
        return arr
            #stds *= np.log(means)
        #return ((arr - means[:, None]) / stds[:, None])
    raise Exception
    
def oldnorm(arr, axis = 0, log_scale = False):
    """
    Log scale: scale the standard deviation along each feature dimension to the
    mean value of that feature
    
    For axis == 1, we scale features to mean and std, but only consider non-zero values.
    """
    # TODO background subtraction
    if axis == 0:
        return ((arr - arr.mean(axis = axis)) / arr.std(axis = axis))
    elif axis == 1:
        xi, yi = np.nonzero(arr)
        xxi, yyi = lambda i: xi[xi == i], lambda i: yi[xi == i]
        means, stds = np.array([(arr[xxi(i), yyi(i)].mean(), arr[xxi(i), yyi(i)].std() )
                                for i in range(arr.shape[0])]).T
        #if log_scale:
            #stds *= np.log(means)
        return ((arr - means[:, None]) / stds[:, None])
    raise Exception
    
def csim_pairs(composition):
    norm = np.linalg.norm(composition, axis = 1)
    similarity = np.dot(composition, composition.T) / np.outer(norm, norm)
    return similarity

def l2_pairs(a):
    b = a.reshape((a.shape[0], 1, a.shape[1]))
    dist_l2 = np.sqrt(np.einsum("ijk, ijk->ij", a - b, a - b))
    return dist_l2

def l2_sim(a):
    return -l2_pairs(a)
    
from scipy.ndimage.measurements import label
from sklearn.cluster import KMeans

def preprocess(patterns, bg_smooth = 80, smooth = 1.7, bgsub = False, threshold_percentile = 60):
    bg = gf(patterns,bg_smooth)
    p = patterns.copy()
    if bgsub:
        p = p - bg

    threshold = np.percentile(patterns, threshold_percentile)
    p[p < threshold] = 0

    #p = np.log(1 + p)
    smoothed = gf(p, smooth)
    
    return smoothed

def refine_and_label(arr, thicken = True, size_thresh = 5, sizetype = 'vertical'):
    if thicken:
        arr = shuffle(arr)
    else:
        arr = np.sign(arr)

#     plt.subplot(a, b, 3)
#     plt.title('ridged(smoothed and expanded)')
#     plt.imshow(arr, cmap = 'jet')

    if len(arr.shape) == 2:
        structure = np.ones((3, 3), dtype=int)  # this defines the connection filter
    if len(arr.shape) == 3:
        structure = np.ones((3, 3, 3), dtype=int)
    labeled, ncomponents = label(arr, structure)
    indices = np.indices(arr.shape)#.T[:,:,[1, 0]]
#     xx, yy = indices

    j = 1
    new_labeled = np.zeros_like(labeled)
    # TODO: thickening and size threshold should be linked
    for i in range(1, ncomponents + 1):
        n_i = get_size(labeled, i, sizetype)
        if n_i >= size_thresh:
            new_labeled[labeled == i] = j
            j += 1

    labeled = new_labeled
    return arr, labeled

def get_ridge_features(patterns, smooth = 1.7, threshold_percentile = 60, thicken = True, size_thresh = 5,
                      bgsub = False, bg_smooth = 80, log_scale_features = False, logscale_heatmap = True,
                      a = 5, b = 1, normf = norm):

    plt.rcParams["figure.figsize"]=(20, 13)
    plt.subplot(a, b, 1)
    plt.title('ridges')
    plt.imshow(get_ridges(patterns), cmap = 'jet')

    smoothed = preprocess(patterns, bg_smooth, bgsub = bgsub, threshold_percentile = threshold_percentile)

    arr = get_ridges(smoothed)
    arr, labeled = refine_and_label(arr, thicken = thicken, size_thresh = size_thresh)
    
    plt.subplot(a, b, 2)
    plt.title('ridges (smoothed)')
    plt.imshow(arr, cmap = 'jet')

    plt.subplot(a, b, 4)
    plt.title('final feature masks')
    plt.imshow(labeled, cmap = 'jet')

    plt.subplot(a, b, 5)
    plt.title('final feature masks (overlayed)')
    if logscale_heatmap:
        plt.imshow(np.log(1 + patterns), cmap = 'jet')
    else:
        plt.imshow(patterns, cmap = 'jet')
    plt.imshow(np.sign(labeled), cmap='Greys', alpha = .5)
    feature_masks = np.array([labeled == i for i in range(1, labeled.max() + 1)])
    print(len(feature_masks))

    activations = (feature_masks * patterns).sum(axis = 2)
    #activations = (feature_masks * p).sum(axis = 2)
    activations_n0 = normf(activations, 0, log_scale = log_scale_features)
    activations_n1 = normf(activations, 1, log_scale = log_scale_features)
    return labeled, feature_masks, activations, activations_n0, activations_n1

def do_clust(patterns, activations, n_clust, ctype = 'agglom'):
    X = activations.T
    
    if ctype == 'kmeans':
        # X = activations_n1.T
        kmeans = KMeans(n_clusters=n_clust, random_state=0).fit(X)
        clust = kmeans.labels_
    elif ctype == 'agglom':
        clustering = AgglomerativeClustering(n_clusters=n_clust).fit(X)
        clust = clustering.labels_
    else:
        raise Exception

    clust_cms = np.zeros_like(clust).astype(float)
    ci = np.indices(patterns.shape)[0]
    for i in range(clust.max() + 1):
        val = (ci[clust == i].astype('float').mean())
        clust_cms[clust == i] = val

    sorter = np.argsort([j * 10000 + i for (i, j) in enumerate(clust_cms)])
    return clust, sorter, clust_cms

def get_boundaries(patterns, activations, n_clust = 7, ctype = 'agglom'):
    clust, sorter, clust_cms = do_clust(patterns, activations, n_clust, ctype = ctype)
    boundaries = np.hstack(((np.diff(clust_cms[sorter]) > 0), [0])).astype(bool)
    return sorter, boundaries, clust_cms


    
    
def plot_pca_explained_variance(X):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X)
    X_train_std = sc.transform(X)
    #X_test_std = sc.transform(X_test)
    #
    # Instantiate PCA
    #
    pca = PCA()
    #
    # Determine transformed features
    #
    X_train_pca = pca.fit_transform(X_train_std)
    #
    # Determine explained variance using explained_variance_ration_ attribute
    #
    exp_var_pca = pca.explained_variance_ratio_
    #
    # Cumulative sum of eigenvalues; This will be used to create step plot
    # for visualizing the variance explained by each principal component.
    #
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    #
    # Create the visualization plot
    #
    plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    return cum_sum_eigenvalues

def simple_heatmap(arr, **kwargs):
    plt.imshow(arr, interpolation = 'none', cmap = 'jet')


def similarity_plot_row(fn, label, patterns, activations, activations_n1, 
                        activations_n1_log = None, offset = 0, swapper = lambda arr: arr.T,
                       plotter = simple_heatmap, a = 2, b = 3):
    # transformer: swap between pattern and activations row ordering
    if swapper is None:
        swapper = lambda arr: arr
    plt.subplot(a, b, 1 + offset)
    raw_csims = fn(patterns)
    plt.title("{} distances (raw signal)".format(label))
    plotter(raw_csims)
    #plt.imshow(raw_csims, interpolation = 'none', cmap = 'jet')
    #plt.colorbar()
    
    plt.subplot(a, b, 2 + offset)
    feature_csims = fn(swapper(activations))
    plt.title("{} distances (feature space, no normalization)".format(label))
    plotter(feature_csims)
    #plt.imshow(feature_csims, interpolation = 'none', cmap = 'jet')

    plt.subplot(a, b, 3 + offset)
    feature_csims1 = fn(swapper(activations_n1))
    plt.title("{} distances (feature space, normalized dim 1)".format(label))
    plotter(feature_csims1)
    #plt.imshow(feature_csims1, interpolation = 'none', cmap = 'jet')

    
    if activations_n1_log is not None:
        plt.subplot(a, b, 4 + offset)
        feature_csims = fn(swapper(activations_n1_log))
        plt.title("{} distances, log scaled (feature space, normalized dim 1)".format(label))
        plt.imshow(feature_csims, interpolation = 'none', cmap = 'jet')
        
def ordered_cuts(activations, n, ctype = 'agglom'):
    tmp = [get_boundaries(activations, n_clust = i, ctype = ctype)[1] for i in range(2, n + 1)]
    tmp = [np.zeros_like(tmp[0]).astype(bool)] + tmp
    cuts = set()
    for i, (cp, cn) in enumerate(zip(tmp, tmp[1:])):
        for cut_i in np.where((cp ^ cn))[0]:
            if cut_i not in cuts:
                cuts.add((i, cut_i))
                cuts.add(cut_i)
    cuts = sorted(list(filter(lambda elt: type(elt) == tuple, cuts)))
    cuts = [elt[1] for elt in cuts]
    return cuts

def draw_cuts(o_cuts, offset, fontsize, vlines = True, extent = 60):
    if extent is None:
        extent = offset
    for i, v in enumerate(o_cuts):
        if v:
#             if i > 3:
#                 sty = 'dashed'
#             else:
            sty = 'solid'
            plt.hlines(v + .5, 0, extent, color = 'k', linestyles=sty)
            if vlines:
                plt.vlines(v + .5, 0, extent, color = 'k', linestyles=sty)
            plt.text(offset, v + .5, "cut {}".format(i), fontsize=fontsize)
            

def cluster_draw_boundaries(patterns, activations, n_clust = 7, ctype = 'agglom', logscale = True):
    # kinda deprecated, use draw_cuts instead
    sorter, boundaries, _ = get_boundaries(patterns, activations, n_clust = n_clust, ctype = ctype)
    p3 = patterns.copy()
    p3[boundaries] = p3.min()
    #p3[:, 550:] = (p3.min() + (p3.max() - p3.min()) * (clust / (clust.max() - clust.min())))[:, None]
    plt.title('{} clusters feat (N = {})'.format(ctype, n_clust))
    if logscale:
        plt.imshow(np.log(1 + p3[sorter]), cmap = 'jet')
    else:
        plt.imshow(p3[sorter], cmap = 'jet')

    return boundaries
