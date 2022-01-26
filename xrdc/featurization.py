from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.ndimage.filters import gaussian_filter as gf
from scipy.ndimage import gaussian_filter1d as gf1d

from . import misc

debug = False

def distortion(X, means, covi, i):
    p = X.shape[1]
    # TODO don't calculate the whole matrix
    return (np.diagonal(np.dot(X - means[i], np.dot(covi, (X - means[i]).T))))

def distortions(X, means, covi):
    K = means.shape[0]
    return np.vstack([distortion(X, means, covi, i) for i in range(K)])

def min_distortions(X, means, covi):
    return distortions(X, means, covi).min(axis = 0)

# JumpMethod(X):
#     Let Y = (p/2)
#     Init a list D, of size n+1
#     Let D[0] = 0
#     For k = 1 ... n:
#         Cluster X with k clusters (e.g., with k-means)
#         Let d = Distortion of the resulting clustering
#         D[k] = d^(-Y)
#     Define J(i) = D[i] - D[i-1]
#     Return the k between 1 and n that maximizes J(k)


from sklearn.cluster import KMeans

def distortion_curve(X, Kmax = 15, **kwargs):
    # TODO the dimensional scaling method seems not to work, so for now we're not using it
    print(kwargs)
    p = X.shape[1]
    Y = p / 2
    D = np.zeros(Kmax)
    grid = list(range(1, Kmax))
    for i in grid:
        if i == 1:
            y_predict = np.zeros(X.shape[0])
            c0 = np.mean(X, axis = 0)
            centroids = np.vstack([c0 for _ in range(X.shape[0])])
        else:
            y_predict = AgglomerativeClustering(n_clusters=i, **kwargs).fit_predict(X)
            #y_predict = gm.predict(X)
            from sklearn.neighbors.nearest_centroid import NearestCentroid
            clf = NearestCentroid()
            clf.fit(X, y_predict)
            #gm = GaussianMixture(n_components=i, random_state=0, covariance_type='tied').fit(X)
            centroids = clf.centroids_
        d = min_distortions(X, centroids, np.eye(p)).mean()
        #d = distortions(X, gm.means_, np.linalg.inv(gm.covariances_)).mean()
        D[i] = d#**(-Y)
    return grid, D[1:]

def color_peaks(fit_list, pattern, imin = 10, fwhm_max = 80, area_min = 0,
        peakwidth = 'auto'):
    """
    Return an array of 1s in indices corresponding to a peak (+- HWHM) and 0s elsewhere. 
    
    fit_list: list of derived peak fit parameters.
    
    Only the first curve in each group of fits is considered.
    """
    res = np.zeros_like(pattern)
    if fit_list is None:
        return res
    for peaks in fit_list:
        primary = peaks['curve 0']
        if peakwidth == 'auto':
            peakwidth_n = int((primary['FWHM'] + 1) / 4)
        else:
            peakwidth_n = peakwidth
        #qwhm = 1
        i0 = int(primary['x0'] + .5)
        if i0 >= imin and primary['FWHM'] <= fwhm_max:# and primary['area'] > area_min:
            res[int(i0 - peakwidth_n): int(i0  + peakwidth_n)] = 1
    return res

def color_peaks_2d(fitlists, patterns, **kwargs):
    """
    Given a 2d array (i.e. a series of 1D patterns) and an array
    of the same length of peak fit parameter lists, return an
    array of the same shape as patterns in which peak regions are
    valued 1 and all else is valued 0.
    """
    assert fitlists.shape == patterns.shape[:-1]
    return np.vstack([color_peaks(fitlist, patterns[i], **kwargs) for i, fitlist in enumerate(fitlists)])

def color_peaks_3d(fitlists_3d, patterns, **kwargs):
    """
    Given a 3d array (i.e. a 2d array of 1D patterns) and an array
    of the same shape of peak fit parameter lists, return a
    matching-shape array in which peak regions are
    valued 1 and all else is valued 0.
    """
    from . import source_separation as sep
    assert len(fitlists_3d.shape) == 2
    assert len(patterns.shape) == 3
    return np.array([color_peaks_2d(fitlists_2d, patterns[i], **kwargs) for i, fitlists_2d in enumerate(fitlists_3d)])

import pdb
# this one comes from head, might be different
def get_ridges(orig, axis = 1, fitlists = None, **kwargs):
    if fitlists is not None:
        print("using fitlists")
        if len(orig.shape) == 2:
            return color_peaks_2d(fitlists, orig, **kwargs)
        elif len(orig.shape) == 3:
            return color_peaks_3d(fitlists, orig, **kwargs)
        else:
            raise ValueError
    # determine the indices of the local maxima
    max_ind = argrelextrema(orig, np.greater, axis = axis)
    edges = np.zeros_like(orig)
    edges[max_ind] = 1
    return edges

def shuffle(bin_img, thicken_ax0 = 1, thicken_ax1 = 1):
    ret = np.zeros_like(bin_img)
    for s0 in range(-round(thicken_ax0), round(thicken_ax0) + 1):
        for s1 in range(-round(thicken_ax1), round(thicken_ax1) + 1):
            ret += np.roll(bin_img, s0, axis = 0)
            ret += np.roll(bin_img, s1, axis = 1)
            if len(bin_img.shape) == 3:
                print(3)
                ret += np.roll(bin_img, s1, axis = 2)
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
            arr[xxi(i), yyi(i)] = (ai ) / ai.std()
            #arr[xxi(i), yyi(i)] = (ai - ai.min()) / ai.std()
            if log_scale:
                arr[xxi(i), yyi(i)] *= (np.log(ai.mean() - global_min + 1))
        
        return arr
            #stds *= np.log(means)
        #return ((arr - means[:, None]) / stds[:, None])
    raise Exception

# same thing for 3d arrays, TODO merge
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
    """
    Calculate L2 distances.
    """
    b = a.reshape((a.shape[0], 1, a.shape[1]))
    dist_l2 = np.sqrt(np.einsum("ijk, ijk->ij", a - b, a - b))
    return dist_l2

def l2_sim(a):
    return -l2_pairs(a)
    
from scipy.ndimage.measurements import label
from sklearn.cluster import KMeans

def preprocess(patterns, bg_smooth = 80, smooth_ax1 = 'FWHM', smooth_ax0 = 2, bgsub = False, threshold_percentile = 50,
        fwhm_finder = None, smooth_factor_ax1 = 0.25):
    """
    fwhm_finder: callback function that, given y, returns mean peak FWHM for most prominent peak
    across all rows of patterns.

    smooth_ax1: 'FWHM' or numeric.
    """
    assert bgsub in [True, False]
    p = patterns.copy()

    #p = np.log(1 + p)
    if smooth_ax1 == 'FWHM':
        fwhm = fwhm_finder(patterns)
        sig1 = smooth_factor_ax1 * fwhm
    else:
        sig1 = smooth_ax1
        hwhm = None

    if len(p.shape) == 2:
        smooth_shape = (smooth_ax0, sig1)
    else:
        smooth_shape = (smooth_ax0, smooth_ax0, sig1)
    bg = gf(patterns, smooth_shape)
    if bgsub:
        p = p - bg
        p = p - min(0, p.min()) # shift so that all values are positive

    threshold = np.percentile(patterns, threshold_percentile)
    p[p < threshold] = 0

    smoothed = gf(p, smooth_shape)
    
#    import pdb
#    pdb.set_trace()
    try:
        return smoothed, fwhm
    except UnboundLocalError:
        return smoothed, None

import pdb
def flood_thicken(labeled, arr, thresh = .95, max_hsize = 50):
    from skimage import data, filters, color, morphology
    from skimage.segmentation import flood, flood_fill
    labeled = labeled.copy()
    for i in range(1, labeled.max()):
        i1 = np.nonzero(labeled == i)
        #i1 = i1[0][::3], i1[1][1::3]
        p = arr.copy()
        peak_values = p[i1]
        above_hm = ((p[i1[0]] > peak_values[:, None] * thresh)).copy()
        filled = flood_fill(above_hm.astype(int), (0, i1[1][0]), -1, tolerance = 0.)
        filled[filled > -1] = 0
        fillx, filly = np.nonzero(filled)
        fillx += i1[0].min()

        new_labeled = labeled.copy()
        new_labeled[fillx, filly] = i
        _, hspan = get_features_spans(new_labeled, i)
        if hspan > max_hsize:
            print("feature {}: expanded feature size {} too large".format(i, hspan))
            continue

        labeled = new_labeled
        labeled[fillx, filly] = i
    return labeled

def refine_and_label(ridges, thicken = True, do_flood_thicken = False, size_thresh = 2, sizetype = 'vertical',
        max_size_flood = 50, flood_threshold = .95, thicken_ax0 = 1, thicken_ax1 = 'FWHM'):
    if thicken and do_flood_thicken:
        raise ValueError("only one of thicken and flood_thicken can be selected")
    if thicken:
        ridges = shuffle(ridges, thicken_ax0, thicken_ax1)
    else:
        ridges = np.sign(ridges)

    if len(ridges.shape) == 2:
        structure = np.ones((3, 3), dtype=int)  # this defines the connection filter
    if len(ridges.shape) == 3:
        structure = np.ones((3, 3, 3), dtype=int)
    labeled, ncomponents = label(ridges, structure)
    indices = np.indices(ridges.shape)#.T[:,:,[1, 0]]
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
    if do_flood_thicken:
        raise NotImplementedError
#        labeled = flood_thicken(labeled, smoothed, max_hsize = max_size_flood, thresh = flood_threshold)
#        labeled, ncomponents = label(labeled, structure) # merge overlapping features
    return ridges, labeled

def imshow_labeled(labeled, **kwargs):
    from matplotlib import colors
    cmap = colors.ListedColormap(list('kbgrcmy'))
    plt.imshow((labeled % 7) + (labeled != 0), cmap= cmap, interpolation = 'none', **kwargs)
    #plt.imshow(labeled, cmap = 'jet')

def get_ridge_features(patterns, threshold_percentile = 50, thicken = True, size_thresh = 2,
                      bgsub = False, bg_smooth = 80, log_scale_features = False, logscale_heatmap = True,
                      smooth_ax1 = 'FWHM', smooth_ax0 = 2, fwhm_finder = None, smooth_factor_ax1 = 0.25,
                      a = 5, b = 1, normf = norm, do_flood_thicken = False, max_size_flood = 50, flood_threshold = .95,
                      thicken_ax0 = 1, thicken_ax1 = 'FWHM', **kwargs):


    packed_pp = preprocess(patterns, bg_smooth, bgsub = bgsub, threshold_percentile = threshold_percentile,
        smooth_ax1 = smooth_ax1, smooth_ax0 = smooth_ax0, fwhm_finder = fwhm_finder, smooth_factor_ax1 = smooth_factor_ax1)

#    if thicken_ax1 == 'FWHM':
#        thicken_ax1 = int(fwhm / 4) # TODO parameterize?
    smoothed, fwhm = packed_pp
#    else:
#        smoothed = packed_pp

    arr = get_ridges(smoothed, **kwargs)
    arr, labeled = refine_and_label(arr, thicken = thicken, do_flood_thicken = do_flood_thicken, size_thresh = size_thresh,
        max_size_flood = max_size_flood, flood_threshold = flood_threshold, thicken_ax0 = thicken_ax0, thicken_ax1 = thicken_ax1)
    
    if len(patterns.shape) == 2:
        # TODO handle 3d case
        plt.rcParams["figure.figsize"]=(20, 18)
        plt.subplot(a, b, 1)
        plt.title('ridges')
        plt.imshow(get_ridges(patterns, **kwargs), cmap = 'jet')
        plt.subplot(a, b, 2)
        plt.title('ridges (smoothed)')
        plt.imshow(arr, cmap = 'jet')

        plt.subplot(a, b, 4)
        plt.title('final feature masks')

        imshow_labeled(labeled)

        plt.subplot(a, b, 5)
        plt.title('final feature masks (overlayed)')
        if logscale_heatmap:
            plt.imshow(np.log(1 + patterns), cmap = 'jet', interpolation = 'none')
        else:
            plt.imshow(patterns, cmap = 'jet', interpolation = 'none')

        plt.imshow(np.sign(labeled), cmap='Greys', alpha = .5)

    feature_masks = np.array([labeled == i for i in range(1, labeled.max() + 1)])
    print(len(feature_masks))

    def _norm(act):
        return normf(act, 1, log_scale = log_scale_features)

    activations = featurize(feature_masks, patterns)#(feature_masks * patterns).sum(axis = 2)
    #activations = (feature_masks * p).sum(axis = 2)
    #activations_n0 = normf(activations, 0, log_scale = log_scale_features)
    activations_n1 = normf(activations, 1, log_scale = log_scale_features)
    return labeled, feature_masks, activations, _norm, activations_n1
    #return labeled, feature_masks, activations, activations_n0, activations_n1

def featurize(feature_masks, patterns):
    return (feature_masks * patterns).sum(axis = 2)

def do_clust(patterns, activations, n_clust, ctype = 'agglom', **kwargs):
    #print(kwargs)
    X = activations.T
    
    if ctype == 'kmeans':
        # X = activations_n1.T
        kmeans = KMeans(n_clusters=n_clust, random_state=0, **kwargs).fit(X)
        clust = kmeans.labels_
    elif ctype == 'agglom':
        if debug:
            print(kwargs)
        clustering = AgglomerativeClustering(n_clusters=n_clust, **kwargs).fit(X)
        clust = clustering.labels_
    elif ctype == 'divk':
        #print(ctype)
        #pdb.set_trace()
        from kmeans import bisecting_kmeans as divk
        clustering = divk.KMeans(n_clust)
        clustering.fit(list(X))
        clust = np.array(clustering.labels_)
    else:
        raise Exception

    clust_cms = np.zeros_like(clust).astype(float)
    ci = np.indices(patterns.shape)[0]
    for i in range(clust.max() + 1):
        val = (ci[clust == i].astype('float').mean())
        clust_cms[clust == i] = val

    sorter = np.argsort([j * 10000 + i for (i, j) in enumerate(clust_cms)])
    return clust, sorter, clust_cms

def get_boundaries(patterns, activations, n_clust = 7, ctype = 'agglom', **kwargs):
    clust, sorter, clust_cms = do_clust(patterns, activations, n_clust, ctype = ctype, **kwargs)
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
        
def ordered_cuts(patterns, activations, n, ctype = 'agglom', cut_type = 'clustering', simfn = None, **kwargs):
    if cut_type == 'clustering':
        tmp = [get_boundaries(patterns, activations, n_clust = i, ctype = ctype, **kwargs)[1] for i in range(2, n + 1)]
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
    elif cut_type == 'pairwise':
        # TODO: compute this efficiently
        feature_csims = simfn(activations.T)
        psims = np.hstack((np.diagonal(np.roll(feature_csims, -1, axis = 1))[:-1], np.ones(1)))
        return np.argsort(psims)[:n - 1]

def draw_cuts(o_cuts, offset, fontsize, vlines = True, extent = 60):
    if extent is None:
        extent = offset
    for i, v in enumerate(o_cuts):
        if v:
            if i == 0:
                sty = 'solid'
            elif i == 1:
                sty = 'dashed'
            elif i == 2:
                sty = 'dashdot'
            else:
                sty = 'dotted'
            plt.hlines(v + .5, 0, extent, color = 'k', linestyles=sty, linewidth = 5)
            if vlines:
                plt.vlines(v + .5, 0, extent, color = 'k', linestyles=sty, linewidth = 5)
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

def sims_with_boundaries(patterns, clustering_mat, visualization_mat, n = 5, simtype = 'Cosine', extra_label = '',
                        cut_offset_similarity = 62, cut_offset_raw = 598,
                        lines = True, ctype = 'agglom', cut_type = 'clustering',
                        plot_distortion = True, **kwargs):
    if simtype == 'Cosine':
        simfn = csim_pairs
    elif simtype == 'L2':
        simfn = l2_sim
    else:
        raise Exception
    if plot_distortion:
        a, b = 2, 2
    else:
        a, b = 1, 2
    plt.rcParams["figure.figsize"]=(20, 20)

    # todo bug when clustering_mat and visualization_mat are different, and cut type is 'pairwise'
    feature_csims1 = simfn(visualization_mat.T)

    plt.subplot(a, b, 1)
    plt.imshow(np.log(1 + patterns), aspect=patterns.shape[1] / patterns.shape[0], cmap = 'jet')
    o_cuts = ordered_cuts(patterns, clustering_mat, n, simfn = simfn, cut_type = cut_type,
                               ctype = ctype, **kwargs)
    
    if lines:
        draw_cuts(o_cuts, cut_offset_raw, 25, vlines = False, extent = cut_offset_raw * .95)

    plt.subplot(a, b, 2)

    _, boundaries, clust_cms = get_boundaries(patterns, clustering_mat, n, ctype = ctype, **kwargs)

    N, M = patterns.shape
    # feature_csims1 = csim_pairs(clustering_mat.T)
    plt.title("{}, normalized{}".format(simtype, extra_label))
    plt.imshow(feature_csims1, interpolation = 'none', cmap = 'jet')
    if lines:
        draw_cuts(o_cuts, cut_offset_similarity, 25, extent = cut_offset_similarity * .95)
        
    if plot_distortion:
        plt.subplot(a, b, 3)
        X = clustering_mat.T
        d0 = X - (X).mean(axis = 0)
        grid, D = distortion_curve(clustering_mat.T, **kwargs)
        #D = np.hstack(([np.linalg.norm(d0)**2], D))
#        print(D)
#        print(np.diff(np.diff(D)))
        plt.plot(grid, D, label = 'distortion')
        plt.xlabel('Number of clusters', fontsize = 20)
        plt.ylabel('expected min distortion', fontsize = 20)
        plt.grid()
        plt.subplot(a, b, 4)
        plt.plot(grid[:-1], np.hstack(([np.nan], np.diff(np.diff(D)))), c = 'orange')
        plt.xlabel('Number of clusters', fontsize = 20)
        plt.ylabel('Double difference of expected min distortion', fontsize = 20)
        plt.grid()
    return feature_csims1, o_cuts

from .dataproc.dataproc.operations import hitp
def fwhm_finder(patterns, peakShape = 'Voigt', numCurves = 1):
    M = patterns.shape[1]
    all_curves = [hitp.fit_peak(np.arange(M), y, peakShape=peakShape, numCurves=numCurves)[1] for y in patterns[::10]]
    def get_fwhm(curves):
        return np.mean([v['FWHM'] for v in curves.values()])
    return np.median([get_fwhm(curves) for curves in all_curves])

def peakfit_featurize(patterns_pp, fitlists, size_thresh = 5, smooth_ax1 = 'FWHM', smooth_ax0 = 2, threshold_percentile = 50, thicken = True, bgsub=False,
        log_scale_features = False, fwhm_finder=fwhm_finder, do_flood_thicken = False, max_size_flood = 20,
        thicken_ax1 = 0, flood_threshold=.95, smooth_factor_ax1 = .125, 
        peakwidth = 2, **kwargs):
    """
    Given patterns and a list of peak fit parameters, return peak
    shift-corrected features.

    patterns_pp should preferably be background-subtracted.
    """
    # TODO move these parameters to the function declaration
    labeled, feature_masks, activations, norm_, activations_n1 = get_ridge_features(
        patterns_pp,
       smooth_ax1 = smooth_ax1, smooth_ax0 = smooth_ax0, threshold_percentile = threshold_percentile, thicken = thicken, size_thresh = size_thresh, bgsub=bgsub,
        log_scale_features =log_scale_features, fwhm_finder=fwhm_finder, do_flood_thicken =do_flood_thicken, max_size_flood =max_size_flood,
        thicken_ax1 =thicken_ax1, flood_threshold=flood_threshold, smooth_factor_ax1 =smooth_factor_ax1, fitlists = fitlists,
        peakwidth =peakwidth, **kwargs)
    return labeled, feature_masks, activations, norm_, activations_n1

simtype, scaling = 'Cosine', 'linear'
from IPython.display import Markdown as md
def printinfo(simtype, scaling, linkage_type, clustering_type):
    return md("""
    Similarity type: {}

    Scaling: {}
    
    Clustering type: {}
    
    Linkage: {}
    """.format(simtype, scaling, clustering_type, linkage_type))

def get_features_range(labeled, i):
    """
    Return (min vertical index, max vertical index, horizontal index at
    max vertical position)
    """
    indices = np.indices(labeled.shape)#.T[:,:,[1, 0]]
    if len(labeled.shape) == 2:
        vi, hi = indices[:, labeled == i]
        return (vi.min(), vi.max(), np.floor(hi[(vi == vi.max())].mean()))
    
def get_peakshift_corrected_heatmap(patterns, activations, labeled):    
    """
    Given XRD array, corresponding feature label map, and an array of
    feature intensities for each pattern/feature pair, return a heatmap
    of 'straightened" features. To be used for visualization purposes,
    basically.
    """
    straight_feature_map = np.zeros_like(patterns)
    thickness = 2
    for i in range(1, labeled.max() + 1):
        vmin, vmax, hcenter = get_features_range(labeled, i)
        hstart, hend = int(hcenter - thickness / 2), int(hcenter + thickness / 2)
        vi, hi = np.indices(labeled.shape)
        fillmask = ((hi >= hstart) & (hi <= hend)) & ((vi <= vmax) & (vi >= vmin))
        vi_fill, hi_fill = vi[fillmask], hi[fillmask]
        for vidx, hidx in zip(vi_fill, hi_fill):
            straight_feature_map[vidx, hidx] = activations[i - 1][vidx]
    return straight_feature_map

def count_bblocks(fitlists, dim = 2):
    npeak_grid2 = np.array(misc.deepmap(fitlists, lambda lst: len(lst) if lst is not None else 0, dim))
    npeak_grid2[npeak_grid2 == 0] = np.min(npeak_grid2[npeak_grid2 > 0])
    return npeak_grid2

def count_peaks(fitlists, dim = 2):
    npeak_grid2 = np.array(misc.deepmap(fitlists, lambda lst: sum(len(l) for l in lst) if lst is not None else 0, dim))
    npeak_grid2[npeak_grid2 == 0] = np.min(npeak_grid2[npeak_grid2 > 0])
    return npeak_grid2
