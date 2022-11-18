import numpy as np
from xrdc import xrdutils as xdu
from xrdc import simulation as sim

from xrdc import simulation
from xrdc import datasets as xd
import matplotlib.pyplot as plt

from .. import featurization as feat
from .. import source_separation as sep
from ..datasets import d2d


def sample_peak_intensity():
    return np.random.uniform()

# https://stackoverflow.com/questions/35282456/find-local-maximums-in-numpy-array
import pandas as pd
from scipy.signal import argrelextrema

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


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

patterns = d2d.patterns
# TODO move this into sep
patterns = patterns / patterns.mean(axis = 1)[:, None]
slow_q, fast_q, slow_T, fast_T = sep.separate_signal(patterns, cutoff = .25, threshold = 20, smooth_q = 1.7, background_after_filter = False)
background = slow_q

scale = 1 / (1 * fast_q.mean(axis = 1)[:, None])
patterns =  patterns * scale

background = background * scale
fast_T = fast_T * scale
fast_q = fast_q * scale


def default_featurize(patterns_pp):
    #patterns_pp = gf(patterns, (1, 1.7))
    labeled, feature_masks, activations, norm_, activations_n1 = feat.get_ridge_features(
        patterns_pp,
       smooth_ax1 = 'FWHM', smooth_ax0 = 1, threshold_percentile = 75, thicken = True, size_thresh = 5, bgsub=False,
        log_scale_features = False, fwhm_finder=feat.fwhm_finder, do_flood_thicken = False, max_size_flood = 20,
        thicken_ax0 = 0, thicken_ax1 = 1, flood_threshold=.95, smooth_factor_ax1 = .125)
    return labeled, feature_masks, activations, norm_, activations_n1

patterns_pp_simple = fast_q * (fast_q > 0)
labeled_simple, feature_masks_simple, activations_simple, _norm, activations_n1_simple =\
    default_featurize(patterns_pp_simple);

straightened_heatmap_simple = feat.get_peakshift_corrected_heatmap(patterns_pp_simple, activations_simple, labeled_simple)

def get_activations(patterns_pp_simple):
    activations_n1_simple = feat.norm((feature_masks_simple * patterns_pp_simple).sum(axis = 2), axis = 1)
    return activations_n1_simple

act = get_activations(fast_q * (fast_q > 0))
#act = get_activations(fast_q * (fast_q > 0))

simtype = 'Cosine'
scaling = 'log'
ctype = 'agglom'

#feature_csims1, o_cuts = feat.sims_with_boundaries(patterns, act, act, n = 7, simtype = simtype, extra_label='',
#                    ctype = ctype, linkage = 'ward', affinity = 'euclidean')
#
##fsub_stop_2d = pf.curvefit_2d((patterns - 0), stdratio_threshold = 2, noise_estimate = fast_T,
##                   background = background, bg_shift_pos = False)
#
#def fitpeaks(stdratio_threshold = 4):
#    # This enforces one peak per Bayesian block
#    from .. import peak_fitting as pf
#    pf.cfg['fitInfo']['numCurves'] = 1
#    fsub_stop_2d = pf.curvefit_2d((patterns - 0), stdratio_threshold = stdratio_threshold, noise_estimate = fast_T,
#                       background = background, bg_shift_pos = False, bounds = None)
#    return fsub_stop_2d
#
#def pp_features(activations, log_scale_features = False):
#    act_min = activations.copy()
#    idxzero = np.where((act_min == 0))
#    iizero = idxzero[0]
#    act_min[act_min == 0] = np.inf
#    mins = act_min.min(axis = -1)
#    # TODO move this into feat.peakfit_featurize if it's going to be the standard transformation
#    activations_n1 = feat.norm(activations + mins[:, None] / 10, 1,
#        log_scale = log_scale_features)
#    return activations_n1
#
#def featurize_fits(fsub_stop_2d, peakwidth = 1.1, log_scale_features = False):
#    patterns_pp = fast_q #- fast_q.min()
#    #patterns_pp /= patterns_pp.mean()
#    fitlists = fsub_stop_2d[1]
#
#    labeled, feature_masks, activations, norm_, _ = feat.peakfit_featurize(patterns_pp, fitlists,
#        size_thresh = 5, peakwidth = peakwidth,
#        log_scale_features= log_scale_features)
#
##    act_min = activations.copy()
##    idxzero = np.where((act_min == 0))
##    iizero = idxzero[0]
##    act_min[act_min == 0] = np.inf
##
##    mins = act_min.min(axis = -1)
#
#    # TODO move this into feat.peakfit_featurize if it's going to be the standard transformation
#    #activations_n1 = feat.norm(activations + mins[:, None] / 2, 1)
#    activations_n1 = pp_features(activations)
#    return labeled, feature_masks, activations, norm_, activations_n1 
#
