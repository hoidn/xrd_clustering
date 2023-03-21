"""
Feature extraction for 3d datasets.
"""
# should eventually be merged into featurization.py

import numpy as np
from . import featurization as feat

def get_activations(patterns_pp, feature_masks, axis = 3,
                   thicken_ax0 = 0, size_thresh = 25,
                            area_min = 5, thicken_ax1 = 0):
#     #patterns_pp = patterns_pp * (patterns_pp > 0)
#     labeled, feature_masks, activations, _norm, activations_n1 = peakfit_featurize(patterns_pp, fitlists,
#                                                                 thicken_ax0 = thicken_ax0,
#                                                                 size_thresh = size_thresh,
#                                                                 area_min = area_min,
#                                                                 thicken_ax1 = thicken_ax1)

#     mask = (patterns_pp.sum(axis = 2) != 0)
#     feature_masks, labeled = consolidate_feature_maps(feature_masks)

#     #activations = (feature_masks * (patterns_pp * (patterns_pp > 0))).sum(axis = axis)
    activations = (feature_masks * patterns_pp).sum(axis = axis)
    #scale = np.log(1 + np.mean(activations, axis = (1, 2)))
    scale = np.mean(activations, axis = (1, 2))
    activations_n1 = feat.norm3d(activations)
    return activations / scale[:, None, None] #activations_n1
