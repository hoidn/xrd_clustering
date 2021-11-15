"""
Peak fitting for XRD datasets with proximity information.
"""

import time
import json
from pathlib import Path
import re
import pyFAI.detectors as dets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from importlib import reload
from scipy.ndimage.filters import gaussian_filter as gf

import pyFAI.detectors as dets
import pyFAI

from dataproc.operations import hitp
from dataproc.operations.utils import single_select, folder_select

from xrdc import source_separation as sep

plt.rcParams["figure.figsize"]=(10, 8)

template = ''
configPath = "workflows/alanConfig"

# Configuration setup
# Grab configs
print(configPath)
with open(configPath) as jp:
    cfg = json.load(jp)
#cfg['fitInfo']['blockBounds'] = boundaries


def workflow(y, boundaries, downsample_int = 10, noise_estimate = None, background = None,
             **kwargs):
    """
    kwargs are passed to hitp.fit_peak
    """

    # Fill out experimental information
    expInfo = {}


    expInfo['blockBounds'] = cfg['fitInfo']['blockBounds']

    print('Experimental Info used: \n')
    print(expInfo)

    # Pull out Fit info
    fitInfo = cfg['fitInfo']

    # Start processing loop =======================================================
    run_enable = True
    
    # restrict range?
    subx, suby = np.arange(len(y)) + 1, y
    #pdb.set_trace()
    
    if background is None:
        # Background subtract/move to zero
        suby = suby - np.min(suby)
        subx, suby = hitp.bkgd_sub(subx, suby, downsample_int)
    else:
        suby = y - background
        if suby.min() < 0:
            print('negative values in background-subtracted pattern. taking absolute value.')
            suby = suby - suby.min()

    # segment rangeinto two...
    xList = []
    yList = []
    noiseList = []
    paramsList = []
    bnds = expInfo['blockBounds']
    for leftBnd in range(len(bnds) - 1): # indexes
        selector = np.where((subx >= bnds[leftBnd]) & (subx < bnds[leftBnd + 1]))
        xList.append(subx[selector])
        yList.append(suby[selector])
        if noise_estimate is not None:
            noiseList.append(noise_estimate[selector] + 1e-9) 
        else:
            noiseList.append(None)
    for i, (xbit, ybit, noisebit) in enumerate(zip(xList, yList, noiseList)):
        # Restrict range and fit peaks
        curveParams, derivedParams = hitp.fit_peak(xbit, ybit,
                            peakShape=fitInfo['peakShape'],
                            fitMode=fitInfo['fitMode'],
                            numCurves=fitInfo['numCurves'],
                            noise_estimate = noisebit,
                                             **kwargs)
        print(f'    ----Saving data for block between {np.min(xbit):.2f} - {np.max(xbit):.2f}')
        # output/saving of blocks
        hitp.save_dict(curveParams, cfg['exportPath'], template + f'_block{i}_curve')
        hitp.save_dict(derivedParams, cfg['exportPath'], template + f'_block{i}_derived')
        hitp.save_curve_fit(xbit, ybit, curveParams, cfg['exportPath'], 
                        template + f'_block{i}', peakShape=fitInfo['peakShape'])
        paramsList.append(derivedParams)
    return suby, paramsList


def fit_curves(y, **kwargs):
    if y.sum() != 0:
        x = np.arange(len(y))
        boundaries = hitp.bayesian_block_finder(x, gf(y, 1.5))
        #boundaries = [b for b in boundaries if b >= boundaries_min and b <= boundaries_max]
        print(boundaries)
        cfg['fitInfo']['blockBounds'] = boundaries
        suby, derivedParams = workflow(y, boundaries, **kwargs)
        return suby, derivedParams
    return np.zeros_like(y), None


def curvefit_2d(patterns, background = None, noise_estimate = None, **kwargs):
    def _background(i):
        if background is not None:
            return background[i]
        return None
    def _noise_estimate(i):
        if noise_estimate is not None:
            return noise_estimate[i]
        return None
    
    arrays = np.zeros(patterns.shape)
    params = np.empty(patterns.shape[:-1], object)
    
    for indices in np.ndindex(patterns.shape[:-1]):
        suby, derivedParams = fit_curves(patterns[indices], background = background[indices],
                        noise_estimate = noise_estimate[indices],
                        **kwargs)
        arrays[indices] = suby
        params[indices] = derivedParams
    
    return arrays, params
