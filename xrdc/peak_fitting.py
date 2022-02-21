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

from .dataproc.dataproc.operations.utils import single_select, folder_select
from .dataproc.dataproc.operations import hitp

from xrdc import source_separation as sep
from .utils.utils import utils

from pathos.pools import ParallelPool as Pool

plt.rcParams["figure.figsize"]=(10, 8)

# TODO proper path management
template = ''
configPath = utils.resource_path( "dataproc/dataproc/workflows/basicConfig")

# Configuration setup
# Grab configs
print(configPath)
with open(configPath) as jp:
    cfg = json.load(jp)
#cfg['fitInfo']['blockBounds'] = boundaries


def _fit_peak(xbit, ybit, noisebit, fitInfo, kwargs):
    if 'guess_params_dict' in kwargs:
        kwargs.pop('guess_params_dict', None)
    # Restrict range and fit peaks
    curveParams, derivedParams = hitp.fit_peak(xbit, ybit,
                        peakShape=fitInfo['peakShape'],
                        fitMode=fitInfo['fitMode'],
                        numCurves=fitInfo['numCurves'],
                        noise_estimate = noisebit,
                                         **kwargs)
    return curveParams, derivedParams

def workflow(y, boundaries, downsample_int = 10, noise_estimate = None, background = None, bg_shift_pos = True,
             parallel = False, cb = _fit_peak, param_guesses = None, **kwargs):
    """
    kwargs are passed to hitp.fit_peak
    """
    print(bg_shift_pos)

    # Fill out experimental information
    expInfo = {}


    #expInfo['blockBounds'] = cfg['fitInfo']['blockBounds']
    expInfo['blockBounds_list'] = cfg['fitInfo']['blockBounds_list']

    print('Experimental Info used: \n')
    print(expInfo)

    # Pull out Fit info
    fitInfo = cfg['fitInfo']

    # Start processing loop =======================================================
    run_enable = True
    
    # restrict range?
    subx, suby = np.arange(len(y)) + 1, y
    
    if background is None:
        # Background subtract/move to zero
        suby = suby - np.min(suby)
        subx, suby = hitp.bkgd_sub(subx, suby, downsample_int)
    else:
        suby = y - background
        if suby.min() < 0:
            if bg_shift_pos:
                print('negative values in background-subtracted pattern. shifting first percentile to zero and setting values below it to zero.')
                suby = suby.copy()
                suby = suby - np.percentile(suby, .5)
                suby[suby < np.percentile(suby, 1)] = 0
                #suby = suby - suby.min()
            else:
                suby = suby - (suby * (suby < 0))

    # segment rangeinto two...
    xList = []
    yList = []
    noiseList = []
    paramsList = []
    curve_paramsList = []
    guessList = []

    xListNew = []
    yListNew = []
    noiseListNew = []
    bnds_list = expInfo['blockBounds_list']
    for i, (leftBnd, rightBnd) in enumerate(bnds_list): # indexes
        selector = np.where((subx >= leftBnd) & (subx < rightBnd))
        xList.append(subx[selector])
        yList.append(suby[selector])
        if noise_estimate is not None:
            noiseList.append(noise_estimate[selector] + 1e-9) 
        else:
            noiseList.append(None)
        if param_guesses is not None:
            guessList.append(param_guesses[i]) 
        else:
            guessList.append(None) 

    def _store_peakfit_outputs(outputs):
        for i, ((curveParams, derivedParams), xbit, ybit, noisebit) in enumerate(zip(outputs, xList, yList, noiseList)):
            if curveParams is None or derivedParams is None:
                print('dropping block with index', i)
            else:
                print(f'    ----Saving data for block between {np.min(xbit):.2f} - {np.max(xbit):.2f}')
                # output/saving of blocks
                hitp.save_dict(curveParams, cfg['exportPath'], template + f'_block{i}_curve')
                hitp.save_dict(derivedParams, cfg['exportPath'], template + f'_block{i}_derived')
#                TODO there's a memory leak here
#                hitp.save_curve_fit(xbit, ybit, curveParams, cfg['exportPath'], 
#                                template + f'_block{i}', peakShape=fitInfo['peakShape'])
                paramsList.append(derivedParams)
                curve_paramsList.append(curveParams)
                xListNew.append(xbit)
                yListNew.append(ybit)
                noiseListNew.append(noisebit)

    kwlist = [kwargs] * len(xList)
    for i, guess_params_dict in enumerate(guessList):
        kwlist[i]['guess_params_dict'] = guess_params_dict
    if parallel:
        print('submitting parallel job')
        pool = Pool()
        fitoutputs = list(pool.map(cb, xList, yList, noiseList, [fitInfo] * len(xList), kwlist))
        _store_peakfit_outputs(fitoutputs)
        print("done")
        pool.clear()
    else:
        # TODO refactor
        print('running a single process')
        fitoutputs = list(map(cb, xList, yList, noiseList, [fitInfo] * len(xList), kwlist))
        _store_peakfit_outputs(fitoutputs)
        print("done")
    pdb.set_trace()
    return suby, paramsList, noiseListNew, xListNew, yListNew, curve_paramsList


def _refine_peak(xbit, ybit, noisebit, fitInfo, kwargs):
    # Restrict range and fit peaks
    guess_params_dict = kwargs['guess_params_dict']
    guess = paramdict_to_list(guess_params_dict)
    print(guess)
    print('dict')
    print(guess_params_dict)
    curveParams, derivedParams = hitp.refine_peaks(xbit, ybit,
                        peakShape=fitInfo['peakShape'],
                        fitMode=fitInfo['fitMode'],
                        numCurves=fitInfo['numCurves'],
                        noise_estimate = noisebit,
                        guess = guess,
                         **kwargs)
    return curveParams, derivedParams

def mk_bnd_list(bounds, overlap = 1):
    return [[a, b] for a, b in list(zip(bounds, bounds[overlap:]))]

def fit_curves(y, bba_smooth = 1.5, cb = _fit_peak, overlap = 1, **kwargs):
    if y.sum() != 0:
        x = np.arange(len(y))
        boundaries = hitp.bayesian_block_finder(x, gf(y, bba_smooth))
        #boundaries = [b for b in boundaries if b >= boundaries_min and b <= boundaries_max]
        print(boundaries)
        cfg['fitInfo']['blockBounds'] = boundaries
        cfg['fitInfo']['blockBounds_list'] = mk_bnd_list(boundaries, overlap)
        suby, derivedParams, noiseList, xList, yList, curve_params = workflow(y, boundaries, cb = cb, **kwargs)
        return suby, derivedParams, noiseList, xList, yList, curve_params
    return np.zeros_like(y), None, None, None, None, None

import pdb
def curvefit_2d(patterns: np.ndarray, background = None, noise_estimate = None, **kwargs):
    """
    Run BBA and peak-fitting routine for each XRD pattern in a
    multidimensional dataset whose last axis is the q dimension.
    """
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
    curveparams = np.empty(patterns.shape[:-1], object)
    noiselists = np.empty(patterns.shape[:-1], object)
    xLists = np.empty(patterns.shape[:-1], object)
    yLists = np.empty(patterns.shape[:-1], object)
    
    for indices in np.ndindex(patterns.shape[:-1]):
        if background is not None:
            background_selected = background[indices]
        else:
            background_selected = None
        if noise_estimate is not None:
#            pdb.set_trace()
            noise_estimate_selected = noise_estimate[indices]
        suby, derivedParams, noiseList, xList, yList, cparams =\
            fit_curves(patterns[indices], background = background_selected,
                        noise_estimate = noise_estimate_selected, **kwargs)
        arrays[indices] = suby
        params[indices] = derivedParams
        curveparams[indices] = cparams
        noiselists[indices] = noiseList
        xLists[indices] = xList
        yLists[indices] = yList
        
    return arrays, params, noiselists, xLists, yLists, curveparams

from copy import deepcopy
def refine_2d(patterns, fitlists, background = None, **kwargs):
    """
    Run BBA and peak-fitting routine for each XRD pattern in a
    multidimensional dataset whose last axis is the q dimension.
    """
    fitlists = deepcopy(fitlists)
    arrays, params, noiselists, xLists, yLists, curveparams = fitlists

    def _background(i):
        if background is not None:
            return background[i]
        return None
    def _noise_estimate(i):
        if noise_estimate is not None:
            return noise_estimate[i]
        return None
    
    for indices in np.ndindex(patterns.shape[:-1]):
        if background is not None:
            background_selected = background[indices]
        else:
            background_selected = None
        if noise_estimate is not None:
#            pdb.set_trace()
            noise_estimate_selected = noise_estimate[indices]
        suby, derivedParams, noiseList, xList, yList, cparams =\
            fit_curves(patterns[indices], background = background_selected,
                        noise_estimate = noise_estimate_selected, cb = _refine_peak, **kwargs)
        arrays[indices] = suby
        params[indices] = derivedParams
        curveparams[indices] = cparams
        noiselists[indices] = noiseList
        xLists[indices] = xList
        yLists[indices] = yList
        
    return arrays, params, noiselists, xLists, yLists, curveparams

###
# Functions for analyzing peak fitting results
###

def get_curves(fitoutput, i, j, k = None, peak_range_only = True):
    """
    j: profile index within a single peak fit region
    i: dataset row index
    k: (if 3d) dataset col index
    """
    # TODO clean up
    arrays, paramsLists, noiselists, xLists, yLists, curveparams = fitoutput

    if k is None:
        x, y, cparams = xLists[i][j], yLists[i][j], curveparams[i][j]

        # y values for individual Voigts
        ys = hitp.gen_curve_fits(x, y, cparams, 'Voigt')

        X, Y = np.hstack(xLists[i]), np.hstack(yLists[i])
        bounds = [onex[0] for onex in xLists[i]]
        if peak_range_only:
            mask = (X >= x.min()) & (X <= x.max())
            return x, y, X[mask], Y[mask], ys, bounds
        else:
            return x, y, X, Y, ys, bounds
    # TODO better handling of different-dimension arrays
    else:
        x, y, cparams = xLists[i][k][j], yLists[i][k][j], curveparams[i][k][j]

        # y values for individual Voigts
        ys = hitp.gen_curve_fits(x, y, cparams, 'Voigt')

        X, Y = np.hstack(xLists[i][k]), np.hstack(yLists[i][k])
        bounds = [onex[0] for onex in xLists[i][k]]
        if peak_range_only:
            mask = (X >= x.min()) & (X <= x.max())
            return x, y, X[mask], Y[mask], ys, bounds
        else:
            return x, y, X, Y, ys, bounds
    
def mean2d(arr2d):
    return arr2d.mean(axis = 0)[:, None].T
