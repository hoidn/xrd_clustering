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
    print('peak fit kwargs', kwargs)
    curveParams, derivedParams = hitp.fit_peak(xbit, ybit,
                        peakShape=fitInfo['peakShape'],
                        fitMode=fitInfo['fitMode'],
                        numCurves=fitInfo['numCurves'],
                        noise_estimate = noisebit,
                                         **kwargs)
    return curveParams, derivedParams

from copy import deepcopy
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
    
#    if background is None:
#        # Background subtract/move to zero
##        suby = suby - np.min(suby)
##        subx, suby = hitp.bkgd_sub(subx, suby, downsample_int)
#        pass
#    else:
#        suby = y - background
#        if suby.min() < 0:
#            if bg_shift_pos:
#                print('negative values in background-subtracted pattern. shifting first percentile to zero and setting values below it to zero.')
#                suby = suby.copy()
#                suby = suby - np.percentile(suby, .5)
#                suby[suby < np.percentile(suby, 1)] = 0
#                #suby = suby - suby.min()
#            else:
#                suby = suby - (suby * (suby < 0))
    def adj_suby(suby):
        if suby.min() < 0:
            if bg_shift_pos:
                print('negative values in background-subtracted pattern. shifting to create all non-negative values.')
                suby = suby.copy()
#                suby = suby - np.percentile(suby, .5)
#                suby[suby < np.percentile(suby, 1)] = 0
                suby = suby - suby.min()
            else:
                pass
                #suby = suby - (suby * (suby < 0))
        return suby

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
    #for i, (leftBnd, rightBnd) in zip(boundaries, boundaries[1:]): # indexes
    for i, (leftBnd, rightBnd) in enumerate(bnds_list): # indexes
#        if i == 21:
#            pdb.set_trace()
        selector = np.where((subx >= leftBnd) & (subx < rightBnd))
        xList.append(subx[selector])
        if background is not None and len(background) != len(subx):
            yList.append(adj_suby(suby[selector] - background[i]))
            suby[selector] -= background[i]
            print('background segments')
        elif background is not None:
            yList.append(adj_suby(suby[selector] - background[selector]))
            suby[selector] -= background[selector]
            print('global background')
        else:
            yList.append(suby[selector])
            
        print(suby[selector])
        print(suby[selector])
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

    kwlist = [deepcopy(kwargs) for _ in range(len(xList))]
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
    return suby, paramsList, noiseListNew, xListNew, yListNew, curve_paramsList

def iter_cnames():
    i = 0
    while True:
        yield 'curve {}'.format(i)
        i += 1
def dslice(d, i):
    res = dict()
    for j, k in enumerate(d.keys()):
        if j >= i:
            res[k] = d[k]
    return res

def _merge_eq(dlist):
    """
    round robin merge dicts of equal length
    """
    res = dict()
    vals = [pl.values() for pl in dlist]
    interleaved = [x for t in zip(*vals) for x in t]
    for d, s in zip(interleaved, iter_cnames()):
        res[s] = d
    return res

def _stack_dicts(d1, d2):
    """
    """
    res = dict()
    vals = list(d1.values()) + list(d2.values())
    for d, s in zip(vals, iter_cnames()):
        res[s] = d
    return res

def _refine_peak(xbit, ybit, noisebit, fitInfo, kwargs):
    # Restrict range and fit peaks
    guess_params_dict = kwargs['guess_params_dict']
    guess = hitp.paramdict_get_params(guess_params_dict)
    bounds = hitp.paramdict_get_bounds(guess_params_dict)
    if 'guess_params_dict' in kwargs:
        kwargs.pop('guess_params_dict', None)
#    print(guess)
#    print('dict')
#    print(guess_params_dict)
    curveParams, derivedParams = hitp.refine_peaks(xbit, ybit,
                        peakShape=fitInfo['peakShape'],
                        fitMode=fitInfo['fitMode'],
                        numCurves=fitInfo['numCurves'],
                        noise_estimate = noisebit,
                        guess = guess,
                        bounds = bounds,
                         **kwargs)
    return curveParams, derivedParams

def _merge_dicts(dlist):
    if len(dlist) == 0:
        return dlist
    if len(dlist) == 1:
        return dlist[0]
    if max(len(d) for d in dlist) == 0:
        return dlist[0]
    minlength = min(len(d) for d in dlist)
    part0 = _merge_eq(dlist)
    tails = []
    for d in dlist:
        if len(d) > minlength:
            tails.append(dslice(d, minlength))
    if len(tails) > 0:
        part1 = _merge_dicts(tails)
        part0 = _stack_dicts(part0, part1)
    return part0
def merge_dicts(*dlist):
    return _merge_dicts(dlist)

def mk_bnd_list(bounds, overlap = 1):
    return [[a, b] for a, b in list(zip(bounds, bounds[overlap:]))]

def merge_fitoutput_blocks(fitoutputs, overlap = 1, lists_only = False,
        params_only = False):
    """
    Merge peak fit parameters from adjacent blocks (defaults to overlap
    == 1, i.e. no merging)
    """
    arrays, params, noiselists, xLists, yLists, plists =\
        fitoutputs
    plists_new = []
    params_new = []

    def _iter(sources, merger):
        accum = []
        iterlist = []
        for source in sources:
            for shift in range(overlap):
                iterlist.append(source[shift:])
            groups = zip(*iterlist)
            merged = [merger(*pl) for pl in groups]
            accum.append(merged)
        res = np.empty(len(accum), object)
        for k, elt in enumerate(accum):
            res[k] = elt
        return res
        #return np.vstack(accum)

    def _merge_arrs(*arrs):
        return np.hstack(arrs)

    if lists_only:
        return arrays, params, _iter(noiselists, _merge_arrs),\
            _iter(xLists, _merge_arrs), _iter(yLists, _merge_arrs),\
            plists
    elif params_only:
        return arrays, _iter(params, merge_dicts), noiselists,\
            xLists, yLists,\
            _iter(plists, merge_dicts) 
    return arrays, _iter(params, merge_dicts), _iter(noiselists, _merge_arrs),\
        _iter(xLists, _merge_arrs), _iter(yLists, _merge_arrs),\
        _iter(plists, merge_dicts) 

def fit_curves(y, bba_smooth = 1.5, cb = _fit_peak, overlap = 1, bounds = None, **kwargs):
    if y.sum() != 0:
        x = np.arange(len(y))
        if bounds is None:
            boundaries = hitp.bayesian_block_finder(x, gf(y, bba_smooth), norm = cfg['fitInfo']['bba_norm'])
        else:
            boundaries = bounds
        #boundaries = [b for b in boundaries if b >= boundaries_min and b <= boundaries_max]
        print(boundaries)
        cfg['fitInfo']['blockBounds'] = boundaries
        cfg['fitInfo']['blockBounds_list'] = mk_bnd_list(boundaries, overlap)
        suby, derivedParams, noiseList, xList, yList, curve_params = workflow(y, boundaries, cb = cb, **kwargs)
        return suby, derivedParams, noiseList, xList, yList, curve_params
    return np.zeros_like(y), None, None, None, None, None

import pdb
def curvefit_2d(patterns: np.ndarray, background = None, noise_estimate = None,
        bounds = None, **kwargs):
    """
    Run BBA and peak-fitting routine for each XRD pattern in a
    multidimensional dataset whose last axis is the q dimension.
    """
#    if bounds is None and patterns.shape[0] != 1:
#        raise ValueError("bounds must be provided to process multiple samples")
#    else:
#        try:
#            len(bounds[0])
#        except TypeError:
#            bounds = np.vstack([bounds for _ in len(patterns)])
            
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
        else:
            noise_estimate_selected = None
        suby, derivedParams, noiseList, xList, yList, cparams =\
            fit_curves(patterns[indices], background = background_selected,
                        noise_estimate = noise_estimate_selected, bounds = bounds,# TODO hack
                        **kwargs)
        arrays[indices] = suby
        params[indices] = derivedParams
        curveparams[indices] = cparams
        noiselists[indices] = noiseList
        xLists[indices] = xList
        yLists[indices] = yList
        
    return arrays, params, noiselists, xLists, yLists, curveparams

from copy import deepcopy
def refine_2d(patterns, fitoutputs, noise_estimate = None, background = True, **kwargs):
    """
    Run BBA and peak-fitting routine for each XRD pattern in a
    multidimensional dataset whose last axis is the q dimension.
    """
    fitoutputs = deepcopy(fitoutputs)
    arrays, params, noiselists, xLists, yLists, curveparams = fitoutputs

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
        else:
            noise_estimate_selected = None
        suby, derivedParams, noiseList, xList, yList, cparams =\
            fit_curves(patterns[indices], background = background_selected,
                        noise_estimate = noise_estimate_selected, cb = _refine_peak,
                        param_guesses = curveparams[indices], **kwargs)
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
