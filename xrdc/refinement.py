import numpy as np
from . import featurization as feat
from . import plotting
from . import misc
from functools import reduce
from copy import deepcopy as dc

def iter_cnames():
    i = 0
    while True:
        yield 'curve {}'.format(i)
        i += 1
        
def list_to_dict(lst):
    keygen = iter_cnames()
    return {next(keygen): v for v in lst}


def extract_paramlist_attribute(fitlist, attribute, truncate_end = 0):
    return np.array(
        reduce(lambda a, b: a + b,
              map(lambda a: list([v[attribute] for v in
                                  list(a.values())[:len(a.values()) - truncate_end]]),
                                      fitlist)))

def flat_extract(fitlists, attribute, truncate_end = 0):
    from functools import reduce
    return np.array([extract_paramlist_attribute(fitlists[i], attribute, truncate_end)
               for i in range(len(fitlists))])


def add_to_offsets(paramsdict_list, increment):
    paramsdict_list = dc(paramsdict_list)
    ncurves = len(paramsdict_list)
    for curveparams in paramsdict_list:
        curveparams['y0'] += increment / ncurves
        curveparams['leftBnd'][1] += increment / ncurves
        curveparams['rightBnd'][1] += increment / ncurves
    return paramsdict_list

#def _insert_background(paramsdict_list, background_list):
#    """
#    Put backgrounds into curve parameter dicts for a single sequence of peak fits.
#    """
#    paramsdict_list = dc(paramsdict_list)
#    ncurves = len(paramsdict_list)
#    for bg, curveparams in zip(background_list, paramsdict_list):
#        curveparams['background'] = bg
#    return paramsdict_list

def insert_background(fitoutput, background_array):
    """
    Put peak-level backgrounds into curve parameter dictionaries.
    """
    suby, summaryList, noiseListNew, xListNew, yListNew, curve_paramsList = fitoutput
    curve_paramsList = dc(curve_paramsList)
    # TODO 3d
    for i, row in enumerate(curve_paramsList):
        for j, d in enumerate(row):
            for curvename, curveparams in d.items():
                row[j][curvename]['background'] = background_array[i][j]
    return suby, summaryList, noiseListNew, xListNew, yListNew, curve_paramsList

def filter_peaks(fitoutput, bounds, i, boundoffset = 1, dim = 1, buffer = 0, invert = False,
        clip_edges = True):
    suby, summaryList, noiseListNew, xListNew, yListNew, curve_paramsList = fitoutput
    params, summary = curve_paramsList[i], summaryList[i]

    params_filt = []#list(paramsList[:boundoffset - 1]) # Voigt parameters
    summary_filt = []#list(curve_paramsList[:boundoffset - 1]) # Human-readable peak parameters
    for i in range(len(params)):
        j = i + boundoffset

        peakcenters = flat_extract([[params[i]]], 'x0')
        bndLeft, bndRight = bounds[j], (bounds + [1e9])[j + 1]
        #print(bndLeft, bndRight)
        mask = ((peakcenters > (bndLeft + buffer))\
            & (peakcenters < (bndRight - buffer)))[0]
        if invert:
            mask = ~mask
        else:
            if mask.sum() == 0:
                mask[:] = True
                print('encountered empty peak set. skipping filtering at location', i)
        retained_peaks = np.array(list(params[i].values()))[mask]
        removed_peaks = np.array(list(params[i].values()))[~mask]
        
        removed_peaks_offset = np.sum([curveparams['y0'] for curveparams in removed_peaks])
        retained_peaks = add_to_offsets(retained_peaks, removed_peaks_offset)
        
        params_filt.append(list_to_dict(
            list(retained_peaks)
        ))
        summary_filt.append(list_to_dict(list(np.array(list(summary[i].values()))[mask])))

    summary_filt_ndarr = np.empty(1, dtype = object)
    summary_filt_ndarr[0] = summary_filt

    params_filt_ndarr = np.empty(1, dtype = object)
    params_filt_ndarr[0] = params_filt
    print('initial peak count:', np.sum(feat.count_peaks([summary], dim = dim)))
    print('final peak count:', np.sum(feat.count_peaks([summary_filt], dim = dim)))

    if clip_edges:
        return suby, summary_filt_ndarr, take(noiseListNew, boundoffset),\
            take(xListNew, boundoffset), take(yListNew, boundoffset), params_filt_ndarr
    else:
#        print([elt for elt in [suby, summary_filt_ndarr, noiseListNew,\
#            xListNew, yListNew, params_filt_ndarr]])
        return suby, summary_filt_ndarr, noiseListNew,\
            xListNew,  yListNew,  params_filt_ndarr

def take(arr, offset = 1):
    return misc.deepmap(arr, lambda arr: arr[offset:-offset], 1)

def get_segment(arr, x):
#    leftBnd = x[0]
#    rightBnd = x[-1]
#    selector = np.where((x >= leftBnd) & (x <= rightBnd))
#    print(selector)
#    print(arr[selector])
#    print(type(arr[selector]))
    return arr[x]
    
import pdb
def get_tails(fitoutput, bounds, i, boundoffset = 1, dim = 1, buffer = 0, background = None):
    fitoutput = filter_peaks(fitoutput, bounds, i, boundoffset = boundoffset, dim = dim, buffer = buffer, invert = True, clip_edges = False)

    XX = []
    YY = []
    for j in range(len(fitoutput[1][i])):
        x, y, X, Y, ys, _, total_offset = plotting._get_curves(fitoutput, i, j, bounds = bounds)
#        if j == 21:
#            pdb.set_trace()
        XX += [x]
        #print(len(x), len(ys[-1]), len(X))
        if len(ys) <= 1:
            increment = np.zeros_like(x)
        else:
            #increment = ys[-1] - ys[-1].mean() # Don't include offset components of adjacent peak regions
            increment = ys[-1] - total_offset # Don't include offset components of adjacent peak regions
        if background is not None:
            #increment -= increment.mean() # 
            increment += get_segment(background[i], x)
            print(total_offset)
        YY += [increment]
    return XX, YY
