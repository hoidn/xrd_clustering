import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

from . import source_separation as sep
from . import featurization as feat
from . import visualization as vis
from . import peak_fitting as pf
from . import workflows
from . import geometry
from . import xrdutils

from scipy.ndimage.filters import gaussian_filter as gf
from matplotlib.pyplot import figure

def get_total_offsets(paramsdict):
    res = 0
    for curveparams in paramsdict.values():
        res += curveparams['y0']#_get_attribute(curveparams, 'y0', i, j + 2)
    return res

def _get_curves(fitoutput, i, j, peak_range_only = True, bounds = None):
    # TODO clean up
    arrays, paramsLists, noiselists, xLists, yLists, curveparams = fitoutput

    x, y, cparams = xLists[i][j], yLists[i][j], curveparams[i][j]

    # y values for individual Voigts
    ys = pf.hitp.gen_curve_fits(x, y, cparams, 'Voigt')

    X, Y = np.hstack(xLists[i]), np.hstack(yLists[i])
    if bounds is None:
        bounds = [onex[0] for onex in xLists[i]]
    if peak_range_only:
        mask = (X >= x.min()) & (X <= x.max())
        return x, y, X[mask], Y[mask], ys, bounds, get_total_offsets(curveparams[i][j])
    else:
        return x, y, X, Y, ys, bounds, get_total_offsets(curveparams[i][j])

def get_curves(fitoutput, i, j, peak_range_only = True, bounds = None):
    # TODO clean up
    x, y, X, Y, ys, bounds, total_offset = _get_curves(fitoutput, i, j, peak_range_only = True, bounds = None)
    return x, y, X, Y, ys, bounds

        
def plot_one_fit(fsub_stop_2d_1, i, j):
    fig1 = figure(1)
    frame1=fig1.add_axes((.1,.3,.8,.6))

    arrays, paramsLists, noiselists, xLists, yLists, curveparams = fsub_stop_2d_1

    Ys = []
    yss = []
    x, y, X, Y, ys, bounds = get_curves(fsub_stop_2d_1, i, j)
    yss.append(ys)
    Ys.append(Y)
    plt.gca().set_prop_cycle(None)
    for k, yk in enumerate(ys[:-1]):
        plt.plot(x, yk, label = 'Peak {}'.format(k))
    plt.plot(x, ys[-1], label = 'total fits', color = 'blue')

    _, _, X, Y, _, _ = get_curves(fsub_stop_2d_1, i, j, peak_range_only=True)
    plt.plot(X, Y, 'o', color = 'k', label = 'Data')
    plt.grid()
    
    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
    plt.legend()
    
    #Residual plot
    frame2=fig1.add_axes((.1,.1,.8,.2)) 
    plt.plot(xLists[i][j], np.abs(noiselists[i][j]), 'or', label = 'Noise estimate (L1)')
    plt.plot(xLists[i][j], 
        np.abs(yLists[i][j] - ys[-1]), 'ok', label = 'Fit residual (L1)')

    plt.grid()
    plt.legend(loc = 'upper left')
    return Ys, yss

def plot_all_fits(fsub_stop_2d_1, i):
    arrays, paramsLists, noiselists, xLists, yLists, curveparams = fsub_stop_2d_1

    Ys = []
    yss = []
    for j in range(len(paramsLists[i])):
        x, y, X, Y, ys, bounds = get_curves(fsub_stop_2d_1, i, j)
        yss.append(ys)
        Ys.append(Y)
        plt.gca().set_prop_cycle(None)
        for k, yk in enumerate(ys[:-1]):
            plt.plot(x, yk, label = k)
        plt.plot(x, ys[-1], label = 'total fits', color = 'blue')

    _, _, X, Y, _, _ = get_curves(fsub_stop_2d_1, i, j, peak_range_only=False)
    plt.plot(X, Y, '--', color = 'k')
    plt.grid()
    plt.vlines(bounds, 0, Y.max(), color = 'red', linestyles = 'dotted')
    return Ys, yss

class Block(object):
    def __init__(self, fitoutput, j, i = 0):
        x, y, X, Y, ys, bounds = get_curves(fitoutput, 0, j)
        array, params, noise, x, y, curveparams = [elt[i][j] for elt in fitoutput]
        self.x = x
        self.y = y
        self.peaks = [{'y': yy, 'params': pp} for yy, pp in zip(ys, params.values())]
        self.fit_total = ys[-1]

def extract_block(fitoutput, j):
    x, y, X, Y, ys, bounds = fitoutput
    return Block(x, y, ys, [row[j] for row in fitoutputs])

def extract_blocks(fitoutput, i = 0):
    fitoutput_row = [elt[i] for elt in fitoutput]
    for j in range(len(fitoutput_row[2])):
        yield Block(fitoutput, j, i)
