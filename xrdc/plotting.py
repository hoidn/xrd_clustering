import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

from . import source_separation, featurization, visualization, peak_fitting as sep, feat, vis, pf
from . import workflows
from . import geometry
from . import xrdutils

from scipy.ndimage.filters import gaussian_filter as gf
from matplotlib.pyplot import figure

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
    for j in range(len(xLists[i])):
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
    plt.vlines(bounds, 0, 800, color = 'red', linestyles = 'dotted')
    return Ys, yss

