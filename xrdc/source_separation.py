import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter
from scipy import signal
from scipy.signal import convolve2d as conv2
from scipy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift, fftn, ifftn
from scipy.signal import blackman
from scipy.ndimage import gaussian_filter as gf
from scipy.interpolate import interp1d
from scipy.interpolate import NearestNDInterpolator
from scipy import ndimage as nd

from ipywidgets import interactive

import pdb

#def plot_df(*args):
#    df = pd.DataFrame([p for p, _ in args]).T
#    df.columns = [l for _, l in args ]
#    return df.plot()

#def lowpass_g(size, y):
#    gwindow = signal.gaussian(len(y), std = size)
#    L = power(fftshift(fft(gwindow)))
#    L /= L.max()
#    return L

#def highpass_g(size, y):
#    return 1 - lowpass_g(size, y)

def clip_low(x, frac_zero, invert = False):
    N = len(x)
    nz = int(frac_zero * N)
    x2  = x.copy()
    mask = np.ones_like(x)
    mask[:( nz) // 2 ] = 0
    mask[(-nz) // 2:] = 0
    return mask * x

def spec_fft_2(pattern, pad = 1000, roll = 0, do_conv_window = False, do_window = True, log = False):
    if log:
        y = np.pad(np.log(pattern + 1), pad, mode = 'edge')
    else:
        y = np.pad(pattern, pad, mode = 'edge')
    y = np.roll(y, roll)
    # Number of sample points
    N = y.shape[0]
    w = blackman(N)
    #w = 1
    #yf = fft(y * w)
    if do_window:
        ywf = fft(y*w)
    else:
        ywf = fft(y)
    if do_conv_window:
        ywf = conv_window(ywf)
    return w, ywf

def power(arr):
    """
    Return squared amplitude of an array.
    """
    ampsq = arr * np.conjugate(arr)
    return np.real(ampsq)

def mag(x):
    return np.sqrt(power(x))

def lorenz(gamma, x, x0):
    return ( 1. / (np.pi * gamma)) * (gamma**2) / ((x - x0)**2 + gamma**2)

def do_rl(sig, window_width = 4, peak_width = 2, window_type = 'gaussian',
         bwindow = None, norm = False):
    from skimage import color, data, restoration
    """
    Richardson-Lucy deconvolution
    """
    if window_type == 'gaussian':
        gwindow = signal.gaussian(len(y), std = window_width)
        #gwindow = lorenz(peak_width, np.arange(len(sig)), len(sig) // 2)
        L = power(fftshift(fft(gwindow)))
        L /= L.max()
        H = 1 - L
    elif window_type == 'step':
        H = clip_low_window(sig, .001) * bwindow
    else:
        raise ValueError
    
    g = signal.gaussian(len(y), std = peak_width)
    gfft = fftshift(fft(g))
    
    psf = mag(ifft(gfft * H))[:, None].T
    psf_1d = psf[:, 1275:1324]
    deconvolved_RL = restoration.richardson_lucy((sig[:, None].T) / (10 * sig.max()), psf_1d, iterations=120)
    if not norm:
        return deconvolved_RL[0]
    else:
        return deconvolved_RL[0] / deconvolved_RL[0].mean()

def conv_window(sig, mode = 'same'):
    tmp = np.real(np.sqrt(fftshift(fft(window)) * np.conjugate(fftshift(fft(window)))))
    return np.convolve(sig, tmp / tmp.max(), mode =mode)

def filter_bg(pattern, smooth = 1.5, window_type = 'gaussian', blackman = True,
    q_cutoff = .001, deconvolve = False, invert = False, **kwargs):
    """
    Extract high-frequency component (in q) from a 2d XRD dataset by
    high-pass filtering with a Blackman window and/or step function,
    taking the IFFT amplitude, and applying a gaussian smoothing.

    The option for deconvolution should be considered deprecated since
    it doesn't help with extraction.
    """
    # TODO how necessary is padding?
    qsize = pattern.shape[0]
    npad = int(1.66 * qsize) # TODO shouldn't be hardcoded
    window, ywf = spec_fft_2(pattern, npad)
    if window_type == 'gaussian': 
        raise NotImplementedError
        #sig = if_mag(patterns, highpass_g(q_cutoff, ywf) * ywf, **kwargs)
    elif window_type == 'step': # hard step
        clipped = clip_low(ywf, q_cutoff, invert = invert)
        if blackman:
            if invert:
                window = 1 - window
            sig = if_mag(clipped * window, **kwargs)
        else:
            sig = if_mag(clipped, **kwargs)
    else:
        raise ValueError
    if deconvolve:
        sig = do_rl(sig, q_cutoff, 2.2)
    sig = gf(sig, smooth)
    #print(npad)
    return sig[npad: -npad]#, mask[1000: -1000]


def iplot_rows(*patterns_list, X_list = None, styles = None, labels = None,
              log = False, offset = 0, height = '550px',
              nested_label_callback = None):
    """
    Plot a series of curves interactively.
    """
    # TODO X_list argument doesn't work
    plt.rcParams["figure.figsize"]=(12, 9)
    #labels = [label1, label2]
    if labels is None:
        labels = [''] * len(patterns_list)
    def f(i):
        if log:
            plt.semilogy()
        for j, patterns in enumerate(patterns_list):
            if styles is not None:
                extra_args = (styles[j],)
            else:
                extra_args = ()
            try:
                #assert x is not None
                #print(len(patterns[i]))
                if X_list is not None:
                    X = X_list[j]
                else:
                    X = np.arange(len(patterns[i][0]))
                for k in range(len(patterns[i])):
                    len(patterns[i][k]) # TODO hack
                    if nested_label_callback is not None:
                        label = nested_label_callback(patterns[i], k)
                    else:
                        label = k
                    plt.plot(X, patterns[i][k], *extra_args, label = label)
            except: # TODO except what?
                if X_list is not None:
                    X = X_list[j]
                else:
                    X = np.arange(len(patterns[i]))
                if j < 2:
                    plt.plot(X, patterns[i] + offset, *extra_args, label = labels[j])
                else:
                    plt.plot(X, patterns[i] + offset, *extra_args)
        plt.legend()
        plt.grid()

    interactive_plot = interactive(f, i=(0, len(patterns_list[0]) - 1), step = 1)
    output = interactive_plot.children[-1]
    output.layout.height = height
    return interactive_plot

def logim(arr, offset = 1):
    plt.imshow(np.log(offset + arr), cmap = 'jet')

def if_mag(arr, phase = 0, truncate = False, toreal = 'psd', **kwargs):
    """
    Return the amplitude or real component of the inverse fourier
    transform, with optional phase shift.
    """
    phase = np.exp(1j * phase)
    tmp = ifft(arr)
    if toreal == 'psd':
        real = np.real(np.sqrt(np.conjugate(tmp) * tmp))
    elif toreal == 'real':
        real = np.real(tmp)
    else:
        raise ValueError
    if truncate:
        raise NotImplementedError
    return real

def extract_single(row, q_cutoff = .001, smooth_q = 1.7):
    """
    Default procedure for extracting the high-frequency component of a
    single 1d diffraction pattern.
    """
    return filter_bg(row, smooth_q, window_type = 'step', deconvolve = False, toreal = 'psd', q_cutoff = q_cutoff)

def apply_bottom(func, arr, axis = None, **kwargs):
    """
    apply 1d array-transforming function to bottom (q) dimension
    """
    def new_f(*args):
        """ bind kwargs """
        return func(*args, **kwargs)

    if axis is None:
        axis = len(arr.shape) - 1
    return np.apply_along_axis(new_f, axis, arr)

def ffill(arr):
    """
    forward-fill nans in a 1d np array
    """
    mask = np.isnan(arr)
    idx = np.where(~mask,np.arange(mask.shape[0]),0)
    np.maximum.accumulate(idx,axis=0, out=idx)
    out = arr[idx]#[np.arange(idx.shape[0])[:,None], idx]
    return out

def bfill(arr):
    """
    backwards-fill nans in a 1d np array
    """
    return ffill(arr[::-1])[::-1]

def fill(arr):
    """
    fill consecutive nans at the edge of a 1d np array with the nearest
    non-nan value.
    """
    return ffill(bfill(arr))

def fill_nd(arr):
    """
    fill consecutive nans at the edge of a np array with the nearest
    non-nan value.
    """
    return apply_bottom(fill, arr)

def mk_smooth(patterns, smooth_neighbor, smooth_q):
    n = len(patterns.shape)
    return (smooth_neighbor,) * (n - 1) + (smooth_q,)

def reference_bgsub(patterns, smooth_q = 1.7, smooth_neighbor_background = 1,
        q_cutoff = .001, **kwargs):
    """
    Extract high-frequency component (in q) from a 2d XRD dataset. This
    method distorts peak intensities but is good at identifying their
    locations.
    """
    bgsubbed_nosmooth = apply_bottom(extract_single, patterns,
        q_cutoff = q_cutoff, smooth_q = smooth_q)
    bgsubbed_final = gf(bgsubbed_nosmooth, mk_smooth(patterns, smooth_neighbor_background, smooth_q))
    bgsubbed_final *= patterns.max() / bgsubbed_final.max() #np.percentile(patterns, 99.9) / np.percentile(bgsubbed_final, 99.9)
    return bgsubbed_final

def interprows(arr, mask, fn = None, **kwargs):
    # TODO refactor, generalize
    if fn is None:
        def fn(*args):
            return interp1d(*args, bounds_error = False, **kwargs)
    if len(arr.shape) == 2:
        res = []
        for row, rowmask in zip(arr, mask):
            x = np.indices(row.shape)[0][rowmask]
            y = row[rowmask]
            f = fn(x, y, **kwargs)
            res.append(f(np.indices(row.shape)[0]))
        return np.vstack(res)
    elif len(arr.shape) == 3:
        res = np.zeros_like(arr)
        n, m, _ = arr.shape
        res = np.zeros_like(arr)
        for i in range(n):
            for j in range(m):
                row = arr[i, j, :]
                rowmask = mask[i, j, :]
                x = np.indices(row.shape)[0][rowmask]
                y = row[rowmask]
                f = fn(x, y, **kwargs)
                res[i, j, :] = f(np.indices(row.shape)[0])
        return res

def get_bgmask(patterns, threshold, **kwargs):
    """
    Find peak regions and return a mask that identifies them.
    Peak pixels map to False and background pixels map to True.

    This function returns a boolean array
    """
    bgsubbed = reference_bgsub(patterns, **kwargs)
    percentiles = np.percentile(bgsubbed, threshold, axis = len(patterns.shape) - 1)
    mask = (bgsubbed >
            percentiles[..., None])
    bgsubbed[mask] = np.nan
    bkgmask = ~np.isnan(bgsubbed)
    return bkgmask

def get_background_nan(patterns, threshold = 50, smooth_q = 1.7,
        smooth_q_background = 10, smooth_neighbor_background = 1, q_cutoff = .001):
    """
    Mask peak entries in patterns to np.nan and optionally do some Gaussian smoothing.
    """
    # TODO smooth or not?
    smooth = mk_smooth(patterns, smooth_neighbor_background, smooth_q_background)
    print(smooth)
    bkgmask = get_bgmask(patterns, threshold, smooth_q = smooth_q, smooth_neighbor_background = smooth_neighbor_background, q_cutoff = q_cutoff)
    filled_bg = interprows(patterns, bkgmask)
    smooth_bg = gf(filled_bg, smooth)
    return smooth_bg

def get_background(patterns, threshold = 50, bg_fill_method = 'simple',
        smooth_q = 1.7, smooth_neighbor_background = 1, q_cutoff = .001,
        smooth_q_background = 10, smooth_before = True, smooth_after = True):
    smooth = mk_smooth(patterns, smooth_neighbor_background, smooth_q)
    """
    If smooth_before, smooth background values before interpolation.
    If smooth_after, smooth background estimate post-interpolation.

    Background smoothing is applied *before* interpolation but not
    after. The returned background array is not smoothed.
    """
    if bg_fill_method in ['none', 'simple', 'extrap_1d']:
        smooth_bg = get_background_nan(patterns, threshold = threshold,
            smooth_q_background = 0,
            smooth_neighbor_background = 0, q_cutoff = q_cutoff)
        if bg_fill_method == 'none':
            mask = get_bgmask(patterns, threshold, smooth_q_background = 0,
                smooth_neighbor_background = 0, q_cutoff = q_cutoff)
            filled_data = smooth_bg
        elif bg_fill_method == 'simple':
            # TODO am i getting the higher-dimensional nearest neighbor?
            mask = np.where(~np.isnan(smooth_bg))
            interp = NearestNDInterpolator(np.transpose(mask), smooth_bg[mask])
            filled_data = interp(*np.indices(smooth_bg.shape))
        elif bg_fill_method == 'extrap_1d':
            filled_data = fill_nd(smooth_bg) 
        else:
            raise ValueError
    elif bg_fill_method == 'cloughtocher':
        mask = get_bgmask(patterns, threshold)
        filled_data = CTinterpolation(mask * patterns)
    else:
        raise ValueError
    if smooth_before is False:
        raise NotImplementedError
    if smooth_after:
        filled_data = gf(filled_data, mk_smooth(filled_data, smooth_neighbor_background, smooth_q_background))
    return filled_data

def draw_circle(arr,diamiter):
    '''
    Input:
    shape    : tuple (height, width)
    diameter : scalar
    
    Output:
    np.array of shape that says True within a circle with diamiter =
    around center
    '''
    shape = arr.shape
    assert len(shape) == 2
    TF = np.zeros(shape)
    center = np.array(TF.shape)/2.0

    for iy in range(shape[0]):
        for ix in range(shape[1]):
            TF[iy,ix] = (iy- center[0] + .5)**2 + (ix - center[1] + .5)**2 < diamiter **2
    return(TF)

def gaussNd(sigma):
    """
    Returns a function that's a gaussian over a cube of coordinates
    of any dimension.
    """
    N = 1 / (sigma * np.sqrt(2 * np.pi))
    def f(*args):
        n = args[0].shape[0]
        def g(*args2):
            args_sq = (np.array(args2) * np.array(args2)).sum(axis = 0)
            return np.exp(- args_sq / (2 * sigma**2))
        # TODO check the offset
        x0 = (n  -1) / 2
        return N * g(*(arg - x0 for arg in args))
    return f

def gauss_low_Nd(arr, cutoff):
    # TODO assert cubic shape
    n = len(arr)
    args = np.indices(arr.shape)
    sigma = cutoff * n
    return (gaussNd(sigma)(*args))

def lowpassNd(arr, cutoff, mode = 'gaussian'):
    """
    Low pass filter with a circular step aperture
    """
    if mode == 'step':
        raise NotImplementedError # TODO update this
        mask = draw_circle(arr, int(cutoff * ((arr.shape[0] + arr.shape[1]) / 2)))
    elif mode == 'gaussian':
        mask = gauss_low_Nd(arr, cutoff)
        mask /= mask.max()
    else:
        raise ValueError
    arrfft = fftshift(fftn(arr))
    arr_filtered = ifftn(ifftshift(mask * arrfft))
    return arr_filtered

def imshow_log(arr):
    """plot log of absolute value heatmap, with an offset
    """
    arr = arr.copy()
    floor = np.abs(arr.min())
    arr[arr < 0] = 0
    plt.imshow(np.log(floor + arr ), cmap = 'jet')


#https://stackoverflow.com/questions/3662361/fill-in-missing-values-with-nearest-neighbour-in-python-numpy-masked-arrays
def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """
    #import numpy as np
    #import scipy.ndimage as nd

    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]

def CTinterpolation(imarray, smoothing = 0):
    """
    Do a 2d interpolation to fill in zero values of a 2d ndarray.
    
    Uses scipy.interpolate import CloughTocher2DInterpolator.
    
    Arguments:
        imarray : np.ndarray
        detid : string
        smoothing : numeric
    """
    from scipy.interpolate import CloughTocher2DInterpolator as ct
    dimx, dimy = np.shape(imarray)
    gridx, gridy = np.indices(imarray.shape)#np.arange(dimx), np.arange(dimy)
    

    def interp_2d(imarray):
        # flattened values of all pixels
        z = imarray.flatten()
        z_good = np.where(z != 0)[0]
        if len(z_good) == 0:
            return np.zeros_like(imarray)
        else:
            x, y = gridx.flatten(), gridy.flatten()
            xgood, ygood = x[z_good], y[z_good]

            points = np.vstack((xgood, ygood)).T
            values = z[z_good]
            interpolator = ct(points, values)
            return interpolator(x, y).reshape(imarray.shape)

    # Input to the CT interpolation is a smoothed NN interpolation
    # This pre-interpolation step, combined with a sufficiently large value of
    # smoothing, is often necessary to prevent the interpolation from
    # oscillating/overshooting.
    smoothNN = gf(fill(imarray), smoothing)
    smooth_masked = np.where(np.isclose(imarray, 0), 0., smoothNN)
    CTinterpolated = interp_2d(smooth_masked)
    
    # Fill in NAN values from outside the convex hull of the interpolated points
    combined = fill(CTinterpolated)# np.where(np.isnan(CTinterpolated), smoothNN, CTinterpolated)
    return combined

def NDinterpolation(imarray, smoothing = 0):
    """
    Do a 2d interpolation to fill in zero values of a 2d ndarray.
    
    Uses scipy.interpolate import CloughTocher2DInterpolator.
    
    Arguments:
        imarray : np.ndarray
        detid : string
        smoothing : numeric
    """
    from scipy.interpolate import NearestNDInterpolator as ct
    dimx, dimy = np.shape(imarray)
    gridx, gridy = np.indices(imarray.shape)#np.arange(dimx), np.arange(dimy)
    

    def interp_2d(imarray):
        # flattened values of all pixels
        z = imarray.flatten()
        z_good = np.where(z != 0)[0]
        if len(z_good) == 0:
            return np.zeros_like(imarray)
        else:
            x, y = gridx.flatten(), gridy.flatten()
            xgood, ygood = x[z_good], y[z_good]

            points = np.vstack((xgood, ygood)).T
            values = z[z_good]
            interpolator = ct(points, values)
            return interpolator(x, y).reshape(imarray.shape)

    # Input to the CT interpolation is a smoothed NN interpolation
    # This pre-interpolation step, combined with a sufficiently large value of
    # smoothing, is often necessary to prevent the interpolation from
    # oscillating/overshooting.
    smoothNN = gf(fill(imarray), smoothing)
    smooth_masked = np.where(np.isclose(imarray, 0), 0., smoothNN)
    CTinterpolated = interp_2d(smooth_masked)
    
    # Fill in NAN values from outside the convex hull of the interpolated points
    combined = fill(CTinterpolated)# np.where(np.isnan(CTinterpolated), smoothNN, CTinterpolated)
    return combined

# This function is the main entry point
def separate_signal(patterns, cutoff = .2, mode = 'gaussian',
        background_after_filter = True, q_cutoff = .001, **kwargs):
    """
    Decompose a dataset into high- and low-frequency components in the
    non-q dimensions. Any rows that sum to zero are neglected.

    If background_after_filter, the background is estimated without
    first removing high-frequency components.

    The most important keyword arguments are:
        -cutoff: frequency cutoff for noise extraction in non-q dimensions.
        -threshold: percentage of pixels to use in the background
            interpolation. A lower value excludes more points in and
            surrounding peak regions and therefore gives a more conservative
            estimate of the background.
        -background_after_filter: **IMPORTANT** this can be set to True
            in the case of truly continuous datasets, but otherwise it
            must be False. Else noise removal will corrupt both the
            background estimate and the background-subtracted signal
            (fast_q).
        -smooth_q_background: gaussian smoothing standard deviation for
            the interpolated background. Should be smaller than the
            scale of background variations.
        -smooth_q: gaussian smoothing standard deviation for peak
            extraction (should be of order peak FWHM / 2).
        -smooth_neighbor_background: gaussian smoothing standard
            deviation for non-q dimensions of the background estimate.
            Can be of order unity for connected datasets but should
            be set to 0 if the background is discontinuous accross
            neighboring patterns.

    Other arguments:
        -mode: == 'gaussian' or 'step'; kernel to use for the non-q
            frequency filtering.
        -q_cutoff: frequency cutoff for q peak filtering (deprecated;
            should be a value close to 0 since peak extraction uses a
            Blackman window by default, which is sufficient on its own)
        -bg_fill_method. fill method for background values outside of
            interpolation range; should equal one of the following:
                'simple': nearest neighbor matching across non-q dimensions
                'none': np.nan values outside the interpolation range
                'extrap_1d': 1d extrapolation using the nearest non-nan
                    value
                'cloughtocher': cubic 2d interpolation

    Returns tuple:
        (interpolated background (excluding high-frequency non-q component),
        signal (excluding high-frequency non-q component) - interpolated background,
        low-frequency non-q signal,
        high-frequency non-q signal))
    """
    for ii, jj in zip((patterns.shape[0],) + patterns.shape[:-1], patterns.shape[:-1]):
        # array must be square for fourier filtering to work in the non-q dimensions
        assert ii == jj 
    # TODO filter T before background or after? need to do a more careful comparison
    # calculate the low-frequency component in xy
    nq = patterns.shape[-1]
    low_xy = np.zeros_like(patterns)
    wafer_mask = (patterns.sum(axis = (len(patterns.shape) - 1)) != 0)
    for i in range(nq):
        low_xy[..., i] = np.real(lowpassNd(fill(patterns[..., i], patterns[..., i] == 0), cutoff, mode)) * wafer_mask
    high_xy = patterns - low_xy

#    # TODO take cutoff parameter for q filtering as well
    if background_after_filter:
        interpolated_background = get_background(low_xy, q_cutoff = q_cutoff,
            **kwargs)
        fast_q = low_xy - interpolated_background
    else:
        interpolated_background = get_background(patterns, q_cutoff = q_cutoff,
            **kwargs)
        fast_q = patterns - interpolated_background
    return interpolated_background, fast_q, low_xy, high_xy
