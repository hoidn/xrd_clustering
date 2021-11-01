import pandas as pd
import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift, fftn, ifftn
from scipy.signal import blackman
import matplotlib.pyplot as plt
from scipy.signal import butter
from scipy import signal
from scipy.signal import convolve2d as conv2

from skimage import color, data, restoration
from scipy.ndimage import gaussian_filter as gf
from scipy.interpolate import interp1d
from scipy.interpolate import NearestNDInterpolator

from xrdc import featurization as feat
from scipy import ndimage as nd

#def plot_df(*args):
#    df = pd.DataFrame([p for p, _ in args]).T
#    df.columns = [l for _, l in args ]
#    return df.plot()

def lowpass_g(size, y):
    gwindow = signal.gaussian(len(y), std = size)
    L = power(fftshift(fft(gwindow)))
    L /= L.max()
    return L

#def clip_high(x, frac_zero):
#    x2  = x.copy()
#    mask = clip_high_filter(x, frac_zero)
#    return x2 * mask

#def clip_high_window(x, frac_zero):
#    """ low pass filter
#    """
#    x = ifftshift(x)
#    N = len(x)
#    nz = int(frac_zero * N)
#    x2 = np.ones_like(x)
#    x2[(N - nz) // 2: (N + nz) // 2] = 0
#    #x2[(-nz) // 2:] = 0
#    return fftshift(x2)

def clip_low(x, frac_zero, invert = False, inc_window = False):
    x2  = x.copy()
    mask = clip_low_window(x, frac_zero, invert = invert, inc_window = inc_window)
    return x2, mask

def clip_low_window(x, frac_zero, invert = False, inc_window = False):
    x = ifftshift(x)
    N = len(x)
    nz = int(frac_zero * N)
    x2  = x.copy()
    mask = np.ones_like(x)
    mask[:( nz) // 2 ] = 0
    mask[(-nz) // 2:] = 0
    if inc_window:
        window = np.blackman(N)
        mask *= window
    if invert:
        mask = 1 - mask
    return fftshift(mask)
    return x2, mask

def highpass_g(size, y):
    return 1 - lowpass_g(size, y)

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
             deconvolve = False, invert = False, **kwargs):
    """
    Extract high-frequency component (in q) from a 2d XRD dataset by
    high-pass filtering, taking the IFFT amplitude, and applying a
    gaussian smoothing.

    Optionally multiply the FFT output by a Blackman window envelope.

    The option for deconvolution should be considered deprecated since
    it doesn't help with extraction.
    """
    cutoff = 4
    window, ywf = spec_fft_2(pattern, 1000)
    if window_type == 'gaussian': #todo inversion
        sig = if_mag(patterns, highpass_g(cutoff, ywf) * ywf, **kwargs)
    elif window_type == 'step': # hard step
        clipped, mask = clip_low(ywf, .001, invert = invert)
        if blackman:
            if invert:
                window = 1 - window
            mask *= window
            sig = if_mag(clipped * window, **kwargs)
        else:
            sig = if_mag(clipped, **kwargs)
    else:
        raise ValueError
    if deconvolve:
        sig = do_rl(sig, cutoff, 2.2)
    sig = gf(sig, smooth)
    return sig[1000: -1000]#, mask[1000: -1000]

from ipywidgets import interactive
import matplotlib.pyplot as plt
import numpy as np

def iplot_rows(patterns, patterns2 = None, label1 = 'raw', label2 = 'curve fit subtraction',
              log = False, offset = 0, height = '550px'):
    """
    Plot a series of curves interactively.
    """
    plt.rcParams["figure.figsize"]=(12, 9)
    def f(i):
        if log:
            plt.semilogy()
        plt.plot(patterns[i] + offset, label = label1)
        if patterns2 is not None:
            plt.plot(patterns2[i] + offset, label = label2)
        plt.legend()
        plt.grid()

    interactive_plot = interactive(f, i=(0, len(patterns)), step = 1)
    output = interactive_plot.children[-1]
    output.layout.height = height
    return interactive_plot

def logim(arr, offset = 1):
    plt.imshow(np.log(offset + arr), cmap = 'jet')

#def mk_black2d(y2d, fraction):
#    """
#    Make a two-dimensional Blackman filter.
#    """
#    N, M = y2d.shape
#    n = m = fraction
#    nwin, mwin = 2 * (int(n * N) // 2), 2 * (int(m * M) // 2)
#    w2d = blackman(nwin)[:, None] * blackman(mwin)
#    w2d = np.pad(w2d, (M - mwin) // 2)
#    w2d = np.sqrt(1e-9 + w2d)
#    trim = w2d.shape[0] - N
#    w2d = w2d[trim // 2: -trim // 2]
#    return w2d

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

def extract_single(row):
    """
    Default procedure for extracting the high-frequency component of a
    single 1d diffraction pattern.
    """
    return filter_bg(row, 0, window_type = 'step', deconvolve = False, toreal = 'psd')

def apply_bottom(func, arr):
    """
    apply 1d function to bottom (q) axis
    """
    return np.apply_along_axis(func, len(arr.shape) - 1, arr)

def mk_smooth(patterns, smooth_neighbor, smooth_q):
    n = len(patterns.shape)
    return (smooth_neighbor,) * (n - 1) + (smooth_q,)

#def default_smooth(arr):
#    """
#    Gaussian smoothing kernels used in the extraction of peak regions.
#    The q-smoothing parameter should be set according to the typical peak
#    width, while the neighbor smoothing shouldn't have to change if
#    standard assumptions apply.
#    """
#    # TODO move this to configuration
#    if len(arr.shape) == 3:
#        return (1, 1, 1.7)
#    elif len(arr.shape) == 2:
#        return (1, 1.7)
#    else:
#        return ValueError

def reference_bgsub(patterns, smooth_q = 1.7, smooth_neighbor = 1, **kwargs):
    """
    Extract high-frequency component (in q) from a 2d XRD dataset. This
    method distorts peak intensities but is good at identifying their
    locations.
    """
    bgsubbed_nosmooth = apply_bottom(extract_single, patterns)
    bgsubbed_final = gf(bgsubbed_nosmooth, mk_smooth(patterns, smooth_neighbor, smooth_q))
    bgsubbed_final *= patterns.max() / bgsubbed_final.max() #np.percentile(patterns, 99.9) / np.percentile(bgsubbed_final, 99.9)
    return bgsubbed_final

def interprows(arr, mask):
    # TODO refactor
    if len(arr.shape) == 2:
        res = []
        for row, rowmask in zip(arr, mask):
            x = np.indices(row.shape)[0][rowmask]
            y = row[rowmask]
            f = interp1d(x, y, bounds_error = False)
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
                f = interp1d(x, y, bounds_error = False)
                res[i, j, :] = f(np.indices(row.shape)[0])
        return res

def get_bgmask(patterns, threshold, **kwargs):
    """
    Find peak regions and return a mask that identifies them.
    """
    bgsubbed = reference_bgsub(patterns, **kwargs)
    bgsubbed[bgsubbed > np.percentile(bgsubbed, threshold)] = np.nan
    bkgmask = ~np.isnan(bgsubbed)
    return bkgmask

def get_background_nan(patterns, threshold = 50, smooth_q = 1.7, smooth_neighbor = 1):
    # TODO smooth or not?
    smooth = mk_smooth(patterns, smooth_neighbor, smooth_q)
    bkgmask = get_bgmask(patterns, threshold, smooth_q = smooth_q, smooth_neighbor = smooth_neighbor)
    filled_bg = interprows(patterns, bkgmask)
    smooth_bg = gf(filled_bg, smooth)
    return smooth_bg

def get_background(patterns, threshold = 50, method = 'simple', smooth_q = 1.7, smooth_neighbor = 1):
    smooth = mk_smooth(patterns, smooth_neighbor, smooth_q)
    if method == 'simple':
        smooth_bg = get_background_nan(patterns, threshold = threshold,
            smooth_q = smooth_q, smooth_neighbor = smooth_neighbor)
        mask = np.where(~np.isnan(smooth_bg))
        # TODO am i getting the higher-dimensional nearest neighbor?
        interp = NearestNDInterpolator(np.transpose(mask), smooth_bg[mask])
        filled_data = interp(*np.indices(smooth_bg.shape))
    elif method == 'cloughtocher':
        mask = get_bgmask(patterns, threshold)
        filled_data = CTinterpolation(mask * patterns)
    else:
        raise ValueError
    return filled_data


def draw_circle(arr,diamiter):
    '''
    Input:
    shape    : tuple (height, width)
    diameter : scalar
    
    Output:
    np.array of shape  that says True within a circle with diamiter =  around center 
    '''
    shape = arr.shape
    assert len(shape) == 2
    TF = np.zeros(shape)
    center = np.array(TF.shape)/2.0

    for iy in range(shape[0]):
        for ix in range(shape[1]):
            TF[iy,ix] = (iy- center[0] + .5)**2 + (ix - center[1] + .5)**2 < diamiter **2
    return(TF)


# from functools import reduce
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

# This function is the main entry point
def separate_signal(patterns, cutoff = .2, mode = 'gaussian', **kwargs):
    """
    Decompose a dataset into high- and low-frequency components in  the
    non-q dimensions.
    Any rows that sum to zero are neglected.

    Returns interpolated background, raw patterns - interpolated background,
        low-frequency non-q signal, high-frequency non-q signal)
    """
    # TODO take cutoff parameter for q filtering as well
    interpolated_background = get_background(patterns, **kwargs)
    fast_q = patterns - interpolated_background
    # calculate the low-frequency component in xy
    nq = patterns.shape[-1]
    low_xy = np.zeros_like(patterns)
    wafer_mask = (patterns.sum(axis = (len(patterns.shape) - 1)) != 0)
    for i in range(nq):
        low_xy[..., i] = np.absolute(lowpassNd(fill(patterns[..., i], patterns[..., i] == 0), cutoff, mode)) * wafer_mask
    high_xy = patterns - low_xy
    return interpolated_background, fast_q, low_xy, high_xy
