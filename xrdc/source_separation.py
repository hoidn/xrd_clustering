import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq, ifft, fft2, ifft2, ifftshift
from scipy.signal import blackman
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
from scipy.signal import butter
from scipy import signal
from scipy.signal import convolve2d as conv2

from skimage import color, data, restoration
from scipy.ndimage import gaussian_filter as gf

from xrdc import featurization as feat
from xrdc import fourier
def plot_df(*args):
    df = pd.DataFrame([p for p, _ in args]).T
    df.columns = [l for _, l in args ]
    return df.plot()

def lowpass_g(size, y):
    gwindow = signal.gaussian(len(y), std = size)
    L = power(fftshift(fft(gwindow)))
    L /= L.max()
    return L

def clip_high(x, frac_zero):
    x2  = x.copy()
    mask = clip_high_filter(x, frac_zero)
    return x2 * mask

def clip_high_window(x, frac_zero):
    """ low pass filter
    """
    x = ifftshift(x)
    N = len(x)
    nz = int(frac_zero * N)
    x2  = np.ones_like(x)
    x2[(N - nz) // 2 : (N + nz) // 2] = 0
    #x2[(-nz) // 2:] = 0
    return fftshift(x2)

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
        
#     x2[:( nz) // 2 ] = 0
#     x2[(-nz) // 2:] = 0
    return x2, mask

def highpass_g(size, y):
    return 1 - lowpass_g(size, y)

def if_mag(patterns, arr, phase = 0, truncate = False):
    trunc = len(arr) - len(patterns[0])
    phase = np.exp(1j * phase)
    tmp = ifft(ifftshift(arr))
    if truncate:
        return np.real(np.sqrt(np.conjugate(tmp * phase) * tmp))[trunc // 2: -trunc // 2]
    return np.real(np.sqrt(np.conjugate(tmp * phase) * tmp))

def spec_fft(patterns, i, pad = 1000, roll = 0, do_conv_window = False, do_window = True, log = False, dat = None):
    if dat is not None:
        pi = dat
    else:
        pi = patterns[i]
    if log:
        y = np.pad(np.log(pi + 1), pad, mode = 'edge')
    else:
        y = np.pad(pi, pad, mode = 'edge')
    y = np.roll(y, roll)
    # Number of sample points
    N = y.shape[0]
    w = blackman(N)
    #w = 1
    #yf = fft(y * w)
    if do_window:
        ywf = fftshift(fft(y*w))
    else:
        ywf = fftshift(fft(y))
    if do_conv_window:
        ywf = conv_window(ywf)
    return w, ywf

def power(arr):
    ampsq = arr * np.conjugate(arr)
    return np.real(ampsq)

def mag(x):
    return np.sqrt(power(x))

def lorenz(gamma, x, x0):
    return ( 1. / (np.pi * gamma)) * (gamma**2) / ((x - x0)**2 + gamma**2)

def do_rl(sig, window_width = 4, peak_width = 2, window_type = 'gaussian',
         bwindow = None, norm = False):
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
    return np.convolve(sig, tmp / tmp.max(), mode =mode)#if_mag(clip_low(ywf, .01) * window)

def filter_bg(patterns, i, smooth = 1.5, window_type = 'gaussian', blackman = True,
             deconvolve = False, invert = False):
    cutoff = 4
    window, ywf = spec_fft(patterns, i, 1000)
    if window_type == 'gaussian': #todo inversion
        sig = if_mag(patterns, highpass_g(cutoff, ywf) * ywf)
    elif window_type == 'step': # hard step
        clipped, mask = clip_low(ywf, .001, invert = invert)
        if blackman:
            if invert:
                window = 1 - window
            mask *= window
            sig = if_mag(patterns, clipped * window)
        else:
            sig = if_mag(patterns, clipped)
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

def mk_black2d(y2d, fraction):
    N, M = y2d.shape
    n = m = fraction
    nwin, mwin = 2 * (int(n * N) // 2), 2 * (int(m * M) // 2)
    w2d = blackman(nwin)[:, None] * blackman(mwin)
    w2d = np.pad(w2d, (M - mwin) // 2)
    w2d = np.sqrt(1e-9 + w2d)
    trim = w2d.shape[0] - N
    w2d = w2d[trim // 2: -trim // 2]
    return w2d

def clip_low2d(x, frac_zero):
    N = len(x)
    nz = int(frac_zero * N)
    x2  = x.copy()
    x2[:( nz) // 2 ] = 0
    x2[(-nz) // 2:] = 0
    return x2

def spec_fft2(do_window = True):
    if do_window:
        ywf = fft2(y*w)
    else:
        ywf = fft2(y)
    if do_conv_window:
        ywf = conv_window(ywf)
    return w, ywf

def if_mag(patterns, arr, phase = 0, truncate = False):
    trunc = len(arr) - len(patterns[0])
    phase = np.exp(1j * phase)
    tmp = ifft2(arr)
    if truncate:
        return np.real(np.sqrt(np.conjugate(tmp * phase) * tmp))[trunc // 2: -trunc // 2]
    return np.real(np.sqrt(np.conjugate(tmp * phase) * tmp))

def spec_separate(patterns, pad = None, i = 0, ftype = 'step', std = 4, cutoff = .9,
        window_high = False):
    """
    separate signal into low and high-frequency components
    """
    N = len(patterns[0])
    if pad is None:
        pad = N
    
    g = y = np.pad(patterns[i], pad, mode = 'edge')

    window, ywf = spec_fft(patterns, i, pad)

    
    if ftype == 'step':
        L = clip_high_window(ywf, cutoff)
        H = 1 - L
    elif ftype == 'gaussian':
        gwindow = signal.gaussian(len(y), std = std)

        L = np.absolute(fftshift(fft(gwindow))) * window
        L /= np.absolute(L).max()
        H = 1 - L
    else:
        raise Exception
    if window_high:
        H = H * window
    gfft = fftshift(fft(g))

    
    low = np.real(ifft(ifftshift(gfft * L)))[pad: -pad]
    high = np.real(ifft(ifftshift(gfft * H)))[pad: -pad]

    return high, low, (g, gfft, H, L)

def separate_2d(arr, **kwargs):
    high = []
    low = []
    for i in range(len(arr)):
        h, l, _ = spec_separate(arr, i = i, **kwargs)
        high.append(h)
        low.append(l)
    return np.vstack(high), np.vstack(low)

def reference_bgsub(patterns):
    bgsubbed_nosmooth = np.vstack([fourier.filter_bg(
    patterns, i, 0, window_type = 'step', deconvolve = False, toreal = 'psd')
                               for i in range(len(patterns))])
    bgsubbed_final = gf(bgsubbed_nosmooth, (1, 1.7))
    bgsubbed_final *= patterns.max() / bgsubbed_final.max() #np.percentile(patterns, 99.9) / np.percentile(bgsubbed_final, 99.9)
    return bgsubbed_final
from scipy.interpolate import interp1d
from scipy.interpolate import NearestNDInterpolator

def interprows(arr, mask):
    res = []
    for row, rowmask in zip(arr, mask):
        x = np.indices(row.shape)[0][rowmask]
        y = row[rowmask]
        f = interp1d(x, y, bounds_error = False)
        res.append(f(np.indices(row.shape)[0]))
    return np.vstack(res)

def get_background(patterns, threshold = 50, smooth = (1, 1.7)):
    bgsubbed = reference_bgsub(patterns)
    bgsubbed[bgsubbed > np.percentile(bgsubbed, threshold)] = np.nan
    bkgmask = ~np.isnan(bgsubbed)
    filled_bg = interprows(patterns, bkgmask)
    smooth_bg = gf(filled_bg, smooth)

    mask = np.where(~np.isnan(smooth_bg))
    interp = NearestNDInterpolator(np.transpose(mask), smooth_bg[mask])
    filled_data = interp(*np.indices(smooth_bg.shape))
    return filled_data

def separate_signal(patterns):
    filled_data = get_background(patterns)
    # interpolation of slow-varying background
    slow_q = get_background(patterns)
    # crystalline diffraction + high-frequency noise (original data minus interpolated background)
    fast_q = patterns - filled_data
    # high-T frequency noise, low-T frequency signal
    fast_T, slow_T = separate_2d(fast_q.T, ftype = 'gaussian', std = .5)
    return slow_q, fast_q, slow_T, fast_T

def imshow_log(arr):
    """plot log of absolute value heatmap, with an offset
    """
    arr = arr.copy()
    floor = np.abs(arr.min())
    arr[arr < 0] = 0
    plt.imshow(np.log(floor + arr ), cmap = 'jet')
