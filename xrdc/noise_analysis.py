#from xrdc.source_separation import gf
from . import featurization as feat
from . import fourier
from sklearn.utils import resample
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def mk_pairs_mask(arr):
    """
    Select which q points count as background, then return a corresponding square boolean matrix
    """
    rowmask = (arr < np.percentile(arr, 50))
    #rowmask[:1300] = False

    pairs_mask = np.einsum('i,j->ij', rowmask, rowmask)
    return pairs_mask

def resample_2d(arr):
    N = len(arr)
    idx = np.arange(N).astype(int)
    return arr[resample(idx), :]

def boot_stderr(arr2d):
    return np.vstack([resample_2d(arr2d).mean(axis = 0) for _ in range(100)]).std(axis = 0)

from statsmodels.distributions.empirical_distribution import ECDF

#def boot_cdfs(arr2d):
#    """
#    return mean-subtracted CDFs
#    """
#    samples = np.vstack([resample_2d(arr2d).mean(axis = 0) for _ in range(100)])
#    means = samples.mean(axis = 0)
#    centered = np.abs(samples - means)
#    cdfs = np.array([ECDF(centered[:, i]) for i in range(samples.shape[1])])
#    return cdfs
#def boot_pvalue(arr2d, signed_zscore, cdfs):
#    stderr = reduce_stderr(pairwise_stderr)
#    #diffs = pairwise_diffs.mean(axis = 0)
#    print('pval', arr2d.shape, signed_zscore.shape)
#    #deviations = signed_zscore * stderr
#    cdfs = boot_cdfs(np.abs(pairwise_diffs))
#    tails_values = np.array([cdf(dev) for cdf, dev in zip(cdfs, deviations)])
#    return 1 - tails_values
#    #return 2 * np.min([tails_values, 1 - tails_values], axis = 0)

def boot_mean(arr2d, condense = True):
    samples = np.vstack([resample_2d(arr2d).mean(axis = 0) for _ in range(100)])
    if condense:
        return samples.mean(axis = 0)
    else:
        return samples

def pairdiffs(arr2d):
    return arr2d[:, None, :] - arr2d[..., None]

def _boot_diffs_pairwise(arr2d, N = 60):
    """
    Generates a single bootstrap sample
    """
    # Assumes first dimension is the sample dimension
    sample_ixs = np.arange(len(arr2d)).astype(int)
    sampled_ixs = resample(sample_ixs, n_samples=N)
    diffs_3d = pairdiffs(arr2d)
    boot_sample = diffs_3d[sampled_ixs]
    boot_mean = boot_sample.mean(axis = 0)
    return boot_mean

# TODO maintain bootstrap distribution, mask without aggregating
# or, easier, make the independence assumption
def boot_diffs_pairwise(arr2d, agg_dim = 0, mask = None):
    print('pairwise', arr2d.shape)
    samples = np.array([_boot_diffs_pairwise(arr2d) for _ in range(100)])
    if agg_dim == 0: # condense into bootstrap means and standard deviations
        return samples.mean(axis = 0), samples.std(axis = 0)
    elif agg_dim in (1, 2): # retain all the samples, but sum in one of the block index dimensions
        return (samples * mask).sum(axis = agg_dim)
#        cdfs = np.array(
#                [[ECDF(samples[:, i, j])
#                    for i in range(samples.shape[0])]
#                for j in range(samples.shape[1])]
#            )
    else:
        raise ValueError

#def agg_cdf(samples_b_j, mask):



def c2d(arr, kernel):
    arr_conv = np.vstack([np.convolve(p, kernel, 'same') for p in arr])
    idxs = np.arange(len(arr[0]))
    return idxs, arr_conv

def get_aggregate_mean(arr, kernel, blocksize):
    idxs, arr_conv = c2d(arr, kernel)
    return idxs[::blocksize], np.mean(arr_conv[:, ::blocksize], axis = 0)

def get_aggregate_std(arr, kernel, blocksize):
    arr_conv = np.vstack([np.convolve(p, kernel, 'same') for p in arr])
    idxs = np.arange(len(arr))
    return idxs[::blocksize], np.std(arr_conv[:, ::blocksize], axis = 0)

def get_aggregate_stderr(arr, kernel, blocksize):
    idxs, arr_conv = c2d(arr, kernel)
    return idxs[::blocksize], arr_conv[:, ::blocksize],\
        arr_conv[:, ::blocksize].mean(axis = 0), boot_stderr(arr_conv[:, ::blocksize])

def cartesian_quad(a, b):
    b = b.reshape((b.shape[0], 1))
    return np.sqrt(a**2 + b**2)

def cartesian_diff(a, b):
    b = b.reshape((b.shape[0], 1))
    return b - a

def reduce_stderr(arr2d):
    return np.sqrt((arr2d**2 + 1e-9).sum(axis = 0))

def cdfs_from_boot_samples(arr2d, sample_axis = 0):
    assert sample_axis == 0
    return np.array([ECDF(arr2d[:, i] * np.sign(arr2d[:, i].mean()))
        for i in range(arr2d.shape[1])])

def get_agged_zscores(fast_q, peak, blocksize, mask, mode = 'agg_first',
        signed = False):
    print('signed', signed)
    if mode == 'agg_first':
        _, agged, mean, sterr = get_aggregate_stderr(fast_q, peak, blocksize)
        pairwise_diffs = cartesian_diff(mean, mean) * mask
        pairwise_stderr = cartesian_quad(sterr, sterr) * mask
    elif mode == 'agg_last':
        _, arr_conv = c2d(fast_q, peak)
        arr_conv = arr_conv[:, ::blocksize]
        pairwise_diffs, pairwise_stderr = boot_diffs_pairwise(arr_conv)
        pairwise_diffs, pairwise_stderr = pairwise_diffs * mask, pairwise_stderr * mask
    elif mode == 'cdf':
        _, arr_conv = c2d(fast_q, peak)
        arr_conv = arr_conv[:, ::blocksize]
        pairwise_diffs, pairwise_stderr = boot_diffs_pairwise(arr_conv)
        pairwise_diffs, pairwise_stderr = pairwise_diffs * mask, pairwise_stderr * mask
        samples_j = boot_diffs_pairwise(arr_conv, agg_dim = 1, mask = mask)
        cdfs = cdfs_from_boot_samples(samples_j)
        if not signed:
            # TODO standard error aggregation wrong? TODO
            zscores = (np.sum(np.abs(pairwise_diffs), axis = 0) / (pairwise_stderr + 1e-9).sum(axis = 0))
        else:
            zscores = (np.sum((pairwise_diffs), axis = 0) / (pairwise_stderr + 1e-9).sum(axis = 0))
        return zscores, cdfs
    else:
        raise ValueError
    if not signed:
        # TODO standard error aggregation wrong? TODO
        zscores = (np.sum(np.abs(pairwise_diffs), axis = 0) / (pairwise_stderr + 1e-9).sum(axis = 0))
    else:
        zscores = (np.sum((pairwise_diffs), axis = 0) / (pairwise_stderr + 1e-9).sum(axis = 0))
    return zscores, (pairwise_diffs, pairwise_stderr)

def cmp_zscores(fast_q, scores, blocksize):
    a, b = 2, 1
    mpl.rcParams['figure.figsize'] =(10, 10)
    plt.subplot(a, b, 1)
    plt.grid()
    plt.plot(np.log(np.abs(100 * fast_q.mean(axis = 0) / 20))[::blocksize], label = 'N ={}'.format(fast_q.shape[0]))
    plt.grid()
    plt.ylim(0, None)
    plt.grid()

    plt.subplot(a, b, 2)
    plt.plot(np.log(1 + scores) , label = 'N ={}'.format(fast_q.shape[0]))

    plt.ylim(0., None)
    plt.ylabel('log(1 + Z score)')
    plt.grid()

    plt.hlines(np.log(4), 0, len(scores), color = 'k')


def plot_zscores_subsample_rows(fast_q, row_blocksize = 1, blocksize = 10, peak = None, transpose = False,
                               **kwargs):
    _, agged, mean, sterr = get_aggregate_stderr(fast_q[::row_blocksize], peak, blocksize)
    pairwise_diffs = cartesian_diff(mean, mean)
    pairwise_stderr = cartesian_quad(sterr, sterr)
    zscores = (np.abs(pairwise_diffs) / (pairwise_stderr + 1e-9))

    arr = np.median(zscores, axis = 0)
    rowmask = (arr < np.percentile(arr, 50))
    select_zscores = np.einsum('i,j->ij', rowmask, ~rowmask)
    bgmask_1d = (select_zscores == False).all(axis = 0)
    off_diag_mask = (np.roll(np.eye(len(fast_q[0][::blocksize])), -1, axis = 0) +\
                np.roll(np.eye(len(fast_q[0][::blocksize])), 1, axis = 0)).astype(bool)

    if not transpose:
        cmp_zscores(fast_q[::row_blocksize], get_agged_zscores(
            fast_q[::row_blocksize], peak, blocksize, off_diag_mask, **kwargs)[0],
            blocksize)
    else:
        off_diag_mask = (np.roll(np.eye(len(fast_q[0][::blocksize].T)), -1, axis = 0) +\
            np.roll(np.eye(len(fast_q[0][::blocksize].T)), 1, axis = 0)).astype(bool)
        cmp_zscores(fast_q[::row_blocksize].T, get_agged_zscores(
            fast_q[::row_blocksize].T, peak.T, 1, off_diag_mask, **kwargs)[0],
            blocksize)


def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2),
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant')

from scipy.interpolate import interp1d

shape = None

def zipmerge(arrs, shape):
    flat = np.swapaxes(np.array(arrs), 0, 1).ravel()
    res = np.zeros((shape))
    res[:] = flat[:len(res)]
    return res

def padded_zscores_shifted(fast_q, bs, peak, shift = 0, **kwargs):
    fast_q = np.roll(fast_q, shift, axis = 1)
    arr, pairwise = plot_z_blocksize(fast_q, bs, peak, plot = False, **kwargs)
    return arr

def padded_zscores(fast_q, stride, peak, shifts = False, **kwargs):
    # Effective block size is entirely decided by the 'peak' array
    shape = fast_q.shape[1]
    if not shifts:
        # TODO fix this case. Need to call get_agged_zscores
        arr, pairwise_stats = plot_z_blocksize(fast_q, stride, peak, plot = False, **kwargs)
        #pad = stride - (len(arr) % stride)# - (shape % stride)#shape - len(arr)
        qpoints = np.arange(len(arr) * stride).reshape((-1, stride)).T.mean(axis = 0)
        fn = interp1d(qpoints, arr, fill_value = 'extrapolate')
        return fn(np.arange(shape)), pairwise_stats
    else:
        res = []
        for shift in range(stride):
            res.append(padded_zscores_shifted(fast_q, stride, peak, shift = shift, **kwargs))
        return zipmerge(res, shape), pairwise_stats

def plot_z_blocksize(fast_q, blocksize, peak, plot = True, **kwargs):
    padded = np.pad(fast_q[0], (0, blocksize - (len(fast_q[0]) % blocksize)), mode = 'edge')
    qpoints = np.arange(len(padded)).reshape((-1, blocksize)).T.mean(axis = 0)
    off_diag_mask = (np.roll(np.eye(len(fast_q[0][::blocksize])), -1, axis = 0) +\
                np.roll(np.eye(len(fast_q[0][::blocksize])), 1, axis = 0)).astype(bool)

    zloc, pairwise_stats = get_agged_zscores(fast_q, peak, blocksize, off_diag_mask, **kwargs)
    #plt.plot(qpoints[:len(zloc)], zloc, label = blocksize)
    if plot:
        plt.plot(qpoints[:len(zloc)], np.log(1 + zloc), label = blocksize)
    return zloc, pairwise_stats

def mk_peak_blocksize(arr2d, bs, ptype = 'gaussian'):
    """
    peak with width adapted to block size
    """
    if ptype == 'gaussian':
        peak = fourier.lowpass_g(1000 / bs, arr2d[0])
    elif ptype == 'square':
        peak = mk_square(arr2d, bs)
    else:
        raise ValueError
    return peak

def plot_twopanel(arr2d, z_vs_blocksize, aspect = 18.3,
        lineout_indices = [1, 3, 7], lineout_labels = [1, 3, 7], offset_scale = 2.,
        ymax = 19, do_top = True, do_bottom = True):
    shape = arr2d.shape[1]
    # matplotlib.rcParams['figure.figsize'] =(18, 9)

    best = z_vs_blocksize[9]

    size2 = 18
    size1 = 15
    plt.rcParams.update({'font.size': size1})
    plt.rcParams.update({'axes.labelsize': size2})
    plt.rcParams.update({'axes.titlesize': size2})
    plt.rcParams.update({'xtick.labelsize': size2})
    plt.rcParams.update({'ytick.labelsize': size2})

    def plt_lineout(lineout, offset = 0, **kwargs):
        #- lineout.min()
        plt.plot((np.log(1 + lineout - lineout.min()) + offset), **kwargs)

    if do_top and do_bottom:
        a, b = 2, 1
        ax1 = plt.subplot(a, b, 1)
        plt_lineout(np.abs(arr2d).mean(axis = 0), 3 * offset_scale, label = 'raw (averaged)')
        for i, (idx, label) in enumerate(zip(lineout_indices, lineout_labels)):
            plt_lineout(z_vs_blocksize[idx], i * offset_scale, label = 'log (1 + Z score), block width {}'.format(label))
        plt.grid()
        plt.ylim(0, ymax)
        plt.xlim(0, shape - 1)
        plt.legend(loc = (.5, .6))
        plt.title('(a)')
        xticks = ax1.xaxis.get_major_ticks()
        xticks[0].label1.set_visible(False)

        ax2 = plt.subplot(a, b, 2)
        plt.imshow(np.log(1 + z_vs_blocksize), aspect = aspect, cmap = 'jet', interpolation='none')
        plt.xlabel('$q$ index')
        plt.ylabel('block width')
        # plt.title('log (1 + Z score)')

        plt.subplots_adjust(hspace=0)
        yticks = ax2.yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)
        yticks[1].label1.set_visible(False)
    #    plt.setp(ax2.get_yticklabels()[-1], visible=False)
    #    plt.setp(ax2.get_yticklabels()[0], visible=False)

        # plt.colorbar()
        plt.title('(b)')
    elif do_top:
        a, b = 1, 1
        ax1 = plt.subplot(a, b, 1)
        plt_lineout(np.abs(arr2d).mean(axis = 0), 3 * offset_scale, label = 'raw (averaged)')
        for i, (idx, label) in enumerate(zip(lineout_indices, lineout_labels)):
            plt_lineout(z_vs_blocksize[idx], i * offset_scale, label = 'log (1 + Z score), block width {}'.format(label))
        plt.grid()
        plt.ylim(0, ymax)
        plt.xlim(0, shape - 1)
        plt.legend(loc = (.5, .6))
        xticks = ax1.xaxis.get_major_ticks()
        xticks[0].label1.set_visible(False)

    plt.savefig('zscores_60.pdf')

    def plt_lineout(lineout, offset = 0, **kwargs):
        #- lineout.min()
        plt.plot((np.log(1 + lineout - lineout.min()) + offset), **kwargs)

    if do_top and do_bottom:
        a, b = 2, 1
        ax1 = plt.subplot(a, b, 1)
        plt_lineout(np.abs(arr2d).mean(axis = 0), 3 * offset_scale, label = 'raw (averaged)')
        for i, (idx, label) in enumerate(zip(lineout_indices, lineout_labels)):
            plt_lineout(z_vs_blocksize[idx], i * offset_scale, label = 'log (1 + Z score), block width {}'.format(label))
        plt.grid()
        plt.ylim(0, ymax)
        plt.xlim(0, shape - 1)
        plt.legend(loc = (.5, .6))
        plt.title('(a)')
        xticks = ax1.xaxis.get_major_ticks()
        xticks[0].label1.set_visible(False)

        ax2 = plt.subplot(a, b, 2)
        plt.imshow(np.log(1 + z_vs_blocksize), aspect = aspect, cmap = 'jet', interpolation='none')
        plt.xlabel('$q$ index')
        plt.ylabel('block width')
        # plt.title('log (1 + Z score)')

        plt.subplots_adjust(hspace=0)
        yticks = ax2.yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)
        yticks[1].label1.set_visible(False)
    #    plt.setp(ax2.get_yticklabels()[-1], visible=False)
    #    plt.setp(ax2.get_yticklabels()[0], visible=False)

        # plt.colorbar()
        plt.title('(b)')
    elif do_top:
        a, b = 1, 1
        ax1 = plt.subplot(a, b, 1)
        plt_lineout(np.abs(arr2d).mean(axis = 0), 3 * offset_scale, label = 'raw (averaged)')
        for i, (idx, label) in enumerate(zip(lineout_indices, lineout_labels)):
            plt_lineout(z_vs_blocksize[idx], i * offset_scale, label = 'log (1 + Z score), block width {}'.format(label))
        plt.grid()
        plt.ylim(0, ymax)
        plt.xlim(0, shape - 1)
        plt.legend(loc = (.5, .6))
        xticks = ax1.xaxis.get_major_ticks()
        xticks[0].label1.set_visible(False)

    plt.savefig('zscores_60.pdf')

def plot_emv_zscores(arr2d, z_vs_blocksize, q, aspect = 18.3,
        block_sizes = [2, 4, 8, 16], lineout_labels = [2, 4, 8, 16], offset_scale = 2.,
        ymax = 19, do_top = True, do_bottom = True, lineout_indices = [1, 3, 7]):

    shape = arr2d.shape[1]
    # matplotlib.rcParams['figure.figsize'] =(18, 9)

    best = z_vs_blocksize[9]

    size2 = 18
    size1 = 15
    plt.rcParams.update({'font.size': size1})
    plt.rcParams.update({'axes.labelsize': size2})
    plt.rcParams.update({'axes.titlesize': size2})
    plt.rcParams.update({'xtick.labelsize': size2})
    plt.rcParams.update({'ytick.labelsize': size2})


    def plt_lineout(lineout, offset = 0, **kwargs):
        #- lineout.min()
        plt.plot((np.log(1 + lineout - lineout.min()) + offset), **kwargs)


#    ax2 = plt.subplot(a, b, 2)
#    #plt_lineout(np.abs(arr2d).mean(axis = 0), 3 * offset_scale, label = 'raw (averaged)')
#    for i, (idx, label) in enumerate(zip(lineout_indices, lineout_labels)):
#        plt_lineout(z_vs_blocksize[idx], i * offset_scale, label = 'log (1 + Z score), block width {}'.format(label))
#    plt.grid()
#    plt.ylim(0, ymax)
#    plt.xlim(0, shape - 1)
#    plt.legend(loc = (.5, .6))
#    plt.title('(a)')
#    xticks = ax1.xaxis.get_major_ticks()
#    xticks[0].label1.set_visible(False)
#    plt.title('(b)')
#
    plt.savefig('zscores_60.pdf')

    def plt_lineout(lineout, offset = 0, **kwargs):
        #- lineout.min()
        plt.plot((np.log(1 + lineout - lineout.min()) + offset), **kwargs)

    a, b = 2, 1
#    ax1 = plt.subplot(a, b, 1)
#    scale1 = 1
#    for i, bs in enumerate(block_sizes):
#        print(i)
#        kernel = mk_square(arr2d, bs)
#        idxs, arr_conv = c2d(arr2d, kernel)
#        stderr = boot_stderr(arr_conv[:, ::bs])
#        plt.plot(q[::bs], stderr + i * scale1, label = 'block size {}'.format(bs))
#    plt.grid()
#    plt.ylim(0, ymax)
#    #plt.xlim(0, shape - 1)
#    plt.legend()
#    plt.autoscale(enable=True, axis='x', tight=True)
#    plt.title('(a)')
#    xticks = ax1.xaxis.get_major_ticks()
#    xticks[0].label1.set_visible(False)

    a, b = 2, 1
    ax1 = plt.subplot(a, b, 1)
    scale1 = 1
    for i, bs in enumerate(block_sizes):
        print(i)
        kernel = mk_square(arr2d, bs)
        idxs, arr_conv = c2d(arr2d, kernel)
        stderr = boot_stderr(arr_conv[:, ::bs])
        plt.plot(q[::bs], stderr + i * scale1, label = 'block size {}'.format(bs))
    plt.grid()
    plt.ylim(0, ymax)
    #plt.xlim(0, shape - 1)
    plt.legend()
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.title('(b)')
    xticks = ax1.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)

    ax2 = plt.subplot(a, b, 2)
    #plt_lineout(np.abs(arr2d).mean(axis = 0), 3 * offset_scale, label = 'raw (averaged)')
    for i, (idx, label) in enumerate(zip(lineout_indices, lineout_labels)):
        plt_lineout(z_vs_blocksize[idx], i * offset_scale, label = 'log (1 + Z score), block size {}'.format(label))
    plt.grid()
    plt.ylim(0, ymax)
    plt.xlim(0, shape - 1)
    plt.legend(loc = 'upper right')
    plt.title('(a)')
    xticks = ax2.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)

    plt.savefig('zscores_60.pdf')

def twopanel_generic(sterr_vs_blocksize, z_vs_blocksize, q, aspect = 18.3,
        block_sizes = [2, 4, 8, 16], lineout_labels = [2, 4, 8, 16], offset_scale = 2.,
        ymax = 19, do_top = True, do_bottom = True, lineout_indices = [1, 3, 7]):

    shape = sterr_vs_blocksize.shape[1]
    # matplotlib.rcParams['figure.figsize'] =(18, 9)

    best = z_vs_blocksize[9]

    size2 = 18
    size1 = 15
    plt.rcParams.update({'font.size': size1})
    plt.rcParams.update({'axes.labelsize': size2})
    plt.rcParams.update({'axes.titlesize': size2})
    plt.rcParams.update({'xtick.labelsize': size2})
    plt.rcParams.update({'ytick.labelsize': size2})


    plt.savefig('zscores_60.pdf')

    def plt_lineout(lineout, offset = 0, **kwargs):
        #- lineout.min()
        plt.plot((np.log(1 + lineout - lineout.min()) + offset), **kwargs)

    a, b = 2, 1
#    ax1 = plt.subplot(a, b, 1)
#    scale1 = 1
#    for i, bs in enumerate(block_sizes):
#        print(i)
#        kernel = mk_square(arr2d, bs)
#        idxs, arr_conv = c2d(arr2d, kernel)
#        stderr = boot_stderr(arr_conv[:, ::bs])
#        plt.plot(q[::bs], stderr + i * scale1, label = 'block size {}'.format(bs))
#    plt.grid()
#    plt.ylim(0, ymax)
#    #plt.xlim(0, shape - 1)
#    plt.legend()
#    plt.autoscale(enable=True, axis='x', tight=True)
#    plt.title('(a)')
#    xticks = ax1.xaxis.get_major_ticks()
#    xticks[0].label1.set_visible(False)

    a, b = 2, 1
    ax1 = plt.subplot(a, b, 1)
    scale1 = 1
    for i, bs in enumerate(block_sizes):
        plt.plot(sterr_vs_blocksize + i * scale1, label = 'block size {}'.format(bs))
    plt.grid()
    plt.ylim(0, ymax)
    #plt.xlim(0, shape - 1)
    plt.legend()
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.title('(b)')
    xticks = ax1.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)

    ax2 = plt.subplot(a, b, 2)
    #plt_lineout(np.abs(arr2d).mean(axis = 0), 3 * offset_scale, label = 'raw (averaged)')
    for i, (idx, label) in enumerate(zip(lineout_indices, lineout_labels)):
        plt_lineout(z_vs_blocksize[idx], i * offset_scale, label = 'log (1 + Z score), block size {}'.format(label))
    plt.grid()
    plt.ylim(0, ymax)
    plt.xlim(0, shape - 1)
    plt.legend(loc = 'upper right')
    plt.title('(a)')
    xticks = ax2.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)

    plt.savefig('zscores_60.pdf')

def fill(arr, bs, shape):# TODO refactor
    qpoints = np.arange(len(arr) * bs).reshape((-1, bs)).T.mean(axis = 0)
    fn = interp1d(qpoints, arr, fill_value = 'extrapolate')
    return fn(np.arange(shape))
#     return np.array([arr for _ in range(bs)]).swapaxes(0, 1).ravel()

def pp(patterns, bs, kernel):
    idxs, arr_conv = c2d(patterns, kernel)
    return arr_conv[:, ::bs - 1]

def mk_delta(patterns):
    peak = fourier.lowpass_g(100, patterns[0], sym=False)
    peak = (peak == 1).astype(float)
    return peak

def mk_square(patterns, bs):
    delta = mk_delta(patterns)
    return feat.shuffle(np.array([delta]), 0, (bs - 1) / 2)[0]
