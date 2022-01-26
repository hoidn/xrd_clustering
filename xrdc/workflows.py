from xrdc import peak_fitting as pf
def twod():
    """
    source separation and peak fitting for the temperature progression dataset.
    """
    from xrdc.datasets import d2d
    from xrdc import source_separation as sep

    patterns = d2d.patterns
    slow_q, fast_q, slow_T, fast_T = sep.separate_signal(patterns, background_after_filter=False,
                           smooth_q_background = 0, threshold = 25, smooth_q = 1.7)
    background = slow_q
    fsub_stop_2d = pf.curvefit_2d((patterns - 0), stdratio_threshold = 3, noise_estimate = fast_T,
                           background = background, bg_shift_pos = False)
    return slow_q, fast_q, slow_T, fast_T, fsub_stop_2d

def threed(patterns, smooth_q = 1.7, threshold = 50):
    from xrdc import source_separation as sep
    background, fast_q, slow_T, fast_T = sep.separate_signal(patterns, background_after_filter = True,
                                    threshold = threshold, smooth_q = smooth_q,
                                    bg_fill_method = 'simple')

    # slow_q, fast_q, slow_T, fast_T = sep.separate_signal(patterns)

    # #background = sep.get_background(patterns, threshold = 25, smooth_q = 1.7, method = 'simple')
    # background = sep.get_background(slow_T, threshold = 25, smooth_q = 1.7, bg_fill_method = 'simple')

    fsub_stop_2d = pf.curvefit_2d(patterns, background = background,
                             noise_estimate = fast_T, stdratio_threshold = 2)
    return background, fast_q, slow_T, fast_T, fsub_stop_2d
