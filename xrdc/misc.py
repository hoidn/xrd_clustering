"""
Miscelaneous utilities
"""

import numpy as np

def deepmap(lst, func, level):
    if level <= 0:
        return func(lst)
    else:
        return [deepmap(sub, func, level - 1) for sub in lst]

def transpose_x_3d(arr):
    return np.swapaxes(
            np.swapaxes(arr, 0, 2), 
                0, 1)
