"""
Miscelaneous utilities
"""

def deepmap(lst, func, level):
    if level <= 0:
        return func(lst)
    else:
        return [deepmap(sub, func, level - 1) for sub in lst]
