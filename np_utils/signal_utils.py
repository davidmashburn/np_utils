from __future__ import absolute_import
from __future__ import division
from builtins import range
import numpy as np

from .np_utils import partitionNumpy, map_along_axis

# This faster, but more memory-heavy than it needs to be:
# (aka, np.fromiter could be lower resource, but eh...)
def _quick_fft(a, chunk_size):
    '''Partion an array into chunks of a certain size and run fft's on
       each chunk'''
    return np.fft.fft(partitionNumpy(a, chunk_size))

# This is slower, but more flexible and COULD be more memery conscious
# by using np.fromiter
# Timing example (8331264 samples):
#     import time
#     t = time.time()
#     ff = project_herschel.audio_video.rolling_fft(a, framerate//10) # 100 ms
#     print time.time()-t
#   0.583346843719
#     t = time.time()
#     ff = project_herschel.audio_video.rolling_fft(a, framerate//10, 1) # 100 ms
#     print time.time()-t
#   2.62382698059
def _smooth_fft(a, chunk_size, smoothing_factor=4):
    '''Run fft's over multiple sections of a signal.
       chunk_size is the number of samples to include in each fft
       smoothing factor increases the number of samples multiplicatively
       (must me an integer >=1)'''
    n, c = len(a), chunk_size
    cs = c // smoothing_factor
    return np.array([np.fft.fft(a[i:i + c])
                     for i in range(0, n, cs)
                     if i + c <= n])

def rolling_fft(a, chunk_size=10000, smoothing_factor=None):
    '''Return the amplitude of the chunked fft over an array
       using either _quick_fft or _smooth_fft
       Essentially returns a spectrogram'''
    _fft = _quick_fft if smoothing_factor is None else _smooth_fft
    return np.absolute(_fft(a, chunk_size))

def entropy(arr, axis=None, handle_non_integers=True):
    """Computes the Shannon entropy of the elements of A. Assumes A is
    an array-like of nonnegative ints whose max value is approximately
    the number of unique values present.

    >>> a = [0, 1]
    >>> entropy(a)
    1.0
    >>> A = np.c_[a, a]
    >>> entropy(A)
    1.0
    >>> A                   # doctest: +NORMALIZE_WHITESPACE
    array([[0, 0], [1, 1]])
    >>> entropy(A, axis=0)  # doctest: +NORMALIZE_WHITESPACE
    array([ 0., 0.])
    >>> entropy(A, axis=1)  # doctest: +NORMALIZE_WHITESPACE
    array([1., 1.])
    >>> entropy([0, 0, 0])
    0.0
    >>> entropy([])
    0.0
    >>> entropy([5])
    0.0

    Use the option "handle_non_integers" in the case that array values
    are non-integer (such as strings) or when integers are very large
    and vary widely.

    Modified version of this StackOverflow post, answer by "Dave":
    http://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python"""
    if arr is None or len(arr) < 2:
        return 0.

    arr = np.asanyarray(arr)

    if axis is None:
        arr = arr.flatten()
        if handle_non_integers:
            # Replace values in arr with compacted integers:
            _, arr = np.unique(arr, return_inverse=True)

        _counts = np.bincount(arr) # needs small, non-negative ints
        counts = _counts[_counts > 0]

        if len(counts) == 1:
            return 0. # avoid returning -0.0 to prevent weird doctests
        probs = counts / float(arr.size)
        return -np.sum(probs * np.log2(probs))
    else:
        entropies = map_along_axis(entropy, arr, axis)
        return entropies
