# coding: utf-8

from __future__ import division, absolute_import, print_function

import numpy.core.numeric as _nx
import numpy as np

from numpy import asanyarray, newaxis

def stack(arrays, axis=0):
    """
    Join a sequence of arrays along a new axis.
    The `axis` parameter specifies the index of the new axis in the dimensions
    of the result. For example, if ``axis=0`` it will be the first dimension
    and if ``axis=-1`` it will be the last dimension.
    .. versionadded:: 1.10.0
    Parameters
    ----------
    arrays : sequence of array_like
        Each array must have the same shape.
    axis : int, optional
        The axis in the result array along which the input arrays are stacked.
    Returns
    -------
    stacked : ndarray
        The stacked array has one more dimension than the input arrays.
    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    split : Split array into a list of multiple sub-arrays of equal size.
    Examples
    --------
    >>> arrays = [np.random.randn(3, 4) for _ in range(10)]
    >>> np.stack(arrays, axis=0).shape
    (10, 3, 4)
    >>> np.stack(arrays, axis=1).shape
    (3, 10, 4)
    >>> np.stack(arrays, axis=2).shape
    (3, 4, 10)
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([2, 3, 4])
    >>> np.stack((a, b))
    array([[1, 2, 3],
           [2, 3, 4]])
    >>> np.stack((a, b), axis=-1)
    array([[1, 2],
           [2, 3],
           [3, 4]])
    """
    arrays = [asanyarray(arr) for arr in arrays]
    if not arrays:
        raise ValueError('need at least one array to stack')

    shapes = set(arr.shape for arr in arrays)
    if len(shapes) != 1:
        raise ValueError('all input arrays must have the same shape')

    result_ndim = arrays[0].ndim + 1
    if not -result_ndim <= axis < result_ndim:
        msg = 'axis {0} out of bounds [-{1}, {1})'.format(axis, result_ndim)
        raise IndexError(msg)
    if axis < 0:
        axis += result_ndim

    sl = (slice(None),) * axis + (_nx.newaxis,)
    expanded_arrays = [arr[sl] for arr in arrays]
    return _nx.concatenate(expanded_arrays, axis=axis)