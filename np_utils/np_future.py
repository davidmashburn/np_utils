# coding: utf-8

from __future__ import division, absolute_import, print_function
from builtins import range

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

#######################
## stride_tricks.py: ##
#######################

def _maybe_view_as_subclass(original_array, new_array):
    if type(original_array) is not type(new_array):
        # if input was an ndarray subclass and subclasses were OK,
        # then view the result as that subclass.
        new_array = new_array.view(type=type(original_array))
        # Since we have done something akin to a view from original_array, we
        # should let the subclass finalize (if it has it implemented, i.e., is
        # not None).
        if new_array.__array_finalize__:
            new_array.__array_finalize__(original_array)
    return new_array

def _broadcast_to(array, shape, subok, readonly):
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    array = np.array(array, copy=False, subok=subok)
    if not shape and array.shape:
        raise ValueError('cannot broadcast a non-scalar to a scalar array')
    if any(size < 0 for size in shape):
        raise ValueError('all elements of broadcast shape must be non-'
                         'negative')
    needs_writeable = not readonly and array.flags.writeable
    extras = ['reduce_ok'] if needs_writeable else []
    op_flag = 'readwrite' if needs_writeable else 'readonly'
    broadcast = np.nditer(
        (array,), flags=['multi_index', 'refs_ok', 'zerosize_ok'] + extras,
        op_flags=[op_flag], itershape=shape, order='C').itviews[0]
    result = _maybe_view_as_subclass(array, broadcast)
    if needs_writeable and not result.flags.writeable:
        result.flags.writeable = True
    return result


def broadcast_to(array, shape, subok=False):
    """Broadcast an array to a new shape.
    Parameters
    ----------
    array : array_like
        The array to broadcast.
    shape : tuple
        The shape of the desired array.
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise
        the returned array will be forced to be a base-class array (default).
    Returns
    -------
    broadcast : array
        A readonly view on the original array with the given shape. It is
        typically not contiguous. Furthermore, more than one element of a
        broadcasted array may refer to a single memory location.
    Raises
    ------
    ValueError
        If the array is not compatible with the new shape according to NumPy's
        broadcasting rules.
    Notes
    -----
    .. versionadded:: 1.10.0
    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> np.broadcast_to(x, (3, 3))
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])
    """
    return _broadcast_to(array, shape, subok=subok, readonly=True)


def _broadcast_shape(*args):
    """Returns the shape of the ararys that would result from broadcasting the
    supplied arrays against each other.
    """
    if not args:
        raise ValueError('must provide at least one argument')
    # use the old-iterator because np.nditer does not handle size 0 arrays
    # consistently
    b = np.broadcast(*args[:32])
    # unfortunately, it cannot handle 32 or more arguments directly
    for pos in range(32, len(args), 31):
        # ironically, np.broadcast does not properly handle np.broadcast
        # objects (it treats them as scalars)
        # use broadcasting to avoid allocating the full array
        b = broadcast_to(0, b.shape)
        b = np.broadcast(b, *args[pos:(pos + 31)])
    return b.shape


def broadcast_arrays(*args, **kwargs):
    """
    Broadcast any number of arrays against each other.
    Parameters
    ----------
    `*args` : array_likes
        The arrays to broadcast.
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise
        the returned arrays will be forced to be a base-class array (default).
    Returns
    -------
    broadcasted : list of arrays
        These arrays are views on the original arrays.  They are typically
        not contiguous.  Furthermore, more than one element of a
        broadcasted array may refer to a single memory location.  If you
        need to write to the arrays, make copies first.
    Examples
    --------
    >>> x = np.array([[1,2,3]])
    >>> y = np.array([[1],[2],[3]])
    >>> np.broadcast_arrays(x, y)
    [array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]]), array([[1, 1, 1],
           [2, 2, 2],
           [3, 3, 3]])]
    Here is a useful idiom for getting contiguous copies instead of
    non-contiguous views.
    >>> [np.array(a) for a in np.broadcast_arrays(x, y)]
    [array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]]), array([[1, 1, 1],
           [2, 2, 2],
           [3, 3, 3]])]
    """
    # nditer is not used here to avoid the limit of 32 arrays.
    # Otherwise, something like the following one-liner would suffice:
    # return np.nditer(args, flags=['multi_index', 'zerosize_ok'],
    #                  order='C').itviews

    subok = kwargs.pop('subok', False)
    if kwargs:
        raise TypeError('broadcast_arrays() got an unexpected keyword '
                        'argument {}'.format(kwargs.pop()))
    args = [np.array(_m, copy=False, subok=subok) for _m in args]

    shape = _broadcast_shape(*args)

    if all(array.shape == shape for array in args):
        # Common case where nothing needs to be broadcasted.
        return args

    # TODO: consider making the results of broadcast_arrays readonly to match
    # broadcast_to. This will require a deprecation cycle.
    return [_broadcast_to(array, shape, subok=subok, readonly=False)
            for array in args]
