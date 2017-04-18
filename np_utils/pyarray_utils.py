"""Basic utilities to support conversion between array.array <-> numpy.array
"""
from __future__ import print_function

import array

import numpy as np

_TYPECODE_DICT = {np.dtype(i): i for i in 'cUbBhHiIlLfd'}

def to_pyarray(arr):
    '''Convert numpy.array to a flattened array.array
       Supports only the limited type choices offered by array.array,
       fails otherwise'''
    try:
        return array.array(_TYPECODE_DICT[arr.dtype], arr.ravel())
    except KeyError:
        print('Array type of {} is not supported by to_pyarray'.format(arr.dtype))
        raise

def from_pyarray(arr):
    '''Convert array.array to numpy.array'''
    return np.frombuffer(arr, arr.typecode)
