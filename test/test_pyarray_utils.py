import numpy as np

from np_utils import *

def test_from_to_pyarray_1():
    a = np.arange(100)
    b = from_pyarray(to_pyarray(a))
    assert np.array_equal(a, b)
    assert a.dtype == b.dtype

if __name__ == '__main__':
    test_from_to_pyarray_1()
