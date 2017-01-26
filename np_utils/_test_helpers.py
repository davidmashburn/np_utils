'''Helper functions for use in tests of the other sub-modules'''
from __future__ import print_function
from builtins import zip

import numpy as np

cache = {}

# Unused, but pretty useful function (and there was nowhere else to put it)
def assert_index_groups_same(x, y):
    assert np.array_equal(x[0], y[0])
    assert len(x[1])==len(y[1])
    for i, j in zip(x[1], y[1]):
        try:
            assert np.array_equal(i, j)
        except:
            print(i, j)
            raise

def _get_sample_rec_array():
    '''Build a dummy record array to test with. Has fields m,n,o,p.
       Cache the result as "cache['sample_recarray']" to save time.'''
    if 'sample_recarray' in cache:
        arr = cache['sample_recarray']
    else:
        l = 10000
        ID = np.arange(l)
        np.random.seed(0)
        m = np.array([['This', 'That'][j] for j in np.random.randint(2, size=l)])
        n = np.random.randint(100, size=l)
        o = np.random.normal(loc=300, scale=100, size=l)
        p = np.random.logistic(0, 20, size=l)
        arr = np.rec.fromarrays([ID, m, n, o, p], names=list('imnop'))
        cache['sample_recarray'] = arr
    return arr
