from __future__ import print_function
from builtins import map
from future.utils import lmap

import np_utils
from np_utils import *

from operator import add
increment = lambda x: x + 1

def test_mapf():
    x = [1,2,3]
    y = [4,5,6]
    assert list(mapf(increment)(x)) == lmap(increment, x)
    assert list(mapf(add)(x, y)) == lmap(add, x, y)

def test_lmapf():
    x = [1,2,3]
    y = [4,5,6]
    assert lmapf(increment)(x) == lmap(increment, x)
    assert lmapf(add)(x, y) == lmap(add, x, y)

def test_mapd():
    d = {1:2, 2:3, 3:4}
    d1 = {1:3, 2:4, 3:5}
    assert mapd(increment, d) == d1
    assert mapd(add, d, 1) == d1

def test_map_in_place():
    x = [1,2,3]
    x1 = [2,3,4]
    map_in_place(increment, x)
    assert x == x1

def test_mapd_in_place():
    d = {1:2, 2:3, 3:4}
    d1 = {1:3, 2:4, 3:5}
    mapd_in_place(increment, d)
    assert d == d1

def test_double_wrap():
    '''This is not really a test of doublewrap as much as
       a demo of how to use it'''
    @doublewrap
    def decc(f, **dec_kwds):
        def newf(*args, **kwds):
            print('dec_kwds, kwds:', dec_kwds, kwds)
            that = kwds.pop('extra', dec_kwds.get('extra', 0))
            return f(*args, **kwds) + that
        
        return newf
    
    @decc
    def g(x):
        return x+1
    
    @decc(extra=1)
    def h(x):
        return x+1
    
    assert g(10) == 11
    assert g(10, extra=1) == 12
    assert g(10) == 11
    assert h(10) == 12
    assert h(10, extra=1) == 12
    assert h(10) == 12

if __name__ == '__main__':
    test_mapf()
    test_mapd()
    test_map_in_place()
    test_mapd_in_place()
    test_double_wrap()
