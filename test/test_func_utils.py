import np_utils
from np_utils import *

def test_double_wrap():
    '''This is not really a test of doublewrap as much as
       a demo of how to use it'''
    @doublewrap
    def decc(f, **dec_kwds):
        def newf(*args, **kwds):
            print 'dec_kwds, kwds:', dec_kwds, kwds
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
    test_double_wrap()
