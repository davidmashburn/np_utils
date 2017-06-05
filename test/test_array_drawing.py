#!/usr/bin/env python
'''Tests for some functions in np_utils.py. Use nose to run them.
   Fair warning, this is NOT an exhaustive suite.'''

import numpy as np
from copy import copy
from collections import Counter
import np_utils
from np_utils import *

def test_nd_gradient_1():
    shape = [2,3,4]
    origin_val = 1
    stopvals = [5,7,11]
    a = nd_gradient(shape, origin_val, stopvals)
    print a.shape
    assert a.shape == tuple(shape)
    assert a[0, 0, 0] == origin_val
    assert [a[-1, 0, 0], a[0, -1, 0], a[0, 0, -1]] == stopvals

def test_nd_radial_gradient_1():
    assert np.array_equal(nd_radial_gradient((4, 5))[:, 2], [1.5, 0.5, 0.5, 1.5])
    assert np.array_equal(nd_radial_gradient((3, 3)),
                          np.sqrt([[2, 1, 2], [1, 0, 1], [2, 1, 2]]))
    a = nd_radial_gradient((9, 9))
    assert np.array_equal(a[4, :], a[:, 4])
    assert np.array_equal(a[4, :], [4, 3, 2, 1, 0, 1, 2, 3, 4])

def test_GetDirectionsOfSteepestSlope_A():
    assert GetDirectionsOfSteepestSlope([1,0,0],[0,2,0],[0,0,3])==[2,2,1]

def test_GetDirectionsOfSteepestSlope_B():
    assert GetDirectionsOfSteepestSlope([1,0,0],[0,2,0],[0,0,2])==[1,2,1]

def test_GetDirectionsOfSteepestSlope_C():
    assert GetDirectionsOfSteepestSlope([1,0,0,0],[0,2,0,0],[0,0,3,1])==[2,2,1,1]

def test_GetDirectionsOfSteepestSlope_D():
    assert GetDirectionsOfSteepestSlope([3,0,0,-4],[1,0,2,0],[1,2,2,0])==[1,3,1,1]

def test_GetDirectionsOfSteepestSlope_E():
    assert GetDirectionsOfSteepestSlope([0,0,0],[0,1,0],[1,2,0])==[1,0,None]

#Not tested yet
#def GetDirectionsOfSteepestSlope_BorderCheckingVersion(borderPts):

#Not tested yet
#def BresenhamTriangle(p0,p1,p2): # Generalization for triangle

#Not tested yet
#def ImageCircle(r):

#Not tested yet
#def ImageSphere(r):

#Not tested yet
#def blitCircleToArray(arr,x,y,r,val):

#Not tested yet
#def blitSphereToArray(arr,x,y,z,r,val):

if __name__ == '__main__':
    test_nd_gradient_1()
    test_nd_radial_gradient_1()
    test_GetDirectionsOfSteepestSlope_A()
    test_GetDirectionsOfSteepestSlope_B()
    test_GetDirectionsOfSteepestSlope_C()
    test_GetDirectionsOfSteepestSlope_D()
    test_GetDirectionsOfSteepestSlope_E()
