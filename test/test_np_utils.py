#!/usr/bin/env python
'''Tests for some functions in np_utils.py. Use nose to run them.
   Fair warning, this is NOT an exhaustive suite.'''

import numpy as np
from copy import copy
from collections import Counter
import np_utils
from np_utils import *

def test_addBorder_0():
    v = [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 2, 0], [0, 2, 0], [0, 2, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 2, 0], [0, 2, 0], [0, 2, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    assert np.all( addBorder(2*np.ones([2,3,1])) == v )

def test_addBorder_0_th2():
    v = np.zeros([6,5,5])
    v[2:-2,2,2]=2
    assert np.all( addBorder(2*np.ones([2,1,1]),borderThickness=2)==v )

def test_addBorder_5():
    v = [[[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]],
         [[5, 5, 5], [5, 2, 5], [5, 2, 5], [5, 2, 5], [5, 5, 5]],
         [[5, 5, 5], [5, 2, 5], [5, 2, 5], [5, 2, 5], [5, 5, 5]],
         [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]]]
    assert np.all( addBorder(2*np.ones([2,3,1]),borderValue=5) == v )

def test_addBorder_axis1():
    v = [[[0], [2], [2], [2], [0]],
         [[0], [2], [2], [2], [0]]]
    assert np.all( addBorder(2*np.ones([2,3,1]),axis=1) == v )

def test_addBorder_axism1():
    v = [[[0, 2, 0], [0, 2, 0], [0, 2, 0]],
         [[0, 2, 0], [0, 2, 0], [0, 2, 0]]]
    assert np.all( addBorder(2*np.ones([2,3,1]),axis=-1) == v )

def test_limitInteriorPoints_r5_0_True():
    assert limitInteriorPoints(range(5),0,uniqueOnly=True) == [0,4]

def test_limitInteriorPoints_r5_2_True():
    assert limitInteriorPoints(range(5),2,uniqueOnly=True) == [0,1,3,4]

def test_limitInteriorPoints_r5_7_True():
    assert limitInteriorPoints(range(5),7,uniqueOnly=True) == range(5)
    
def test_limitInteriorPoints_r5_7_False():
    assert limitInteriorPoints(range(5),7,uniqueOnly=False) == [0,0,1,2,2,2,3,4,4]

def test_limitInteriorPointsInterpolating_04_3():
    assert limitInteriorPointsInterpolating([0,4],3) == [0,1.,2,3.,4]

def test_partitionNumpy_1234_2():
    assert (partitionNumpy([1,2,3,4],2)==[[1,2],[3,4]]).all()

def test_partitionNumpy_123_2():
    assert (partitionNumpy([1,2,3],2)==[[1,2]]).all()

def test_shape_multiply_123_12():
    assert np.all(shape_multiply([[1,2,3]],[2,3])==[[1,1,1,2,2,2,3,3,3]]*2)

def test_shape_multiply_zero_fill_123_12():
    assert np.all(shape_multiply_zero_fill([[1,2,3]],[3,3]) == [[0]*9,[0,1,0,0,2,0,0,3,0],[0]*9])

def test_interpNaNs_0nnn4_r5():
    assert np.all(interpNaNs(np.array([0,np.nan,np.nan,np.nan,4]))==range(5))

# A couple working but not-that-great interpolation functions:
def interpNumpy_1345_half():
    assert interpNumpy([1,3,4,5],0.5)==2

def interpNumpy_11354050_half():
    assert np.all(interpNumpy([[1,1],[3,5],[4,0],[5,0]],0.5)==[2,3])

def interpolatePlane_113540_half_1():
    assert np.all(interpolatePlane(np.array([[1,1],[3,5],[4,0]]),0.5,axis=1)==[1,4,2])

def interpolateSumBelowAbove_113540_1p5_1():
    b,a = interpolateSumBelowAbove(np.array([[1,1],[3,5],[4,0]]),1.5,axis=1)
    assert np.all(b==[ 1.5, 5.5, 4. ])
    assert np.all(a==[ 0.5, 2.5, 0. ])

def limitInteriorPointsInterpolating_1234_1():
    assert limitInteriorPointsInterpolating([1,2,3,4],1)==[1, 2.5, 4]

def limitInteriorPointsInterpolating_1234_5():
    assert limitInteriorPointsInterpolating([1,2,3,4],5)==[1, 1.5, 2, 2.5, 3, 3.5, 4]

# Not tested yet
#def BresenhamFunctionOld(p0,p1):

# Not tested yet
#def BresenhamFunction(p0,p1): # Generalization to n-dimensions

def test_linearTransform_11_012m2_False():
    assert np.all(linearTransform([[1,1]],[[0,1],[2,-2]],False)==[[5,0]])

def test_linearTransform_11_012m2_True():
    assert np.all(linearTransform([[1,1]],[[0,1],[2,-2]],True)==[[-1,-4]])

def test_reverseLinearTransform_m1m41_012m2_False():
    assert np.all(reverseLinearTransform([[5,0]],[[0,1],[2,-2]],False)==[[1,1]])

def test_reverseLinearTransform_11_012m2_True():
    assert np.all(reverseLinearTransform([[-1,-4]],[[0,1],[2,-2]],True)==[[1,1]])

def test_FindOptimalScaleAndTranslationBetweenPointsAndReference():
    r,c = FindOptimalScaleAndTranslationBetweenPointsAndReference([[0,0],[0,1],[2,3]],[[3.1,0.1],[2.8,2],[4.5,7.1]])
    assert r-1.786363636363636 < 1e-12
    assert c[0]-2.275757575757576 < 1e-12
    assert c[1]-684848484848485 < 1e-12

def test_polyArea_A():
    assert polyArea([[0,0],[1,0],[1,1],[0,1]])==1

def test_polyArea_B():
    assert polyArea([[0,0],[1,0],[4,0.5],[1,1.5]])==3

def polyCentroid_A():
    assert polyCentroid([[0,0],[1,0],[1,1],[0,1]])==(0.5,0.5)

def polyCentroid_B():
    assert polyCentroid([[0,0],[4,0],[7,1.5],[3,1.5]])==(3.5, 0.75)

def test_pointDistance_A():
    assert pointDistance([0,0],[-12,5])==13

def test_pointDistance_B():
    assert pointDistance([[2,0],[1,1]],[[2,1],[4,5]])==[1,5]

def test_polyPerimeter_closeLoop():
    assert polyPerimeter([[0,0],[0,5],[8,11],[0,11]])==34

def test_polyPerimeter_openLoop():
    assert polyPerimeter([[0,0],[0,5],[8,11],[0,11]],closeLoop=False)==23

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
