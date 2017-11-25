#!/usr/bin/env python
'''Tests for some functions in np_utils.py. Use nose to run them.
   Fair warning, this is NOT an exhaustive suite.'''
from __future__ import print_function, division
from builtins import range
from future.utils import lrange

import numpy as np

import np_utils
from np_utils import *
from np_utils._test_helpers import _get_sample_rec_array

def test_haselement_1A():
    assert haselement([1,2,3,4,1,2,3,5],3)

def test_haselement_1B():
    assert not haselement([1,2,3,4,1,2,3,5],6)

def test_haselement_1C():
    assert not haselement([1,2,3,4,1,2,3,5],[3])

def test_haselement_2A():
    assert haselement([[1,2,3,4],[1,2,3,5]],[1,2,3,4])

def test_haselement_2B():
    assert not haselement([[1,2,3,4],[1,2,3,5]],[1,2,3,6])

def test_haselement_2C():
    assert not haselement([[1,2,3,4],[1,2,3,5]],[1,2,3])

def test_haselement_3A():
    assert haselement([[[1,2,3,4]],[[1,2,3,5]]],[[1,2,3,4]])

def test_haselement_3B():
    assert not haselement([[[1,2,3,4]],[[1,2,3,5]]],[[1,2,3,4],[1,2,3,5]])

def test_haselement_3D():
    assert not haselement( [[[1,2,3,4],[5,6,7,8]],[[1,2,3,5],[10,12,14,15]]], [[1,2,3,4],[10,12,14,15]] )

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

def test_reshape_repeating_1():
    '''Might be worth refactoring this into smaller tests'''
    a = np.arange(100)
    assert np.all(reshape_repeating(a, 10) == a[:10])
    assert np.all(reshape_repeating(a, 1000) == np.tile(a, 10))
    fif = reshape_repeating(a, (50, 50))
    assert np.all(fif[::2] == np.arange(50))
    assert np.all(fif[1::2] == np.arange(50, 100))
    b = reshape_repeating(a, 100)
    c = reshape_repeating(a, (10,10))
    a[-1] = 6
    assert c[-1, -1] == 6
    assert reshape_repeating(a, 100).base is a
    assert reshape_repeating(a, (10, 10)).base is a
    assert reshape_repeating(a, 99).base is a
    assert reshape_repeating(a, (4, 5)).base is a
    assert reshape_repeating(a, (10,11)).base.base is None

def test_limitInteriorPoints_r5_0_True():
    assert limitInteriorPoints(range(5),0,uniqueOnly=True) == [0,4]

def test_limitInteriorPoints_r5_2_True():
    assert limitInteriorPoints(range(5),2,uniqueOnly=True) == [0,1,3,4]

def test_limitInteriorPoints_r5_7_True():
    assert limitInteriorPoints(range(5),7,uniqueOnly=True) == lrange(5)

def test_limitInteriorPoints_r5_7_False():
    assert limitInteriorPoints(range(5),7,uniqueOnly=False) == [0,0,1,2,2,2,3,4,4]

def test_limitInteriorPointsInterpolating_04_3():
    assert limitInteriorPointsInterpolating([0,4],3) == [0,1.,2,3.,4]

def test_reshape_smaller_1():
    a = [1,2,3,4,6]
    assert np.array_equal(reshape_smaller(a, 5), a)
    assert np.array_equal(reshape_smaller(a, 3), a[:3])

def test_reshape_smaller_2():
    a44, a33 = np.arange(16).reshape((4, 4)), np.arange(9).reshape((3, 3))
    assert np.array_equal(reshape_smaller(a44, (3,3)), a33)

def test_partitionNumpy_1234_2():
    assert (partitionNumpy([1,2,3,4],2)==[[1,2],[3,4]]).all()

def test_partitionNumpy_123_2():
    assert (partitionNumpy([1,2,3],2)==[[1,2]]).all()

def test_shape_multiply_123_12():
    assert np.array_equal(shape_multiply([[1,2,3]],[2,3]),
                          [[1,1,1,2,2,2,3,3,3]]*2)

def test_shape_multiply_zero_fill_123_12():
    assert np.array_equal(shape_multiply_zero_fill([[1,2,3]],[3,3]),
                          [[0]*9,[0,1,0,0,2,0,0,3,0],[0]*9])

def test_shape_divide_1234_12_mean():
    assert np.array_equal(shape_divide([[1,2,3,4]], [1,2]),
                          [[1.5, 3.5]])

def test_shape_divide_i24_22_mean():
    assert np.array_equal(shape_divide(np.arange(8).reshape(2,4), [2,2]),
                          [[10*0.25, 18*0.25]])

def test_shape_divide_i24_22_median():
    assert np.array_equal(shape_divide(np.arange(8).reshape(2,4), [2,2], reduction='median'),
                          np.median([[[0,1,4,5],[2,3,6,7]]], axis=-1))

def test_shape_divide_i24_22_first():
    assert np.array_equal(shape_divide(np.arange(8).reshape(2,4), [2,2], reduction='first'),
                          [[0,2]])

def test_shape_divide_i24_22_all():
    assert np.array_equal(shape_divide(np.arange(8).reshape(2,4), [2,2], reduction='all'),
                          np.reshape([0,2,1,3,4,6,5,7], [2,1,2,2]))

def test_cartesian_1():
    assert np.array_equal(cartesian(([1, 2, 3], [4, 5], [6, 7])),
                          np.array([[1, 4, 6],
                                    [1, 4, 7],
                                    [1, 5, 6],
                                    [1, 5, 7],
                                    [2, 4, 6],
                                    [2, 4, 7],
                                    [2, 5, 6],
                                    [2, 5, 7],
                                    [3, 4, 6],
                                    [3, 4, 7],
                                    [3, 5, 6],
                                    [3, 5, 7]]))

def test_sliding_window_1():
    # Doesn't actually test the sliding part of the sliding windows, but that's ok for now
    a = np.arange(24).reshape(4, 6)
    b = sliding_window(a, (2, 2))
    assert b.shape == (6, 2, 2)
    assert np.array_equal(b,
                          a.reshape(2, 2, 3, 2)
                           .transpose(0,2,1,3)
                           .reshape(6, 2, 2))

    c = sliding_window(a, (2, 2), flatten=False)
    assert c.shape == (2, 3, 2, 2)
    assert np.array_equal(c,
                          a.reshape(2, 2, 3, 2)
                           .transpose(0,2,1,3))

def test_interpNaNs_0nnn4_r5():
    assert np.all(interpNaNs(np.array([0,np.nan,np.nan,np.nan,4]))==lrange(5))

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
    assert np.all( pointDistance([[2,0],[1,1]],[[2,1],[4,5]])==[1,5] )

def test_polyPerimeter_closeLoop():
    assert polyPerimeter([[0,0],[0,5],[8,11],[0,11]])==34

def test_polyPerimeter_openLoop():
    assert polyPerimeter([[0,0],[0,5],[8,11],[0,11]],closeLoop=False)==23

def test_build_grid_A():
    assert build_grid((0.7,0.15),(10,11),(1,1)==[[[0.7]],[[0.15]]])

def test_build_grid_B():
    assert np.all(build_grid((0,0),(1,1),(3,3))==np.array([[[-1,-1,-1],[0,0,0],[1,1,1]],
                                                           [[-1,0,1]]*3]))

def test_build_grid_C():
    assert np.all(np.isclose(build_grid((0,1),(1,4.1),(3,3)),
                  [[[-1,-1,-1],[0,0,0],[1,1,1]],[[-3.1,1,5.1]]*3]))

def test_reverse_broadcast_1():
    assert np.array_equal(reverse_broadcast(np.add)([100,200],
                                                    np.arange(6).reshape(2,3)),
                          np.array([[100, 101, 102], [203, 204, 205]]))

def test_box_1():
    a = np.arange(10)
    aaa = np.arange(24).reshape([2, 3, 4])
    assert np.array_equal(box(a)[0], a)
    assert np.array_equal(box(a, 0)[0], a)
    assert np.array_equal(box(a, 1)[0, 0], 0)
    assert np.array_equal(box(aaa, 0)[0], aaa)
    assert np.array_equal(box(aaa, 1)[0, 0], aaa[0])
    assert np.array_equal(box(aaa, 2)[0, 0, 0], aaa[0, 0])
    assert np.array_equal(box(aaa, 3)[0, 0, 0, 0], aaa[0, 0, 0])
    assert box_shape(box(aaa, 0)) == ()
    assert box_shape(box(aaa, 1)) == (2,)
    assert box_shape(box(aaa, 2)) == (2, 3)
    assert box_shape(box(aaa, 3)) == (2, 3, 4)

def _unbox_box_equals(arr, depth=0):
    return np.array_equal(arr, unbox(box(arr, depth)))

def test_unbox_1():
    assert _unbox_box_equals([1,2,3,4])
    assert _unbox_box_equals([[1,2,3,4]])
    assert _unbox_box_equals([[1,2,3,4]], depth=1)
    assert _unbox_box_equals(np.arange(24).reshape(2,3,4))
    assert _unbox_box_equals(np.arange(24).reshape(2,3,4), 1)
    assert _unbox_box_equals(np.arange(24).reshape(2,3,4), 2)

def test_apply_at_depth_0():
    # Verify that the kwd handling is correct
    a = np.arange(24).reshape([2, 3, 4])
    b = np.arange(6).reshape([2, 3])
    assert np.array_equal(a - 1,
                          apply_at_depth(np.subtract, a, np.array([1]), depth=2))
    assert np.array_equal(a - 1,
                          apply_at_depth(np.subtract, a, np.array([1]), depths=2))
    assert np.array_equal(a - 1,
                          apply_at_depth(np.subtract, a, np.array([1]), depths=[2, 2]))

    fail = False
    try:    apply_at_depth(np.subtract, a, np.array([1]), depth=2, depths=2)
    except: fail = True
    assert fail


def test_apply_at_depth_1():
    a = np.arange(24).reshape([2, 3, 4])

    np.array_equal(apply_at_depth(np.sum, a),
                   apply_at_depth_ravel(np.sum, a))

    for depth in [0] + lrange(-4,4):
        assert np.array_equal(apply_at_depth(np.sum, a, depth=depth),
                              apply_at_depth_ravel(np.sum, a, depth=depth))
    assert apply_at_depth(np.sum, a, depth=0) == np.sum(a)
    assert np.array_equal(apply_at_depth(np.sum, a, depth=-1),
                          np.sum(a, axis=-1))
    assert np.array_equal(apply_at_depth(np.subtract, a, np.array([1]), depth=0),
                          a - 1)
    assert np.array_equal(apply_at_depth(np.subtract, a, np.array([1]), depth=0),
                          [i - 1 for i in a])
    assert np.array_equal(apply_at_depth(np.subtract, a, np.array([1]), depth=0),
                          [[j - 1 for j in i]
                           for i in a])
    for depth in [0] + lrange(-4,4):
        assert np.array_equal(apply_at_depth(np.subtract, a, np.array([1]), depth=depth),
                              a - 1)

def test_apply_at_depth_2():
    def f(a, b):
        assert a.ndim == 2
        assert b.ndim == 1
        return np.vstack([a, b])

    arr1 = np.arange(4*9).reshape(4, 1, 9)
    arr2 = np.array([[[1,2,3,4,5,6,7,8,9],
                      [10,20,30,40,50,60,70,80,90],
                      [100]*9,
                      [1000]*9]])
    for i in range(4):
        assert np.array_equal(apply_at_depth(f, arr1, arr2, depths=[1, 2])[0, i],
                              f(arr1[i], arr2[0, i]))
    images = np.array([i * np.arange(12).reshape(3, 4)
                       for i in range(5)])
    add_ons = np.array([[1,2,3,4],
                        [10,20,30,40],
                        [10]*4,
                        [100]*4,
                        [15]*4])

    assert np.array_equal(apply_at_depth(f, images, add_ons, depths=[1, 1]),
                          np.array([f(images[i], add_ons[i]) for i in range(5)]))

def test_apply_at_depth_3():
    failed = False
    try: a + b
    except: failed = True
    assert failed

    a, b = np.ones([4, 6]), [1,10,100,1000]
    assert np.array_equal(apply_at_depth(np.add, a, b, depths=1),
                          reverse_broadcast(np.add)(a, b))

    inds = (0,0,1,2)
    a, b = np.arange(120).reshape(1,2,3,4,5), [1,10,100,1000]
    assert np.array_equal(apply_at_depth(np.add, a, b, depths=[4, 1]).__getitem__(inds),
                         a.__getitem__(inds) + b[inds[-1]])

if __name__ == '__main__':
    test_haselement_1A()
    test_haselement_1B()
    test_haselement_1C()
    test_haselement_2A()
    test_haselement_2B()
    test_haselement_2C()
    test_haselement_3A()
    test_haselement_3B()
    test_haselement_3D()
    test_addBorder_0()
    test_addBorder_0_th2()
    test_addBorder_5()
    test_addBorder_axis1()
    test_addBorder_axism1()
    test_reshape_repeating_1()
    test_limitInteriorPoints_r5_0_True()
    test_limitInteriorPoints_r5_2_True()
    test_limitInteriorPoints_r5_7_True()
    test_limitInteriorPoints_r5_7_False()
    test_limitInteriorPointsInterpolating_04_3()
    test_reshape_smaller_1()
    test_reshape_smaller_2()
    test_partitionNumpy_1234_2()
    test_partitionNumpy_123_2()
    test_shape_multiply_123_12()
    test_shape_multiply_zero_fill_123_12()
    test_shape_divide_1234_12_mean()
    test_shape_divide_i24_22_mean()
    test_shape_divide_i24_22_median()
    test_shape_divide_i24_22_first()
    test_shape_divide_i24_22_all()
    test_cartesian_1()
    test_sliding_window_1()
    test_interpNaNs_0nnn4_r5()
    interpNumpy_1345_half()
    interpNumpy_11354050_half()
    interpolatePlane_113540_half_1()
    interpolateSumBelowAbove_113540_1p5_1()
    limitInteriorPointsInterpolating_1234_1()
    limitInteriorPointsInterpolating_1234_5()
    test_linearTransform_11_012m2_False()
    test_linearTransform_11_012m2_True()
    test_reverseLinearTransform_m1m41_012m2_False()
    test_reverseLinearTransform_11_012m2_True()
    test_FindOptimalScaleAndTranslationBetweenPointsAndReference()
    test_polyArea_A()
    test_polyArea_B()
    polyCentroid_A()
    polyCentroid_B()
    test_pointDistance_A()
    test_pointDistance_B()
    test_polyPerimeter_closeLoop()
    test_polyPerimeter_openLoop()
    test_build_grid_A()
    test_build_grid_B()
    test_build_grid_C()
    test_reverse_broadcast_1()
    test_box_1()
    test_unbox_1()
    test_apply_at_depth_0()
    test_apply_at_depth_1()
    test_apply_at_depth_2()
    test_apply_at_depth_3()



