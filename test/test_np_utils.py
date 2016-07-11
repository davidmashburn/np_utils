#!/usr/bin/env python
'''Tests for some functions in np_utils.py. Use nose to run them.
   Fair warning, this is NOT an exhaustive suite.'''

import numpy as np
from copy import copy
from collections import Counter
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

def test_element_of_3D():
    assert not haselement( [[[1,2,3,4],[5,6,7,8]],[[1,2,3,5],[10,12,14,15]]], [[1,2,3,4],[10,12,14,15]] )

def test_split_at_boundaries_1(): # test for arrays as well
    x = np.array([0, 3, 8, 6, 5, 9, 8, 9, 9, 2])
    s = split_at_boundaries(x, [2,3,4])
    s2 = split_at_boundaries(x, [0,2,3,4,10])
    r = [[0, 3], [8], [6], [5, 9, 8, 9, 9, 2]]
    assert all(np.array_equal(i, j)
               for i, j in zip(s, r))
    assert all(np.array_equal(i, j)
               for i, j in zip(s, r))

def test_np_groupby_1():
    a = _get_sample_rec_array()
    g = np_groupby(a['n'], a['o'], np.max)
    assert g.shape == (100,)
    assert g.dtype == [('f0', '<i8'), ('f1', '<f8')]
    assert np.all(g['f0'] == np.arange(100))
    assert np.all(g['f1'] > 450)

def test_np_groupby_2():
    a = _get_sample_rec_array()
    g = np_groupby(a['n'], a['o'], np.max, names=['n', 'max_o'])
    assert g.shape == (100,)
    assert g.dtype == [('n', '<i8'), ('max_o', '<f8')]

def test_np_groupby_3():
    a = _get_sample_rec_array()
    g = np_groupby(a['n'], a['o'], np.max, np.min,
                   names=['n', 'max_o', 'min_o'])
    assert g.max_o[0] != g.min_o[0]

def test_np_groupby_4():
    a = _get_sample_rec_array()
    g = np_groupby(a[['m', 'n']], a,
                   lambda x: np.mean(x['o']),
                   lambda x: np.std(x['o']),
                   lambda x: np.min(x['p']),
                   names=['m', 'n', 'mean_o', 'std_o', 'min_p'])
    assert g.shape == (200,)
    assert g.dtype == [('m', '|S4'), ('n', '<i8'), ('mean_o', '<f8'),
                       ('std_o', '<f8'), ('min_p', '<f8')]

def test_np_groupby_5():
    a = _get_sample_rec_array()
    def compute_some_thing(x):
        o, p = x['o'], x['p']
        return np.mean(o) / np.std(o) * np.min(p)
    g = np_groupby(a[['m', 'n']], a, compute_some_thing,
                   names=['m', 'n', 'its_complicated'])
    assert g.shape == (200,)
    assert g.dtype == [('m', '|S4'), ('n', '<i8'), ('its_complicated', '<f8')]

def test_rec_groupby_1():
    a = _get_sample_rec_array()
    assert np.all(rec_groupby(a, 'n', (np.max, 'o', 'max_o')) ==
                  np_groupby(a['n'], a['o'], np.max, names=['n', 'max_o']))

def test_rec_groupby_2():
    a = _get_sample_rec_array()
    assert np.all(rec_groupby(a, 'n', (np.max, 'o', 'max_o'),
                                      (np.min, 'o', 'min_o')) ==
                  np_groupby(a['n'], a['o'], np.max, np.min,
                             names=['n', 'max_o', 'min_o']))

def test_rec_groupby_3():
    a = _get_sample_rec_array()
    def compute_some_thing(x):
        o, p = x['o'], x['p']
        return np.mean(o) / np.std(o) * np.min(p)
    
    g_r = rec_groupby(a, ['m', 'n'], (compute_some_thing, ['o', 'p'], 'its_complicated'))
    g_n = np_groupby(a[['m', 'n']], a, compute_some_thing,
                     names=['m', 'n', 'its_complicated'])
    
    assert np.all(g_r == g_n)

def test_get_first_indices_1():
    a = [0,1,2,3,3,4,5,12,4,8]
    assert np.all(get_first_indices(a, [1, 4, 12]) == [1, 5, 7])
    assert np.all(get_first_indices(a, [1, 4, 12,13]) == [1, 5, 7, None])
    assert np.all(get_first_indices(a, [1, 4, 12, 13], missing='len') == [1, 5, 7, len(a)])
    assert np.all(get_first_indices(a, [1, 4, 12, 13], missing=-1) == [1, 5, 7, -1])
    
    try:
        np.all(get_first_indices(a, [1, 4, 12, 13], missing='fail') == [1, 5, 7, -1])
        intentioned_fail = False
    except:
        intentioned_fail = True
    assert intentioned_fail

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
    assert np.all(
        reverse_broadcast(np.add)([100,200], np.arange(6).reshape(2,3)) ==
        np.array([[100, 101, 102], [203, 204, 205]])
    )

if __name__ == '__main__':
    test_haselement_1A()
    test_haselement_1B()
    test_haselement_1C()
    test_haselement_2A()
    test_haselement_2B()
    test_haselement_2C()
    test_haselement_3A()
    test_haselement_3B()
    test_element_of_3D()
    test_split_at_boundaries_1()
    test_np_groupby_1()
    test_np_groupby_2()
    test_np_groupby_3()
    test_np_groupby_4()
    test_np_groupby_5()
    test_rec_groupby_1()
    test_rec_groupby_2()
    test_rec_groupby_3()
    test_get_first_indices_1()
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
    test_partitionNumpy_1234_2()
    test_partitionNumpy_123_2()
    test_shape_multiply_123_12()
    test_shape_multiply_zero_fill_123_12()
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
