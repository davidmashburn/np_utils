from builtins import range
from future.utils import lrange

import np_utils
from np_utils import *

TEST_SET = lrange(10)

SAMPLE_LD_1 = [{'a': 1, 'b': 2, 'c': 3},
               {'a': 2, 'b': 4, 'c': 6},
               {'a': 3, 'b': 6, 'c': 9},
               {'a': 4, 'b': 8},
               {'a': 5, 'b': 10}]

REV_LD_1 = {'a': [1, 2, 3, 4, 5],
            'b': [2, 4, 6, 8, 10],
            'c': [3, 6, 9, None, None],}

SAMPLE_DL_1 = {'a': [1, 2, 3, 4, 5],
               'b': [2, 4, 6, 8, 10],
               'c': [3, 6, 9]}

# Made from np.arange(24).reshape((2, 3, 4)):
SAMPLE_NESTED_2_3_4 = (((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)),
                       ((12, 13, 14, 15), (16, 17, 18, 19), (20, 21, 22, 23)))

def test_totuple_1():
    test_arr = np.array(SAMPLE_NESTED_2_3_4)
    test_list = test_arr.tolist()
    assert totuple(test_arr) == SAMPLE_NESTED_2_3_4
    assert totuple(test_list) == SAMPLE_NESTED_2_3_4

def test_totuple_2():
    abc, abc_tuple = 'abcdef', ('a', 'b', 'c', 'd', 'e', 'f')
    assert totuple(abc) == abc_tuple
    assert totuple([abc]) == (abc_tuple,)
    assert totuple(abc, break_strings=False) == abc
    assert totuple([abc], break_strings=False) == (abc,)

def test_tolist_1():
    test_arr = np.array(SAMPLE_NESTED_2_3_4)
    test_list = test_arr.tolist()
    assert tolist(test_arr) == test_list
    assert tolist(test_list) == test_list
    assert tolist(SAMPLE_NESTED_2_3_4) == test_list

def test_tolist_2():
    abc, abc_list = 'abcdef', ['a', 'b', 'c', 'd', 'e', 'f']
    assert tolist(abc) == abc_list
    assert tolist([abc]) == [abc_list]
    assert tolist(abc, break_strings=False) == abc
    assert tolist([abc], break_strings=False) == [abc]

def test_iterToX_1():
    test_arr = np.array(SAMPLE_NESTED_2_3_4)
    test_list = test_arr.tolist()
    assert iterToX(list, test_arr) == test_list
    assert iterToX(list, test_list) == test_list
    assert iterToX(list, SAMPLE_NESTED_2_3_4) == test_list

def test_iterToX_2():
    abc, abc_list = 'abcdef', ['a', 'b', 'c', 'd', 'e', 'f']
    assert iterToX(list, abc) == abc_list
    assert iterToX(list, [abc]) == [abc_list]
    assert iterToX(list, abc, break_strings=False) == abc
    assert iterToX(list, [abc], break_strings=False) == [abc]

def test_iterToX_3():
    deepshape = lambda x: ([] if not (hasattr(x, '__len__') and len(x)>0) else
                           [x.shape] + deepshape(x[0]))
    assert deepshape(iterToX(box, SAMPLE_NESTED_2_3_4[0][0])) == [(1,), (4,)]
    assert deepshape(iterToX(box, SAMPLE_NESTED_2_3_4[0])) == [(1,), (3, 1), (1,), (4,)]
    assert deepshape(iterToX(box, SAMPLE_NESTED_2_3_4)) == [(1,), (2, 1), (1,), (3, 1), (1,), (4,)]

def test_iterToX_splat_1():
    list_splat = lambda *x: list(x)
    test_arr = np.array(SAMPLE_NESTED_2_3_4)
    test_list = test_arr.tolist()
    assert iterToX_splat(list_splat, test_arr) == test_list
    assert iterToX_splat(list_splat, test_list) == test_list
    assert iterToX_splat(list_splat, SAMPLE_NESTED_2_3_4) == test_list

def test_iterToX_splat_2():
    abc, abc_list = 'abcdef', ['a', 'b', 'c', 'd', 'e', 'f']
    list_splat = lambda *x: list(x)
    assert iterToX_splat(list_splat, abc) == abc_list
    assert iterToX_splat(list_splat, [abc]) == [abc_list]
    assert iterToX_splat(list_splat, abc, break_strings=False) == abc
    assert iterToX_splat(list_splat, [abc], break_strings=False) == [abc]

def test_partition_range_1():
    assert list(partition_range(29, 10)) == [range(0, 10), range(10, 20), range(20, 29)]
    assert list(partition_range(30, 10)) == [range(0, 10), range(10, 20), range(20, 30)]
    assert list(partition_range(31, 10)) == [range(0, 10), range(10, 20), range(20, 30), range(30, 31)]
    assert list(partition_range(35, 10)) == [range(0, 10), range(10, 20), range(20, 30), range(30, 35)]

def test_split_list_on_condition_1():
    assert split_list_on_condition(TEST_SET, lambda x: x % 2 == 0) == ([0, 2, 4, 6, 8], [1, 3, 5, 7, 9])

def test_split_list_on_condition_2():
    assert split_list_on_condition(TEST_SET, False) == ([], lrange(10))

def test_split_list_on_condition_3():
    manual = [0, 0, 0, 1, 0, 0, 1, 1, 1, 0]
    assert split_list_on_condition(TEST_SET, manual) == ([3, 6, 7, 8], [0, 1, 2, 4, 5, 9])

def test_split_at_boundaries_1():
    s = split_at_boundaries([1,2,3,4,5,6,7,8,9], [2,5,6])
    assert s == [[1, 2], [3, 4, 5], [6], [7, 8, 9]]

def test_values_sorted_by_keys_1():
    assert list(values_sorted_by_keys({2: 'b', 1: 'a'}, key=None)) == ['a', 'b']

def test_values_sorted_by_keys_2():
    np.random.seed(0)
    keys = np.random.randint(0, 100000, size=100)
    assert len(np.unique(keys)) == len(keys), 'Unit test bug, choose new keys'
    values = np.random.rand(100)
    assert np.array_equal(list(values_sorted_by_keys(dict(zip(keys, values)), key=None)),
                          values[np.argsort(keys)])

def test_rotate_list_of_dicts_1():
    assert rotate_list_of_dicts(SAMPLE_LD_1) == REV_LD_1

def test_rotate_dict_of_lists_1():
    assert rotate_dict_of_lists(SAMPLE_DL_1) == SAMPLE_LD_1

def test_shallowAdd_1():
    assert shallowAdd([2, 3], [1, 1]) == [3, 4]

def test_shallowAdd_2():
    assert shallowAdd([[2], [3]], [1, 1]) == [[3], [4]]

def test_shallowAdd_3():
    assert shallowAdd([[2, 3, 4], [5, 6, 7]], [10, 20]) == [[12, 13, 14], [25, 26, 27]]

def test_shallowAdd_4():
    assert (shallowAdd([[2, [3, 4]], [5, [6, 7]]], [[10, 20], [30, 40]]) ==
            [[12, [23, 24]], [35, [46, 47]]])

def test_shallowAdd_blank():
    assert shallowAdd([], []) == []

def test_deepAdd_1():
    assert deepAdd([2, 3], [1, 1]) == [3, 4]

def test_deepAdd_2():
    assert deepAdd([[2], [3]], [1]) == [[3], [4]]

def test_deepAdd_3():
    assert deepAdd([[2, 3, 4], [5, 6, 7]], [10, 20, 30]) == [[12, 23, 34], [15, 26, 37]]

def test_deepAdd_4():
    assert (deepAdd([[2, [3, 4]], [5, [6, 7]]], [10, [20, 30]]) ==
            [[12, [23, 34]], [15, [26, 37]]])


def test_deepAdd_blank():
    assert deepAdd([], []) == []

def test_shallowAdd_fail_1():
    failed = None
    try:
        shallowAdd([1], [])
        failed = False
    except TypeError:
        failed = True

    assert failed

def test_deepAdd_fail_1():
    failed = None
    try:
        deepAdd([1], [])
        failed = False
    except TypeError:
        failed = True

    assert failed

def test_fancyIndexingList_1():
    assert fL([[1],[2,3],[4,5,6],[7,8,9,10]])[:,0] == [1, 2, 4, 7]
    assert fL([[1],[2,3],[4,5,6],[7,8,9,10]])[2:4,0] == [4, 7]
    assert fL([1,[2,3],[[4,5],[6,7]]])[2,0,1] == 5
    assert fL([[1,7],[2,3],[4,5,6],[7,8,9,10]])[:,(0,1)] == [[1, 7], [2, 3], [4, 5], [7, 8]]
    assert fL([[1,7],[2,3],[4,5,6],[7,8,9,10]])[(0,2),(0,1)] == [[1, 7], [4, 5]]
    
    failed = False
    try:
        fL([1,2,3,4,5])[(1,2)]
    except TypeError:
        failed = True
    assert failed
    
    failed = False
    try:
        fL([1,2,3,4,5])[[1,2]]
    except TypeError:
        failed = True
    assert failed
    
    assert fL([1,2,3,4,5])[(1,2),] == [2,3]
    assert fL([1,2,3,4,5])[[1,2],] == [2,3]
    assert fL([1,2,3,4,5])[((1,2),)] == [2,3]
    assert fL([1,2,3,4,5])[([1,2],)] == [2,3]
    assert fL([1,2,3,4,5])[[[1,2]]] == [2,3]

    assert fL( [[1,7],[2,3],[4,5,6],[7,8,9,10]] )[((0,0),(2,1)),] == [1,5]
    assert fL( [[[[1,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]] )[:, ((0,0),(0,1),1), 1] == \
           [[2, 4, [7,8]], [10, 12, [15,16]]]

def test_fancyIndexingList_2():
    assert fL([{'a': 1}, {'a': 2}])[:, 'a'] == [1, 2]
    assert fL([{'a': [1, 2, 3, 4, 5]}, {'a': [2, 3, 4, 5, 6]}])[:, 'a', 1:3] == [[2, 3], [3, 4]]
    assert fD({'a': [[1, 10], [2, 20], 3, 4, 5]})['a', :2, 1] == [10, 20]
    
    assert fD({'a':[5, {'c': 15}]})['a', 1, 'c'] == 15
    
    failed = False
    try:
        fD({'a': [1], 'b': [2], 'c': [3]})[('a', 'b'), 0]
    except KeyError:
        failed = True
    assert failed
    
    failed = False
    try:
        fD({('a', 'b'): 1})['a', 'b']
    except KeyError:
        failed = True
    assert failed
    
    assert fD({('a', 'b'): 1})[('a', 'b'), ] == 1

def test_fancyIndexingListM1_1():
    assert fLm1([[1],[2,3],[4,5,6],[7,8,9,10]])[:, 1] == [1, 2, 4, 7]
    assert fLm1([[1],[2,3],[4,5,6],[7,8,9,10]])[1:, 1] == [1, 2, 4, 7]
    assert fLm1([[1],[2,3],[4,5,6],[7,8,9,10]])[1:5, 1] == [1, 2, 4, 7]
    assert fLm1([[1],[2,3],[4,5,6],[7,8,9,10]])[3:5, 1] == [4, 7]
    assert fLm1([1,[2,3],[[4,5],[6,7]]] )[3,1,2] == 5
    assert fLm1([[1,7],[2,3],[4,5,6],[7,8,9,10]])[:,(1,2)] == [[1, 7], [2, 3], [4, 5], [7, 8]]
    assert fLm1([[1,7],[2,3],[4,5,6],[7,8,9,10]])[(1,3),(1,2)] == [[1, 7], [4, 5]]
    
    failed = False
    try:
        fLm1([1,2,3,4,5])[(2,3)]
    except TypeError:
        failed = True
    assert failed
    
    failed = False
    try:
        fLm1([1,2,3,4,5])[[2,3]]
    except TypeError:
        failed = True
    assert failed
    
    assert fLm1([1,2,3,4,5])[(2,3),] == [2,3]
    assert fLm1([1,2,3,4,5])[[2,3],] == [2,3]
    assert fLm1([1,2,3,4,5])[((2,3),)] == [2,3]
    assert fLm1([1,2,3,4,5])[([2,3],)] == [2,3]
    assert fLm1([1,2,3,4,5])[[[2,3]]] == [2,3]

    assert fLm1( [[1,7],[2,3],[4,5,6],[7,8,9,10]] )[((1,1),(3,2)),] == [1,5]
    assert fLm1( [[[[1,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]] )[:, ((1,1),(1,2),2), 2] == \
           [[2, 4, [7,8]], [10, 12, [15,16]]]

def test_fancyIndexingListM1_2():
    assert fLm1([{'a': 1}, {'a': 2}])[1:3, 'a'] == [1, 2]
    assert fLm1([{'a': [1, 2, 3, 4, 5]}, {'a': [2, 3, 4, 5, 6]}])[:, 'a', 2:4] == [[2, 3], [3, 4]]
    assert fDm1({'a': [[1, 10], [2, 20], 3, 4, 5]})['a', 1:3, 2] == [10, 20]
    
    assert fDm1({'a':[5, {'c': 15}]})['a', 2, 'c'] == 15
    
    failed = False
    try:
        fDm1({'a': [1], 'b': [2], 'c': [3]})[('a', 'b'), 1]
    except KeyError:
        failed = True
    assert failed
    
    failed = False
    try:
        fDm1({('a', 'b'): 1})['a', 'b']
    except KeyError:
        failed = True
    assert failed
    
    assert fDm1({('a', 'b'): 1})[('a', 'b'), ] == 1

if __name__ == '__main__':
    test_totuple_1()
    test_totuple_2()
    test_tolist_1()
    test_tolist_2()
    test_iterToX_1()
    test_iterToX_2()
    test_iterToX_3()
    test_iterToX_splat_1()
    test_iterToX_splat_2()
    test_partition_range_1()
    test_split_list_on_condition_1()
    test_split_list_on_condition_2()
    test_split_list_on_condition_3()
    test_split_at_boundaries_1()
    test_values_sorted_by_keys_1()
    test_values_sorted_by_keys_2()
    test_rotate_list_of_dicts_1()
    test_rotate_dict_of_lists_1()
    test_shallowAdd_1()
    test_shallowAdd_2()
    test_shallowAdd_3()
    test_shallowAdd_4()
    test_deepAdd_1()
    test_deepAdd_2()
    test_deepAdd_3()
    test_deepAdd_4()
    test_shallowAdd_blank()
    test_deepAdd_blank()
    test_shallowAdd_fail_1()
    test_deepAdd_fail_1()
    test_fancyIndexingList_1()
    test_fancyIndexingList_2()
    test_fancyIndexingListM1_1()
    test_fancyIndexingListM1_2()
