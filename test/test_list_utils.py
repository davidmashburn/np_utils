import np_utils
from np_utils import *

TEST_SET = range(10)

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

def test_split_list_on_condition_1():
    assert split_list_on_condition(TEST_SET, lambda x: x % 2 == 0) == ([0, 2, 4, 6, 8], [1, 3, 5, 7, 9])

def test_split_list_on_condition_2():
    assert split_list_on_condition(TEST_SET, False) == ([], range(10))

def test_split_list_on_condition_3():
    manual = [0, 0, 0, 1, 0, 0, 1, 1, 1, 0]
    assert split_list_on_condition(TEST_SET, manual) == ([3, 6, 7, 8], [0, 1, 2, 4, 5, 9])

def test_split_at_boundaries_1():
    s = split_at_boundaries([1,2,3,4,5,6,7,8,9], [2,5,6])
    assert s == [[1, 2], [3, 4, 5], [6], [7, 8, 9]]

def test_rotate_list_of_dicts_1():
    assert rotate_list_of_dicts(SAMPLE_LD_1) == REV_LD_1

def test_rotate_dict_of_lists_1():
    assert rotate_dict_of_lists(SAMPLE_DL_1) == SAMPLE_LD_1

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
    test_split_list_on_condition_1()
    test_split_list_on_condition_2()
    test_split_list_on_condition_3()
    test_split_at_boundaries_1()
    test_rotate_dict_of_lists_1()
    test_rotate_list_of_dicts_1()

