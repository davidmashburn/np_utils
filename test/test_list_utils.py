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
    test_split_list_on_condition_1()
    test_split_list_on_condition_2()
    test_split_list_on_condition_3()
    test_split_at_boundaries_1()
    test_rotate_dict_of_lists_1()
    test_rotate_list_of_dicts_1()

