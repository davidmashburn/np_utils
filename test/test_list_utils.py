import np_utils
from np_utils import *

TEST_SET = range(10)

def test_split_list_on_condition_1():
    assert split_list_on_condition(TEST_SET, lambda x: x % 2 == 0) == ([0, 2, 4, 6, 8], [1, 3, 5, 7, 9])

def test_split_list_on_condition_2():
    assert split_list_on_condition(TEST_SET, False) == ([], range(10))

def test_split_list_on_condition_3():
    manual = [0, 0, 0, 1, 0, 0, 1, 1, 1, 0]
    assert split_list_on_condition(TEST_SET, manual) == ([3, 6, 7, 8], [0, 1, 2, 4, 5, 9])

if __name__ == '__main__':
    test_split_list_on_condition_1()
    test_split_list_on_condition_2()
    test_split_list_on_condition_3()
