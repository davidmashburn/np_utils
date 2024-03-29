import np_utils
from np_utils import *

TEST_SET = list(range(10))

SAMPLE_LD_1 = [
    {"a": 1, "b": 2, "c": 3},
    {"a": 2, "b": 4, "c": 6},
    {"a": 3, "b": 6, "c": 9},
    {"a": 4, "b": 8},
    {"a": 5, "b": 10},
]

REV_LD_1 = {
    "a": [1, 2, 3, 4, 5],
    "b": [2, 4, 6, 8, 10],
    "c": [3, 6, 9, None, None],
}

SAMPLE_DL_1 = {"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10], "c": [3, 6, 9]}

# Made from np.arange(24).reshape((2, 3, 4)):
SAMPLE_NESTED_2_3_4 = (
    ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)),
    ((12, 13, 14, 15), (16, 17, 18, 19), (20, 21, 22, 23)),
)


def test_totuple_1():
    test_arr = np.array(SAMPLE_NESTED_2_3_4)
    test_list = test_arr.tolist()
    assert totuple(test_arr) == SAMPLE_NESTED_2_3_4
    assert totuple(test_list) == SAMPLE_NESTED_2_3_4


def test_totuple_2():
    abc, abc_tuple = "abcdef", ("a", "b", "c", "d", "e", "f")
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
    abc, abc_list = "abcdef", ["a", "b", "c", "d", "e", "f"]
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
    abc, abc_list = "abcdef", ["a", "b", "c", "d", "e", "f"]
    assert iterToX(list, abc) == abc_list
    assert iterToX(list, [abc]) == [abc_list]
    assert iterToX(list, abc, break_strings=False) == abc
    assert iterToX(list, [abc], break_strings=False) == [abc]


def test_iterToX_3():
    deepshape = lambda x: (
        []
        if not (hasattr(x, "__len__") and len(x) > 0)
        else [x.shape] + deepshape(x[0])
    )
    assert deepshape(iterToX(box, SAMPLE_NESTED_2_3_4[0][0])) == [(1,), (4,)]
    assert deepshape(iterToX(box, SAMPLE_NESTED_2_3_4[0])) == [(1,), (3, 1), (1,), (4,)]
    assert deepshape(iterToX(box, SAMPLE_NESTED_2_3_4)) == [
        (1,),
        (2, 1),
        (1,),
        (3, 1),
        (1,),
        (4,),
    ]


def test_iterToX_splat_1():
    list_splat = lambda *x: list(x)
    test_arr = np.array(SAMPLE_NESTED_2_3_4)
    test_list = test_arr.tolist()
    assert iterToX_splat(list_splat, test_arr) == test_list
    assert iterToX_splat(list_splat, test_list) == test_list
    assert iterToX_splat(list_splat, SAMPLE_NESTED_2_3_4) == test_list


def test_iterToX_splat_2():
    abc, abc_list = "abcdef", ["a", "b", "c", "d", "e", "f"]
    list_splat = lambda *x: list(x)
    assert iterToX_splat(list_splat, abc) == abc_list
    assert iterToX_splat(list_splat, [abc]) == [abc_list]
    assert iterToX_splat(list_splat, abc, break_strings=False) == abc
    assert iterToX_splat(list_splat, [abc], break_strings=False) == [abc]


def test_partition_range_1():
    assert list(partition_range(29, 10)) == [range(0, 10), range(10, 20), range(20, 29)]
    assert list(partition_range(30, 10)) == [range(0, 10), range(10, 20), range(20, 30)]
    assert list(partition_range(31, 10)) == [
        range(0, 10),
        range(10, 20),
        range(20, 30),
        range(30, 31),
    ]
    assert list(partition_range(35, 10)) == [
        range(0, 10),
        range(10, 20),
        range(20, 30),
        range(30, 35),
    ]


def test_split_list_on_condition_1():
    assert split_list_on_condition(TEST_SET, lambda x: x % 2 == 0) == (
        [0, 2, 4, 6, 8],
        [1, 3, 5, 7, 9],
    )


def test_split_list_on_condition_2():
    assert split_list_on_condition(TEST_SET, False) == ([], list(range(10)))


def test_split_list_on_condition_3():
    manual = [0, 0, 0, 1, 0, 0, 1, 1, 1, 0]
    assert split_list_on_condition(TEST_SET, manual) == (
        [3, 6, 7, 8],
        [0, 1, 2, 4, 5, 9],
    )


def test_split_at_boundaries_1():
    s = split_at_boundaries([1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 5, 6])
    assert s == [[1, 2], [3, 4, 5], [6], [7, 8, 9]]


def test_unshuffle_indices():
    # x = list(range(len(10)))
    # random.seed(0)
    # random.shuffle(x)
    x = [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]
    y = [7, 2, 6, 4, 5, 3, 9, 0, 1, 8]
    assert unshuffle_indices(x) == y
    assert unshuffle_indices(y) == x


def test_get_ranks():
    x = [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]
    ranks = [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]
    assert get_ranks(x) == ranks


def test_get_generator_ranks():
    x = [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]
    ranks = [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]
    assert get_generator_ranks(len(x), (i for i in x)) == ranks


def test_group_by_first_elements():
    input = [("A", 1, 4), ("A", 2, 7), ("B", 5, 9), ("A", 0, 3), ("B", 1, 1)]
    output = {"A": [(1, 4), (2, 7), (0, 3)], "B": [(5, 9), (1, 1)]}
    assert group_by_first_elements(input) == output


def test_values_sorted_by_keys_1():
    assert list(values_sorted_by_keys({2: "b", 1: "a"}, key=None)) == ["a", "b"]


def test_values_sorted_by_keys_2():
    np.random.seed(0)
    keys = np.random.randint(0, 100000, size=100)
    assert len(np.unique(keys)) == len(keys), "Unit test bug, choose new keys"
    values = np.random.rand(100)
    assert np.array_equal(
        list(values_sorted_by_keys(dict(zip(keys, values)), key=None)),
        values[np.argsort(keys)],
    )


def test_rotate_list_of_dicts_1():
    assert rotate_list_of_dicts(SAMPLE_LD_1) == REV_LD_1


def test_rotate_dict_of_lists_1():
    assert rotate_dict_of_lists(SAMPLE_DL_1) == SAMPLE_LD_1


def test_append_rank_to_dict_of_lists_1():
    test = {"a": [2], "c": [1], "b": [3]}
    out = {"a": [2, 1], "c": [1, 0], "b": [3, 2]}
    assert append_rank_to_dict_of_lists(test) == out
    assert test == out


def test_append_rank_to_dict_of_lists_2():
    test = {"a": [2], "c": [1], "b": [3]}
    out_rev = {"a": [2, 1], "c": [1, 2], "b": [3, 0]}
    assert append_rank_to_dict_of_lists(test, reverse=True) == out_rev
    assert test == out_rev


def test_get_index_by_value_mapping():
    out = {4: 0, 5: 1, 6: 2, 7: 3}
    assert out == get_index_by_value_mapping(range(4, 8))


def test_shallowAdd_1():
    assert shallowAdd([2, 3], [1, 1]) == [3, 4]


def test_shallowAdd_2():
    assert shallowAdd([[2], [3]], [1, 1]) == [[3], [4]]


def test_shallowAdd_3():
    assert shallowAdd([[2, 3, 4], [5, 6, 7]], [10, 20]) == [[12, 13, 14], [25, 26, 27]]


def test_shallowAdd_4():
    assert shallowAdd([[2, [3, 4]], [5, [6, 7]]], [[10, 20], [30, 40]]) == [
        [12, [23, 24]],
        [35, [46, 47]],
    ]


def test_shallowAdd_blank():
    assert shallowAdd([], []) == []


def test_deepAdd_1():
    assert deepAdd([2, 3], [1, 1]) == [3, 4]


def test_deepAdd_2():
    assert deepAdd([[2], [3]], [1]) == [[3], [4]]


def test_deepAdd_3():
    assert deepAdd([[2, 3, 4], [5, 6, 7]], [10, 20, 30]) == [[12, 23, 34], [15, 26, 37]]


def test_deepAdd_4():
    assert deepAdd([[2, [3, 4]], [5, [6, 7]]], [10, [20, 30]]) == [
        [12, [23, 34]],
        [15, [26, 37]],
    ]


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


if __name__ == "__main__":
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
    test_unshuffle_indices()
    test_get_ranks()
    test_get_generator_ranks()
    test_group_by_first_elements()
    test_split_at_boundaries_1()
    test_values_sorted_by_keys_1()
    test_values_sorted_by_keys_2()
    test_rotate_list_of_dicts_1()
    test_rotate_dict_of_lists_1()
    test_append_rank_to_dict_of_lists_1()
    test_append_rank_to_dict_of_lists_2()
    test_get_index_by_value_mapping()
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
