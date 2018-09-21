#!/usr/bin/env python
'''Tests for some functions in recarray_utils.py. Use nose to run them.
   Fair warning, this is NOT an exhaustive suite.'''
from __future__ import print_function, division

from builtins import zip
from future.utils import lrange

import numpy as np

import np_utils
from np_utils import *
from np_utils._test_helpers import _get_sample_rec_array


def test_multi_where_1d_1():
    np.random.seed(0)
    x = np.arange(1000)
    np.random.shuffle(x)
    subx = np.arange(100)
    inds = multi_where_1d(x, subx)
    assert np.array_equal(x[inds], subx)

def test_multi_where_1d_2():
    np.random.seed(0)
    x = np.arange(1000)
    np.random.shuffle(x)
    subx = np.arange(-2, 100)

    try:
        multi_where_1d(x, subx)
        failed = False
    except AssertionError as E:
        failed = True

    assert failed

def test_true_where_1():
    assert np.array_equal(true_where(4, [0, 2]),
                          [1, 0, 1, 0])
    assert np.array_equal(true_where([4] * 2, [lrange(4)] * 2),
                          np.eye(4))

def test_true_where_2():
    np.random.seed(0)
    arr = np.random.rand(2, 3, 4)
    bool_arr = (arr < 0.5)
    assert np.array_equal(true_where(bool_arr.shape, np.where(bool_arr)),
                          bool_arr)
    assert np.array_equal(np.where(true_where(arr.shape, np.where(arr))),
                          np.where(arr))

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
    dtype1 = a.dtype[1] # Either '<S4' or '<U4'
    g = np_groupby(a[['m', 'n']], a,
                   lambda x: np.mean(x['o']),
                   lambda x: np.std(x['o']),
                   lambda x: np.min(x['p']),
                   names=['m', 'n', 'mean_o', 'std_o', 'min_p'])
    assert g.shape == (200,)
    assert g.dtype == [('m', dtype1), ('n', '<i8'), ('mean_o', '<f8'),
                       ('std_o', '<f8'), ('min_p', '<f8')]

def test_np_groupby_5():
    a = _get_sample_rec_array()
    dtype1 = a.dtype[1] # Either '<S4' or '<U4'
    def compute_some_thing(x):
        o, p = x['o'], x['p']
        return np.mean(o) / np.std(o) * np.min(p)
    g = np_groupby(a[['m', 'n']], a, compute_some_thing,
                   names=['m', 'n', 'its_complicated'])
    assert g.shape == (200,)
    assert g.dtype == [('m', dtype1), ('n', '<i8'), ('its_complicated', '<f8')]

def test_group_transform_1():
    arr = _get_sample_rec_array()
    # Sort items based on 'o' in groups based on 'm':
    sorted_o = group_transform(arr['m'], arr['o'], np.sort, np.float)
    assert len(sorted_o) == len(arr)
    assert not np.array_equal(sorted_o, arr['o'])
    assert not np.array_equal(sorted_o, np.sort(arr['o']))
    assert np.array_equal(np.sort(sorted_o), np.sort(arr['o']))

def test_group_transform_2():
    arr = _get_sample_rec_array()
    # Rank items based on 'o' and 'p' in groups based on 'm':
    simple_rank = lambda x: np.argsort(x) + 1
    ranked_op = group_transform(arr['m'], arr[['o', 'p']], simple_rank, np.int)
    assert ranked_op.max() < len(arr)

def test_group_transform_3():
    arr = _get_sample_rec_array()
    # Subtract the group mean (background) for 'p' in groups based on 'm':
    background_subtract = lambda x: x - x.mean()
    bg_removed_p = group_transform(arr['m'], arr['p'], background_subtract, np.float)

    # Normalize groups (divide by mean) for 'o' in groups based on 'm' and 'n':
    simple_normalize = lambda x: x / x.mean()
    normalized_o = group_transform(arr[['m', 'n']], arr['o'], simple_normalize, np.float)

    assert not np.array_equal(bg_removed_p, background_subtract(arr['p']))
    assert np.isclose(np.mean(bg_removed_p), 0)
    assert not np.array_equal(normalized_o, simple_normalize(arr['o']))
    assert np.isclose(np.mean(normalized_o), 1)

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

def test_rec_groupby_4():
    a = _get_sample_rec_array()
    def compute_some_thing(x):
        o, p = x['o'], x['p']
        return np.mean(o) / np.std(o) * np.min(p)

    g_mn = rec_groupby(a, ['m', 'n'], (compute_some_thing, ['o', 'p'], 'its_complicated'))
    g_nm = rec_groupby(a, ['n', 'm'], (compute_some_thing, ['o', 'p'], 'its_complicated'))

    assert not np.array_equal(g_mn, g_nm)

def test_np_and_rec_groupby_full_1():
    arr = _get_sample_rec_array()
    simple_rank = lambda x: np.argsort(x) + 1
    background_subtract = lambda x: x - x.mean()
    simple_normalize = lambda x: x / x.mean()

    n = np_groupby_full(arr[['m', 'n']], arr,
       (lambda x: simple_rank(x['o']), np.int),
       (lambda x: simple_rank(x[['o', 'p']]), np.int),
       (lambda x: background_subtract(x['o']), np.float),
       (lambda x: simple_normalize(x['p']), np.float),
       names=['m', 'n', 'rank_o', 'rank_op', 'bg_sub_o', 'norm_p']
    )

    r = rec_groupby_full(arr, ['m', 'n'],
        (simple_rank,         np.int,   'o',        'rank_o'),
        (simple_rank,         np.int,   ['o', 'p'], 'rank_op'),
        (background_subtract, np.float, 'o',        'bg_sub_o'),
        (simple_normalize,    np.float, 'p',        'norm_p')
    )

    assert np.array_equal(n, r)

def test_rec_groupby_full_2():
    arr = _get_sample_rec_array()
    simple_rank = lambda x: np.argsort(x) + 1
    background_subtract = lambda x: x - x.mean()
    simple_normalize = lambda x: x / x.mean()

    r = rec_groupby_full(arr, ['m', 'n'],
        #(simple_rank,         np.int,   'o',        'rank_o'),
        #(simple_rank,         np.int,   ['o', 'p'], 'rank_op'),
        (background_subtract, np.float, 'o',        'bg_sub_o'),
        (simple_normalize,    np.float, 'p',        'norm_p')
    )
    assert(np.isclose(r['bg_sub_o'].mean(), 0))
    assert(np.isclose(r['norm_p'].mean(), 1))
    assert(not np.array_equal(r['bg_sub_o'], background_subtract(arr['o'])))
    assert(not np.array_equal(r['norm_p'], simple_normalize(arr['p'])))

_is_first_occurrence_1_dat = [[0, 1, 0, 1, 0, 2, 3, 4],
                              [1, 1, 0, 0, 0, 1, 1, 1]]

def test_is_first_occurrence_1():
    a, b = _is_first_occurrence_1_dat
    assert np.array_equal(is_first_occurrence(a), b)

def test_is_first_occurrence_1d_1():
    a, b = _is_first_occurrence_1_dat
    assert np.array_equal(is_first_occurrence_1d(a), b)

def test_is_first_occurrence_2():
    assert np.array_equal(is_first_occurrence([[[0,1]], [[0,1]], [[0,2]],[[0,1]], [[1,1]]]),
                          [1,0,1,0,1])

def test_is_first_occurrence_and_1d_3():
    def stupid_version(x):
        return np.array([(p not in x[:i]) for i,p in enumerate(x)]) # This will work, but is very slow

    t = reshape_repeating(np.arange(6), (40, 3))
    assert np.array_equal(stupid_version(t), is_first_occurrence(t))
    assert np.array_equal(is_first_occurrence(t.ravel()), is_first_occurrence_1d(t))

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

def test_cat_recarrays_on_columns_1():
    r = _get_sample_rec_array()
    cat = cat_recarrays_on_columns([r, r], ['i', 'o'])
    assert cat.dtype == [('i', '<i8'), ('o', '<f8')]

def test_cat_recarrays_on_columns_2():
    r = _get_sample_rec_array()
    r = r.reshape(10, -1)
    cat = cat_recarrays_on_columns([r, r], ['i', 'o'])
    assert cat.dtype == [('i', '<i8'), ('o', '<f8')]
    assert cat.ndim == 2
    assert cat.shape[0] == 20
    assert cat.shape[1] == r.shape[1]

def test_cat_recarrays_1():
    r = _get_sample_rec_array()
    r = r.reshape(10, -1)
    cat = cat_recarrays(r, r)
    assert cat.dtype == r.dtype
    assert cat.ndim == 2
    assert cat.shape[0] == 20
    assert cat.shape[1] == r.shape[1]
    assert np.array_equal(cat,
                          cat_recarrays_on_columns([r, r], r.dtype.names))

def test_cartesian_records_1():
    a = np.array([(1.,), (2.,), (3.,)], dtype=[('a', np.float)])
    b = np.array([(4,), (5,)], dtype=[('b', np.int)])
    c = np.array([(6,), (7,)], dtype=[('c', np.int)])

    r = cartesian_records([a, b, c])
    expected_dtype = [('a', np.float), ('b', np.int), ('c', np.int)]
    expected_result = np.array([(1., 4, 6),
                                (1., 4, 7),
                                (1., 5, 6),
                                (1., 5, 7),
                                (2., 4, 6),
                                (2., 4, 7),
                                (2., 5, 6),
                                (2., 5, 7),
                                (3., 4, 6),
                                (3., 4, 7),
                                (3., 5, 6),
                                (3., 5, 7)
                               ], dtype=expected_dtype)
    assert np.array_equal(r, expected_result)

def test_rec_inner_join_on_one_input():
    a = _get_sample_rec_array()
    r = rec_inner_join('m', a)
    for i in 'imnop':
        assert np.array_equal(np.sort(a[i]), np.sort(r[i]))

def test_rec_inner_join_1():
    a = np.array([('x', 1.), ('x', 2.), ('y', 3.)], dtype=[('s', 'S20'), ('a', np.float)])
    b = np.array([('x', 4), ('y', 5), ('x', 0)], dtype=[('s', 'S20'), ('b', np.int)])
    c = np.array([(6, 'x'), (7, 'y'), (9, 'z')], dtype=[('c', np.int), ('s', 'S20')])

    r = rec_inner_join('s', a, b, c)

    expected_dtype = [('s', 'S20'), ('a', np.float), ('b', np.int), ('c', np.int)]
    expected_result = np.array([('x', 1., 4, 6),
                                ('x', 1., 0, 6),
                                ('x', 2., 4, 6),
                                ('x', 2., 0, 6),
                                ('y', 3., 5, 7),
                               ], dtype=expected_dtype)

    assert np.array_equal(r, expected_result)

if __name__ == '__main__':
    test_multi_where_1d_1()
    test_multi_where_1d_2()
    test_true_where_1()
    test_true_where_2()
    test_split_at_boundaries_1()
    test_np_groupby_1()
    test_np_groupby_2()
    test_np_groupby_3()
    test_np_groupby_4()
    test_np_groupby_5()
    test_group_transform_1()
    test_group_transform_2()
    test_group_transform_3()
    test_rec_groupby_1()
    test_rec_groupby_2()
    test_rec_groupby_3()
    test_rec_groupby_4()
    test_np_and_rec_groupby_full_1()
    test_rec_groupby_full_2()
    test_is_first_occurrence_1()
    test_is_first_occurrence_1d_1()
    test_is_first_occurrence_2()
    test_is_first_occurrence_and_1d_3()
    test_get_first_indices_1()
    test_cat_recarrays_on_columns_1()
    test_cat_recarrays_on_columns_2()
    test_cat_recarrays_1()
    test_cartesian_records_1()
    test_rec_inner_join_on_one_input()
    test_rec_inner_join_1()
