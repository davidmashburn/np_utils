#!/usr/bin/env python
'''Utilities for record array manipulation by David Mashburn.
Also covers features like indexing and grouping

Notable functions:

group_count ->
    Efficient element counting (but safe/ more flexible than np.bincount)

get_index_groups ->
    Efficient, generalized grouping operation for arrays (1D)
    Produces aligned keys (unique values)
    and index_groups (array of locations for each key in the original array)
    Essentially like an efficient version of:
        [np.where(arr==i) for i in np.unique(arr)])
    Based on code from np.unique

np_groupby ->
    Group-by operation using get_index_groups

rec_groupby ->
    Similar to matplotlib's rec_groupby but much more efficient

np_groupby_full, rec_groupby_full ->
    Group by for non-reducing operations

get_array_subgroups ->
    Form a dictionary of subgroups of an array based on a different grouping array

fields_view ->
    Efficient view of only certain fields in a record array

find_first_occurrence, find_first_occurrence_1d, is_first_occurrence,
is_first_occurrence_1d, get_first_indices ->
    Functions to determine the first occurrence (index)
    of each unique value in an array (ND or 1D)
'''

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from builtins import map, zip
from future.utils import lmap

import numpy as np

from .gen_utils import islistlike
from .list_utils import split_at_boundaries

def _get_index_groups_old(arr):
    '''For a 1D array, get all unique values (keys)
        and the locations of each value in the original array (index_groups).
        keys and index_groups are aligned so that dict(zip(keys, index_groups))
        creates a dictionary that maps from each unique value in arr
        to a list of all the locations for that value.
        index_groups can be thought of as a much more efficient variant of:
            [np.where(arr==i) for i in np.unique(arr)]
        A simple count can be achieved by:
            keys, index_groups = get_index_groups(arr)
            counts = map(len, index_groups)
        which would be equivalent to this pseudo-sql:
            select count(x) from arr group by x
        '''
    keys, inv = np.unique(arr, return_inverse=True)
    split_points = np.cumsum(np.bincount(inv)[:-1])
    index_groups = np.split(np.argsort(inv), split_points)
    return keys, index_groups

def _index_helper(arr, sort_arr):
    '''Compute some basic things needed for indexing functions below
       Inputs
       arr:      Any flat array
       sort_arr: Sorted version of arr
       
       Returns
       isdiff:       Boolean array with same length as arr
                     True if each element is different from its lefthand
                     False if it is the same
                     (called "flag" in np.unique)
       
       keys:         Unique values of arr (sorted)
       
       split_points: Locations where the isdiff is True
                     (or where the sorted array has change points)
                     This is what makes it possible to determine the
                     size of each group'''
    isdiff = np.concatenate(([True], sort_arr[1:] != sort_arr[:-1]))
    keys = sort_arr[isdiff]
    split_points = np.flatnonzero(isdiff)
    return isdiff, keys, split_points

def group_count(arr, use_argsort=False):
    '''Get the count of all the groups in the array
       This is much faster than what could be acheived using the tools below,
       i.e. map(len, get_index_groups(arr)[0])'''
    arr = np.ravel(arr)
    sort_arr = np.sort(arr)
    isdiff, keys, split_points = _index_helper(arr, sort_arr)
    counts = np.diff(np.concatenate((split_points, [arr.size])))
    return keys, counts

def get_index_groups(arr):
    '''For a 1D array, get all unique values (keys)
       and the locations of each value in the original array (index_groups).
       keys and index_groups are aligned so that dict(zip(keys, index_groups))
       creates a dictionary that maps from each unique value in arr
       to a list of all the locations for that value.
       
       "index_groups" can be thought of as a much more efficient variant of:
           [np.where(arr==i) for i in np.unique(arr)]
       Example: a group by "count" can be achieved by:
           keys, index_groups = get_index_groups(arr)
           counts = map(len, index_groups)
       which would be equivalent to this pseudo-sql:
           select count(x) from arr group by x
       
       The algorithm can be summarized as follows:
       * Form a list of unique values (keys)
       
       * Find the locations where the sorted array changes values (split_points)
       
       * Replace every value in arr with an index into the unique keys (inv)
         - keys and inv are calculated in the exact same way as in np.unique
       
       * Argsort "inv" to cluster the indices of arr into groups and
         split these groups at the split points
         These indices will then be indices for the original values as well since
         the positions of elements in "inv" represent values in "arr"

       Note: The reason for using _index_base (above) instead of "np.unique"
             is that we can reuse "flag" (isdiff here) to calculate the
             split_points directly and avoid calling np.bincount and
             then np.cumsum on inv
       
       Internal variable details:
       sort_ind:        Argsort of arr -- indices of the array rearranged such
                                          that arr[sort_ind] == np.sort(arr)
       
       sort_arr:        Sorted version of arr
       
       sorted_key_inds: A list the unique value's indices in keys (so just 0-n)
                        repeated the number of times if occurs in the array
       
       inv:             The inverse mapping from unique to arr, an array
                        of the indices of keys such that keys[inv] == arr
                        (same as the optional return value from np.unique)
                        
                        Note that np.argsort(inv) gives the indices of arr
                        sorted into groups based on keys; a 1d array where
                        indices in the same group are "clustered" contiguously
       
       index_groups:    The indices of arr grouped into list of arrays
                        where each group (array) matches a specific key
                        This property will then be true:
                        np.all(arr[index_groups[i]] == keys[i])
       
    '''
    arr = np.ravel(arr)
    sort_ind = np.argsort(arr) #,kind='mergesort')
    sort_arr = arr[sort_ind]
    isdiff, keys, split_points = _index_helper(arr, sort_arr)
    sorted_key_inds = np.cumsum(isdiff) - 1
    inv = np.empty(arr.shape, dtype=np.intp)
    inv[sort_ind] = sorted_key_inds
    index_groups = split_at_boundaries(np.argsort(inv), split_points)
    return keys, index_groups

def get_array_subgroups(arr, grouping_arr):
    '''Form a dictionary of subgroups of arr
       where keys are members of grouping_arr
       (grouping_arr is usually something like arr[field])
       and values are rows where grouping_arr matches each key

       A typical usage would be for a record array with a column like 'foreign_id':
       d = get_array_subgroups(arr, arr['foreign_id'])
       Then d[id] will be a subarray of arr where arr['foreign_id']==id'''
    assert len(arr) == len(grouping_arr), 'Arrays must have the same length!'
    keys, index_groups = get_index_groups(grouping_arr)
    return {k: arr[inds] for k, inds in zip(keys, index_groups)}

def _split_records(arr):
    '''Split out individal arrays from records as a list of views'''
    arr = np.asanyarray(arr)
    if arr.dtype.names is None:
        return [arr]
    else:
        return [arr[name] for name in arr.dtype.names]

def np_groupby(keyarr, arr, *functions, **kwds):
    '''A really simple, relatively fast groupby for numpy arrays.
       Takes two arrays of the same length and a function:
         keyarr: array used to generate the groups (argument to np.unique)
         arr: array used to compute the metric
         functions: functions that computes the metrics
         names (optional): names for each column in the resulting record array
       This applies f to groups of values from arr where values in keyarr are the same
       
       Returns a 2-column array dictionary with keys taken from the keyfun(a)
       
       Example use case for a record array with columns 'i' and 'j':
           np_groupby(a['i'], a['j'], np.max)
       In psqudo-sql, this would be:
           select i, max(j) from a groupby i
       Multiple columns can also be used.
       For instance, if there are fields 'm', 'n', 'o', 'p',
       we could write:
           np_groupby(a[['m', 'n']], a,
                     lambda x: np.mean(x['o']),
                     lambda x: np.std(x['o']),
                     lambda x: np.min(x['p']),
                     names=['m', 'n', 'mean_o', 'std_o', 'min_p'])
       In psqudo-sql, this would be:
           select m,n,mean(o),std(o),min(p) from a groupby m,n
       
       We could also easily use a compound function like this:
           def compute_some_thing(x):
               o, p = x['o'], x['p']
               return np.mean(o) / np.std(o) * np.min(p)
           np_groupby(a[['m', 'n']], a, compute_some_thing,
                     names=['m', 'n', 'its_complicated'])
       
       There are more memory and time efficient ways to do this in special
       cases, but this is flexible for any functions and gets the job done
       (in 4 lines).
       
       Other great options for groupby include:
           pandas
           numpy_groupies
           matplotlib.mlab.rec_groupby
           itertools.groupby
           groupByFunction in the list_utils sub-package here
           np_unique + np.bincount (handle with care)
       
       Adapted from multiple places, most notably:
       http://stackoverflow.com/questions/16856470/is-there-a-matlab-accumarray-equivalent-in-numpy
       http://stackoverflow.com/questions/8623047/group-by-max-or-min-in-a-numpy-array
       https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/mlab.py
       '''
    names = kwds.pop('names', None)
    keys, index_groups = get_index_groups(keyarr)
    groups = [np.fromiter((f(arr[i]) for i in index_groups), dtype=None, count=len(keys))
              for f in functions]
    return np.rec.fromarrays(_split_records(keys) + groups, names=names)

def _group_transform(arr, index_groups, fun, result_dtype):
    '''Helper function for group_transform'''
    result = np.empty(len(arr), dtype=result_dtype)
    for g in index_groups:
        result[g] = fun(arr[g])
    
    return result

def group_transform(keyarr, arr, fun, result_dtype):
    '''Perform a non-reducing operation that acts on groups
       This results in a new array the same length as the original
       (rank, normalize, sort, etc.)
       
       Examples:
       (assuming arr has fields 'm', 'n', 'o', 'p')
       
       # Sort items based on 'o' in groups based on 'm':
       sorted_o = group_transform(arr['m'], arr['o'], np.sort, np.float)
       
       # Rank items based on 'o' and 'p' in groups based on 'm':
       simple_rank = lambda x: np.argsort(x) + 1
       ranked_op = group_transform(arr['m'], arr[['o', 'p']], simple_rank, np.int)
       
       # Subtract the group mean (background) for 'p' in groups based on 'm':
       background_subtract = lambda x: x - x.mean()
       bg_removed_p = group_transform(arr['m'], arr['p'], background_subtract, np.float)
       
       # Normalize groups (divide by mean) for 'o' in groups based on 'm' and 'n':
       simple_normalize = lambda x: x / x.mean()
       normalized_o = group_transform(arr[['m', 'n']], arr['o'], simple_normalize, np.float)
       '''
    keys, index_groups = get_index_groups(keyarr)
    return _group_transform(arr, index_groups, fun, result_dtype)

def np_groupby_full(keyarr, arr, *functions_result_dtypes, **kwds):
    '''Special case of np_groupby where the end result is an array the
       same length as the original (rank, normalize, etc.)
       
       Example:
       
       simple_rank = lambda x: np.argsort(x) + 1
       background_subtract = lambda x: x - x.mean()
       simple_normalize = lambda x: x / x.mean()
   
       result = np_groupby_full(arr[['m', 'n']], arr,
           (lambda x: simple_rank(x['o']), np.int),
           (lambda x: simple_rank(x[['o', 'p']]), np.int),
           (lambda x: background_subtract(x['o']), np.float),
           (lambda x: simple_normalize(x['p']), np.float),
           names=['m', 'n', 'rank_o', 'rank_op', 'bg_sub_o', 'norm_p'])
       '''
    names = kwds.pop('names', None)
    keys, index_groups = get_index_groups(keyarr)
    results = [_group_transform(arr, index_groups, fun, result_dtype)
               for fun, result_dtype in functions_result_dtypes]
    return np.rec.fromarrays(_split_records(arr) + results, names=names)

def fields_view(arr, fields):
    '''Select fields from a record array without a copy
       Taken from:
       http://stackoverflow.com/questions/15182381/how-to-return-a-view-of-several-columns-in-numpy-structured-array
       '''
    fields = fields if islistlike(fields) else [fields]
    newdtype = np.dtype({name: arr.dtype.fields[name] for name in fields})
    return np.ndarray(arr.shape, newdtype, arr, 0, arr.strides)

def _outfielder(fun, fields):
    '''Helper function to generate a function that takes an array as the argument'''
    def f(arr):
        return fun(arr[fields]) # This does not work here: fields_view(arr, fields)
    return f

def rec_groupby(a, keynames, *fun_fields_name):
    '''A special version of np_groupby for record arrays, somewhat similar
       to the function found in matplotlib.mlab.rec_groupby.
       
       This is basically a wrapper around np_groupy that automatically
       generates lambda's like the ones in the np_groupby doc string.
       That same call would look like this using rec_grouby:
       
       rec_groupby(a, ['m', 'n'], (np.mean, 'o', 'mean_o'),
                                  (np.std, 'o', 'std_o'),
                                  (np.min, 'p', 'min_p'))
       and the second function could be written as:
           def compute_some_thing(x):
               o, p = x['o'], x['p']
               return np.mean(o) / np.std(o) * np.min(p)
           rec_groupby(a, ['m', 'n'],
                       (compute_some_thing, ['o', 'p'], 'its_complicated'))
       
       In general, this function is faster than matplotlib.mlab, but not
       as fast as pandas and probably misses some corner cases for each :)
       '''
    keynames = list(keynames) if islistlike(keynames) else [keynames]
    keyarr = fields_view(a, keynames)
    functions = [_outfielder(fun, fields)
                 for fun, fields, name in fun_fields_name]
    names = [i[-1] for i in fun_fields_name]
    return np_groupby(keyarr, a, *functions, names=(keynames + names))

def rec_groupby_full(a, keynames, *fun_dtype_fields_name):
    '''A special version of np_groupby for record arrays, somewhat similar
       to the function found in matplotlib.mlab.rec_groupby.
       
       This is basically a wrapper around np_groupy_full that automatically
       generates lambda's like the ones in the np_groupby_full doc string.
       That same call would look like this using rec_grouby_full:
       
       simple_rank = lambda x: np.argsort(x) + 1
       background_subtract = lambda x: x - x.mean()
       simple_normalize = lambda x: x / x.mean()
       
       rec_groupby_full(a, ['m', 'n'],
           (simple_rank,         np.int,   'o',        'rank_o'),
           (simple_rank,         np.int,   ['o', 'p'], 'rank_op'),
           (background_subtract, np.float, 'o',        'bg_sub_o'),
           (simple_normalize,    np.float, 'p',        'norm_p')
       )
       
       
       In general, this function is faster than matplotlib.mlab, but not
       as fast as pandas and probably misses some corner cases for each :)
       '''
    keynames = list(keynames) if islistlike(keynames) else [keynames]
    keyarr = fields_view(a, keynames)
    functions_result_dtypes = [(_outfielder(fun, fields), dtype)
                               for fun, dtype, fields, name in fun_dtype_fields_name]
    names = [i[-1] for i in fun_dtype_fields_name]
    return np_groupby_full(keyarr, a, *functions_result_dtypes, names=(keynames + names))

def find_first_occurrence(arr):
    '''Find the first occurrence (index) of each unique value (subarray) in arr.
       Modified version of code found here:
       http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order
       Relies on hashing the string of the subarray, which *should* be fine...
       For non-array iterables, use list_utils.find_first_occurrence_in_list'''
    arr = np.asanyarray(arr)
    seen = set()
    return np.array([i for i, a in enumerate(arr)
                     for s in (a.tostring(),)     # with s as a.tostring()   <-- would be this if python supported it
                     if s not in seen and not seen.add(s)])

def find_first_occurrence_1d(arr, get_keys=True):
    '''Equivalent to find_first_occurrence(arr.ravel()), but should be much faster
       (uses the very fast get_index_groups function)'''
    keys, index_groups = get_index_groups(arr)
    first_occurrences = lmap(np.min, index_groups)
    return (keys, first_occurrences) if get_keys else first_occurrences

def true_where(shape, true_locs):
    """Return a new boolean array with the values at locations
       "true_inds" set to True, and all others False
       This function is the inverse of np.where for boolean arrays:
       true_where(bool_arr.shape, np.where(bool_arr)) == bool_arr
       or in general:
       np.where(true_where(arr.shape, np.where(arr))) == np.where(arr)"""
    bool_arr = np.zeros(shape, dtype=np.bool)
    bool_arr[true_locs] = True
    return bool_arr

def is_first_occurrence(arr):
    '''For each sub-array arr, whether or not this is equivalent to another
       subarray with a lower index'''
    return true_where(len(arr), find_first_occurrence(arr))

def is_first_occurrence_1d(arr):
    '''Flattening version of is_first_occurrence (uses find_first_occurrence_1d)'''
    return true_where(np.size(arr), find_first_occurrence_1d(arr, get_keys=False))

def get_first_indices(arr, values, missing=None):
    '''Get the index of the first occurrence of the list of values in
       the (flattened) array
       
       The missing argument determines how missing values are handled:
       None: ignore them, leave them None
       -1: make them all -1
       'len': replace them with the length of the array (aka outside the array)
       'fail': throw an error'''
    bad_str = """Bad value for "missing", choose one of: None, -1, 'len', 'fail'"""
    assert missing in [None, -1, 'len', 'fail'], bad_str
    arr = np.asanyarray(arr)
    
    first_inds = dict(zip(*find_first_occurrence_1d(arr)))
    inds = lmap(first_inds.get, values)
    
    if missing == 'fail' and None in inds:
        raise Exception('Value Error! One of the values is not in arr')
    elif missing == 'len':
        default = arr.size
    elif missing == -1:
        default = -1
    else:
        default = None
    
    if default is not None:
        inds = [default if i is None else i for i in inds]
    
    return np.array(inds)

def merge_recarrays(arrays):
    '''Fast version of join for structured arrays
       from http://stackoverflow.com/questions/5355744/numpy-joining-structured-arrays
       (Similar to numpy.lib.recfunctions.merge_array)
       
       <CURRENTLY UNTESTED HERE>
    '''
    sizes = numpy.array([a.itemsize for a in arrays])
    offsets = numpy.r_[0, sizes.cumsum()]
    n = len(arrays[0])
    joint = numpy.empty((n, offsets[-1]), dtype=numpy.uint8)
    for a, size, offset in zip(arrays, sizes, offsets):
        joint[:,offset:offset+size] = a.view(numpy.uint8).reshape(n,size)
    dtype = sum((a.dtype.descr for a in arrays), [])
    return joint.ravel().view(dtype)
