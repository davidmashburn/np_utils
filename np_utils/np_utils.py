#!/usr/bin/env python
'''Utilities for array and list manipulation by David Mashburn.
Notably:
    shape_multiply, shape_multiply_zero_fill ->
        functions to scale arrays by integer multiples (without interpolation)
    
    polyArea -> get polygon area from points
    
    MORE HERE
    
    '''

import numpy as np

from gen_utils import islistlike
from list_utils import totuple, flatten, zipflat, assertSameAndCondense, split_at_boundaries

one = np.array(1) # a surprisingly useful little array; makes lists into arrays by simply one*[[1,2],[3,6],...]

def haselement(arr,subarr):
    '''Test if subarr is equal to one of the elements of arr.
       This is the equivalent of the "in" operator when using lists instead of arrays.'''
    arr = np.asarray(arr)
    subarr = np.asarray(subarr)
    if subarr.shape!=arr.shape[1:]:
        return False
    elif arr.ndim<2:
        return (subarr==arr).any()
    else:
        boolArr = (subarr==arr)
        boolArr.resize([arr.shape[0],np.prod(arr.shape[1:])])
        return boolArr.all(axis=1).any()

def np_has_duplicates(arr):
    '''For a 1D array, test whether there are any duplicated elements.'''
    return len(arr) > len(np.unique(arr))

def multidot(*args):
    '''Multiply multiple arguments with np.dot:
       reduce(np.dot,args,1)'''
    return reduce(np.dot,args,1)

def map_along_axis(f, axis, arr):
    '''Apply a function to a specific axis of an array
       This is slightly different from np.apply_along_axis when used
       in more than 2 dimensions.
       apply_along_axis applies the function to the 1D arrays which are associated with that axis
       map_along_axis transposes the original array so that that dimension is first
       and then applies the function to each entire (N-1)D array
       
       Example:
       >>> arr = np.arange(8).reshape([2,2,2])
       >>> arr
       array([[[0, 1],
               [2, 3]],
              [[4, 5],
               [6, 7]]])
       >>> np.apply_along_axis(np.sum, 1, arr)
       array([[ 2,  4],
              [10, 12]])
       >>> map_along_axis(np.sum, 1, arr)
       array([10, 18])
    '''
    arr = np.asanyarray(arr)
    new_dim_order = [axis] + range(axis) + range(axis+1,arr.ndim)
    return np.array([f(a) for a in arr.transpose(new_dim_order)])

def fields_view(arr, fields):
    '''Select fields from a record array without a copy
       Taken from:
       http://stackoverflow.com/questions/15182381/how-to-return-a-view-of-several-columns-in-numpy-structured-array
       '''
    fields = fields if islistlike(fields) else [fields]
    newdtype = np.dtype({name: arr.dtype.fields[name] for name in fields})
    return np.ndarray(arr.shape, newdtype, arr, 0, arr.strides)

def _split_records(arr):
    '''Split out individal arrays from records as a list of views'''
    arr = np.asanyarray(arr)
    if arr.dtype.names is None:
        return [arr]
    else:
        return [arr[name] for name in arr.dtype.names]

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
       
       The algorithm is as follows:
       * Form a list of the unique "keys"
       * Replace every value in the array with an index into the unique keys, "inv"
       (keys and inv are calculated using the exact same technique as np.unique)
       * Determine the locations where the sorted array changes values (split_points)
       * Argsort "inv" to cluster the indices of arr into groups and
         split these groups at the split points
         These indices will then be indices for the original values as well since
         the positions of elements in "inv" represent values in "arr"

       Note: The reason for not just calling "np.unique" is that we can reuse
             "flag" (isdiff here) to calculate the split_points directly and
             avoid calling np.bincount and then np.cumsum on inv'''

    arr = np.asanyarray(arr).flatten()                                  # flat version of the array
    sort_ind = arr.argsort()#kind='mergesort')                          # The indices of the array rearranged
    sort_arr = arr[sort_ind]                                            # such that arr[sort_ind] is the sorted array
    isdiff = np.concatenate(([True], sort_arr[1:] != sort_arr[:-1]))    # True if each element is different from its lefthand neighbor, False if it is the same
    keys = sort_arr[isdiff]                                             # The unique values in the array
    sorted_key_inds = np.cumsum(isdiff) - 1                             # A list the unique value's indices in keys (so just 0-n) repeated the number of times if occurs in the array
    inv = np.empty(arr.shape, dtype=np.intp)                            # The inverse mapping from unique to arr -- a list
    inv[sort_ind] = sorted_key_inds                                     # of the indices of uniq such that keys[inv] == arr
    split_points = np.nonzero(isdiff)[0]                                # the locations where the array has change points
    index_groups = split_at_boundaries(np.argsort(inv), split_points)   # cluster indices of arr into groups based on keys and then split index groups according to split points
    return keys, index_groups

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
    funs, fields_list, names = zip(*fun_fields_name)
    functions = [_outfielder(fun, fields)
                 for fun, fields, name in fun_fields_name]
    names = [i[-1] for i in fun_fields_name]
    return np_groupby(keyarr, a, *functions, names=keynames + names)

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
    
    keys, index_groups = get_index_groups(arr)
    first_inds = dict(zip(keys, [i[0] for i in index_groups]))
    inds = map(first_inds.get, values)
    
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

def limitInteriorPoints(l,numInteriorPoints,uniqueOnly=True):
    '''return the list l with only the endpoints and a few interior points (uniqueOnly will duplicate if too few points)'''
    inds = np.linspace(0,len(l)-1,numInteriorPoints+2).round().astype(np.integer)
    if uniqueOnly:
        inds = np.unique(inds)
    return [ l[i] for i in inds ]

# The active version of this function is defined below (this version is broken/ possibly never finished?)
#def limitInteriorPointsInterpolatingBAD(l,numInteriorPoints):
#    '''Like limitInteriorPoints, but interpolates evenly instead; this also means it never clips'''
#    l=np.array(l)
#    if l.ndim==1:
#        l=l[None,:]
#    return [ [ np.interp(ind, range(len(l)), i)
#              for i in l.transpose() ]
#            for ind in np.linspace(0,len(sc),numInteriorPoints+2) ]

def linrange(start, step, length):
    '''Includes "length" points, including "start", each separated by "step".
       (a hybrid between linspace and arange)'''
    return start + np.arange(length) * step
    
#def linrange_OLD(start, step, length):
#    return np.arange(start, start + step * (length - 0.5), step) # More efficient, but more complicated too

def partitionNumpy(l,n):
    '''Like partition, but always clips and returns array, not list'''
    a=np.array(l)
    a.resize(len(l)//n,n)
    return a

def shapeShift(arr,newShape,offset=None,fillValue=0):
    '''Create a new array with a specified shape and element value
       and paste another array into it with an optional offset.
       In 2D image processing, this like changing the canvas size and
       then moving the image in x and y.
       
       In the simple case of expanding the shape of an array, this is
       equivalent to the following standard procedure:
         newArray = zeros(shape)
         newArray[:arr.shape[0],:arr.shape[1],...] = arr
       
       However, shapeShift is more flexible because it can safely
       clip for any shape and any offset (but using it just for cropping
       an array is more efficiently done with slicing).
       
       A more accurate name for this might be "copyDataToArrayOfNewSize", but
       "shapeShift" is much easier to remember (and cooler).
       '''
    oldArr = np.asarray(arr)
    newArr = np.zeros(newShape,dtype=oldArr.dtype)+fillValue
    oldShape,newShape = np.array(oldArr.shape), np.array(newArr.shape)
    offset = ( 0*oldShape if offset==None else np.array(offset) )
    
    assert len(oldShape)==len(newShape)==len(offset)
    
    oldStartEnd = np.transpose([ np.clip(i-offset,0,oldShape) for i in [0,newShape] ])
    newStartEnd = np.transpose([ np.clip(i+offset,0,newShape) for i in [0,oldShape] ])
    oldSlice = [ slice(start,end) for start,end in oldStartEnd ]
    newSlice = [ slice(start,end) for start,end in newStartEnd ]
    newArr[newSlice] = oldArr[oldSlice]
    return newArr

def addBorder(arr,borderValue=0,borderThickness=1,axis=None):
    '''For an ND array, create a new array with a border of specified
       value around the entire original array; optional thickness (default 1)'''
    # a slice of None or a single int index; handles negative index cases
    allOrOne = ( slice(None) if axis==None else
               ( axis        if axis>=0 else 
                 arr.ndim+axis ))
    # list of border thickness around each dimension (as an array)
    tArr = np.zeros(arr.ndim)
    tArr[allOrOne] = borderThickness
    # a new array stretched to accommodate the border; fill with border Value
    arrNew = np.empty( 2*tArr+arr.shape, dtype=arr.dtype )
    arrNew[:]=borderValue
    # a slice object that cuts out the new border
    if axis==None:
        sl = [slice(borderThickness,-borderThickness)]*arr.ndim
    else:
        sl = [slice(None)]*arr.ndim
        sl[axis] = slice(borderThickness,-borderThickness)
    # use the slice object to set the interior of the new array to the old array
    arrNew[sl] = arr
    return arrNew

def shape_multiply(arr, shapeMultiplier, oddOnly=False, adjustFunction=None):
    '''Works like tile except that it keeps all like elements clumped
       Essentially a non-interpolating, multi-dimensional image up-scaler'''
    # really only 7 lines without checks...
    arr=np.asarray(arr)
    
    sh = arr.shape
    ndim = arr.ndim # dimensional depth of the array
    shm = ( shapeMultiplier if hasattr(shapeMultiplier,'__len__') else
           [shapeMultiplier] )
    
    if not len(shm)==len(arr.shape):
        print 'Length of shapeMultipler must be the same as the array shape!'
        return
    if not sum([i>0 for i in shm])==len(arr.shape):
        print 'All elements of shapeMultiplier must be integers greater than 0!'
        return
    if oddOnly:
        if not sum([i%2==1 for i in shm])==ndim:
            print 'All elements of shapeMultiplier must be odd integers greater than 0!'
            return
    
    t=np.tile(arr,shm)
    t.shape = zipflat(shm,sh)
    t = t.transpose(*zipflat(range(1,ndim*2,2),range(0,ndim*2,2)))
    
    if adjustFunction!=None:
        t=adjustFunction(t,arr,shm)
    
    return t.reshape(*[sh[i]*shm[i] for i in range(ndim)])

def shape_multiply_zero_fill(arr,shapeMultiplier):
    '''Same as shape_muliply, but requires odd values for\n'''
    '''shapeMultiplier and fills around original element with zeros.'''
    def zeroFill(t,arr,shm):
        ndim=arr.ndim
        t*=0
        s = [slice(None,None,None)]*ndim , [i//2 for i in shm] # construct a slice for the middle
        t[zipflat(*s)]=arr
        return t
        # This is a nice idea, but it doesn't work very easily...
        #c = np.zeros(shm,arr.dtype)
        #c[tuple([(i//2) for i in shm])]=1
    return shape_multiply(arr,shapeMultiplier,oddOnly=True,adjustFunction=zeroFill)

def reshape_repeating(arr, new_shape):
    """A forgiving version of np.reshape that allows the resulting
    array size to be larger or smaller than the original
    When the new size is larger, values are filled by repeating the original (flattened) array
    A smaller or equal size always returns a view.
    A larger size always returns a copy.
    """
    arr_flat = arr.reshape(arr.size)
    new_shape = (new_shape,) if not hasattr(new_shape, '__iter__') else new_shape
    new_size = np.prod(new_shape)
    if new_size <= arr.size:
        return arr_flat[:new_size].reshape(new_shape)
    else:
        repeats = np.ceil(new_size / arr.size)
        s = np.lib.stride_tricks.as_strided(arr_flat, (repeats, arr.size), (0, arr.itemsize))
        assert arr_flat.base is arr
        assert s.base.base is arr_flat
        assert s.flat.base is s
        f = s.flat[:new_size] # This final slicing op is the first time a new array is created... hmmm
        assert f.base is None # poo
        r = f.reshape(new_shape)
        assert r.base is f
        return r

def interpNaNs(a):
    '''Changes the underlying array to get rid of NaN values
       If you want a copy instead, just use interpNaNs(a.copy())
       Adapted from http://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array'''
    nans = np.isnan(a)
    nz_1st = lambda z: z.nonzero()[0]
    a[nans] = np.interp( nz_1st(nans), nz_1st(~nans), a[~nans] )
    return a

# A couple working but not-that-great interpolation functions:
def interpNumpy(l,index):
    '''Just like interp except that it uses numpy instead of lists.
       If both l and index are integer, this function will also return integer values.
       The more canonical way to this operation would be:
           np.interp(index,range(len(l)),l)
       (where l is already an array)'''
    l = np.asarray(l)
    m = index % 1
    if m==0:
        return l[index]
    else:
        indexA,indexB = int(index), int(index) + (1 if index>=0 else -1)
        return l[indexA]*(1-m) + l[indexB]*(m)

def interpolatePlane(arr,floatIndex,axis=0):
    '''Interpolate a hyperplane in an numpy array (basically just like normal indexing but with floats)
       And yes, I know that scipy.interpolate would do a much better job...'''
    assert axis<len(arr.shape),'Not enough dimensions in the array'
    maxInd = arr.shape[axis]
    assert 0<=floatIndex<=maxInd-1,'floatIndex ('+str(floatIndex)+') is out of bounds '+str([0,maxInd-1])+'!'
    
    slicer = ( slice(None), )*axis
    ind = int(floatIndex)
    rem = floatIndex-ind
    indPlane = arr[slicer+(ind,)]
    indp1Plane = (arr[slicer+(ind+1,)] if ind+1<arr.shape[axis] else 0)
    plane = indPlane*(1-rem) + indp1Plane*rem
    return plane

def interpolateSumBelowAbove(arr,floatIndex,axis=0):
    '''For any given index, interpolate the sum below and above this value'''
    assert axis<len(arr.shape),'Not enough dimensions in the array'
    maxInd = arr.shape[axis]
    assert 0<=floatIndex<=maxInd,'floatIndex ('+str(floatIndex)+') is out of bounds ('+str([0,maxInd])+')!'
    
    slicer = ( slice(None), )*axis
    ind = int(floatIndex)
    rem = floatIndex-ind
    indPlane = (arr[slicer+(ind,)] if ind<arr.shape[axis] else 0)
    indp1Plane = (arr[slicer+(ind+1,)] if ind+1<arr.shape[axis] else 0)
    below = np.sum(arr[slicer+(slice(None,ind),)],axis=axis)   + indPlane*rem
    above = indp1Plane + np.sum(arr[slicer+(slice(ind+2,None),)],axis=axis) + indPlane*(1-rem)
    return below,above

def limitInteriorPointsInterpolating(l,numInteriorPoints):
    '''Like limitInteriorPoints, but interpolates evenly instead; this also means it never clips'''
    l=np.asarray(l)
    return [ interpNumpy(l,ind)
            for ind in np.linspace(0,len(l)-1,numInteriorPoints+2) ]

def getValuesAroundPointInArray(arr,point,wrapX=False,wrapY=False):
    '''Given any junction in a 2D array, get the set of unique values that surround it:
       Junctions always have 4 pixels around them, and there is one more junction
          in each direction than there are pixels (but one less internal junction)
       The "point" arguments must be an internal junction unless wrapX / wrapY
          are specified; returns None otherwise.'''
    s = arr.shape
    x,y = point
    xi,xf = ( (x-1,x) if not wrapX else ((x-1)%s[0],x%s[0]) )
    yi,yf = ( (y-1,y) if not wrapY else ((y-1)%s[1],y%s[1]) )
    if ( 0<x<s[0] or wrapX ) and ( 0<y<s[1] or wrapY ):
        return np.unique( arr[([xi,xf,xi,xf],[yi,yi,yf,yf])] )
    else:
        print "Point must be interior to the array!"
        return None

def linearTransform( points,newRefLine,mirrored=False ):
    '''Transforms points from a (0,0) -> (1,0) reference coordinate system
       to a (x1,y1) -> (x2,y2) coordinate system.
       "points" can also be a single point.
       Optional mirroring with "mirrored" argument.'''
    points,newRefLine = np.asarray(points), np.asarray(newRefLine)
    dx,dy = newRefLine[1] - newRefLine[0]
    mir = -1 if mirrored else 1
    mat = [[ dx      , dy      ],
           [-dy * mir, dx * mir]]
    return np.dot(points, mat) + newRefLine[0]

def reverseLinearTransform(points,oldRefLine,mirrored=False):
    '''Transforms points from a (x1,y1) -> (x2,y2) reference coordinate system
       to a (0,0) -> (1,0) coordinate system.
       "points" can also be a single point.
       Optional mirroring the "mirrored" argument.'''
    points, oldRefLine = np.asarray(points), np.asarray(oldRefLine)
    dx, dy = oldRefLine[1] - oldRefLine[0]
    points = points - oldRefLine[0]
    mir = -1 if mirrored else 1
    matT = [[dx,-dy * mir],
           [dy, dx * mir]]
    dx2dy2 = np.square(dx) + np.square(dy)
    return np.dot(points, matT) * 1. / dx2dy2

def FindOptimalScaleAndTranslationBetweenPointsAndReference(points,pointsRef):
    '''Find the (non-rotational) transformation that best overlaps points and pointsRef
       aka, minimize the distance between:
       (xref[i],yref[i],...)
       and
       (a*x[i]+x0,a*y[i]+y0,...)
       using linear least squares
       
       return the transformation parameters: a,(x0,y0,...)'''
    # Force to array of floats:
    points = np.asarray(points,dtype=np.float)
    pointsRef = np.asarray(pointsRef,dtype=np.float)

    # Compute some means:
    pm     = points.mean(axis=0)
    pm2    = np.square(pm)
    prefm  = pointsRef.mean(axis=0)
    p2m    = np.square(points).mean(axis=0)
    pTpref = (points * pointsRef).mean(axis=0)
    
    a = ((   (pm*prefm).sum() - pTpref.sum()   ) /
         #   -------------------------------     # fake fraction bar...
         (        pm2.sum() - p2m.sum()        ))
    p0 = prefm - a*pm
    return a,p0
    
    # More traditional version (2 dimensions only):
    # xm,ym = pm
    # xrefm,yrefm = prefm
    # x2m,y2m = p2m
    # xTxref,yTyref = pTpref
    # a = 1.*(xm*xrefm - xCxref + ym*yrefm - yCyref) / (xm**2 - x2m + ym**2 - y2m)
    # x0,y0 = xrefm - a*xm,yrefm - a*ym

def polyArea(points):
    '''This calculates the area of a general 2D polygon
       Algorithm adapted from Darius Bacon's post:
       http://stackoverflow.com/questions/451426/how-do-i-calculate-the-surface-area-of-a-2d-polygon
       Updated to a numpy equivalent for speed'''
    # Get all pairs of points in (x0,x1),(y0,y1) format
    # Then compute the area as half the abs value of the sum of the cross product
    xPairs,yPairs = np.transpose([points,np.roll(points,-1,axis=0)])
    return 0.5 * np.abs(np.sum(np.cross(xPairs,yPairs)))
    
    # Old version
    #x0,y0 = np.array(points).T # force to numpy array and transpose
    #x1,y1 = np.roll(points,-1,axis=0).T
    #return 0.5*np.abs(np.sum( x0*y1 - x1*y0 ))
    
    # Non-numpy version:
    #return 0.5*abs(sum( x0*y1 - x1*y0
    #                   for ((x0, y0), (x1, y1)) in zip(points, roll(points)) ))

def polyCirculationDirection(points):
    '''This algorithm calculates the overall circulation direction of the points in a polygon.
       Returns 1 for counter-clockwise and -1 for clockwise
       Based on the above implementation of polyArea
         (circulation direction is just the sign of the signed area)'''
    xPairs,yPairs = np.transpose([points,np.roll(points,-1,axis=0)])
    return np.sign(np.sum(np.cross(xPairs,yPairs)))

def polyCentroid(points):
    '''Compute the centroid of a generic 2D polygon'''
    xyPairs = np.transpose([points,np.roll(points,-1,axis=0)]) # in (x0,x1),(y0,y1) format
    c = np.cross(*xyPairs)
    xySum = np.sum(xyPairs,axis=2) # (x0+x1),(y0+y1)
    xc,yc = np.sum( xySum*c, axis=1 )/(3.*np.sum(c))
    return xc,yc
    
    # Old version
    #x0,y0 = np.array(points).T # force to numpy array and transpose
    #x1,y1 = np.roll(x0,-1), np.roll(y0,-1)     # roll to get the following values
    #c = x0*y1-x1*y0                            # the cross-term
    #area6 = 3.*np.sum(c)                       # 6*area
    #x,y = np.sum((x0+x1)*c), np.sum((y0+y1)*c) # compute the main centroid calculation
    #return x/area6, y/area6
    
    # Non-numpy version:
    #x = sum( (x0+x1)*(x0*y1-x1*y0)
    #        for ((x0, y0), (x1, y1)) in zip(points, roll(points)) )
    #y = sum( (y0+y1)*(y0*x1-y1*x0)
    #        for ((x0, y0), (x1, y1)) in zip(points, roll(points)) )
    #return x/area6,y/area6
    
    # Single function version:
    #def _centrX(points):
    #    '''Compute the x-coordinate of the centroid
    #       To computer y, reverse the order of x and y in points'''
    #    return sum( (x0+x1)*(x0*y1-x1*y0)
    #               for ((x0, y0), (x1, y1)) in zip(points, roll(points)) )
    #return _centrX(points)/area6,_centrX([p[::-1] for p in points])/area6

def diffmean(x,*args,**kwds):
    '''Subtract off the mean from x'''
    x = np.asarray(x)
    return x - x.mean(*args,**kwds)

def sum_sqr_diff(a, b):
    '''Compute the sum of square differences between a and b:
          Sum      (a[i] - b[i]) ** 2
       i = 0 to n
    '''
    a, b = map(np.asanyarray, (a, b))
    return np.sum(np.square(a - b))

def sqrtSumSqr(x,axis=-1):
    '''Compute the sqrt of the sum of the squares'''
    return np.sqrt(np.sum(np.square(np.asanyarray(x)),axis=axis))

def sqrtMeanSqr(x,axis=-1):
    '''Compute the sqrt of the mean of the squares (RMS)'''
    return np.sqrt(np.mean(np.square(np.asanyarray(x)),axis=axis))

def vectorNorm(x,axis=-1):
    '''Normalize x by it's length (sqrt of the sum of the squares)
       x can also be a list of vectors'''
    x = np.asarray(x)
    sh = list(x.shape)
    sh[axis]=1
    return x/sqrtSumSqr(x,axis=axis).reshape(sh)

def pointDistance(point0,point1):
    '''Compute the distance between two points.
       point0 and point1 can also be lists of points'''
    return sqrtSumSqr(np.asarray(point1)-np.asarray(point0))

def polyPerimeter(points,closeLoop=True):
    '''This calculates the length of a (default closed) poly-line'''
    if closeLoop:
        points = np.concatenate([points,[points[0]]])
    return sum(pointDistance(points[1:],points[:-1]))

def build_grid(center, steps, nsteps):
    '''Build a meshgrid based on:
       * a center point,
       * a step size in each dimension, and
       * a number of steps in each dimension
       "steps" and "nsteps" can be single numbers
       but otherwise their dimensions must match "center"
       The output will be a list of ND arrays with shape equal to nsteps'''
    steps = steps if hasattr(steps, '__iter__') else [steps] * len(center)
    nsteps = nsteps if hasattr(nsteps, '__iter__') else [nsteps] * len(center)
    assertSameAndCondense(map(len, (center, steps, nsteps)),
                          'All the arguments must have the same length!')
    return np.meshgrid(*[np.linspace(c-(n-1.)/2 * s, c+(n-1.)/2 * s, n)
                         for c, s, n in zip(center, steps, nsteps)],
                       indexing='ij')

def reverse_broadcast(f):
    """Change a function to use "reverse broadcasting"
    which amounts to calling np.transpose on all arguments and again on the output.
    """
    newf = g_inv_f_g(f, np.transpose)
    newf.__doc__ = '\n'.join(['Transpose the arguments before and after running the function f:',
                             f.__doc__])
    return newf
