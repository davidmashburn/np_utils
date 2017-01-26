#!/usr/bin/env python
'''Utilities for array and list manipulation by David Mashburn.
Notable functions by category:
    
    Simple utilities and array generators
    -------------------------------------
    multidot ->
        np.dot with multiple arguments
    linrange, build_grid ->
        alternatives to numpy builtins (arange, mgrid)
        with different parameter options
    true_where, np_has_duplicates, haselement ->
        functions to test arrays
    
    Transformations: reshaping, splitting, filtering
    ------------------------------------------------
    ravel_at_depth, reshape_smaller, reshape_repeating ->
        reshape helpers/ enhancements
    partitionNumpy ->
        like list_utils.partition for arrays
    shapeShift ->
        In one call, construct a new array of different shape with the same contents
    addBorder ->
        add a border aroung an ND-array
    shape_multiply, shape_multiply_zero_fill ->
        scale arrays by integer multiples (without interpolation)
    remove_duplicate_subarrays->
        efficient set-like operation for arrays -- like an order-perserving
        version of np.unique that deals with subarrays instead of elements
    
    Basic numerical calculations
    ----------------------
    diffmean, sum_sqr_diff, sqrtSumSqr, sqrtMeanSqr ->
        common combinations of numpy operations
    vectorNorm, pointDistance ->
        vector norm and euclidean distance
    
    Polygon metrics
    ---------------
    polyArea, polyCirculationDirection, polyCentroid, polyPerimeter ->
        basic metrics over polygons represented as arrays of (x, y) points
        (the final point is always assumed to connect to the first)
    
    Interpolation
    -------------
    interpNaNs ->
        Fill any NaN values with interpolated values (1D)
    linearTransform, reverseLinearTransform ->
        Linear transformations based on starting and ending points
    FindOptimalScaleAndTranslationBetweenPointsAndReference ->
        Helper function, wrapper around linear least squares
    
    Broadcasting
    ------------
    concatenate_broadcasting ->
        Broadcasting version of concatenate (alias broad_cat)
    reverse_broadcast ->
        change any function to broadcast its arguments using reversed shapes
        (matches leftmost dimensions instead of right)
    
    Indexing, grouping, and record arrays
    -------------------------------------
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
    get_array_subgroups ->
        Form a dictionary of subgroups of an array based on a different grouping array
    fields_view ->
        Efficient view of only certain fields in a record array
    find_first_occurrence, find_first_occurrence_1d, is_first_occurrence,
    is_first_occurrence_1d, get_first_indices ->
        Functions to determine the first occurrence (index)
        of each unique value in an array (ND or 1D)
    
    Boxing (nested arrays)
    ----------------------
    box, unbox -> Generate an ndarray of ndarrays (or vice-versa)
    
    Generalized map/apply
    ---------------------
    map_along_axis ->
        Map a function along any axis of an array
    apply_at_depth_ravel ->
        Apply a function that actions on 1D arrays at any depth in an array
    apply_at_depth ->
        Apply a function at any depth in an array
        Also works for functions with multiple arguments (different depths per argument)
    '''

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import str, map, zip, range

from future.utils import lmap, lrange

import numpy as np

from .gen_utils import islistlike
from .func_utils import g_inv_f_g
from .list_utils import (totuple, flatten, zipflat, all_equal,
                        split_at, split_at_boundaries, coerce_to_target_length
                       )

from distutils.version import StrictVersion
from functools import reduce
if StrictVersion(np.__version__) < StrictVersion('1.11.0'):
    from .np_future import broadcast_arrays
else:
    from numpy import broadcast_arrays

###########################################
## Simple utilities and array generators ##
###########################################

one = np.array(1) # a surprisingly useful little array; makes lists into arrays by simply one*[[1,2],[3,6],...]

def multidot(*args):
    '''Multiply multiple arguments with np.dot:
       reduce(np.dot, args, 1)'''
    return reduce(np.dot, args, 1)

def linrange(start, step, length):
    '''Includes "length" points, including "start", each separated by "step".
       (a hybrid between linspace and arange)'''
    return start + np.arange(length) * step
    
#def linrange_OLD(start, step, length):
#    return np.arange(start, start + step * (length - 0.5), step) # More efficient, but more complicated too

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
    assert all_equal(map(len, (center, steps, nsteps))), \
           'All the arguments must have the same length!'
    return np.meshgrid(*[np.linspace(c-(n-1.)/2 * s, c+(n-1.)/2 * s, n)
                         for c, s, n in zip(center, steps, nsteps)],
                       indexing='ij')

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

def np_has_duplicates(arr):
    '''For a 1D array, test whether there are any duplicated elements.'''
    arr = np.asanyarray(arr)
    return arr.size > np.unique(arr).size

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

# This function is pretty out-dated...
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
        print("Point must be interior to the array!")
        return None

######################################################
## Transformations: reshaping, splitting, filtering ##
######################################################

def ravel_at_depth(arr, depth=0):
    '''Ravel subarrays at some depth of arr
       The default, depth=0, is equivalent to arr.ravel()'''
    arr = np.asanyarray(arr)
    return arr.reshape(arr.shape[:depth] + (-1,))

def reshape_smaller(arr, new_shape):
    '''Like reshape, but allows the array to be smaller than the original'''
    return np.ravel(arr)[:np.prod(new_shape)].reshape(new_shape)

def partitionNumpy(l, n):
    '''Like partition, but always clips and returns array, not list'''
    return reshape_smaller(l, (len(l) // n, n))

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
    oldArr = np.asanyarray(arr)
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
    arrNew = np.empty( (2*tArr+arr.shape).astype(np.int), dtype=arr.dtype )
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

def _transpose_interleaved(arr):
    '''Helper function for shape_multiply and shape_divide
       Transposes an array so that all odd axes are first, i.e.:
       [1, 3, 5, 7, ..., 0, 2, 4, ...]'''
    return arr.transpose(*zipflat(range(1, arr.ndim, 2),
                                  range(0, arr.ndim, 2)))

def shape_multiply(arr, scale, oddOnly=False, adjustFunction=None):
    '''Works like tile except that it keeps all like elements clumped
       Essentially a non-interpolating, multi-dimensional image up-scaler
       Similar to scipy.ndimage.zoom but without interpolation'''
    arr = np.asanyarray(arr)
    scale = coerce_to_target_length(scale, arr.ndim)
    if oddOnly:
        assert all([i%2 == 1 for i in scale]), \
            'All elements of scale must be odd integers greater than 0!'
    t = np.tile(arr, scale)
    t.shape = zipflat(scale, arr.shape)
    t = _transpose_interleaved(t)
    if adjustFunction != None:
        t = adjustFunction(t, arr, scale)
    new_shape = [sh * sc for sh, sc in zip(arr.shape, scale)]
    return t.reshape(new_shape)

def shape_multiply_zero_fill(arr, scale):
    '''Same as shape_muliply, but requires odd values for
       scale and fills around original element with zeros.'''
    def zeroFill(t, a, sc):
        t *= 0
        middle_slice = flatten([slice(None), i//2] for i in sc)
        t[middle_slice] = a
        return t
    
    return shape_multiply(arr, scale, oddOnly=True, adjustFunction=zeroFill)

def shape_divide(arr, scale, reduction='mean'):
    '''Scale down an array (shape N x M x ...) by the specified scale
       in each dimension (n x m x ...)
       Each dimension in arr must be divisible by its scale
       (throws an error otherwise)
       This is reduces each sub-array (n x m x ...) to a single element
       according to the reduction parameter, which is one of:
        * mean (default): mean of each sub-array
        * median: median of each sub-array
        * first: the [0,0,0, ...] element of the sub-array
        * all: all the possible (N x M x ...) sub-arrays;
               returns an array of shape (n, m, ..., N, M, ...)
       This is a downsampling operation, similar to
       scipy.misc.imresize and scipy.ndimage.interpolate'''
    arr = np.asanyarray(arr)
    reduction_options = ['mean', 'median', 'first', 'all']
    assert reduction in reduction_options, \
        'reduction must be one of: ' + ' '.join(reduction_options)
    scale = coerce_to_target_length(scale, arr.ndim)
    assert all([sh % sc == 0 for sh, sc in zip(arr.shape,scale)]), \
        'all dimensions must be divisible by their respective scale!'
    new_shape = flatten([sh//sc, sc] for sh, sc in zip(arr.shape, scale))
    # group pixes into smaller sub-arrays that can then be modified by standard operations
    subarrays = _transpose_interleaved(arr.reshape(new_shape))
    flat_subarrays = subarrays.reshape([np.product(scale)] + new_shape[::2])
    return (np.mean(flat_subarrays, axis=0) if reduction == 'mean' else
            np.median(flat_subarrays, axis=0) if reduction == 'median' else
            flat_subarrays[0] if reduction == 'first' else
            subarrays if reduction == 'all' else
            None)

def _iterize(x):
    '''Ensure that x is iterable or wrap it in a tuple'''
    return x if hasattr(x, '__iter__') else (x,)

def reshape_repeating(arr, new_shape):
    '''A forgiving version of np.reshape that allows the resulting
       array size to be larger or smaller than the original
       When the new size is larger, values are filled by repeating the original (flattened) array
       A smaller or equal size always returns a view.
       A larger size always returns a copy.'''
    arr = np.asanyarray(arr)
    new_shape = _iterize(new_shape)
    new_size = np.prod(new_shape)
    if new_size <= arr.size:
        return reshape_smaller(arr, new_shape)
    else:
        arr_flat = np.ravel(arr)
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

def remove_duplicate_subarrays(arr):
    '''Order preserving duplicate removal for arrays.
       Modified version of code found here:
       http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order
       Relies on hashing the string of the subarray, which *should* be fine...'''
    seen = set()
    return np.array([i for i in arr
                     for s in (i.tostring(),)
                     if s not in seen and not seen.add(s)])

def limitInteriorPoints(l,numInteriorPoints,uniqueOnly=True):
    '''return the list l with only the endpoints and a few interior points (uniqueOnly will duplicate if too few points)'''
    inds = np.linspace(0, len(l)-1, numInteriorPoints + 2).round().astype(np.integer)
    if uniqueOnly:
        inds = np.unique(inds)
    return [ l[i] for i in inds ]

##################################
## Basic numerical calculations ##
##################################

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

#####################
## Polygon metrics ##
#####################

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

def polyPerimeter(points,closeLoop=True):
    '''This calculates the length of a (default closed) poly-line'''
    if closeLoop:
        points = np.concatenate([points,[points[0]]])
    return sum(pointDistance(points[1:],points[:-1]))

###################
## Interpolation ##
###################

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
        return l[int(index)]
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

# The active version of this function is defined below (this version is broken/ possibly never finished?)
#def limitInteriorPointsInterpolatingBAD(l,numInteriorPoints):
#    '''Like limitInteriorPoints, but interpolates evenly instead; this also means it never clips'''
#    l=np.array(l)
#    if l.ndim==1:
#        l=l[None,:]
#    return [ [ np.interp(ind, range(len(l)), i)
#              for i in l.transpose() ]
#            for ind in np.linspace(0,len(sc),numInteriorPoints+2) ]

def limitInteriorPointsInterpolating(l,numInteriorPoints):
    '''Like limitInteriorPoints, but interpolates evenly instead; this also means it never clips'''
    l=np.asarray(l)
    return [ interpNumpy(l,ind)
            for ind in np.linspace(0,len(l)-1,numInteriorPoints+2) ]

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

##################
## Broadcasting ##
##################

def concatenate_broadcasting(*arrs, **kwds):
    '''Broadcasting (i.e. forgiving) version of concatenate
    axis is passed to concatenate (default 0)
    other kwds are passed to broadcast_arrays (subok in new version of numpy)
    
    Docs for concatenate:
    '''
    axis = kwds.pop('axis', 0)
    return np.concatenate(np.broadcast_arrays(*arrs, **kwds), axis=axis)

concatenate_broadcasting.__doc__ += np.concatenate.__doc__
broad_cat = concatenate_broadcasting # alias

def reverse_broadcast(f):
    """Change a function to use "reverse broadcasting"
    which amounts to calling np.transpose on all arguments and again on the output.
    """
    newf = g_inv_f_g(f, np.transpose)
    newf.__doc__ = '\n'.join(['Transpose the arguments before and after running the function f:',
                             f.__doc__])
    return newf

###########################################
## Indexing, grouping, and record arrays ##
###########################################

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
    funs, fields_list, names = zip(*fun_fields_name)
    functions = [_outfielder(fun, fields)
                 for fun, fields, name in fun_fields_name]
    names = [i[-1] for i in fun_fields_name]
    return np_groupby(keyarr, a, *functions, names=keynames + names)

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

############################
## Boxing (nested arrays) ##
############################

def box_list(l, box_shape=None):
    '''Convert a list to boxes (object array of arrays)
       with shape optionally specified by box_shape'''
    box_shape = len(l) if box_shape is None else box_shape
    assert np.prod(box_shape) == len(l), 'shape must match the length of l'''
    boxed = np.empty(box_shape, dtype=np.object)
    boxed_flat = boxed.ravel()
    boxed_flat[:] = lmap(np.asanyarray, l)
    return boxed

def box(arr, depth=0):
    '''Make nested array of arrays from an existing array
       depth specifies the point at which to split the outer array from the inner
       depth=0 denotes that the entire array should be boxed
       depth=-1 denotes that all leaf elements 
       box always adds an outer singleton dimension to ensure that arr values with shape[0]==1 are handled properly
       '''
    box_shape, inner_shape = split_at(np.shape(arr), depth)
    box_shape = (1,) + box_shape # always add an extra outer dimension for the boxing
    arr_flat = np.reshape(arr, (-1,) + inner_shape)
    return box_list(arr_flat, box_shape)

def box_shape(boxed_arr):
    '''Get the shape of a box
       Same as np.shape except that we need to ignore
       the first dimension since it gets added during boxing'''
    return np.shape(boxed_arr)[1:]

def is_boxed(a):
    return (hasattr(a, 'shape') and
            hasattr(a, 'dtype') and
            a.dtype == object and
            a.shape[0] == 1)

def _broadcast_arr_list(l):
    '''Helper function to broadcast all elements in a list to arrays with a common shape
       Uses broadcast_arrays unless there is only one box'''
    arr_list = lmap(np.asanyarray, l)
    return (broadcast_arrays(*arr_list)
            if len(arr_list) > 1 else
            arr_list)

def unbox(boxed):
    '''Convert an array of arrays to one large array.
       If arr is not actually an array, just return it'''
    if not is_boxed(boxed):
        return boxed # already unboxed
    arr = np.array(_broadcast_arr_list(boxed.ravel()))
    return arr.reshape(box_shape(boxed) + arr.shape[1:])

###########################
## Generalized map/apply ##
###########################

def map_along_axis(f, arr, axis):
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
       >>> np.apply_along_axis(np.sum, arr, 1)
       array([[ 2,  4],
              [10, 12]])
       >>> map_along_axis(np.sum, arr, 1)
       array([10, 18])
    '''
    arr = np.asanyarray(arr)
    axis = axis + arr.ndim if axis < 0 else axis
    new_dim_order = [axis] + lrange(axis) + lrange(axis+1, arr.ndim)
    return np.array([f(a) for a in arr.transpose(new_dim_order)])

def apply_at_depth_ravel(f, arr, depth=0):
    '''Take an array and any single-array-based
       associative function with an axis argument
       (sum, product, max, cumsum, etc)
       and apply f to the ravel of arr at a particular depth
       This means f will act on arrays (N-depth)D arrays.
       This is very efficient when acting on numpy ufuncs, much more so than the
       generic "apply_at_depth" function below.'''
    return f(ravel_at_depth(arr, depth=depth), axis=-1)

def apply_at_depth(f, *args, **kwds):
    '''Takes a function and its arguments (assumed to 
       all be arrays) and applies boxing to the arguments so that various re-broadcasting can occur 
       Somewhat similar to vectorize and J's rank conjunction (")
       
       f: a function that acts on arrays and returns an array
       args: the arguments to f (all arrays)
             depending on depths, various subarrays from these are what actually get passed to f
       kwds:
       depths (or depth): an integer or list of integers with the same length as args (default 0)
       broadcast_results: a boolean that determines if broadcasting should be applied to the results (default False)
       
       Returns: a new array based on f mapped over various subarrays of args
       
       Examples:
       
       One way to think about apply_at_depth is as replacing this kind of construct:
       a, b = args
       l = []
       for i in range(a.shape[0]):
           ll = []
           for j in range(a.shape[1]):
               ll.append(f(a[i, j], b[j]))
           l.append(ll)
       result = np.array(l)
       
       This would simplify to:
       
       apply_at_depth(f, a, b, depths=[2, 1])
       
       except that apply_at_depth handles all sorts of
       other types of broadcasting for you.
       
       Something like this could be especially useful if the
       "f" in question depends on its arguments having certain
       shapes but you have data structures with those as subsets.
       
       
       The algorithm itself is as follows:
        * box each arg at the specified depth (box_list)
          See docs for "box" for more details
        * broadcast each boxed argument to a common shape
          (bbl, short for broadcasted box_list)
          Note that box *contents* can still have any  shape
        * flatten each broadcasted box (bbl_flat)
          Each element of bbl_flat will be a 1D list
          of arrays where each list had the same length
          (for clarity, lets call these lists l0, l1, l2, etc)
        * map f over these flat boxes like so:
          [f(l0[i], l1[i], ...)
           for i in range(arg_size)]
          or just map(f, *bbl_flat)
          Again, arg0[i] will still be an array that can have arbitrary shape
          and will be some subarray of args[0] (ex: args[0][2,1])
        * Optionally broadcast the results (otherwise 
          force all outpus to have the same shape) and 
          construct a single array from all the outputs
        * Reshape the result to account for the flattening that
          happened to the broadcasted boxes
          This is the same way that unboxing works.
        * Celebrate avoiding unnecessarily complex loops :)
       
       This function is as efficient as it can be considering the generality;
       if f is reasonably slow and the arrays inside the boxes are
       fairly large it should be fine.
       However, performance may be a problem if applying it to single elements
       In other words, with:
       a = np.arange(2000).reshape(200, 2, 5)
       do this:
       apply_at_depth_ravel(np.sum, depth=1)
       instead of this:
       apply_at_depth(np.sum, a, depth=1)
       The latter is just essentially calling map(np.sum, a)'''
    assert not ('depth' in kwds and 'depths' in kwds), (
           'You can pass either kwd "depth" or "depths" but not both!')
    depths = kwds.pop('depths', kwds.pop('depth', 0)) # Grab depths or depth, fall back to 0
    broadcast_results = kwds.pop('broadcast_results', False)
    depths = (depths if hasattr(depths, '__len__') else
              [depths] * len(args))
    assert len(args) == len(depths)
    boxed_list = lmap(box, args, depths)
    bbl = _broadcast_arr_list(boxed_list)
    bb_shape = box_shape(bbl[0])
    bbl_flat = lmap(np.ravel, bbl)
    results = lmap(f, *bbl_flat)
    results = (results if not broadcast_results else
               _broadcast_arr_list(results))
    arr = np.array(results)
    return arr.reshape(bb_shape + arr.shape[1:])
