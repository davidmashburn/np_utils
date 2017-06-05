'''A collection of array drawing functions:

   BresenhamFunction, BresenhamTriangle ->
       draw lines and planes in N-dimensional space

   NDRectangle ->
       draw square, cubes, hyper-cubes, and any other ND-rectangular
       solids within arrays

   ImageCircle, ImageSphere (also blitCircleToArray, blitSphereToArray) ->
       draw circles and spheres (2D or 3D)

   The drawing functions only use integer arithmetic and return a
   list of the coordinates that can be used as array indices
   '''
from __future__ import absolute_import
from __future__ import division
from builtins import zip, map, range
from future.utils import lmap

from copy import copy
import numpy as np

from .list_utils import totuple, flatten, getMostCommonVal, fL

# Thanks Wikipedia!!
# A line-drawing algorithm by the pixel...
def BresenhamFunctionOld(p0,p1):
    [x0,y0], [x1,y1] = p0, p1
    steep = abs(y1-y0) > abs(x1-x0)
    if steep:
        x0,y0 = y0,x0 # swap
        x1,y1 = y1,x1 # swap
    if x0 > x1:
        x0,x1 = x1,x0 # swap
        y0,y1 = y1,y0 # swap
    deltax = x1-x0
    deltay = abs(y1-y0)
    error = deltax // 2
    y = y0
    if y0 < y1:
        ystep=1
    else:
        ystep=-1
    l=[]
    for x in range(x0,x1+1):
        if steep:
            l.append([y,x])
        else:
            l.append([x,y])
        error = error-deltay
        if error < 0:
            y = y+ystep
            error = error+deltax
    return l

def BresenhamFunction(p0,p1): # Generalization to n-dimensions
    ndim = len(p0)
    delta = [ p1[i]-p0[i] for i in range(ndim) ]
    signs = [ (-1 if d<0 else 1) for d in delta ]
    delta = [ abs(d) for d in delta ]
    imax = delta.index(max(delta)) # The dimension that we go the farthest in
    err = [ delta[imax]//2 ] * ndim

    p = list(p0) # make a shallow copy, and ensure it's a list
    l=[]
    for pmax in range(delta[imax]+1): # in the longest dimension, step evenly
        l.append(copy(p))
        for i in range(ndim):
            err[i] -= delta[i]
            if i==imax or err[i]<0:
                p[i] += signs[i]
                err[i] += delta[imax]
    return l

def NDRectangle(start,end):
    '''Given any two points in an ND grid, return all the points that
       would fill an ND rectangle using these points as distant corners'''
    start,end = [f([start,end],axis=0)     # go from small to large in each dimension
                 for f in (np.min,np.max)]
    nDim, nPts = len(start), np.prod(end+1-start)
    slicelist = [slice(i,j+1) for i,j in zip(start,end)]
    # These two ::-1's just make it so the the results are already sorted
    # (otherwise the results are the same without them):
    pts = np.transpose(np.mgrid[slicelist[::-1]]).reshape(nPts,nDim)[:,::-1]
    return pts

def ImageCircle(r):
    """Create a binary image of a circle radius r"""
    im = np.zeros([2*r-1]*2,dtype=np.int)
    r2 = r**2
    for i in range(2*r-1):
        for j in range(2*r-1):
            if (i-r+1)**2+(j-r+1)**2 < r2:
                im[i,j] = 1
            else:
                im[i,j]=0
    return im

def ImageSphere(r):
    """Create a binary image of a circle radius r"""
    im = np.zeros([2*r-1]*3,dtype=np.int)
    r2 = r**2
    for i in range(2*r-1):
        for j in range(2*r-1):
            for k in range(2*r-1):
                if (i-r+1)**2+(j-r+1)**2+(k-r+1)**2 < r2:
                    im[i,j,k] = 1
                else:
                    im[i,j,k]=0
    return im

def blitCircleToArray(arr,x,y,r,val):
    xL,xH = np.clip(x-r+1,0,arr.shape[0]), np.clip(x+r,0,arr.shape[0])
    yL,yH = np.clip(y-r+1,0,arr.shape[1]), np.clip(y+r,0,arr.shape[1])
    xcL,xcH = xL-(x-r+1), 2*r-1 + xH-(x+r)
    ycL,ycH = yL-(y-r+1), 2*r-1 + yH-(y+r)
    #print (xL,xH,yL,yH),(xcL,xcH,ycL,ycH)
    #print xH-xL,yH-yL,xcH-xcL,ycH-ycL
    c=ImageCircle(r)[xcL:xcH,ycL:ycH]
    #print arr[xL:xH,yL:yH].shape,c.shape
    arr[xL:xH,yL:yH] *= (1-c)
    arr[xL:xH,yL:yH] += (val*c)

def blitSphereToArray(arr,x,y,z,r,val):
    xL,xH = np.clip(x-r+1,0,arr.shape[0]), np.clip(x+r,0,arr.shape[0])
    yL,yH = np.clip(y-r+1,0,arr.shape[1]), np.clip(y+r,0,arr.shape[1])
    zL,zH = np.clip(z-r+1,0,arr.shape[2]), np.clip(z+r,0,arr.shape[2])

    xcL,xcH = xL-(x-r+1), 2*r-1 + xH-(x+r)
    ycL,ycH = yL-(y-r+1), 2*r-1 + yH-(y+r)
    zcL,zcH = zL-(z-r+1), 2*r-1 + zH-(z+r)

    #print (xL,xH,yL,yH),(xcL,xcH,ycL,ycH)
    #print xH-xL,yH-yL,xcH-xcL,ycH-ycL
    c=ImageSphere(r)[xcL:xcH,ycL:ycH,zcL:zcH]
    #print arr[xL:xH,yL:yH].shape,c.shape
    arr[xL:xH,yL:yH,zL:zH] *= (1-c)
    arr[xL:xH,yL:yH,zL:zH] += (val*c)

########################################################################
##                       Gradient Drawing                             ##
########################################################################
def nd_gradient(shape, origin_val, stopvals):
    grids = np.mgrid.__getitem__(lmap(slice, shape))
    ortho_grads = [g * (stop - origin_val) / (s - 1)
                   for s, stop, g in zip(shape, stopvals, grids)]
    return origin_val + sum(ortho_grads)

def nd_radial_gradient(shape, offsets=None):
    if offsets is None: 
        offsets = [0] * len(shape)
    grids = np.mgrid.__getitem__(lmap(slice, shape))
    v = [(g + off + 0.5 - s / 2) ** 2
         for s, off, g in zip(shape, offsets, grids)]
    return np.sqrt(np.sum(v, axis=0))

def interpolated_radial_gradient(shape, stops, values, offsets=None, outer_radius=None, clip=True):
    '''Get a radial gradient with values interpolated.
    
    Stops and values define the interpolation scheme based on distance
    from the origin.
    
    (Needs testing)
    '''
    outer_radius = (np.sqrt(np.sum(np.square(shape))) / 2 if outer_radius is None else
                    outer_radius)
    grad = nd_radial_gradient(shape, offsets)
    if clip:
        grad = np.clip(grad / outer_radius, 0, 1)
    return np.interp(grad.flat, stops, values).reshape(shape)

###############################################################################
##                   Functions for Bresenham Triangles                       ##
###############################################################################

def GetDirectionsOfSteepestSlope(p0,p1,p2):
    '''For a 2-D plane in N-D space, fixing each dimension, identify the
       other dimension with the steepest absolute slope.

       The mathematics behind this function:
       -------------------------------------
       Given the three three points that define a plane:
        (x0,y0,z0,...),(x1,y1,z1,...),(x2,y2,z2,...)
       A 2D planar triangle in N-D space is defined by the equations:
        x = x0 + (x1-x0)*s + (x2-x0)*t
        y = y0 + (y1-y0)*s + (y2-y0)*t
        z = z0 + (z1-z0)*s + (z2-z0)*t
        ...

       where s and t are parametric variables with:
        0<=(s-t)<=1

       so,
        s=0,t=0 -> point 0
        s=1,t=0 -> point 1
        s=0,t=1 -> point 2

       We now look at one dimension (A) which can be any of x,y,z,...
       Obviously if A is the same for all three points (aka, A0==A1==A2), then the plane is flat in that dimension
       If this is not the case, the intersection of the plane (infinite) with any fixed value of A will be a line

       We start by fixing A to any constant value, so dA=0

       Our goal is to find for the other dimension (B) wth the greatest relative slope given that dA=0
       We will compare these by looking at the relative values of dB/dw for some w which is related to the the parametric variables s and t

       Starting with dA=0, we see that:
        0 = dA = (A1-A0)*ds + (A2-A0)*dt
       and
        (A1-A0)*ds = -(A2-A0)*dt

       We define dw as this value:
        dw = (A1-A0)*ds = -(A2-A0)*dt

       Now, in the B direction,
        dB = (B1-B0)*ds + (B2-B0)*dt
       multiplying by (A1-A0)*(A2-A0) and collecting terms we find that:
        (A1-A0)*(A2-A0) * dB = (A1-A0)*(A2-A0) * (B1-B0)*ds + (A1-A0)*(A2-A0) * (B2-B0)*dt
        (A1-A0)*(A2-A0) * dB = (A2-A0) * (B1-B0) * dw - (A1-A0) * (B2-B0) * dw
        (A1-A0)*(A2-A0) * dB = ((A2-A0)*(B1-B0) - (A1-A0)*(B2-B0)) * dw
        (A1-A0)*(A2-A0) * dB = ( A0*(B2-B1) + A1*(B0-B2) + A2*(B1-B0) ) * dw


       So, for any fixed dimension A, the dimension B will have the greatest relative slope (abs value) if it also maximizes the expression:
        abs( A0*(B2-B1) + A1*(B0-B2) + A2*(B1-B0) )
       This expression can be obtained easily using numpy by:
        abs( dot( A_i , (roll(B_i,1) - roll(B_i,-1)) ) )
       This is the expression caclulated below as "slopeMatrix".'''

    pointsT = np.transpose([p0,p1,p2])
    # Calculate the slope matrix (see the docstring)
    bdiff = np.roll(pointsT,1,axis=1)-np.roll(pointsT,-1,axis=1)
    slopeMatrix = abs(np.dot( pointsT,bdiff.T ))

    # Now that we have the slope matrix values, we just look for the dimension(s) with the maximum value for each row
    maxs = [ ( None if i.max()==0 else
               np.where(i==i.max())[0][0].tolist() )
            for i in slopeMatrix ]

    return maxs

def GetDirectionsOfSteepestSlope_BorderCheckingVersion(borderPts):
    '''Taking each dimension as the potential scanning dimension,
       find the other dimension that has the largest slope.
       Needs the list of points of the triangle's boundary.
       Returns a list of integers (one for each dimension).
       '''
    ndim = len(borderPts[0])
    def minMaxDiff(l):
        return max(l)-min(l)

    maxSlopeDim = []
    for i in range(ndim): # loop over all dimensions as the assumed scan dimension (i)
        # Bin the points into hyper-planes with the same value on the i-axis
        binnedPts = {}
        for p in borderPts:
            binnedPts.setdefault(p[i],[]).append(p)

        # Traversing along the i-axis, find the other dimension(s) with the greatest slope
        # Collect the results in dimsOfGreatestExtent, a flat list of integers
        dimsOfGreatestExtent = []
        for pts in binnedPts.values(): # loop through the points in each bin (hyperplane)
            deltas = [ minMaxDiff([ p[j] for p in pts ]) # get the maximum delta in each dimension
                      for j in range(ndim) ]
            maxExtent = max(deltas)
            dimsOfGreatestExtent += [ j for j in range(ndim) if deltas[j]==maxExtent ]

        # Get the dimension that most often has the steepest slope
        maxSlopeDim.append(getMostCommonVal(dimsOfGreatestExtent))
    return maxSlopeDim

def _getMinMaxList(borderPts, iscan, iline):
    '''Helper function for BresenhamTriangle'''
    #Sort the border points according to iscan (x') and then iline (y')
    borderPtsSort = sorted( borderPts,  key = lambda x: (x[iscan],x[iline]) )

    # For each x' plane, select the two most distant points in y' to pass to the Bresenham function
    minMaxList = []
    for i,p in enumerate(borderPtsSort):
        if borderPtsSort[i-1][iscan] != borderPtsSort[i][iscan]:
            minMaxList.append([p,p]) # ensure there are always 2 points, even if they are the same
        else:
            minMaxList[-1][-1] = p
    return minMaxList

def GetTriangleBorderPoints(p0,p1,p2):
    '''Collect all the border points for a raster triangle and remove any duplicates'''
    allPts = [BresenhamFunction(pa,pb)
              for pa,pb in ((p0,p1),(p1,p2),(p2,p0))]
    return list(set(totuple(flatten(allPts))))

def _BresTriBase(p0,p1,p2,dss_rerun=False, iscan_iline=None):
    '''Do grunt work common to both Bresenham Triangle functions'''
    borderPts = GetTriangleBorderPoints(p0,p1,p2)

    if iscan_iline is not None:
        iscan, iline = iscan_iline
        return iscan, iline, borderPts

    # Get the steepest dimension relative to every other dimension:
    dSS = GetDirectionsOfSteepestSlope(p0,p1,p2)

    dSS_notNone = [i for i in dSS if i!=None]
    if len(dSS_notNone) == 0:
        # Degenerate slope matrix (all 0's); points are collinear
        return borderPts
    iscan = getMostCommonVal(dSS_notNone)
    if dss_rerun:
        all_but = [i for i in range(len(p0)) if i != iscan]
        p0a,p1a,p2a = fL([p0,p1,p2])[:, all_but]
        dSS2 = GetDirectionsOfSteepestSlope(p0a,p1a,p2a)
        dSS2_notNone = [i for i in dSS2 if i!=None]
        iline = getMostCommonVal(dSS2_notNone)
        if iline >= iscan:
            iline += 1
    else:
        iline = dSS[iscan]
    assert iline!=None, "iline is <flat>, that can't be right!"

    return iscan, iline, borderPts

def _map_BresLines(point_pairs):
    '''Draw a bunch of Bresenham lines and collect the points'''
    return [p for start, end in point_pairs
              for p in BresenhamFunction(start,end)]

def BresenhamTriangle(p0,p1,p2,doubleScan=True): # Generalization for triangle
    '''Bresenham N-D triangle rasterization
       Uses Bresenham N-D lines to rasterize the triangle.
       Holes are prevented by proper selection of dimensions:
           the 'scan' dimension is fixed for each line (x')
           the 'line' dimension has the maximum slope perpendicular to iscan (y')

       Even with these precautions, diabolical cases can still have holes, so
       the option "doubleScan" also generates the points with x' and y' swapped
       and includes these points as well, ensuring that there are no gaps.
       This does add computation time, so for fastest operation, set it to False.'''
    if p2==None:
        return BresenhamFunction(p0,p1) # In case the last argument is None just use a plane...

    iscan, iline, borderPts = _BresTriBase(p0,p1,p2)

    # Draw Bresenham lines to rasterize the triangle (draw along y' direction for each x' value)
    minMaxList = _getMinMaxList(borderPts,iscan,iline)
    triPts = _map_BresLines(minMaxList)

    if doubleScan:
        minMaxList = _getMinMaxList(borderPts,iline,iscan)
        triPts += _map_BresLines(minMaxList)

    return triPts

def _rounder(x):
    '''Return the nearest int for x
       (where x can also be an array)'''
    return np.round(x).astype(np.int)

def sample_points_from_plane(p, projected_pts, iscan, iline):
    '''Given 3 ND points (p) that define a 2D plane in ND space
       and also given a set of 2D points (projected_pts) and the
       two associated dimensions (iscan, iline),
       return the points in ND space where each (N-2) space
       (associated with each projected point) intersects the 2D plane

       Details:
       All other dimensions are interpolated using the formula for a plane:

       for three points (x0,y0,z0,...), (x1,y1,z1,...), (x2,y2,z2,...):
       x = x0 + (x1-x0)*s + (x2-x0)*t
       y = y0 + (y1-y0)*s + (y2-y0)*t
       z = z0 + (z1-z0)*s + (z2-z0)*t
       ...

       When 2 of these (x' and y') are fixed,
       we can solve for s and t:
       (here using A and B instead of x' and y'):

       A = A0 + (A1-A0)*s + (A2-A0)*t
       B = B0 + (B1-B0)*s + (B2-B0)*t

       Using matrix form and with
       dX01 = X1-X0  and  dX02 = X2-X0:

       [A - A0] = [dA01  dA02] * [s]
       [B - B0] = [dB01  dB02] * [t]

       We can solve by inverting tbe delta's matrix:

       [s] = [dA01  dA02]^-1 * [A - A0]
       [t] = [dB01  dB02]    * [B - B0]

       Lastly, we find values for all remaining coordinates (x,y,z,...) by plugging s and t
       back into the original equations and rounding the result.'''
    p = np.asanyarray(p)
    shifted_pts = projected_pts - p[0, (iscan, iline)]
    pts_diff = p[1:] - p[0] # p1-p0, p2-p0
    mat = pts_diff.T[(iscan, iline),]
    inv = np.linalg.inv(mat)
    s, t = np.dot(inv, shifted_pts.T)
    new_points = _rounder(p[0] + np.outer(s, pts_diff[0]) +
                                 np.outer(t, pts_diff[1]))
    return new_points

def _BresenhamTriangle_PlaneFormulaVersion(p0, p1, p2, iscan_iline=None): # Generalization for triangle
    '''Bresenham N-D triangle rasterization
       Uses a Bresenham-like algorithm, but uses floating-point
       math and rounding for greater accuracy.
       Holes are prevented by proper selection of dimensions:
           the 'scan' dimension is fixed for each line (x')
           the 'line' dimension has the maximum slope perpendicular to iscan (y')
       All other dimensions are interpolated using the formula for a plane:

       for three points (x0,y0,z0,...), (x1,y1,z1,...), (x2,y2,z2,...):
       x = x0 + (x1-x0)*s + (x2-x0)*t
       y = y0 + (y1-y0)*s + (y2-y0)*t
       z = z0 + (z1-z0)*s + (z2-z0)*t
       ...

       When 2 of these (x' and y') are fixed,
       we can solve for s and t:
       (here using A and B instead of x' and y'):

       A = A0 + (A1-A0)*s + (A2-A0)*t
       B = B0 + (B1-B0)*s + (B2-B0)*t

       Using matrix form and with
       dX01 = X1-X0  and  dX02 = X2-X0:

       [A - A0] = [dA01  dA02] * [s]
       [B - B0] = [dB01  dB02] * [t]

       We can solve by inverting tbe delta's matrix:

       [s] = [dA01  dA02]^-1 * [A - A0]
       [t] = [dB01  dB02]    * [B - B0]

       Lastly, we find values for all remaining coordinates (x,y,z,...) by plugging s and t back into the original equations and rounding the result.
    '''
    if p2==None:
        return BresenhamFunction(p0,p1) # In case the last argument is None just use a plane...

    p = np.array([p0,p1,p2])

    iscan, iline, borderPts = _BresTriBase(p0,p1,p2,dss_rerun=False, iscan_iline=iscan_iline)

    # Draw Bresenham lines to rasterize the triangle (draw along y' direction for each x' value)
    minMaxList = _getMinMaxList(borderPts,iscan,iline)

    # Get the points in the iscan/iline projection of the output:
    minMaxProjected = fL(minMaxList)[:, :, (iscan, iline)]
    projectedTriPts = _map_BresLines(minMaxProjected)

    new_points = sample_points_from_plane(p, projectedTriPts, iscan, iline)

    return new_points
