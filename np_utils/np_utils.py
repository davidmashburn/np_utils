#!/usr/bin/env python
'''Utilities for array and list manipulation by David Mashburn.
Notably:
    functions to scale arrays by integer multiples (without interpolation)
    drawing basic graphics objects on (multi-dimensional) numpy arrays:
       line segment, triangle, circle, and sphere
'''

import numpy as np
import scipy.sparse
from copy import copy
from collections import Counter

from list_utils import *

one = np.array(1) # a surprisingly useful little array; makes lists into arrays by simply one*[[1,2],[3,6],...]

def limitInteriorPoints(l,numInteriorPoints,uniqueOnly=True):
    '''return the list l with only the endpoints and a few interior points (uniqueOnly will duplicate if too few points)'''
    inds = np.linspace(0,len(l)-1,numInteriorPoints+2).round().astype(np.integer)
    if uniqueOnly:
        inds = np.unique(inds)
    return [ l[i] for i in inds ]

def limitInteriorPointsInterpolating(l,numInteriorPoints):
    '''Like limitInteriorPoints, but interpolates evenly instead; this also means it never clips'''
    l=np.array(l)
    if l.ndim==1:
        l=l[None,:]
    return [ [ np.interp(ind, range(len(l)), i)
               for i in l.transpose() ]
             for ind in np.linspace(0,len(sc),numInteriorPoints+2) ]

def partitionNumpy(l,n):
    '''Like partition, but always clips and returns array, not list'''
    a=np.array(l)
    a.resize(len(l)//n,n)
    return a

def shape_multiply(arr,shapeMultiplier, oddOnly=False, adjustFunction=None):
    '''Works like tile except that it keeps all like elements clumped\n'''
    '''Essentially a non-interpolating multi-dimensional image up-scaler'''
    # really only 7 lines without checks...
    if not hasattr(arr,'shape'):
        arr=np.array(arr)
    
    sh = arr.shape
    ndim = arr.ndim # dimensional depth of the array
    shm = shapeMultiplier
    
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

def interpNumpy(l,index):
    '''Just like interp except that it uses numpy instead of lists.'''
    l = np.array(l)
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
    l=np.array(l)
    return [ interpNumpy(l,ind)
             for ind in np.linspace(0,len(l)-1,numInteriorPoints+2) ]

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
    error = deltax/2
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

def polyArea(points):
    '''This calculates the area of a general 2D polygon
       Algorithm adapted from Darius Bacon's post:
       http://stackoverflow.com/questions/451426/how-do-i-calculate-the-surface-area-of-a-2d-polygon'''
    return 0.5*abs(sum( x0*y1 - x1*y0
                        for ((x0, y0), (x1, y1)) in zip(points, roll(points)) ))

def _getMostCommonVal(l):
    return Counter( l ).most_common()[0][0]

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
       
       Starting with da=0, we see that:
        0 = dA = (A1-A0)*ds + (A2-A0)*dt
       and
        (A1-A0)*ds = -(A2-A0)*dt
       
       We define dw as this value:
        dw = (A1-A0)*ds = -(A2-A0)*dt
        
       Now, in the B direction,
        dB = (B1-B0)*ds + (B2-B0)*dt
       mltiplying by (A1-A0)*(A2-A0) and collecting terms we find that:
        (A1-A0)*(A2-A0) * dB = (A1-A0)*(A2-A0) * (B1-B0)*ds + (A1-A0)*(A2-A0) * (B2-B0)*dt
        (A1-A0)*(A2-A0) * dB = (A2-A0) * (B1-B0) * dw - (A1-A0) * (B2-B0) * dw
        (A1-A0)*(A2-A0) * dB = ((A2-A0)*(B1-B0) - (A1-A0)*(B2-B0)) * dw
        (A1-A0)*(A2-A0) * dB = ( A0*(B2-B1) + A1*(B0-B2) + A2*(B1-B0) ) * dw


       So, for any fixed dimension A, the dimension B will have the greatest relative slope (abs value) if it also maximizes the expression:
        abs( A0*(B2-B1) + A1*(B0-B2) + A2*(B1-B0) )
       This expression can be obtained easily using numpy by:
        abs( dot( A_i , (roll(B_i,1) - roll(B_i,2)) ) )
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
        maxSlopeDim.append( _getMostCommonVal(dimsOfGreatestExtent) )
    return maxSlopeDim

def BresenhamTriangle(p0,p1,p2): # Generalization for triangle
    '''Bresenham N-D triangle rasterization
       Uses Bresenham N-D lines to rasterize the triangle
       Holes are prevented by proper selection of dimensions:
           the 'scan' dimension is fixed for each line (x')
           the 'line' dimension has the maximum slope perpendicular to iscan (y')'''
    if p2==None:
        return BresenhamFunction(p0,p1) # In case the last argument is None just use a plane...
    
    # Collect all the border points for the triangle and remove duplicates
    borderPts = list(set(totuple(flatten( [ BresenhamFunction(pa,pb)
                                           for pa,pb in ((p0,p1),(p1,p2),(p2,p0)) ] ))))
    
    # Get the steepest dimension relative to every other dimension:
    #dSS = GetDirectionsOfSteepestSlope_BorderCheckingVersion(borderPts)
    dSS = GetDirectionsOfSteepestSlope(p0,p1,p2)
    
    dSS_notNone = [i for i in dSS if i!=None]
    if len(dSS_notNone) == 0:
        # Degenerate slope matrix (all 0's); points are collinear
        return borderPts
    iscan = _getMostCommonVal(dSS_notNone)
    iline = dSS[iscan]
    assert iline!=None,"iline is <flat>, that can't be right!"
    
    #Sort the border points according to iscan (x') and then iline (y')
    borderPtsSort = sorted( borderPts,  key = lambda x: (x[iscan],x[iline]) )
    
    # For each x' plane, select the two most distant points in y' to pass to the Bresenham function
    minMaxList = []
    for i,p in enumerate(borderPtsSort):
        if borderPtsSort[i-1][iscan] != borderPtsSort[i][iscan]:
            minMaxList.append([p,p]) # ensure there are always 2 points, even if they are the same
        else:
            minMaxList[-1][-1] = p
    
    # Draw Bresenham lines to rasterize the triangle (draw along y' direction for each x' value)
    triPts = flatten([BresenhamFunction(start,end) for start,end in minMaxList ])
    return triPts

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
