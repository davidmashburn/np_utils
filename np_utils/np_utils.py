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

from list_utils import *

one = np.array(1) # a surprisingly useful little array; makes lists into arrays by simply one*[[1,2],[3,6],...]

def limitInteriorPoints(l,numInteriorPoints,uniqueOnly=True):
    '''return the list l with only the endpoints and a few interior points (uniqueOnly will duplicate if too few points)'''
    inds = np.linspace(0,len(l)-1,numInteriorPoints+2).round().astype(np.integer)
    if uniqueOnly:
        inds = np.unique(inds)
    return [ l[i] for i in inds ]

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
    
    p = copy(p0)
    l=[]
    for pmax in range(delta[imax]+1): # in the longest dimension, step evenly
        l.append(copy(p))
        for i in range(ndim):
            err[i] -= delta[i]
            if i==imax or err[i]<0:
                p[i] += signs[i]
                err[i] += delta[imax]
    return l

#Deprecated... check and delete
'''def BresenhamPlane(p0,p1,p2): # Generalization for plane instead...
    if p2==None:
        return BresenhamFunction(p0,p1) # In case the last argument is None just use a plane...
    
    ndim = len(p0)
    deltaA = [ p1[i]-p0[i] for i in range(ndim) ]
    signsA = [ (-1 if d<0 else 1) for d in deltaA ]
    deltaA = [ abs(d) for d in deltaA ]
    imaxA = deltaA.index(max(deltaA)) # The dimension that we go the farthest in
    errA = [ deltaA[imaxA]//2 ] * ndim
    
    deltaB = [ p2[i]-p0[i] for i in range(ndim) ]
    signsB = [ (-1 if d<0 else 1) for d in deltaB ]
    deltaB = [ abs(d) for d in deltaB ]
    imaxB = deltaB.index(max(deltaB)) # The dimension that we go the farthest in
    errB = [ deltaB[imaxB]//2 ] * ndim
    
    pA = copy(p0)
    l=[]
    for pmaxA in range(deltaA[imaxA]+1): # in the longest dimension, step evenly
        for i in range(ndim):
            errA[i] -= deltaA[i]
            if i==imaxA or errA[i]<0:
                pA[i] += signsA[i]
                errA[i] += deltaA[imaxA]
        pB = copy(pA)
        for pmaxB in range(deltaB[imaxB]+1): # in the longest dimension, step evenly
            l.append(copy(pB))
            for i in range(ndim):
                errB[i] -= deltaB[i]
                if i==imaxB or errB[i]<0:
                    pB[i] += signsB[i]
                    errB[i] += deltaB[imaxB]
    return l
'''

def BresenhamTriangle(p0,p1,p2,fullPlane=False): # Generalization for triangle
    if p2==None:
        return BresenhamFunction(p0,p1) # In case the last argument is None just use a plane...
    
    ndim = len(p0)
    deltaA = [ p1[i]-p0[i] for i in range(ndim) ]
    signsA = [ (-1 if d<0 else 1) for d in deltaA ]
    deltaA = [ abs(d) for d in deltaA ]
    imaxA = deltaA.index(max(deltaA)) # The dimension that we go the farthest in
    errA = [ deltaA[imaxA]//2 ] * ndim
    stepsA = deltaA[imaxA]+1
    
    deltaB = [ p2[i]-p0[i] for i in range(ndim) ]
    signsB = [ (-1 if d<0 else 1) for d in deltaB ]
    deltaB = [ abs(d) for d in deltaB ]
    imaxB = deltaB.index(max(deltaB)) # The dimension that we go the farthest in
    errB = [ deltaB[imaxB]//2 ] * ndim
    stepsB = deltaB[imaxB]+1
    
    pA = copy(p0)
    l=[]
    for pmaxA in range(stepsA): # in the longest dimension, step evenly
        for i in range(ndim):
            errA[i] -= deltaA[i]
            if i==imaxA or errA[i]<0:
                pA[i] += signsA[i]
                errA[i] += deltaA[imaxA]
        pB = copy(pA)
        if fullPlane:
            stepsB_truncated = stepsB
        else:
            stepsB_truncated = (stepsB*stepsA - stepsB*pmaxA)//stepsA
        # Unlike making a plane, only go a certain percentage of the way out to cut the plane down to a triangle
        for pmaxB in range(stepsB_truncated): # in the longest dimension, step evenly, 
        #for pmaxB in range(int( stepsB*(1.-1.*pmaxA/stepsA) )): # in the longest dimension, step evenly, 
            l.append(copy(pB))
            for i in range(ndim):
                errB[i] -= deltaB[i]
                if i==imaxB or errB[i]<0:
                    pB[i] += signsB[i]
                    errB[i] += deltaB[imaxB]
    return l


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
