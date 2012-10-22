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

one = np.array(1) # a surprisingly useful little array; makes lists into arrays by simply one*[[1,2],[3,6],...]

def totuple(a):
    '''Makes tuples out of nested datastructures like lists and arrays.
       Authored by Bi Rico, http://stackoverflow.com/questions/10016352/convert-numpy-array-to-tuple'''
    try:
        return tuple(totuple(i) for i in a)
    except TypeError: # dig until we can dig no more!
        return a
def makeTuple(a):
    '''Like totuple, but ensures that you get a tuple out.'''
    retVal = totuple(a)
    return ( retVal if retVal.__class__==tuple else (retVal,) )
def iterToX(f,iter):
    '''Replace all iterables in a nested structure (list,tuple,etc...)
       with another type "f" (actually just any function returning iterables)
       Like totuple, but more general, also uses list comprehensions
       instead of generators for more generality (notably sympy.Tuple)'''
    try:
        return f(*[iterToX(f,i) for i in iter])
    except TypeError: # dig until we can dig no more!
        return iter

class fancyIndexingList(list):
    '''fancyIndexingList is overloaded as "fL"
       Gives lists the magical properties of numpy arrays, but without requiring regular shapes...
       Also use it where you would convert a list to a numpy array and back again like: np.array(l)[:,3].tolist()
       Use like: fl(ANY_LIST)[<any valid numpy slice>]
       Examples:
         fl( [[1],[2,3],[4,5,6],[7,8,9,10]] )[:,0]  ->  [1, 2, 4, 7]
         fl( [[1],[2,3],[4,5,6],[7,8,9,10]] )[2:4,0]  ->  [4, 7]
         fl( [1,[2,3],[[4,5],[6,7]]] )[2,0,1]  ->  5
       fl also has the extra feature of being able to use lists-within-lists when indexing
       Examples:
         fl( [[1,7],[2,3],[4,5,6],[7,8,9,10]] )[:,(0,1)]  ->  [[1, 7], [2, 3], [4, 5], [7, 8]]
         fl( [[1,7],[2,3],[4,5,6],[7,8,9,10]] )[(0,2),(0,1)]  ->  [[1, 7], [4, 5]]
       Beware that you will need to nest lists or tuples if you only have a single index:
         fl([1,2,3,4,5])[(1,2)] -> TypeError -- this is equivalent to fl([1,2,3,4,5])[1,2]
         fl([1,2,3,4,5])[[1,2]] -> TypeError -- ""
         fl([1,2,3,4,5])[(1,2),] -> [2,3]
         fl([1,2,3,4,5])[[1,2],] -> [2,3]
         fl([1,2,3,4,5])[((1,2),)] -> [2,3]
         fl([1,2,3,4,5])[([1,2],)] -> [2,3]
         fl([1,2,3,4,5])[[[1,2]]] -> [2,3]
       '''
    def new_fL(self,*args,**kwds):
        '''Just a wrapper around the class constructor for a new instance, fancyIndexingList()'''
        return fancyIndexingList(*args,**kwds)
    def _list_getitem(self,index):
        '''Just a wrapper around list.__getitem__()'''
        return list.__getitem__(self,index) # for a single index, use list's default indexing
    def __getitem__(self,index):
        if not hasattr(index,'__iter__'):
            index=(index,) # make sure index is always a tuple
        if not index.__class__==tuple:
            index = tuple(index) # make sure index is always a tuple
        
        if len(index)==1:
            if hasattr(index[0],'__iter__'):
                return [ self[i] for i in index[0] ] # if the single index is tuple of tuples, 
            else:
                return self._list_getitem(index[0]) # for a single index, use list's default indexing
        elif type(index[0])==slice:
            return [ self.new_fL(i)[index[1:]] for i in self[index[0]] ] # recurse
        elif hasattr(index[0],'__iter__'):
            return [ self.new_fL(self[i])[index[1:]] for i in index[0] ] # recurse
        else:
            return self.new_fL(self[index[0]])[index[1:]] # recurse
    def __getslice__(self,i,j):
        return self.__getitem__(slice(i,j))
fL = fancyIndexingList

class fancyIndexingListM1(fancyIndexingList):
    '''fancyIndexingListM1 is overloaded as "fLm1"
       Just like fancyIndexingList, but changes indices to be 1-indexed: fLm1(i)[1] -> i[0]
       VERY useful if dealing with code from Octave, Matlab, Mathematica, etc... 
       Documentation for fancyIndexingList:\n\n'''+fancyIndexingList.__doc__
    def m1(self,x): # These don't really need to be in the class, but it's cleaner that way...
        '''minus 1 if x>0'''
        return (None if x==None else (x-1 if x>0 else x))

    def m1gen(self,x):
        '''m1, but able to handle iterables (lists, tuples, ...) and slices as well'''
        if hasattr(x,'__iter__'):
            return map(self.m1gen,x)
        elif type(x)==slice:
            return slice(self.m1(x.start),self.m1(x.stop),x.step)
        elif type(x)==int:
            return self.m1(x)
        else:
            print "Not sure what you're feeding me... signed, fancyIndexingListM1.m1gen"
            return self.m1(x)
    def new_fL(self,*args,**kwds):
        '''Just a wrapper around the class constructor for a new instance, fancyIndexingListM1()'''
        return fancyIndexingListM1(*args,**kwds)
    def _list_getitem(self,index):
        '''Just a wrapper around list.__getitem__(), but subtracts 1 from index'''
        return list.__getitem__(self,self.m1gen(index)) # for a single index, use list's default indexing
fLm1 = fancyIndexingListM1

def intOrFloat(string):
    '''Not sure if your string is formatted as an int or a float? Use intOrFloat instead!'''
    try:
        return int(string)
    except ValueError:
        return float(string)

def flatten(l,repetitions=1):
    '''A wrapper around the generator-based list flattener (quite fast)'''
    retVal = l
    for i in range(repetitions):
        gen = ( ( k if hasattr(k,'__iter__') else [k] ) for k in retVal ) # Makes the function able to deal with jagged arrays as well
        retVal = [j for i in gen for j in i]
    return retVal
# Use like: flatten(aList,3)

# Another (nominally faster) version of this is: [ l for i in aList for j in i for k in j for l in k ]
# but it's unable to deal with jagged arrays...

def zipflat(*args):
    '''Like zip, but flattens the result'''
    return [j for i in zip(*args) for j in i]
def removeDuplicates(l):
    '''Order preserving duplicate removal... automatically converts lists and arrays (which are unhashable) to nested tuples.
       Modified version of code found here: http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order'''
    seen = set()
    return [ x for x in totuple(l) if x not in seen and not seen.add(x)]
def limitInteriorPoints(l,numInteriorPoints,uniqueOnly=True):
    '''return the list l with only the endpoints and a few interior points (uniqueOnly will duplicate if too few points)'''
    inds = np.linspace(0,len(l)-1,numInteriorPoints+2).round().astype(np.integer)
    if uniqueOnly:
        inds = np.unique(inds)
    return [ l[i] for i in inds ]
def partition(l,n,clip=True):
    '''Partition list "l" into "n"-sized chunks
       clip chops off whatever does not fit into n-sized chunks at the end'''
    length = ( len(l)//n*n if clip else len(l) )
    return [l[i:i+n] for i in range(0,length,n)]
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

# Create the first 100 circles and the first 20 spheres so we don't have to recalculate them if they are small!
# Relatively negligible amount of memory (364450 total points, about 3MB on a 64-bit computer)
circs = [ImageCircle(i) for i in range(1,100)]
shprs = [ImageSphere(i) for i in range(1,20)]
