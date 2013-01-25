#!/usr/bin/env python
'''Utilities for list and tuple manipulation by David Mashburn.
Notably:
    totuple -> recursive conversion of any nested iterable to tuple
    makeTuple -> like totuple, but for non-iterables, returns (a,)
    iterToX -> like totuple, but for any conversion
    intOrFloat -> converts string to either int if possible, float otherwise
    flatten -> drop 1 (or more if specified) levels of nesting from a data structure
    zipflat -> returns the flattened form of a zip
    removeDuplicates -> order-perserving, recursive duplicate-remover
    partition -> partition a list into n-sized chunks
    fancyIndexingList (alias fL) -> powerful numpy-like indexing for lists,
                                    use like fl(someList)[someIndexingStuff]
    fancyIndexingListM1 (alias fLm1) -> like fL, but subtracts 1 from all indices recursively'''

import operator

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
def intOrFloat(string):
    '''Not sure if your string is formatted as an int or a float? Use intOrFloat instead!'''
    try:
        return int(string)
    except ValueError:
        return float(string)
def floatIntStringOrNone(s):
    '''An even more generic version of intOrFloat... Might be better to use try-except instead of this, but it works for now...'''
    if s=='None':
        return None
    dotC,eC,EC = s.count('.'),s.count('e'),s.count('E')
    weirdDashCount = ( 0 if len(s)==1 else (s[1:].replace('e-','').replace('E-','')).count('-') )
    if dotC>1 or eC>1 or EC>1 or not s.replace('e','').replace('E','').replace('.','').replace('-','').isdigit():
        print 'Error! invalid format!'
        return s
    if dotC>0 or eC>0 or EC>0:
        return float(s)
    else:
        return int(s)
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
def deletecases(l,cases):
    '''Delete all elements of list "cases" from list "l"'''
    if not hasattr(cases,'__iter__'):
        cases=[cases]
    l=copy(l) # work on a copy
    for c in cases:
        for i in range(l.count(c)):
            l.remove(c)
    return l
def partition(l,n,clip=True):
    '''Partition list "l" into "n"-sized chunks
       clip chops off whatever does not fit into n-sized chunks at the end'''
    length = ( len(l)//n*n if clip else len(l) ) # //n*n is a clipping operation...NOT /n**2
    return [l[i:i+n] for i in range(0,length,n)]

def roll(l,n=1):
    '''Roll a list (like numpy.roll) -- For lists only! Uses concatenate with "+"'''
    if hasattr(l,'__iter__'):
        n = n%len(l)
        return l[-n:]+l[:-n]
    else:
        return l

def getMaxDepth(l,depth=0):
    '''Get the maximum depth of any nested structure.
        For a numpy array, this is the same as ndim.'''
    if not hasattr(l,'__iter__'):
        return depth
    else:
        return max([getRecursiveDepth(i,depth+1) for i in l])

def replaceNodesWithNone(l):
    '''A useful way to present the structure of nested objects.
        This can be directly compared from one object to another.'''
    return ( None if not hasattr(l,'__iter__') else
              [replaceNodesWithNone(i) for i in l] )

def applyAtNodes(f,l,*args,**kdws):
    return [ ( applyAtNodes(f,i,*args,**kdws) if hasattr(i,'__iter__') else f(i,*args,**kdws) )
             for i in l ]
def applyAtAllDepths(f,l,*args,**kdws):
    return f([ f( applyAtAllDepths(f,i,*args,**kdws) if hasattr(i,'__iter__') else f(i,*args,**kdws) )
             for i in l ])
def applyAtDepth(f,l,depth,*args,**kdws):
    '''Apply a unary function to any nested structure at a certain depth
        With depth=0, this  is just f(l)).'''
    if depth==0:
        return f(l,*args,**kdws)
    return [ ( ( applyAtDepth(f,i,depth-1,*args,**kdws)
                  if hasattr(i,'__iter__') else
                  i )
                if depth>1 else
                f(i,*args,**kdws) )
             for i in l ]

def applyInfix_ShallowCompare(f,x,y,depth=0):
    '''Apply any function f to the elements of 2 nested list stuctures,
        (like numpy does with + on arrays, but more general).
        Unlike numpy. this defaults to trying to match shallowest structure first:
            a(f,[A,B],[[1,2],[3,4]]) => [[f(A,1),f(A,2)],[f(B,3),f(B,4)]]
        For this to work, x and y must have identical structures down to the point
        where one ends in a node; after that, the other argument can take on any form.
        To put it simpler, if we compare any 2 points in the structures x and y and
        they are both lists, they must have the same length (it is fine if one or both are not a list).'''
    a=applyInfix_ShallowCompare # "a" is just this function, used for recursion
    if hasattr(x,'__iter__'):
        if hasattr(y,'__iter__'):
            if len(x)==len(y):
                return [a(f,x[i],y[i],depth+1) for i in range(len(x))]
            else:
                raise TypeError("Nested structure mismatch at depth "+str(depth)+"!")
        else:
            return [a(f,i,y,depth+1) for i in x]
    else:
        if hasattr(y,'__iter__'):
            return [a(f,x,i,depth+1) for i in y]
        else:
            return f(x,y)
applyInfix = applyInfix_ShallowCompare
def applyInfix_DeepCompare(f,x,y,depth=0,xStructure=None,yStructure=None):
    '''Apply any function f to the elements of 2 nested list stuctures,
        (like numpy does with "+" on arrays, but more general).
        Like numpy, this defaults to trying to match deepest structure first:
            a(f,[A,B],[[1,2],[3,4]]) => [[f(A,1),f(B,2)],[f(A,3),f(B,4)]]
        For this to work, the entire structure of one argument (x or y) must
        be contained at every node in the other argument (y or x)'''
    a=applyInfix_DeepCompare # "a" is just this function, used for recursion
    depthX,depthY = getMaxDepth(x),getMaxDepth(y)
    if xStructure!=None and depthX>depthY:
        raise TypeError("Nested structure mismatch at depth "+str(depth)+"!")
    elif yStructure!=None and depthX<depthY:
        raise TypeError("Nested structure mismatch at depth "+str(depth)+"!")
    elif depthX==depthY:
        xStructure = (replaceNodesWithNone(x) if xStructure==None else xStructure)
        yStructure = (replaceNodesWithNone(y) if yStructure==None else yStructure)
        if xStructure==yStructure:
            return applyInfix_ShallowCompare(f,x,y,depth+1)
        else:
            raise TypeError("Nested structure mismatch at depth "+str(depth)+"!")
    elif depthX==0 or depthY==0:
        return applyFunctionToNestedStructureA(f,x,y,depth+1)
    elif depthX<depthY:
        xStructure = (replaceNodesWithNone(x) if xStructure==None else xStructure)
        return [a(f,x,i,depth+1,xStructure=xStructure) for i in y]
    elif depthX>depthY:
        yStructure = (replaceNodesWithNone(y) if yStructure==None else yStructure)
        return [a(f,i,y,depth+1,yStructure=yStructure) for i in x]

def shallowAdd(x,y):
    return applyInfix_ShallowCompare(operator.add,x,y)
def shallowMul(x,y):
    return applyInfix_ShallowCompare(operator.mul,x,y)
def deepAdd(x,y):
    return applyInfix_DeepCompare(operator.add,x,y)
def deepAdd(x,y):
    return applyInfix_DeepCompare(operator.mul,x,y)

def interp(l,index):
    '''Basically floating point indexing with interpolation in between.'''
    m = index % 1
    if m==0:
        return l[index]
    else:
        indexA,indexB = int(index), int(index) + (1 if index>=0 else -1)
        return l[indexA]*(1-m) + l[indexB]*(m)

def interpGen(l,index):
    '''Just like interp except that it uses the generic shallowAdd and shallowMul.'''
    m = index % 1
    if m==0:
        return l[index]
    else:
        indexA,indexB = int(index), int(index) + (1 if index>=0 else -1)
        return shallowAdd( shallowMul( l[indexA],(1-m) ),
                            shallowMul( l[indexB],m ) )

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
