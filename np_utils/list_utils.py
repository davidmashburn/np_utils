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

def partition(l,n,clip=True):
    '''Partition list "l" into "n"-sized chunks
       clip chops off whatever does not fit into n-sized chunks at the end'''
    length = ( len(l)//n*n if clip else len(l) )
    return [l[i:i+n] for i in range(0,length,n)]


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
