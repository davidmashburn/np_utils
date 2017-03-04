#!/usr/bin/env python
'''Utilities for list and tuple manipulation by David Mashburn.
Most notably:

flatten -> drop 1 (or more if specified) levels of nesting from a data structure
zipflat -> returns the flattened form of a zip
ziptranspose -> pure-function version of zip(*_)
deletecases -> remove all occurrences of each of cases from a list
removeDuplicates -> order-perserving, duplicate-remover
partition -> partition a list into n-sized chunks
roll -> list version of numpy.roll
interp -> floating point "indexing" using linear interpolation when necessary

totuple -> recursive conversion of any nested iterable to tuple
makeTuple -> like totuple, but for non-iterables, returns (a,)
getMaxDepth -> get the maximum depth level of a nested list
iterToX -> like totuple, but for any conversion

fancyIndexingList (alias fL) -> powerful numpy-like indexing for lists,
                                use like fl(someList)[someIndexingStuff]
fancyIndexingListM1 (alias fLm1) -> like fL, but subtracts 1 from all indices recursively

There are other functions here as well, but these are the most useful/novel
'''

from __future__ import print_function
from builtins import next, zip, map, str, range
from past.builtins import basestring

from copy import deepcopy
import operator
from itertools import chain, islice
from collections import Counter

from future.utils import viewitems, lzip, lmap, lrange


#################################
## Nested structure conversion ##
#################################

def totuple(a, break_strings=True):
    '''Makes tuples out of nested datastructures like lists and arrays.
       Modified version of algorithm by Bi Rico:
       http://stackoverflow.com/questions/10016352/convert-numpy-array-to-tuple
       Modified to handle strings without throwing recursion errors
       (since 'a'[0] == 'a')
       
       By default, uses break_strings=True, same handling as tuple itself:
       totuple('abcdef') == totuple('abcdef') == ('a', 'b', 'c', 'd', 'e', 'f')
       
       Using break_strings=False leaves larger strings alone:
       totuple('abcdef', break_strings=False) == 'abcdef'
       
       totuple is especially effective for creating hashable structures
       for use in dict and set'''
    if isinstance(a, basestring) and (len(a) == 1 or not break_strings):
        return a
    try:
        return tuple(totuple(i, break_strings) for i in a)
    except TypeError: # dig until we can dig no more!
        return a

def makeTuple(a, break_strings=True):
    '''Like totuple, but ensures that you get a tuple out.'''
    retVal = totuple(a, break_strings)
    return ( retVal if retVal.__class__==tuple else (retVal,) )

def tolist(a, break_strings=True):
    '''Makes lists out of nested datastructures like tuples, lists, and arrays.
       Based on totuple, see docs for more information.'''
    if isinstance(a, basestring) and (len(a) == 1 or not break_strings):
        return a
    try:
        return [tolist(i, break_strings) for i in a]
    except TypeError:
        return a

def iterToX(f, iterable, break_strings=True):
    '''Generalized version of tolist/totuple.
       Replace all iterables in a nested structure with another type (X) using
       the constructor function "f".
       "f" must take an iterable as it's argument.'''
    if isinstance(iterable, basestring) and (len(iterable) == 1 or not break_strings):
        return iterable
    try:
        return f([iterToX(f, i, break_strings) for i in iterable])
    except TypeError:
        return iterable

def iterToX_splat(f, iterable, break_strings=True):
    '''Generalized version of tolist/totuple.
       Replace all iterables in a nested structure with another type using
       the constructor function "f" that takes multiple arguments.
       "f" must take multiple arguments, like sympy.Tuple, for example.'''
    if isinstance(iterable, basestring) and (len(iterable) == 1 or not break_strings):
        return iterable
    try:
        return f(*[iterToX_splat(f, i, break_strings) for i in iterable])
    except TypeError:
        return iterable

def listify(x):
    '''A simple function to wrap non-lists in []'s
       Helps in the case when arguments can optionally
       be lists or singletons.
       This is explicitly ONLY for lists and tuples, mostly
       to avoid also catching strings.'''
    return (x if isinstance(x, list) else
            list(x) if isinstance(x, tuple) else
            [x])

def coerce_to_target_length(x, target_length):
    '''Coerce singleton values and one-element lists to the target length,
       otherwise test that len(x) == target_length and throw an error otherwise'''
    if not hasattr(x, '__len__'):
        x = [x]
    if len(x) == 1:
        x = [x[0]] * target_length
    assert len(x) == target_length, 'x cannot be coerced to the target length!'
    return x

##################
## List testing ##
##################

def find_first_occurrence_in_list(l):
    '''Find the first occurrence (index) of each unique element in l.
       Modified version of code found here:
       http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order
       For arrays, use np_utils.find_first_occurrence or np_utils.find_first_occurrence_1d'''
    seen = set()
    return np.array([i for i, p in enumerate(l)
                     for s in totuple(p)     # with s as totuple(a)   <-- would be this if python supported it
                     if s not in seen and not seen.add(s)])

def has_duplicates(l):
    '''For a 1D list, test whether there are any duplicated elements.'''
    return len(l) > len(set(l))

def getMostCommonVal(l):
    '''Get the most-occuring value in a list.
       When multiple values occur the same number of times, returns the minimum one
       Example:
           _getMostCommonVal([1,2,4,3,4,5,6,3,5]) -> 3'''
    return Counter(l).most_common()[0][0]

def iter_to_list(l):
    '''Convert any iterable without a length to a list
       Otherwise pass it though unchanged (list, tuple, array, etc)'''
    return l if hasattr(l, '__len__') else list(l)

def all_equal(l, equality_function=operator.eq):
    '''Test if all elements in a list are the same'''
    l = iter_to_list(l)
    if len(l)<2:
        return True
    
    l0 = l[0]
    return all([equality_function(i, l0) for i in l])

def assertSameAndCondense(l, message='List values differ!',
                          equality_function=operator.eq):
    '''Take a list of values that should all be the same, assert that this is true,
       and then return the common value
       This acts as a safe funnel in exploratory data processing,
       cutting a large same-valued list down to a single value.'''
    l = iter_to_list(l)
    assert all_equal(l), message
    return l[0]

#####################################################
# Transformations: reshaping, splitting, filtering ##
#####################################################

# Reshaping:

def flatten(l,repetitions=1):
    '''A wrapper around the generator-based list flattener (quite fast)
       
       Use like: flatten(aList,3)
       
       Another (nominally faster) version of this is: [ l for i in aList for j in i for k in j for l in k ]
       but it's unable to deal with jagged arrays.
       
       A terse but slower version of flatten (1 repetition) can also be acheived with: sum(aList,[])
       
       An even faster version of flatten (1 repetition, non-jagged arrays)
       can be acheived with itertools.chain: list(chain(*l))
       '''
    retVal = l
    for i in range(repetitions):
        gen = ( ( k if hasattr(k,'__iter__') else [k] ) for k in retVal ) # Makes the function able to deal with jagged arrays as well
        retVal = [j for i in gen for j in i]
    return retVal

def zipflat(*args):
    '''Like zip, but flattens the result.
       (itertools.izip is used internally for speed/memory usage)'''
    return [j for i in zip(*args) for j in i]

def ziptranspose(l):
    '''Tranpose the two outer dimensions of a nest list
       (This is just a wrapper around zip(*l) that returns a list)'''
    return lzip(*l)

def partition_range(x, n):
    '''Return a generator for a series of chunked ranges that end at x
       partition_range(x, 1) <==> range(x)
       Examples:
          list(partition_range(19, 10)) -> [range(0, 10), range(10, 19)]
          list(partition_range(20, 10)) -> [range(0, 10), range(10, 20)]
          list(partition_range(21, 10)) -> [range(0, 10), range(10, 20), range(20, 21)]
       '''
    return (range(i, min(x, i+n)) for i in range(0, x, n))

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

def zipIntoPairs(l,cycle=False,offset=1):
    '''Form all adjacent pairs from a list
       If cycle is True, pair the end and beginning as well
       If offset is greater than 1, pair separated elements'''
    if cycle:
        return lzip(l,roll(l,-offset))
    else:
        return lzip(l[:-offset],l[offset:])

def intersperse(l,separator):
    '''Intersperse a separator between all items
       
       Example:
       >>> interperse([1,2,3,4],42)
       [1,42,2,42,3,42,4]
       
       Works by zipping together the list and a list of separators, then
       flattening the result and clipping off the end:
         flatten(zip( l, [sep]*len(l) ))[:-1]
       
       With itertools, this could be written as the longer (but ~3x faster):
         list(islice(chain(*izip( l, [sep]*len(l) )),0,2*len(l)-1))
       
       This horrific list comprehension is also faster:
         [ k for j in izip( l, [sep]*len(l) ) for k in j ][:-1]'''
    
    return flatten(zip( l, [separator]*len(l) ))[:-1]

# Splitting:

def split_at(l, i):
    '''Split a list into the parts before and after i
       Literally just returns l[:i], l[i:]
       For more complex scenarios, use something like split_at_boundaries'''
    return l[:i], l[i:]

def split_at_boundaries(l, boundaries):
    '''Split a list at the boundaries (which must be a sorted list)
       Endpoints are optional and will be ignored
       This function uses itertools in case "boundaries" is very large
       (avoids any intermediate list or array creation)
       This function works on and is fast for any iterable, including arrays'''
    lenl = len(l)
    boundaries = boundaries[(1 if boundaries[0] == 0 else 0):
                            (-1 if boundaries[-1] == lenl else None)]
    if not len(boundaries):
        return [l]
    
    start_end = chain([[0, boundaries[0]]],
                      zip(islice(boundaries, 0, lenl-1),
                          islice(boundaries, 1, None)),
                      [[boundaries[-1], lenl]])
    count = len(boundaries) + 1
    return [l[start:end] for start, end in start_end]

def _function_ize(x):
    '''Ensure that x is a function
       When x is a function, return it
       When x is an iterable, build a function that steps
         through the items sequentially (using a generator)
       When x is a non-iterable value, build a function
         that always returns that value
       (This really just a helper function for split_list_on_condition)'''
    if hasattr(x, '__call__'):
        # x is already a function
        pass
    elif hasattr(x, '__iter__'):
        # x is a list of some kind, so use a generator to step through the items
        g = (i for i in x)
        x = lambda i: next(g)
    else:
        # x is a boolean value, so make a simple function
        bool_val = x
        x = lambda i: bool_val
    
    return x

def split_list_on_condition(l, cond):
    '''Split list "l" based on condition "cond" which can be:
          * a function returning booleans
          * an iterable of booleans
          * a single boolean'''
    cond = _function_ize(cond)
    true_list, false_list = [], []
    for i in l:
        (true_list if cond(i) else false_list).append(i)
    return true_list, false_list

# Filtering:

def deletecases(l,cases):
    '''Delete all elements of list "cases" from list "l"
       For large numbers of cases, pass cases as a set for speed'''
    if not hasattr(cases,'__iter__'):
        cases=[cases]
    return [ i for i in l if i not in cases ]

def removeDuplicates(l):
    '''Order preserving duplicate removal.
       Automatically converts lists and arrays (which are unhashable) to nested tuples.
       Modified version of code found here:
       http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order'''
    seen = set()
    return [ x for x in totuple(l)
               if x not in seen and not seen.add(x) ]

def removeAdjacentDuplicates(l):
    '''Replace any groups of items with the same value with a single occurrence instead.
       Ex:
       removeAdjacentDuplicates([1,2,3,0,0,1,2,0,0,1,1,1,2]) --> [1,2,3,0,1,2,0,1,2]'''
    return [ l[i] for i in range(len(l)-1)
                  if l[i]!=l[i+1] ] + l[-1:]

####################################################
## Grouping, connections, chaining, interpolation ##
####################################################

def groupByFunction(l,f,appendFun=None):
    '''Break up a list into groups (a dict of smaller lists) based on a
       common property (the result of the function f applied to each element).
       Optionally, transform the elements before appending them with appendFun.
       
       Examples:
       >>> groupByFunction([1,2,3,4,5,6,7,8],lambda x:x<5)
       {False: [5, 6, 7, 8], True: [1, 2, 3, 4]})
       >>> groupByFunction([(1,1),(1,2),(-1,3),(-1,4)],lambda x:x[0])
       {1: [(1, 1), (1, 2)], -1: [(-1, 3), (-1, 4)]}
       >>> groupByFunction([(1,1),(1,2),(-1,3),(-1,4)],lambda x:x[0],lambda x:x[1])
       {1: [1, 2], -1: [3, 4]}
       
       where obviously the lambdas in the last 2 examples could be
       replaced with calls to operator.itemgetter instead
       
       This method is based on an example given for collections.defaultdict in:
       http://docs.python.org/2/library/collections.html
       
       Another simple way to acheive this would be using itertools.groupby:
       
       from future.utils import lmap
       from itertools import groupby
       if appendFun==None:
           return {k: list(v) for k, v in groupby(l,f)}
       else:
           return {k: lmap(appendFun, v) for k, v in groupby(l,f)}
       I may change this in a future version after testing it more.
       '''
    groupDict = {}
    for i in l:
        groupDict.setdefault(f(i),[]).append(appendFun(i) if appendFun else i)
    return groupDict

def getElementConnections(connectionSets):
    '''Take a list of connections and return a dictionary of connections to each element.
       Examples:
       >>> getElementConnections([[1,2],[4,5],[2,6]]):
       {1:[2], 2:[1,6], 4:[5], 5:[4], 6:[2]}
       >>> getElementConnections([[1,2],[2,3,4]]):
       {1:[2], 2:[1,3,4], 3:[2,4], 4:[2,3]}
       
       connectionSets must be a nested structure with a depth of 2 (i.e. a list-of-lists)
       and all elements must be immutables (for use in a set)'''
    # Algorithm summary:
    # Form a master list of elements (and make them keys to the dictionary)
    # For any given element, get the connection sets that contain the element
    #   and distill down unique connections from these
    return { k:sorted( set(flatten( i for i in connectionSets
                                      if k in i )).difference([k]) )
            for k in sorted(set(flatten(connectionSets))) }

    # Equivalent but more complex version
    # Speed is similar; better under some circumstances, worse under others
    ##keys = sorted(set(flatten(connectionSets)))
    ##d = { k:set() for k in keys }
    ##for c in connectionSets:
    ##    cs = sorted(set(c))
    ##    for e in cs:
    ##        d[e].update(deletecases(cs,e))
    ##for k,v in d.iteritems():
    ##    d[k] = sorted(v)
    ##return d

def _firstOrOther(l,other=None):
    '''Helper function for getChainsFromConnections
       Returns the first element if it exists and if not,
         returns "other" (default None)'''
    return l[0] if len(l) else other

def getChainsFromConnections(connections,checkConnections=True):
    '''Take a list of connections and return a list of connection chains
       connections is a dictionary of connections between elements (which must be hashable)
         and can be generated using getElementConnections
       The checkConnections option tests that there is only one one path
         through each point (aka 2 or fewer connections, no branching)
       Returns a list of chains (lists of elements)
       '''
    connections = deepcopy(connections) # Protect the input from modification
    if checkConnections: # Check that there is no branching
        assert all( len(v)<3 for k,v in viewitems(connections) ), 'Aborting; this network has branching'
    
    chains = []
    while len(connections): # loop over possible chains
        # Pick a starting point (an end point if possible)
        currPt = _firstOrOther([pt for pt,conn in viewitems(connections)
                                   if len(conn)==1],
                               next(iter(connections))) # was connections.keys()[0]
        # Form a chain and move the current point forward
        chain = [currPt]
        currPt = connections.pop(currPt)[0]
        while currPt: # loop to fill a chain, stop on an invalid
            chain.append(currPt)
            if len(connections)==0:
                break
            connections[currPt] = deletecases(connections[currPt], [chain[-2]])
            currPt = _firstOrOther(connections.pop(currPt,[]))
        chains.append(chain)
    return chains

def interp(l,index):
    '''Basically floating point indexing with interpolation in between.'''
    m = index % 1
    indexA=int(index)
    if m==0:
        return l[indexA]
    else:
        indexB = indexA + (1 if index>=0 else -1)
        return l[indexA]*(1-m) + l[indexB]*(m)

#####################################
## Some utilities for dictionaries ##
#####################################

def dict_apply(f, d):
    return {k: f(v) for k, v in viewitems(d)}

def dict_key_union(dicts):
    '''Form the superset (union) of all keys in all dictionaries'''
    keys = set()
    for d in dicts:
        if d is not None:
            keys.update(d.keys())
    return list(keys)

def rotate_list_of_dicts(ld):
    '''Change from a list of dicts to a dict of lists
       Fill missing entries with "None"
       This is not very efficient, but it works.'''
    return {k: [(None if i is None or
                         k not in i else
                 i[k])
                for i in ld]
            for k in dict_key_union(ld)}

def rotate_dict_of_lists(dl):
    '''Change from a dict of lists to a list of dicts
       Skips entries for shorter lists'''
    num_dicts = max(map(len, dl.values()))
    return [{k: v[i]
             for k, v in viewitems(dl)
             if i < len(v)}
            for i in range(num_dicts)]

def key_collector(x, collection=None):
    '''Recursively collect all the keys that a nested structure at all depths
       x can be a <list/dict> of <list/dict/other> of ...
       with any structure'''
    collection = set()
    
    if hasattr(x, 'keys') and hasattr(x, 'values'):
        collection.update(x.keys())
        for v in x.values():
            collection.update(key_collector(v))
    elif hasattr(x, '__iter__'):
        for i in x:
            collection.update(key_collector(i))
    
    return collection

def directional_key_collector(x, collection=None, last_key='<TOP LEVEL>', skip_list=False):
    '''Recursively collect all the keys that a nested structure at all depths
       x can be a <list/dict> of <list/dict/other> of ...
       with any structure
       This version returns pairs of keys like:
       (last_key, next_key)
       where next_key is inside a nested structure containing last_key
       special keys called <TOP_LEVEL> and <LIST> are used for last_key
       when the key's dictionary is the object or is in inside a list'''
    collection = set()
    
    if hasattr(x, 'keys') and hasattr(x, 'values'):
        for k, v in x.iteritems():
            collection.add((last_key, k))
            collection.update(directional_key_collector(v, last_key=k, skip_list=skip_list))
    elif hasattr(x, '__iter__'):
        lk = last_key if skip_list else '<LIST>'
        for i in x:
            collection.update(directional_key_collector(i, last_key=lk, skip_list=skip_list))
    
    return collection

##########################################
## Some utilities for nested structures ##
##########################################

def getMaxDepth(l,depth=0):
    '''Get the maximum depth of any nested structure.
        For a numpy array, this is the same as ndim.'''
    if not hasattr(l,'__iter__') or not len(l):
        return depth
    else:
        return max([getMaxDepth(i,depth+1) for i in l])

def getBoundingShape(l):
    '''Get the minimum shape of an ND-array that will hold this structure.
       For a numpy array, this is the same as shape.
       This function can only be called on iterables and will fail otherwise;
       (atomic objects have no shape)'''
    shape = []
    childShapes = [ getBoundingShape(i) for i in l if hasattr(i,'__iter__') ]
    if len(childShapes)>0:
        maxLen = max([ len(s) for s in childShapes ])
        for i in range(maxLen):
            shape.append( max( [ s[i] for s in childShapes if len(s)>i] ) )
    return (len(l),)+tuple(shape)

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
        (like numpy does with "+" on arrays, but more general).
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
        return applyInfix_ShallowCompare(f,x,y,depth+1)
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

def deepMul(x,y):
    return applyInfix_DeepCompare(operator.mul,x,y)

def listAdd(x,y,matchTopDown=True):
    '''matchTopDown=True uses ShallowCompare
       matchTopDown=False uses DeepCompare (slower)'''
    _applyInfix = ( applyInfix_ShallowCompare if matchTopDown else
                   applyInfix_DeepCompare )
    return _applyInfix(operator.add,x,y)

def listSub(x,y,matchTopDown=True):
    '''matchTopDown=True uses ShallowCompare
       matchTopDown=False uses DeepCompare (slower)'''
    _applyInfix = ( applyInfix_ShallowCompare if matchTopDown else
                   applyInfix_DeepCompare )
    return _applyInfix(operator.sub,x,y)

def listMul(x,y,matchTopDown=True):
    '''matchTopDown=True uses ShallowCompare
       matchTopDown=False uses DeepCompare (slower)'''
    _applyInfix = ( applyInfix_ShallowCompare if matchTopDown else
                   applyInfix_DeepCompare )
    return _applyInfix(operator.mul,x,y)

def listDiv(x,y,matchTopDown=True,useIntegerDivision=False):
    '''matchTopDown=True uses ShallowCompare
       matchTopDown=False uses DeepCompare (slower)
       
       useIntegerDivision=True uses interger division (//)
       instead of true division'''
    _applyInfix = ( applyInfix_ShallowCompare if matchTopDown else
                   applyInfix_DeepCompare )
    _div = ( operator.div if useIntegerDivision else
             operator.truediv )
    return _applyInfix(_div,x,y)

def listIncr(x):
    return listAdd(x,1)

def listDecr(x):
    return listAdd(x,-1)


def interpGen(l,index):
    '''Just like interp except that it uses the generic shallowAdd and shallowMul.'''
    m = index % 1
    indexA=int(index)
    if m==0:
        return l[indexA]
    else:
        indexB = (1 if index>=0 else -1)
        return shallowAdd( shallowMul( l[indexA],(1-m) ),
                            shallowMul( l[indexB],m ) )

class fancyIndexingList(list):
    '''fancyIndexingList is overloaded as "fL"
       Gives lists the magical properties of numpy arrays, but without requiring regular shapes...
       Also use it where you would convert a list to a numpy array and back again like: np.array(l)[:,3].tolist()
       Use like: fL(ANY_LIST)[<any valid numpy slice>]
       Examples:
         fL( [[1],[2,3],[4,5,6],[7,8,9,10]] )[:,0]  ->  [1, 2, 4, 7]
         fL( [[1],[2,3],[4,5,6],[7,8,9,10]] )[2:4,0]  ->  [4, 7]
         fL( [1,[2,3],[[4,5],[6,7]]] )[2,0,1]  ->  5
       fL also has the extra feature of being able to use lists-within-lists when indexing
       Examples:
         fL( [[1,7],[2,3],[4,5,6],[7,8,9,10]] )[:,(0,1)]  ->  [[1, 7], [2, 3], [4, 5], [7, 8]]
         fL( [[1,7],[2,3],[4,5,6],[7,8,9,10]] )[(0,2),(0,1)]  ->  [[1, 7], [4, 5]]
       Beware that you will need to nest lists or tuples if you only have a single index:
         fL([1,2,3,4,5])[(1,2)] -> TypeError -- this is equivalent to fl([1,2,3,4,5])[1,2]
         fL([1,2,3,4,5])[[1,2]] -> TypeError -- ""
         fL([1,2,3,4,5])[(1,2),] -> [2,3]
         fL([1,2,3,4,5])[[1,2],] -> [2,3]
         fL([1,2,3,4,5])[((1,2),)] -> [2,3]
         fL([1,2,3,4,5])[([1,2],)] -> [2,3]
         fL([1,2,3,4,5])[[[1,2]]] -> [2,3]
       
       And, fL indices can also be nested; this gives a new list which has the (0,0) element and the (2,1) element:
         fL( [[1,7],[2,3],[4,5,6],[7,8,9,10]] )[((0,0),(2,1)),] -> [1,5]
       And in case your head doesn't hurt by now (mine does), here is an example that combines all of the above,
       indexing a 2x2x2x2 square list:
         fL( [[[[1,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]] )[:, ((0,0),(0,1),1), 1]
          -> [[2, 4, [7,8]], [10, 12, [15,16]]]
       This type of usage is NOT recommended, because it is so opaque and covoluted.
       (It's actually just a consequence of the implementation.)
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
            return lmap(self.m1gen,x)
        elif type(x)==slice:
            return slice(self.m1(x.start),self.m1(x.stop),x.step)
        elif type(x)==int:
            return self.m1(x)
        else:
            print("Not sure what you're feeding me... signed, fancyIndexingListM1.m1gen")
            return self.m1(x)
    def new_fL(self,*args,**kwds):
        '''Just a wrapper around the class constructor for a new instance, fancyIndexingListM1()'''
        return fancyIndexingListM1(*args,**kwds)
    def _list_getitem(self,index):
        '''Just a wrapper around list.__getitem__(), but subtracts 1 from index'''
        return list.__getitem__(self,self.m1gen(index)) # for a single index, use list's default indexing
fLm1 = fancyIndexingListM1

def genericSlice(length,start=None,stop=None,step=None,includeStop=False,oneBased=False,checkBounds=False):
    '''A generic version of "slice" supporting:
       * dropping or keeping the stop value
       * 0-based and 1-based indexing
       * optional bounds checking
       * +/-/None indexes (as usual)
       
       Requires the length of the list that will be sliced.
       
       Returns a slice object.
       (failure results in a None value at the moment)'''
    maxStop = (length-1 if includeStop else length)
    
    # Always fail if step is 0
    if step==0:
        return None
    
    # Fail on 0-values in 1-based indexing when checking bounds
    if checkBounds and oneBased and (start==0 or stop==0):
        return None
    
    # Wrap negative values to positive ones and then convert from
    # 1-based to 0-based indexing
    if start!=None:
        if start<0:
            start += length
        elif oneBased:
            start -=1
    if stop!=None:
        if stop<0:
            stop += length
        elif oneBased:
            stop -=1
    
    # If either value goes negative and we are not checking bounds,
    # then set it to None avoid wrap-around
    if not checkBounds:
        if start<0:
            start=None
        if stop<0:
            stop=None
    
    # Handle all the None cases
    if step==None:
        step=1
    if step>0:
        if start==None:
            start = 0
        if stop==None:
            stop = maxStop
    else:
        if start==None:
            start = length-1
        if stop==None:
            stop = (0 if includeStop else -1)
    
    # Fail on out-of-bounds if checkBounds is specified (still somewhat lenient)
    if checkBounds:
        if ( not 0<=start<length or
             not 0<=stop<length or
             step>0 and start-stop>1 or
             step<0 and stop-start>1 ):
            return None
    
    # Increment (or decrement) the stop value to include it
    if includeStop:
        # Add the sign of the step to the stop
        stop += (1 if step>0 else -1)
        
    # If this made stop negative, set it to None avoid wrap-around
    if stop<0:
        stop=None
    
    return slice(start,stop,step)
