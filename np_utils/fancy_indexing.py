def _make_fancy(*args, **kwds):
    '''Return a fancyIndexingList or a fancyIndexingDict

    Returns a fancyIndexingList if the first argument is a not a dict
    Otherwise returns fancyIndexingDict
    Returns fancyIndexingList if no arguments are passed
    '''
    return (fancyIndexingDict(*args, **kwds)
            if len(kwds) or (len(args) and hasattr(args[0], 'keys')) else
            fancyIndexingList(*args, **kwds))


def _make_fancy_m1(*args, **kwds):
    '''Return a fancyIndexingListM1 or a fancyIndexingDictM1

    Returns a fancyIndexingListM1 if the first argument is a not a dict
    Otherwise returns fancyIndexingDictM1
    Returns fancyIndexingListM1 if no arguments are passed
    '''
    return (fancyIndexingDictM1(*args, **kwds)
            if len(kwds) or (len(args) and hasattr(args[0], 'keys')) else
            fancyIndexingListM1(*args, **kwds))


def _fancy_getitem(fancy_x, index):
    '''Generic __getitem__ for fancyIndexingDict and fancyIndexingList

    Takes a fancy object an index object and returns the result as a
    nested list/dict
    '''
    # Make sure index is always a tuple
    index = index if hasattr(index, '__iter__') else (index, )
    index = index if type(index) is tuple else tuple(index)

    if isinstance(fancy_x, fancyIndexingDict):
        x = fancy_x._simple_getitem(index[0])
        return (x if len(index) <= 1 else
                fancy_x._make_fancy(x)[index[1:]])
    else:  # is a fancyIndexingList
        if len(index) == 1:
            # End recursion and return the result
            # Use simple list indexing unless the index is a tuple of tuples
            return ([fancy_x[i] for i in index[0]]
                    if hasattr(index[0], '__iter__') else
                    fancy_x._simple_getitem(index[0]))
        elif type(index[0]) == slice:
            return [fancy_x._make_fancy(f_i)[index[1:]]
                    for f_i in fancy_x[index[0]]]
        elif hasattr(index[0],  '__iter__'):
            return [fancy_x._make_fancy(fancy_x[i])[index[1:]]
                    for i in index[0]]
        else:
            return fancy_x._make_fancy(fancy_x[index[0]])[index[1:]]


class fancyIndexingDict(dict):
    '''Dictionary that supports "nested" indexing

    Expects to wrap a dictionary that includes other dictionaries or lists

    Examples:
      fD({'a':[5, {'c': 15}]})['a', 1, 'c'] == 15

    While it would be nice to support tuple-indexing like:
      fD({'a': [1], 'b': [2], 'c': [3]})[('a', 'b'), 0] == [1, 2]
    this is not possible because dictionary keys can be tuples

    Cooperatively uses fancyIndexingList as needed

    One caveat is that for _normal_ dict tuple indexing,
    you need to wrap in an extra tuple:
      fD({('a', 'b'): 1})['a', 'b'] --> Error
      fD({('a', 'b'): 1})[('a', 'b'), ] == 1
    '''
    def _make_fancy(self, *args, **kwds):
        return _make_fancy(*args, **kwds)

    def _simple_getitem(self, key):
        return dict.__getitem__(self, key)

    def __getitem__(self, index):
        return _fancy_getitem(self, index)


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
    def _make_fancy(self, *args, **kwds):
        return _make_fancy(*args, **kwds)

    def _simple_getitem(self, index):
        '''Just a wrapper around list.__getitem__()'''
        return list.__getitem__(self, index)  # for a single index, use list's default indexing

    def __getitem__(self, index):
        return _fancy_getitem(self, index)

    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))


class fancyIndexingDictM1(fancyIndexingDict):
    def _make_fancy(self, *args, **kwds):
        return _make_fancy_m1(*args, **kwds)


class fancyIndexingListM1(fancyIndexingList):
    '''fancyIndexingListM1 is overloaded as "fLm1"
       Just like fancyIndexingList, but changes indices to be 1-indexed: fLm1(i)[1] -> i[0]
       VERY useful if dealing with code from Octave, Matlab, Mathematica, etc...
       Documentation for fancyIndexingList:\n\n'''+fancyIndexingList.__doc__
    def m1(self,x): # These don't really need to be in the class, but it's cleaner that way...
        '''minus 1 if x > 0'''
        return (None if x is None else
                x - 1 if x > 0 else
                x)

    def m1gen(self, x):
        '''m1, but able to handle iterables (lists, tuples, ...) and slices as well'''
        if hasattr(x, '__iter__'):
            return lmap(self.m1gen, x)
        elif type(x) is slice:
            return slice(self.m1(x.start), self.m1(x.stop), x.step)
        elif type(x) is int:
            return self.m1(x)
        else:
            print("Not sure what you're feeding me... signed, fancyIndexingListM1.m1gen")
            return self.m1(x)

    def _make_fancy(self, *args, **kwds):
        return _make_fancy_m1(*args, **kwds)

    def _simple_getitem(self,index):
        '''Just a wrapper around list.__getitem__(), but subtracts 1 from index'''
        return list.__getitem__(self, self.m1gen(index)) # for a single index, use list's default indexing


fD = fancyIndexingDict
fDm1 = fancyIndexingDictM1
fL = fancyIndexingList
fLm1 = fancyIndexingListM1