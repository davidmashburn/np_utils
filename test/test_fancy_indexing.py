from np_utils import *


def test_fancyIndexingList_1():
    assert fL([[1],[2,3],[4,5,6],[7,8,9,10]])[:,0] == [1, 2, 4, 7]
    assert fL([[1],[2,3],[4,5,6],[7,8,9,10]])[2:4,0] == [4, 7]
    assert fL([1,[2,3],[[4,5],[6,7]]])[2,0,1] == 5
    assert fL([[1,7],[2,3],[4,5,6],[7,8,9,10]])[:,(0,1)] == [[1, 7], [2, 3], [4, 5], [7, 8]]
    assert fL([[1,7],[2,3],[4,5,6],[7,8,9,10]])[(0,2),(0,1)] == [[1, 7], [4, 5]]
    
    failed = False
    try:
        fL([1,2,3,4,5])[(1,2)]
    except TypeError:
        failed = True
    assert failed
    
    failed = False
    try:
        fL([1,2,3,4,5])[[1,2]]
    except TypeError:
        failed = True
    assert failed
    
    assert fL([1,2,3,4,5])[(1,2),] == [2,3]
    assert fL([1,2,3,4,5])[[1,2],] == [2,3]
    assert fL([1,2,3,4,5])[((1,2),)] == [2,3]
    assert fL([1,2,3,4,5])[([1,2],)] == [2,3]
    assert fL([1,2,3,4,5])[[[1,2]]] == [2,3]

    assert fL( [[1,7],[2,3],[4,5,6],[7,8,9,10]] )[((0,0),(2,1)),] == [1,5]
    assert fL( [[[[1,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]] )[:, ((0,0),(0,1),1), 1] == \
           [[2, 4, [7,8]], [10, 12, [15,16]]]

def test_fancyIndexingList_2():
    assert fL([{'a': 1}, {'a': 2}])[:, 'a'] == [1, 2]
    assert fL([{'a': [1, 2, 3, 4, 5]}, {'a': [2, 3, 4, 5, 6]}])[:, 'a', 1:3] == [[2, 3], [3, 4]]
    assert fD({'a': [[1, 10], [2, 20], 3, 4, 5]})['a', :2, 1] == [10, 20]
    
    assert fD({'a':[5, {'c': 15}]})['a', 1, 'c'] == 15
    
    failed = False
    try:
        fD({'a': [1], 'b': [2], 'c': [3]})[('a', 'b'), 0]
    except KeyError:
        failed = True
    assert failed
    
    failed = False
    try:
        fD({('a', 'b'): 1})['a', 'b']
    except KeyError:
        failed = True
    assert failed
    
    assert fD({('a', 'b'): 1})[('a', 'b'), ] == 1

def test_fancyIndexingListM1_1():
    assert fLm1([[1],[2,3],[4,5,6],[7,8,9,10]])[:, 1] == [1, 2, 4, 7]
    assert fLm1([[1],[2,3],[4,5,6],[7,8,9,10]])[1:, 1] == [1, 2, 4, 7]
    assert fLm1([[1],[2,3],[4,5,6],[7,8,9,10]])[1:5, 1] == [1, 2, 4, 7]
    assert fLm1([[1],[2,3],[4,5,6],[7,8,9,10]])[3:5, 1] == [4, 7]
    assert fLm1([1,[2,3],[[4,5],[6,7]]] )[3,1,2] == 5
    assert fLm1([[1,7],[2,3],[4,5,6],[7,8,9,10]])[:,(1,2)] == [[1, 7], [2, 3], [4, 5], [7, 8]]
    assert fLm1([[1,7],[2,3],[4,5,6],[7,8,9,10]])[(1,3),(1,2)] == [[1, 7], [4, 5]]
    
    failed = False
    try:
        fLm1([1,2,3,4,5])[(2,3)]
    except TypeError:
        failed = True
    assert failed
    
    failed = False
    try:
        fLm1([1,2,3,4,5])[[2,3]]
    except TypeError:
        failed = True
    assert failed
    
    assert fLm1([1,2,3,4,5])[(2,3),] == [2,3]
    assert fLm1([1,2,3,4,5])[[2,3],] == [2,3]
    assert fLm1([1,2,3,4,5])[((2,3),)] == [2,3]
    assert fLm1([1,2,3,4,5])[([2,3],)] == [2,3]
    assert fLm1([1,2,3,4,5])[[[2,3]]] == [2,3]

    assert fLm1( [[1,7],[2,3],[4,5,6],[7,8,9,10]] )[((1,1),(3,2)),] == [1,5]
    assert fLm1( [[[[1,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]] )[:, ((1,1),(1,2),2), 2] == \
           [[2, 4, [7,8]], [10, 12, [15,16]]]

def test_fancyIndexingListM1_2():
    assert fLm1([{'a': 1}, {'a': 2}])[1:3, 'a'] == [1, 2]
    assert fLm1([{'a': [1, 2, 3, 4, 5]}, {'a': [2, 3, 4, 5, 6]}])[:, 'a', 2:4] == [[2, 3], [3, 4]]
    assert fDm1({'a': [[1, 10], [2, 20], 3, 4, 5]})['a', 1:3, 2] == [10, 20]
    
    assert fDm1({'a':[5, {'c': 15}]})['a', 2, 'c'] == 15
    
    failed = False
    try:
        fDm1({'a': [1], 'b': [2], 'c': [3]})[('a', 'b'), 1]
    except KeyError:
        failed = True
    assert failed
    
    failed = False
    try:
        fDm1({('a', 'b'): 1})['a', 'b']
    except KeyError:
        failed = True
    assert failed
    
    assert fDm1({('a', 'b'): 1})[('a', 'b'), ] == 1

if __name__ == '__main__':
    test_fancyIndexingList_1()
    test_fancyIndexingList_2()
    test_fancyIndexingListM1_1()
    test_fancyIndexingListM1_2()
