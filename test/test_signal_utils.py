import np_utils
from np_utils import *

def test_entropy_1():
    # I know this is too many conditions in one test -- hey, at least there are tests :)
    a = [0, 1]
    A = np.array([[0, 0], [1, 1]])
    
    assert entropy(a) == 1
    assert entropy(A) == 1
    assert np.all(entropy(A, axis=0) == [0, 0])
    assert np.all(entropy(A, axis=1) == [1, 1])
    assert entropy([0, 0, 0]) == 0
    assert entropy([]) == 0
    assert entropy([5]) == 0

if __name__ == '__main__':
    test_entropy_1()
