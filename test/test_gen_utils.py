import sys

PYTHON2 = sys.version_info < (3, 0)

import np_utils
from np_utils import *

def test_floatIntStringOrNone_1():
    assert floatIntStringOrNone('None') is None

def test_floatIntStringOrNone_2():
    assert floatIntStringOrNone('1') == 1

def test_floatIntStringOrNone_2():
    assert floatIntStringOrNone('1.5') == 1.5

def test_floatIntStringOrNone_3():
    if PYTHON2:
        assert floatIntStringOrNone('15L') == 15

def test_floatIntStringOrNone_4():
    assert floatIntStringOrNone('1e5') == 1e5

def test_floatIntStringOrNone_5():
    assert floatIntStringOrNone('a') == 'a'


if __name__ == '__main__':
    test_floatIntStringOrNone_1()
    test_floatIntStringOrNone_2()
    test_floatIntStringOrNone_2()
    test_floatIntStringOrNone_3()
    test_floatIntStringOrNone_4()
    test_floatIntStringOrNone_5()
