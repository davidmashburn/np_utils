from __future__ import division
import np_utils
from np_utils import *

def test_gaussian_pdf_1():
    assert gaussian_pdf(0, use_coeff=False) == 1

def test_gaussian_pdf_2():
    assert gaussian_pdf(1, 1, use_coeff=False) == 1

def test_gaussian_pdf_3():
    assert gaussian_pdf(10) == gaussian_pdf(-10)

def test_gaussian_pdf_4():
    assert gaussian_pdf(1) == 1 / np.sqrt(np.e * 2 * np.pi)

def test_sample_weights_1():
    n = 10
    w = sample_weights([1./n]*n, 1000)
    assert np.all(0<=w)
    assert np.all(w<n)

def test_sample_weights_2():
    w = sample_weights([0.5, 0, 0.5], 1000)
    assert np.all(w!=1)

def test_sample_from_buckets_1():
    s = sample_from_buckets(['a', 'b', 'c'], [0.5, 0.5, 0], num=1000)
    assert 'a' in s
    assert 'b' in s
    assert 'c' not in s

def test_Bhattacharyya_coefficient_1():
    assert Bhattacharyya_coefficient(0, 1, 0, 1) == 1

def test_Bhattacharyya_coefficient_2():
    assert Bhattacharyya_coefficient(0, 1, 1, 1) == np.exp(-1./8)

def test_Bhattacharyya_coefficient_3():
    assert Bhattacharyya_coefficient(0, 1, 20, 1) < 1e-10

if __name__ == '__main__':
    test_gaussian_pdf_1()
    test_gaussian_pdf_2()
    test_gaussian_pdf_3()
    test_gaussian_pdf_4()
    test_sample_weights_1()
    test_sample_weights_2()
    test_sample_from_buckets_1()
    test_Bhattacharyya_coefficient_1()
    test_Bhattacharyya_coefficient_2()
    test_Bhattacharyya_coefficient_3()
