'''Utilities for noise generation and statistics by David Mashburn'''
from __future__ import division

import numpy as np

def gaussian_pdf(x, mu=0, sig=1, use_coeff=True):
    '''A simple function to compute value(s) from a gaussian PDF'''
    coeff = 1 / sig / np.sqrt(2. * np.pi) if use_coeff else 1
    return coeff * np.exp(-np.square(x - mu) / (2. * np.square(sig)))

def uniformSphericalNoise(*shape):
    '''Creates a uniform distributions within the volume of a hyper-sphere.
       (Implementats the Box-Mueller algorithm)
       
       The last element of shape is the number of dimensions of the hyper sphere'''
    randDir = np.random.randn(*shape).T
    normNoise = (randDir/np.sqrt(np.sum(randDir**2,axis=0))).T
    return np.random.rand(shape[0],1)**(1./shape[-1])*normNoise

def _uniformCenteredNoise(*shape):
    return np.random.rand(*shape)*2-1

def addNoise(points,scales=1,dist='gaussian'):
    '''Adds noise to 2D in a variety of ways
       scales: either a single number or a list of numbers which
               multiply be each dimension separately
       dist: one of: 'uniform', 'gaussian', or 'uniformradial
             - uniform uses np.random.rand
             - gaussian uses np.random.randn
             - uniformradial uses the function uniformSphericalNoise
       '''
    fun = { 'uniform'         : np.random.rand,
            'gaussian'        : np.random.randn,
            'uniformradial'   : uniformSphericalNoise,
            'uniformcentered' : _uniformCenteredNoise,
          }[dist]
    noisyPoints = np.array(points)
    noisyPoints += scales * fun(*noisyPoints.shape)
    return noisyPoints

def pearson(x,y):
    return np.corrcoef(x,y)[0,1]

def sample_weights(weights, num=1):
    '''Given a series of (normalized) weights,
       return a set of N randomly sampled indices'''
    return np.searchsorted(np.cumsum(weights), np.random.random(num))

def sample_from_buckets(buckets, weights, num=1):
    '''Given a grouping of buckets and weights, randomly select N buckets'''
    return np.asanyarray(buckets)[sample_weights(weights, num)]

def Bhattacharyya_coefficient(mu1, sig1, mu2, sig2):
    '''Compute the Bhattacharyya coefficient between two normal distributions
    See https://en.m.wikipedia.org/wiki/Hellinger_distance
    
    <math>\scriptstyle P\,\sim\,\mathcal{N}(\mu_1,\sigma_1^2)</math>
    and
    <math>\scriptstyle Q\,\sim\,\mathcal{N}(\mu_2,\sigma_2^2)</math> is:
    <math>
    BC(P, Q) = \sqrt{\frac{2\sigma_1\sigma_2}{\sigma_1^2+\sigma_2^2}} \,
               e^{-\frac{1}{4}\frac{(\mu_1-\mu_2)^2}{\sigma_1^2+\sigma_2^2}}.
    </math>
    '''
    
    sum_sig_sqr = np.square(sig1) + np.square(sig2)
    sigma_ratio_factor = np.sqrt(2. * sig1 * sig2 / sum_sig_sqr)
    exponent = -np.square(mu1 - mu2) / sum_sig_sqr / 4.
    BC = sigma_ratio_factor * np.exp(exponent)
    return BC

gaussian_similarity = Bhattacharyya_coefficient # an alias since Bhattacharyya is so hard to spell :)
