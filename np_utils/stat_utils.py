'''Utilities for noise generation and statistics by David Mashburn'''

import numpy as np

def uniformSphericalNoise(*shape):
    '''Creates a uniform distributions within the volume of a hyper-sphere.
       (Implementats the Box-Mueller algorithm)'''
    randDir = np.random.randn(*shape).T
    normNoise = (randDir/np.sqrt(np.sum(randDir**2,axis=0))).T
    return np.random.rand(shape[0],1)**(1./shape[1])*normNoise

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
