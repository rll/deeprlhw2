"""
Functions for dealing with categorical distributions,
specified as 2D arrays where each row specifies a distribution,
i.e., each row is a vector of class probabilities.
"""


import numpy as np, numpy.random as nr
from scipy import special

def cat_sample(prob_nk):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    assert np.allclose(prob_nk.sum(axis=1,keepdims=True),1)
    N = prob_nk.shape[0]
    csprob_nk = np.cumsum(prob_nk, axis=1)
    out = np.zeros(N,dtype='i')
    for (n, csprob_k, r) in zip(xrange(N), csprob_nk, nr.rand(N)):
        for (k,csprob) in enumerate(csprob_k):
            if csprob > r:
                out[n] = k
                break
    return out

def cat_entropy(p):
    """
    Entropy of categorical distribution
    """
    # the following version has problems for p near 0
    #   return (-p * np.log(p)).sum(axis=1)
    return special.entr(p).sum(axis=1) #pylint: disable=E1101

def cat_kl(p, q):
    # the following version has problems for p near 0
    #   return (p*np.log(p/q)).sum(axis=1)
    return special.kl_div(p,q).sum(axis=1) #pylint: disable=E1101