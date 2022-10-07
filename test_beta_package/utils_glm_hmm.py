import scipy.special as scsp
import numpy as npn


# ####Including all the simple functions and utilities#####

def sigmoid(x):
    return 1 / (1 + npn.exp(-x))


# rememeber you should give the permutation factor as well but in this case is always 2 (commutative property)

def factorial_perm(x):
    return npn.int32(scsp.factorial(x) / (scsp.factorial(x - 2) * scsp.factorial(2)))
