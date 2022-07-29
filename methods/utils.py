
import numpy as np

from scipy.integrate import simps

from numba import jit
import matplotlib.pyplot as plt

"""
Helper functions for the programs
"""

@jit(nopython=True)
def norm1D(arr, dx):
    """
    Calculates integral of square of discretized wave function arr using a step size of dx
    """
    arr = np.abs(arr)**2
    return 0.5*dx*(sum(2*arr) - arr[0] - arr[-1])


def norm2D(arr, xs, ys):
    """
    Calculates double integral of square of discretized wave function arr
    """
    n = int(np.sqrt(len(arr)))
    mat = np.abs(arr.reshape(n,n))**2
    normX = simps(mat, x=xs, axis=1)
    normXY = simps(normX, x=ys, axis=0)
    return normXY


def find_eigenvalues(M, V):
    """
    Finds eigenvalues of M using eigenvectors given by V
    """
    return np.diag(M.dot(V) / V )
