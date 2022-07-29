import numpy as np
from numpy.linalg import eigvalsh, eigh
import matplotlib.pyplot as plt

from numba import jit

from methods.eigengame import EigenGame
from methods.utils import find_eigenvalues

import time

"""
Takes about 4-5 minutes to run with n=500
"""


# Parameters
OMEGA = 0.2
TOL = 1e-4 # Convergence difference for solution
n = 500 # n needs to be around 500 for accurate eigenvalues

# Domain to solve over
x0 = -14
xf = 14
dx = (xf - x0) / n
xs = np.linspace(x0, xf, n)



@jit(nopython=True)
def matrix(u):
    """
    Returns tridiagonal matrix that we want the eigenvalues of.
    u is a guess for the eigenvector
    """
    n = len(u)
    M = np.zeros((n,n))

    for r in range(n):
        for c in range(n):
            if (c == r-1 or c == r+1):
                M[r][c] = -1/(2*dx**2)
            elif (c == r):
                x = x0 + r*dx
                M[r][c] = (1/dx**2) + (0.5*OMEGA**2)*x**2 + np.abs(u[r])**2
    return M



def main():

    # EigenGame parameters
    k = 3
    rho = 1e-7 # 1e-7 works good I think
    alpha = 0.01
    state = 0 # solution number



    # Loop setup
    guess = np.exp(-OMEGA * xs**2 / 2)
    old = 100*np.ones_like(guess) # dummy value to get into loop
    eigVals = eigVecs = None
    iteration = 0
    

    start = time.time()
    while np.linalg.norm(guess - old) > TOL:
        old = guess
        M = matrix(guess)

        eigVecs = EigenGame(M, k, rho, alpha)
        #eigVals, eigVecs = eigh(M)

        guess = eigVecs.T[state]
        print(f"{iteration}: {np.linalg.norm(guess - old)}")
        iteration += 1
    print(f"Time: {time.time() - start}")


    # Get eigenvalues
    eigVals = find_eigenvalues(matrix(eigVecs.T[state]), eigVecs)



    plt.figure(figsize=[12,5])
    plotnum = 1
    for i in range(k):
        plt.subplot(1,3,plotnum)
        plt.plot(np.linspace(x0, xf, n), eigVecs.T[i])
        lamb = eigVals[i]
        plt.title(rf'$\mu$ = {lamb:.5f}')
        plt.xlabel('x')
        plt.ylabel(r'$\phi(x)$')
        plotnum += 1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

    