import numpy as np
import matplotlib.pyplot as plt
from numba import jit

import time

from methods.eigenvalue import iterate_eig_full
from methods.utils import norm1D as norm

# Parameters
OMEGA = 0.2
TOL = 1e-8 # Convergence difference for solution
n = 500
k = 6 # how many eigenvectors to find

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

    guess = np.exp(-OMEGA * xs**2 / 2)

    start = time.time()
    eigVals, eigVecs = iterate_eig_full(matrix, guess, TOL, k, method='numpy')
    end = time.time()
    print(f"Time: {end - start}")


    # Check norms of solutions (should all be == dx)
    print(f"dx = {dx}")
    print()
    for v in eigVecs.T:
        print(f'L2 norm     : {np.linalg.norm(v)}') # should all be 1
        print(f'Density norm: {norm(v, dx)}')
        print()




    plt.figure(figsize=[11,6])
    plotnum = 1
    for i in range(6):
        plt.subplot(2,3,plotnum)
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

    