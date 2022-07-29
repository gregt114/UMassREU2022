import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from numba import jit

from methods.eigengame import EigenGame as parallel_EG
from methods.utils import find_eigenvalues
from methods.sequential_EG import EigenGame as sequential_EG

import time


# Parameters
OMEGA = 0.2
TOL = 1e-6 # Convergence difference for solution
n = 40 # n needs to be around 500 for accurate eigenvalues

# Domain to solve over
x0 = -14
xf = 14
dx = (xf - x0) / n
xs = np.linspace(x0, xf, n)


# Laplacian + potential
@jit(nopython=True)
def H():
    M = np.zeros((n,n))

    for r in range(n):
        for c in range(n):
            if (c == r-1 or c == r+1):
                M[r][c] = -1/(2*dx**2)
            elif (c == r):
                x = x0 + r*dx
                M[r][c] = (1/dx**2) + (0.5*OMEGA**2)*x**2
    return M


# Full matrix to find eigenvalues for
def F(V):
    return H() + 2*V.dot(V.T)



def main():

    # EigenGame parameters
    k = 40 # number of eigenvectors to find
    rho = 1e-6
    alpha = 0.0005


    # Randomize initial guess (n x k matrix)
    V = np.random.rand(n, k)


    # Loop setup
    eigVals = eigVecs = None
    iteration = 0
    evals = []

    start = time.time()
    for state in range(0, 3):
        guess = V.T[state]
        old = 10*guess # dummy val to get into loop
        while np.linalg.norm(guess - old) > 0.01:
            old = guess
            M = F(V)

            # TODO: is issue that we need all eigenvectors, not just first k?
            #eigVecs = parallel_EG(M, k, rho, alpha) # parallel EigenGame
            eigVecs = sequential_EG(M, k, rho, alpha, mod=20, order='small') # sequential EigenGame (tends to work better than parallel here)

            #eigVals, eigVecs = eigh(M) # solve with regular eigenvalue solver


            guess = eigVecs.T[state]
            V = eigVecs

            print(f"{iteration}: {np.linalg.norm(guess - old)}")
            iteration += 1
        print(f'Done {state}')
        evals.append( find_eigenvalues(F(V), V)[state] )
    print(f"Time: {time.time() - start}")




    plt.figure(figsize=[12,5])
    plotnum = 1
    for i in range(3):
        plt.subplot(1,3,plotnum)
        plt.plot(xs, V.T[i])
        lamb = evals[i]
        plt.title(rf'$\mu$ = {lamb:.5f}')
        plt.xlabel('x')
        plt.ylabel(r'$\phi(x)$')
        plotnum += 1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

    