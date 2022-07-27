import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from numba import jit

from methods.eigengame import EigenGame
from methods.utils import find_eigenvalues

import time



OMEGA = 0.2
TOL = 1e-6 # Convergence difference for solution
n = 150 # n needs to be around 500 for accurate eigenvalues

x0 = -14
xf = 14
dx = (xf - x0) / n
xs = np.linspace(x0, xf, n)



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

def F(V):
    return H() + 2*V @ V.T



def main():

    k = 3
    rho = 1e-7 # 1e-7 works good I think
    alpha = 0.01


    # Initial guess
    V = [
        np.exp(-OMEGA * xs**2 / 2),
        xs * np.exp(-OMEGA * xs**2 / 2),
        xs**2 * np.exp(-OMEGA * xs**2 / 2)
    ]
    V = np.array(V).T


    # Loop setup
    eigVals = eigVecs = None
    iteration = 0
    evals = []

    start = time.time()
    for state in range(0, k):
        guess = V.T[state]
        old = 10*guess # dummy val to get into loop
        while np.linalg.norm(guess - old) > TOL:
            old = guess
            M = F(V)

            eigVecs = EigenGame(M, k, rho, alpha) # issue is that we need all eigenvectors, not just first k?
            # eigVals, eigVecs = eigh(M)


            guess = eigVecs.T[state]
            V = eigVecs

            print(f"{iteration}: {np.linalg.norm(guess - old)}")
            iteration += 1
        evals.append( find_eigenvalues(F(V), V)[state] )
    print(f"Time: {time.time() - start}")




    plt.figure(figsize=[12,5])
    plotnum = 1
    for i in range(k):
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

    