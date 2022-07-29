import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt

from methods.eigenvalue import iterate_eig
from methods.utils import norm1D as norm

from numba import jit


# Parameters
OMEGA = 0.2
TOL = 1e-3
n = 200

# Domain to solve over
x0 = -14
xf = 14
dx = (xf - x0) / n
xs = np.linspace(x0, xf, n)


@jit(nopython=True)
def matrix(u, p):
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
                M[r][c] = (1/dx**2) + (0.5*OMEGA**2)*x**2 + (p/dx)*np.abs(u[r])**2
    return M





def main():

    k = 3 # number of eigenvectors to find

    # Norms to solve over
    p0 = 0
    pf = 1
    n_p = 40
    ps = np.linspace(p0, pf, n_p)

    
    # Initial guesses for each branch
    guess0 = np.exp(-OMEGA*xs**2 / 2)
    guess1 = xs * np.exp(-OMEGA*xs**2 / 2)
    guess2 = xs**2 * np.exp(-OMEGA*xs**2 / 2)
    gs = [guess0, guess1, guess2]


    data = {i: [] for i in range(0, k)} # dict to hold arrays of eigvals, indexed by state
    for state in range(0, k):
        guess = gs[state]
        for p in ps:
            # Solve and update guess
            eigVals, eigVecs = iterate_eig(lambda x: matrix(x,p), guess, TOL, k, method='sparse')
            guess = eigVecs.T[state]
            
            data[state].append(eigVals[state])

            print(f'mu = {eigVals[state]}     norm={p}')
        print(f"Done {state}")


    plt.figure(figsize=[10,6])

    for i in range(0, k):
        evals = data[i]
        plt.plot(evals, ps)


    plt.title('Norm of solutions to 1D NLS')
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$N = \int \psi^2dx$')
    plt.xlim(0, k*0.2 + 0.1)
    
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()


