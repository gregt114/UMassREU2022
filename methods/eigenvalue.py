import numpy as np
from numpy.linalg import eigh, norm
from scipy.sparse.linalg import eigsh

from numba import jit



def iterate_eig_full(F, guess, tol, k, method='numpy'):
    """
    Apply the method of iterated eigenvalues to funtion F
    F: vector --> matrix
    guess: vector
    tol: float
    k: int, number of eigenvectors to compute
    method: string, what method to use
    """
    if method == 'sparse':
        solver = lambda s: _sparse_solve(F, guess, tol, s, k)
    else:
        solver = lambda s: _reg_solve(F, guess, tol, s, k)

    V = np.zeros((k, len(guess)))
    lambs = np.zeros(k)
    for state in range(0, k):
        eigVals, eigVecs = solver(state)
        
        V[state] = eigVecs.T[state]
        lambs[state] = eigVals[state]
    
    V = np.array(V).T
    return (lambs, V)
    

def iterate_eig(F, guess, tol, k, method='numpy'):
    """
    Perform iteration over 1 state rather than all
    """
    if method == 'sparse':
        solver = lambda s: _sparse_solve(F, guess, tol, s, k)
    else:
        solver = lambda s: _reg_solve(F, guess, tol, s, k)

    eigVals, eigVecs = solver(0)
    
    return (eigVals, eigVecs)


@jit(nopython=True)
def _reg_solve(F, guess, tol, state, k):
    old = 100*guess
    eigVals = eigVecs = None
    while np.linalg.norm(guess - old) > tol:
        old = guess
        matrix = F(guess)
        eigVals, eigVecs = eigh(matrix)
        guess = eigVecs.T[state]
    return (eigVals[0:k], eigVecs[:, 0:k])


def _sparse_solve(F, guess, tol, state, k):
    old = 100*guess
    eigVals = eigVecs = None
    while np.linalg.norm(guess - old) > tol:
        old = guess
        matrix = F(guess)
        eigVals, eigVecs = eigsh(matrix, k=k, which='SM')
        guess = eigVecs.T[state]
    return (eigVals, eigVecs)



n = 500
OMEGA = 0.2
TOL = 1e-8 # Convergence difference for solution
k = 6

x0 = -14
xf = 14
dx = (xf - x0) / n
xs = np.linspace(x0, xf, n)



@jit(nopython=True)
def matrix(u):
    """
    Returns tridiagonal matrix that we want the eigenvalues of.
    Used for iterated eigenvalue appraoch.
    u is vector input
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


# Test to make sure it works
def main():
    
    guess = np.exp(-OMEGA * xs**2 / 2)

    eigVals, eigVecs = iterate_eig(matrix, guess, TOL, k, method='numpy')
    #eigVals, eigVecs = iterate_eig(matrix, guess, TOL, k, method='sparse')

    import matplotlib.pyplot as plt

    plt.figure(figsize=[11,6])
    plotnum = 1
    for i in range(0,6):
        plt.subplot(2,3,plotnum)
        plt.plot(xs, eigVecs.T[i])
        lamb = eigVals[i]
        plt.title(rf'$\mu$ = {lamb}')
        plt.xlabel('x')
        plt.ylabel(r'$\phi(x)$')
        plotnum += 1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

    

