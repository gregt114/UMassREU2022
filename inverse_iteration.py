
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

from numba import jit



# Parameters
n = 120
OMEGA = 0.2


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



@jit(nopython=True)
def iter_guess(M, mu, N, T, b=None):
    """
    Perform inverse iteration
    M: matrix
    mu: eigenvalue guess
    N: desired norm
    T: number of iterations
    b: guess of eigenvector, or None
    """
    n = M.shape[0]
    if b is None:
        b = np.random.rand(n)
    
    for t in range(0, T):
        val = inv(M - mu*np.eye(n)) @ b
        val /= np.linalg.norm(val) / N
        b = val
    return (M.dot(b)/b).mean(), b


@jit(nopython=True)
def inverse_iteration(F, mu, n, N, outer, inner, v=None):
    """
    Use inverse iteration to find eigenvectors in iterated eigenvalue method
    """
    if v is None:
        v = np.random.rand(n)
    
    lam = None
    for t in range(outer):
        M = F(v)
        lam, v = iter_guess(M, mu, N, inner, b=v)
    return lam, v


"""
Issue with 1st excited branch - will converge to ground state or 2nd excited states
"""


def main():

    gs = [
        np.exp(-OMEGA * xs**2 / 2),
        xs * np.exp(-OMEGA * xs**2 / 2),
        xs**2 * np.exp(-OMEGA * xs**2 / 2)
    ]

    outer = 5  # outer loop iterations
    inner = 60 # inner loop iterations

    # How many norms to solve over
    NN = 100
    N0 = 0.001
    Nf = 2.5
    Ns = np.linspace(N0, Nf, NN)


    for mu in [0.1, 0.3, 0.5]:
        vals = []
        guess = gs.pop(0)
        for N in Ns:
            lam, vec = inverse_iteration(matrix, mu, n, N, outer, inner, v=guess)
            vals.append(lam)
            mu = lam
            guess = vec
            print(f"mu = {lam}")
        print(f"done {mu}\n")
        plt.plot(vals, Ns)



    plt.show()



    


if __name__ == '__main__':
    main()

