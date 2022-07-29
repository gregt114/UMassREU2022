
import numpy as np


from scipy.sparse.linalg import eigsh
from scipy.sparse import diags

import matplotlib.pyplot as plt
from numba import jit

from methods.eigenvalue import iterate_eig

import time


# Parameters
n = 100
N =  n**2
OMEGA = 0.2
TOL = 1e-4
bc = 0 # Bondary Condition


# Grid to solver over
x0 = -8
xf = 8
y0 = -8
yf = 8
dx = (xf - x0) / n
dy = (yf - y0) / n
xs = np.linspace(x0, xf, n)
ys = np.linspace(y0, yf, n)
X, Y = np.meshgrid(xs, ys)


# Construct the matrix using a given eigenvector guess
def matrix(v):
    dxs = -1/(2*dx**2) * np.ones(N - 1)
    dys = -1/(2*dy**2) * np.ones(N-n)
    
    main = []
    for r in range(0, N):
        i = r // n
        j = r % n

        x = x0 + i*dx
        y = y0 + j*dx

        main.append( 1/dx**2 + 1/dy**2 + np.abs(v[r])**2 + (0.5*OMEGA**2)*(x**2 + y**2) )
    
    res = diags([dys, dxs, main,  dxs, dys], [-n, -1, 0, 1, n])
    return res



def main():

    k = 3 # number of eigenvectors to find
    state = 0 # state to solve for

    # Guesses
    guesses = {
        0: np.exp(-OMEGA * (X**2 + Y**2) / 2), # ground state
        1: Y * np.exp(-OMEGA * (X**2 + Y**2) / 2), # 1st excited
    }
    guess = guesses[state]
    guess = guess.reshape(1, -1)[0]
    

    # Solve
    start = time.time()
    eigVals, eigVecs = iterate_eig(matrix, guess, TOL, k, method='sparse') # numpy method wont work since matrix func isnt jit'ed
    end = time.time()
    print(f"Time: {end - start}")


    # Setup for plotting
    plt.figure(figsize=[11,6])
    soln = eigVecs.T[state].reshape(n, n)
    soln = np.abs(soln)**2

    plt.imshow(soln, origin="lower", extent=(x0, xf, y0, yf))

    lamb = eigVals[state]
    plt.title(rf'$\mu$ = {lamb:.5f}')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

