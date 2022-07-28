import numpy as np
from numpy.linalg import eigvalsh, eigh, cholesky, norm
from numpy.random import randint, rand

from numba import jit

import time



@jit(nopython=True)
def grad(vi, i, V, M):
    """
    Returns gradient of utility for (i+1)-th eigenvector/guess
    vi = vector to find gradient for
    i = what number eigenvector this is for (1st, 2nd, etc...)
    V = matrix of all eigenvector guesses currently
    M = matrix to find eigenvectors for
    """
    grad_i = vi
    vi_M = vi.dot(M) # precompute part of numerator poop poop greg is poop poopy scoop greg is poop
    for j in range(0, i):
        vj = np.ascontiguousarray( V[:, j] )
        numer = vi_M.dot(vj)
        denom = vj.dot(M.dot(vj))
        grad_i = grad_i - (numer / denom) * vj
    return 2*M.dot(grad_i)


@jit(nopython=True)
def EigenGameHelper(M, i, V, rho, alpha, mod):

    iteration = 0

    vi0 = np.ascontiguousarray( V[:, i] )
    vi = vi0
    old = vi * 100 # dummy value to enter loop
    while( norm(vi - old) > rho ):
        old = vi
        if iteration % mod == 0:
            grad_i = grad(vi, i, V, M)
        # No Riemann projection here
        vi = vi + alpha * grad_i
        vi = vi / norm(vi)
        iteration += 1
    return vi



def EigenGame(M, k, rho, alpha, mod=1, order='small'):
    """
    Compute eigenvectors of M
    M: symmetric positive definite matrix
    V: guess matrix of eigenvectors
    k: number of eigenvectors to find
    rho: error tolerance
    alpha: step size
    order: "smallest" to find smallest k-eigenvectors first, else largest
    mod: how many iterations to skip the gradient calculation
    """
    M = np.ascontiguousarray(np.array(M))
    eigVecs = rand(M.shape[0], k)

    if order == 'small':
        M = (eigvalsh(M).max()*1.2) * np.eye(M.shape[0]) - M
    

    for i in range(0, k):
        vi = EigenGameHelper(M, i, eigVecs, rho, alpha, mod)
        eigVecs[:, i] = vi # update matrix of guesses with correct one after found
    return eigVecs[:, 0:k]



def main():
    
    n = 4
    k = 4

    rho = 1e-6
    alpha = 0.01


    def get_mat():
        # Generate random symmetric positive-definite matrix
        while(True):
            M = rand(n, n)
            M = M.T.dot(M) # make symmetric
            try:
                cholesky(M)
                break
            except:
                continue

        M += 0.1*np.eye(n) # in case M is close to being semi-definite rather than positive definite
        return M / M.max()
    
    M = get_mat()
    with np.printoptions(precision=5, suppress=True):

        print("EigenGame Results")
        start = time.time()
        v1 = EigenGame(M, k, rho, alpha, mod=1, order='small')
        end = time.time()
        print(v1)
        print()

        print("Actual Answer")
        vals, vecs = eigh(M)
        #vecs = vecs.T[::-1].T
        vecs = vecs[:, 0:k]

        print(vecs)
        print()

        print("Abs. Error")
        errors = np.abs( np.abs(vecs) - np.abs(v1) )
        print(errors)
        print(f"Max Error: {errors.max()}")

    print()
    print(f"Time: {end - start}")



if __name__ == "__main__":
    main()
