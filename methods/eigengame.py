"""
Implementation of parallel EigenGame
"""
import numpy as np
from numpy.linalg import eigh, cholesky, norm, eigvalsh
from numpy.random import rand

from numba import jit


import time

from multiprocessing import Pool, shared_memory, Lock


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
    vi_M = vi.dot(M) # precompute part of numerator
    for j in range(0, i):
        vj = np.ascontiguousarray( V[:, j] )
        numer = vi_M.dot(vj)
        denom = vj.dot(M.dot(vj))
        grad_i = grad_i - (numer / denom) * vj
    return 2*M.dot(grad_i)



def solve(name, shape, i, M, alpha, rho):
    """ Has access to V through shared memory """
    sm = shared_memory.SharedMemory(name=name)
    lock = Lock() # create lock object
    V = np.ndarray(shape, dtype=np.float64, buffer=sm.buf)

    vi = np.ascontiguousarray( V[:, i] )
    old = vi * 100 # dummy value to enter loop
    while( norm(vi - old) > rho ):
        old = vi
        grad_i = grad(vi, i, V, M)
        # No Riemann projection here
        vi = vi + alpha * grad_i
        vi = vi / norm(vi)

        # Alter shared memory - need lock
        lock.acquire()
        V[:, i] = vi
        lock.release()
    
    sm.close() # child processes only need to close shared memory



def EigenGame(M, k, rho, alpha, order='small'):
    
    if order == 'small':
        M = (eigvalsh(M).max()*1.2) * np.eye(M.shape[0]) - M
    
    M = np.ascontiguousarray(np.array(M))

    V0 = rand(M.shape[0], k)

    sm = shared_memory.SharedMemory(create=True, size=V0.nbytes)
    name = sm.name

    V = np.ndarray(V0.shape, dtype=V0.dtype, buffer=sm.buf)
    V[:] = V0[:]

    pool = Pool()

    result = [ pool.apply(solve, args=(name, V0.shape, i, M, alpha, rho)) for i in range(0, k) ]
    pool.close()
    pool.join()

    ret = V.copy()

    # parent process must close AND unlink shared memory
    sm.close()
    sm.unlink()

    return ret
    


"""
20 vecs of 200x200 --> 68.8 sec
"""

def main():
    
    n = 10
    k = 10
    a = 10

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
        v1 = EigenGame(M, k, rho, alpha, order='small')
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
