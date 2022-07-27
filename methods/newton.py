from ctypes import ArgumentError
import numpy as np
from numpy.linalg import inv

from scipy.optimize import newton_krylov



def newton(func, guess, tol, method, jac=None, verbose=False):
    if method == 'numpy':
        if jac == None: raise ArgumentError(f'Jacobian must be specified for method {method}')
        return _newton_np(func, guess, tol, jac, verbose)
    elif method == 'complex':
        if jac != None: raise Warning("The complex method does not accept a jacobian")
        return _newton_complex(func, guess, tol, verbose)
    else:
        raise ArgumentError(f"{method} is not a valid method")


def _newton_np(func, guess, tol, jac, verbose):
    error = func(guess)
    x = np.array(guess)
    i = 0
    while( np.abs(error).max() > tol):
        if verbose: print(f"{i}: |F(x)| = {np.abs(error).max()}")
        J = jac(x)
        x = x - inv(J).dot(error) # error = func(x), already computed
        error = func(x)
        i+=1
    return x


def _newton_complex(f, guess, tol, verbose):
    
    # Wrapper function
    def func(v2n):
        # Create complex vector of size n
        v = v2n[0:len(v2n)//2] + 1j*v2n[len(v2n)//2:]

        # Pass vector to old function
        res = f(v)

        return np.append(res.real, res.imag)

    # Split guess into real, imag
    guess = np.append(guess.real, guess.imag)

    soln = newton_krylov(func, guess, f_tol=tol, verbose=verbose) # solve
    return soln[0:len(soln)//2] + 1j*soln[len(soln)//2:] # reassemble complex solution



def main():

    def f_np(v):
        return np.array([
            v[0]*v[1]**2 + v[1],
            v[0] - 2*v[1] + 1
        ])
    def jac_np(v):
        return np.array([
            [v[1]**2, 2*v[0]*v[1] + 1],
            [1, -2]])


    def f_complex(v):
        return np.array([
            v[0]**2 + 1,
            v[0] - 2*v[1]  + 1j
        ])
    

  

    # Numpy Test
    v0 = np.array([2,2])
    soln_np = newton(f_np, v0, 1e-7, 'numpy', jac=jac_np, verbose=False)
    print("Numpy")
    print(f'Solution: {soln_np}')
    print(f'Residual: {f_np(soln_np)}')
    print()

    # Complex test
    v0 = np.array([1j,1j])
    soln_c = newton(f_complex, v0, 1e-7, 'complex', jac=None, verbose=False)
    print("Complex")
    print(f'Solution: {soln_c}')
    print(f'Residual: {f_complex(soln_c)}')
    print()




if __name__ == "__main__":
    main()