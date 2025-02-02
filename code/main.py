#!/usr/bin/python3
""" Main function """
import numpy as np
from spectral_lanczos import *
import scipy.io as sio

if __name__ == "__main__":
    m1, m2, m3 = 1000, 20, 30
    spread1 = (1, 10)
    spread2 = (50, 60)
    spread3 = (70, 80)
    shift = 75.0
    n=20

    D= eigenvalue_distribution(m1, m2, m3, spread1, spread2, spread3)
    A, B = generate_matrix(D, delta=1e-7)
    # A =  sio.mmread("./bcsstk13.mtx").toarray()
    # B = sio.mmread("./bcsstk13.mtx").toarray()
    T, _, Q, _ = spectral_lanczos(A, B, m =A.shape[0], n=n, shift=shift)
    evecs, evals = compute_ritz_pairs(T, Q, sigma=shift)
    print(orthogonality_check(evecs)) 
    print(evals)
    print(f"\nCondition number of Y: {np.linalg.cond(evecs)}")