#!/usr/bin/python3
""" Main function """
import numpy as np
from spectral_lanczos import *
from scipy.linalg import eigh

if __name__ == "__main__":
    m1, m2, m3 = 1500, 20, 3
    spread1 = (1, 10)
    spread2 = (50, 60)
    spread3 = (74.999, 75.001)
    shift = 75.0
    n=50

    D = eigenvalue_distribution(m1, m2, m3, spread1, spread2, spread3)
    A, B, L = generate_matrix(D, delta=1e-7)
    T, Q = spectral_lanczos(A, B, L, m =A.shape[0], n=n, shift=shift)
    V, alphas, betas = compute_eigenvalues(T, Q, L, shift)
    residuals = compute_residuals(A, B, alphas=alphas, betas=betas)

    print(f"Eigenvalues computed by Spectral Lanczos: \n{alphas/betas}\n")
    print(f"Residuals: {residuals}")