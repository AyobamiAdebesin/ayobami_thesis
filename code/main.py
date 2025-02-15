#!/usr/bin/python3
""" Main function """
import numpy as np
import sys
from spectral_lanczos import *
from scipy.linalg import eigh

if __name__ == "__main__":
    m1, m2, m3 = 1500, 20, 3
    spread1 = (1, 10)
    spread2 = (50, 60)
    spread3 = (74.9999, 75.0001)
    shift = 76
    n=50
    tol = 1e-7

    D = eigenvalue_distribution(m1, m2, m3, spread1, spread2, spread3)
    D = np.diag(D)
    A, B, L = generate_matrix(D, delta=1e-7)
    T, Q = spectral_lanczos(A, B, L, m =A.shape[0], n=n, shift=shift)

    # Compute the n-generalized eigenvectors and eigenvalues
    V, alphas, betas = compute_eigenvalues(T, Q, L, shift)

    # Compute the converged ritz pairs and their (spectral transformation) relative residuals
    U_converged, theta_converged, residuals = compute_spectral_residuals(A, B, L, T, Q, shift, tol=tol)

    # compute the ritz-pair and residuals for the converged ritz pairs 
    ritz_residuals, v, alpha, beta = compute_ritz_pair_residuals(A, B, L, U_converged, theta_converged, shift)
    
    print(f"Eigenvalues computed by Spectral Lanczos: \n{alphas/betas}\n")
    print(f"Eigenvalues of converged Ritz pairs: \n{alpha/beta}\n")
    print(f"Converged Ritz pair Residuals: \n{la.norm(ritz_residuals, axis=0)}")
    