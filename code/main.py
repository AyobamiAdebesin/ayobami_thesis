#!/usr/bin/python3
""" Main function """
import numpy as np
import sys
from spectral_lanczos import *
from scipy.linalg import eigh

if __name__ == "__main__":
    # m1, m2, m3 = 1500, 20, 3
    # spread1 = (1, 3)
    # spread2 = (59, 60)
    # spread3 = (94.9999, 95.0001)
    # shift = 96
    # n = 100
    # tol = 1e-10
    m1, m2, m3 = 1500, 20, 3
    spread1 = (1, 10)
    spread2 = (50, 60)
    spread3 = (74.9999, 75.0001)
    shift = 76
    n=50
    tol = 1e-7

    D = eigenvalue_distribution(m1, m2, m3, spread1, spread2, spread3)
    D = np.diag(D)
    A, B, L = generate_matrix(D, delta=1e-2)
    T, Q = spectral_lanczos(A, B, L, m=A.shape[0], n=n, shift=shift)

    # Compute eigenvalues
    V, alphas, betas = compute_eigenvalues(T, Q, L, shift)
    residuals = compute_residuals(A, B, alphas, betas, V)

    # Compute the converged ritz pairs and the (spectral transformation) relative ritz residuals
    U_converged, theta_converged, ritz_residuals = compute_ritz_residuals(A, B, L, T, Q, shift, tol)

    # Compute the generalized eigenvectors and eigenvalues and the residuals for the converged ritz pairs
    gen_residuals, v, alpha, beta = compute_generalized_residuals(A, B, L, U_converged, theta_converged, shift)
    
    print(f"Generalized eigenvalues for the converged Ritz pairs: \n{alpha/beta}\n")
    print(f"Generalized Relative Residuals for converged ritz pair: \n{la.norm(gen_residuals, axis=0)}\n")
    print(f"Relative Ritz Residuals for Spectral Transformation: \n{ritz_residuals}\n")

    plot_residuals(eigenvalues=alphas/betas, residuals=la.norm(residuals, axis=0), save_path="residuals")