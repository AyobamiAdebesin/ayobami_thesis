#!/usr/bin/python3
""" Main function """
import numpy as np
import sys
from spectral_lanczos import *
from scipy.linalg import eigh

if __name__ == "__main__":
    m1, m2, m3 = 1500, 400, 100
    spread1 = (1, 199)
    spread2 = (200, 300)
    spread3 = (301, 400)
    shift = 200.000001
    n=1200
    tol = 1e-10

    D = eigenvalue_distribution(m1, m2, m3, spread1, spread2, spread3)
    D = np.diag(D)
    A, B, L = generate_matrix(D, delta=1e-2)
    T, Q, q, x = spectral_lanczos(A, B, L, m=A.shape[0], n=n, shift=shift)

    #compute lanczos decomposition residual
    decomp_res = compute_decomp_residual(A=A, B=B, L=L, T=T, Q=Q, q=q, x=x, shift=shift)

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
    print(f"Decomposition residual: {decomp_res}\n")

    plot_residuals(eigenvalues=theta_converged, residuals=ritz_residuals, save_path="ritz_residuals")
    plot_residuals(eigenvalues=alpha/beta, residuals=la.norm(gen_residuals, axis=0), save_path="converged_ritz_residuals")
