#!/usr/bin/python3
""" Main function """
import numpy as np
import scipy.io as sio
import sys, time
from spectral_lanczos import *
from scipy.linalg import eigh
from scipy.sparse import diags
if __name__ == "__main__":
    start = time.time()
    m1, m2, m3 = 1000, 1000, 1000
    spread1 = (10**-3, 10**1)
    spread2 = (10**1, 10**3)  
    spread3 = (10**3, 10**7)
    shift = 1.5e5
    n=2000
    tol = 1e-10

    D = eigenvalue_distribution_groups(m1, m2, m3, spread1, spread2, spread3)
    D = np.diag(D)

    A, B, L = generate_matrix(D, delta=1e1)
    L = la.cholesky(B)
    T, Q, q, x = spectral_lanczos(A, B, L, m=A.shape[0], n=n, shift=shift)
    
    # compute lanczos decomposition residual
    decomp_res = compute_decomp_residual(A=A, B=B, L=L, T=T, Q=Q, q=q, x=x, shift=shift)

    print(f"Condition number of A: {la.cond(A)}\n")
    print(f"Condition number of B: {la.cond(B)}\n")
    print(f"Decomposition residual: {decomp_res}\n")

    # Compute the converged ritz pairs and the (spectral transformation) relative ritz residuals
    U_converged, theta_converged, ritz_residuals = compute_ritz_residuals(A, B, L, T, Q, shift, tol)

    # Compute the generalized eigenvectors and eigenvalues and the residuals for the converged ritz pairs
    gen_residuals, v, alpha, beta = compute_generalized_residuals(A, B, L, U_converged, theta_converged, shift)

    # Compute the best relative residual
    # res = compute_best_v(A, B, alpha, beta, tol=1e-10)

    # Compute best relative res naive
    #res = compute_best_v_naive(A, B, alpha, beta)

    print(f"Number of converged Ritz pairs: {theta_converged.shape[0]}")

    plot_residuals(eigenvalues_generalized=alpha/beta, residuals_generalized=la.norm(gen_residuals, axis=0),
                   eigenvalues_ritz=theta_converged, residuals_ritz=ritz_residuals,
                   save_path="residuals_plot_well_large")
   

    end = time.time()
    elapsed_time = end - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Runtime: {minutes} minutes and {seconds} seconds")