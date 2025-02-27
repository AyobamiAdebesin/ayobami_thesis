#!/usr/bin/python3

"""
Spectral Transformation Lanczos for Dense Symmetric-Definite 
Generalized Eigenvalue problem Ax = lambda Bx
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy.linalg as la
from scipy.linalg import lu_factor, lu_solve

np.random.seed(256)


def generate_matrix(D, delta):
    """
    Generate A (symmetric) and B (symmetric and positive definite) from known eigenvalues D
    
    D - diagonal matrix of known eigenvalues
    """
    m = D.shape[0]
    Q, _ = la.qr(np.random.randn(m, m))
    C = Q @ D @ Q.T
    
    L0 = np.tril(np.random.randn(m,m))
    B = (L0 @ L0.T ) + (delta * np.eye(m))
    L = la.cholesky(B)
    A = (L @ C ) @ L.T
    return A, B, L

def eigenvalue_distribution(m1, m2, m3, spread1, spread2, spread3):
    """ Generate eigenvalue distribution.
    
    m1 - Number of eigenvalues in the first group.
    m2 - Number of eigenvalues in the second group.
    m3 - Number of eigenvalues in the third group.
    spread1 - (min, max) for the first group.
    spread2 - (min, max) for the second group.
    spread3 - (min, max) for the third group.
    """
    eig1 = np.random.uniform(spread1[0], spread1[1], m1)
    eig2 = np.random.uniform(spread2[0], spread2[1], m2)
    eig3 = np.random.uniform(spread3[0], spread3[1], m3)

    eigenvalues = np.concatenate([eig1, eig2, eig3])

    return eigenvalues

def spectral_lanczos(A, B, L, m, n, shift):
    """
    Compute the Spectral Lanczos decomposition of a Symmetric-Definite GEP
    """
    b = np.random.randn(m)

    # Initialize storage for alpha, beta, and Lanczos vectors
    alphas = np.zeros(n)
    betas = np.zeros(n)
    Q_n = np.zeros((m, n + 1))
    x = np.zeros(n)

    beta_prev = 0
    q_prev = np.zeros(m)
    q_curr = b / np.linalg.norm(b)

    # # Precompute the factorization of (A - shift * B)
    # A_shift_B = A - shift * B
    # lu = lu_factor(A_shift_B)

    # spectral transformation matrix
    C = L.T @ la.inv(A - shift*B) @ L

    # Perform Lanczos iterations
    for j in range(n + 1):
        Q_n[:, j] = q_curr
        v = C @ q_curr
        # u = L @ q_curr
        # v = lu_solve(lu, u)
        # v = L.T @ v
        if j < n:
            # Compute and store alpha
            alpha = np.dot(q_curr, v)
            alphas[j] = alpha
            
            # Orthogonalize v against the current Lanczos vector
            v = v - (beta_prev * q_prev) - (alpha * q_curr)

            # reorthogonalization
            v -= Q_n[:, :j+1] @ (Q_n[:, :j+1].T @ v)

            # Compute beta
            beta = np.linalg.norm(v)
            betas[j] = beta

            if beta < 1e-16:
                break

            # Normalize the new Lanczos vector
            q_prev = q_curr
            q_curr = v / beta
            beta_prev = beta
        else:
            break

    # Construct the tridiagonal matrix T_n of size (n x n)
    T_n = np.zeros((n, n))
    np.fill_diagonal(T_n, alphas)
    np.fill_diagonal(T_n[:-1, 1:], betas[:n - 1])
    np.fill_diagonal(T_n[1:, :-1], betas[:n - 1])

    Q_n = Q_n[:, :n]
    x[n-1] = beta

    return T_n, Q_n, q_curr, x, C

def compute_decomp_residual(A, B, L, T, Q, q, x, shift, C):
    """ Compute the Lanczos decomposition residual """
    # lu = lu_factor(A - shift*B)
    # norm_matrix = la.norm(L.T @ lu_solve(lu, L))
    # aq = L.T @ lu_solve(lu, L @ Q)
    # qt = Q @ T
    # qx = np.outer(q, x.T)
    # decomp_res = la.norm(aq - qt - qx) / norm_matrix
    num = la.norm(C@Q - Q@T - np.outer(q, x.T))
    den = la.norm(C)
    decomp_res = num/den
    return decomp_res

def compute_eigenvalues(T, Q, L, sigma):
    """ Compute the evalues and evectors of tridiagonal matrix T """ 
    eigvals_Tn, eigvecs_Tn = np.linalg.eigh(T)
    theta = eigvals_Tn
    U = Q @ eigvecs_Tn
    alphas = 1.0 + (theta *sigma)
    betas = theta
    V = la.solve(L.T, U)
    return V, alphas, betas

def compute_residuals(A, B, alphas, betas, V):
    """ Compute relative residuals for the Generalized problem """
    residuals = np.zeros_like(V)
    nrma = la.norm(A)
    nrmb = la.norm(B)
    for i in range(len(alphas)):
        residuals[:, i] = (betas[i]*A - alphas[i]*B)@V[:, i] / (
            abs(betas[i]) * nrma + abs(alphas[i]) * nrmb) / la.norm(V[:,i])
    return residuals

def compute_ritz_residuals(A, B, L, T, Q, shift, tol, C):
    """ Compute relative residuals for the spectral transformation problem to obtain converged Ritz pairs based on tolerance """
    eigvals_T, eigvecs_T = la.eigh(T)
    theta = eigvals_T
    U = Q @ eigvecs_T
    ritz_residuals = []
    U_converged = []
    theta_converged = []

    # # Precompute the factorization of (A - shift * B)
    # A_shift_B = A - shift * B
    # lu = lu_factor(A_shift_B)
    
    # # Compute the norm of L.T @ (inv(A - shift * B) @ L)
    # norm_matrix = la.norm(L.T @ lu_solve(lu, L))
    norm_matrix = la.norm(C)

    for i in range(U.shape[1]):
        # q = L @ U[:, i]
        # v = lu_solve(lu, q)
        # v = L.T @ v
        v = C @ U[:, i]
        num = la.norm(v - theta[i] * U[:, i])
        den = (norm_matrix + abs(theta[i])) * la.norm(U[:, i])
        residual = num / den if den != 0 else np.inf
        # Check if the residual is within the tolerance
        if residual <= tol:
            ritz_residuals.append(residual)
            U_converged.append(U[:, i])
            theta_converged.append(theta[i])
    ritz_residuals = np.array(ritz_residuals)
    U_converged = np.array(U_converged).T
    theta_converged = np.array(theta_converged)
    return U_converged, theta_converged, ritz_residuals


def compute_generalized_residuals(A, B, L, U_converged, theta_converged, shift):
    """ Compute the residuals, generalized eigvalues and eigvectors for the converged Ritz pairs """
    # Check for converged ritz pairs
    if len(U_converged) != 0 and len(theta_converged) != 0:
        converged_alphas = 1.0 + (theta_converged * shift)
        converged_betas = theta_converged
        converged_V = la.solve(L.T, U_converged)
        residuals = compute_residuals(A, B, converged_alphas, converged_betas, converged_V)
        return residuals, converged_V, converged_alphas, converged_betas
    else:
        print("No converged Ritz pair!")
        sys.exit(1)

def plot_residuals(eigenvalues, residuals, save_path=None):
    
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6), sharey=True)
    
    # Plot for the residuals of computed eigenvalues
    ax1.scatter(eigenvalues, residuals, color='blue', label='λ', s=10)
    #ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$\lambda$', fontsize=12)
    ax1.set_ylabel('Residual', fontsize=12)
    ax1.set_title('Residual vs $\lambda$', fontsize=12)
    ax1.legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
