#!/usr/bin/python3

"""
Spectral Transformation Lanczos for Dense Symmetric-Definite 
Generalized Eigenvalue problem Ax = lambda Bx
"""

import numpy as np
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

    return np.diag(eigenvalues)

def spectral_lanczos(A, B, L, m, n, shift):
    """
    Compute the Spectral Lanczos decomposition of a Symmetric-Definite GEP
    """
    b = np.random.randn(m)

    # Initialize storage for alpha, beta, and Lanczos vectors
    alphas = np.zeros(n)
    betas = np.zeros(n)
    Q_n = np.zeros((m, n + 1))

    beta_prev = 0
    q_prev = np.zeros(m)
    q_curr = b / np.linalg.norm(b)

    # Precompute the factorization of (A - shift * B)
    A_shift_B = A - shift * B
    lu = lu_factor(A_shift_B)

    # Perform Lanczos iterations
    for j in range(n + 1):
        Q_n[:, j] = q_curr
        u = L @ q_curr
        v = lu_solve(lu, u)
        v = L.T @ v
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

    return T_n, Q_n

def compute_eigenvalues(T, Q, L, sigma):
    """ Compute the evalues and evectors of tridiagonal matrix T """ 
    eigvals_Tn, eigvecs_Tn = np.linalg.eigh(T)
    theta = eigvals_Tn
    U = Q @ eigvecs_Tn
    alphas = 1.0 + (theta *sigma)
    betas = theta
    V = la.solve(L.T, U)
    return V, alphas, betas

def compute_residuals(A, B, alphas, betas):
    """ Compute relative residuals"""
    norm_A = la.norm(A)
    norm_B = la.norm(B)
    residuals = np.zeros_like(alphas, dtype=float)

    for i in range(len(alphas)):
        alpha, beta = alphas[i], betas[i]
        M = beta * A - alpha * B
        sigma_n = la.svdvals(M)[-1]
        denominator = abs(beta) * norm_A + abs(alpha) * norm_B
        
        residuals[i] = sigma_n / denominator if denominator > 0 else np.inf 
    return residuals
