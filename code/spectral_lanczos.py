#!/usr/bin/python3

"""
Spectral Transformation Lanczos for Dense Symmetric-Definite 
Generalized Eigenvalue problem Ax = lambda Bx
"""

import numpy as np
import numpy.linalg as la
from utils import *


def generate_matrix(D, delta):
    """
    Generate A (symmetric) and B (symmetric and positive definite) from known eigenvalues D
    
    D - diagonal matrix of known eigenvalues
    """
    m= D.shape[0]
    Q, _ = la.qr(np.random.randn(m, m))
    C = Q @ D @ Q.T

    L0 = np.tril(np.random.randn(m,m))
    B = (L0 @ L0.T ) + (delta * np.eye(m))
    L = la.cholesky(B)
    A = L @ C @ L.T
    return A, B


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

def spectral_lanczos(A, B, m, n, shift):
    """
    Compute the Spectral Lanczos decomposition of a Symmetric-Definite GEP
    """
    b = np.random.randn(m)
    L = la.cholesky(B)

    # Initialize storage for alpha, beta, and Lanczos vectors
    alphas = np.zeros(n)
    betas = np.zeros(n)
    Q_n_plus_1 = np.zeros((m, n + 1))

    beta_prev = 0
    q_prev = np.zeros(m)
    q_curr = b / np.linalg.norm(b)

    # Perform Lanczos iterations
    for j in range(n + 1):
        Q_n_plus_1[:, j] = q_curr
        u = L @ q_curr
        v = la.solve(A-shift*B, u)
        v = L.T @ v
        if j < n:
            # Compute and store alpha
            alpha = np.dot(q_curr, v)
            alphas[j] = alpha
            
            # Orthogonalize v against the current Lanczos vector
            v = v - (beta_prev * q_prev) - (alpha * q_curr)

            # reorthogonalization
            v -= Q_n_plus_1[:, :j+1] @ (Q_n_plus_1[:, :j+1].T @ v)

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

    # Construct the extended tridiagonal matrix T_n_tilde of size (n+1 x n)
    T_n_tilde = np.zeros((n + 1, n))
    np.fill_diagonal(T_n_tilde[:n, :], alphas)
    np.fill_diagonal(T_n_tilde[:n, 1:], betas[:n - 1])
    np.fill_diagonal(T_n_tilde[1:, :n - 1], betas[:n - 1])

    # Add the last beta_n for q_{n+1}
    T_n_tilde[n, n - 1] = betas[n - 1] if n > 1 else betas[0] 

    Q_n = Q_n_plus_1[:, :n]

    return T_n, T_n_tilde, Q_n, Q_n_plus_1

def compute_ritz_pairs(T, Q, sigma):
    """ Compute the evalues and evectors of tridiagonal matrix T """ 
    eigvals_Tn, eigvecs_Tn = np.linalg.eigh(T)
    eigvecs_A = Q @ eigvecs_Tn
    eigvals_A = sigma + 1/eigvals_Tn
    return eigvecs_A, eigvals_A
