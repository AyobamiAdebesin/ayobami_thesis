#!/usr/bin/python3

"""
This script contains the implementation of the Shift-Invert Lanczos algorithm
for finding the eigenvalues of sparse matrix near a shift. 

Author: Ayobami Adebesin
"""
import numpy as np
import scipy.sparse as sp
import argparse
from scipy.sparse import csr_matrix
from typing import Callable, List, Sequence, Tuple
from collections import Counter

def make_mul_inv(A: csr_matrix, sigma: float, m: int) -> Callable:
    refine = True
    F = sp.linalg.splu(A - sigma * sp.eye(m))
    def mul_invA(B):
        Y = F.solve(B)
        if refine:
            R = A@Y - sigma * Y -B
            return Y-F.solve(R)
        else:
            return Y
    return mul_invA

def construct_A(l: int):
    """ Construct the matrix A """
    # form matrices
    v = np.ones(l**2)
    A1 = sp.spdiags([-v, 2*v, -v], [-1, 0, 1], l, l)
    I_l = sp.eye(l)
    A = sp.kron(I_l, A1, format='csc') + sp.kron(A1, I_l)
    return A

def lanczos(mul_invA: Callable, m, n, b=None):
    """
    Compute the Lanczos decomposition of an mxm symmetric matrix A
    """
    if b is None:
        b = np.random.randn(m)

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
        v = mul_invA(q_curr)
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

