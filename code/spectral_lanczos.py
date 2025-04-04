#!/usr/bin/env python3

"""
Spectral Transformation Lanczos for Dense Symmetric-Definite 
Generalized Eigenvalue problem Ax = lambda Bx
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import numpy.linalg as la
from scipy.linalg import lu_factor, lu_solve, qz, qr, eigh
from scipy.linalg import solve, solve_triangular

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


def eigenvalue_distribution_groups(m1, m2, m3, spread1, spread2, spread3):
    """ Generate eigenvalue distribution within specified magnitude groups.
    
    m1 - Number of eigenvalues in the first group.
    m2 - Number of eigenvalues in the second group.
    m3 - Number of eigenvalues in the third group.
    spread1 - (min, max) for the first group.
    spread2 - (min, max) for the second group.
    spread3 - (min, max) for the third group.
    """
    # Use logspace to generate eigenvalues
    eig1 = np.logspace(np.log10(spread1[0]), np.log10(spread1[1]), m1)
    eig2 = np.logspace(np.log10(spread2[0]), np.log10(spread2[1]), m2)
    eig3 = np.logspace(np.log10(spread3[0]), np.log10(spread3[1]), m3)

    # Combine the eigenvalues into one array
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
    x[n-1] = beta

    return T_n, Q_n, q_curr, x

def compute_decomp_residual(A, B, L, T, Q, q, x, shift):
    """ Compute the Lanczos decomposition residual """
    lu = lu_factor(A - shift*B)
    norm_matrix = la.norm(L.T @ lu_solve(lu, L))
    aq = L.T @ lu_solve(lu, L @ Q)
    qt = Q @ T
    qx = np.outer(q, x.T)
    decomp_res = la.norm(aq - qt - qx) / norm_matrix
    return decomp_res

def compute_residuals(A, B, alphas, betas, V):
    """ Compute relative residuals for the Generalized problem """
    residuals = np.zeros_like(V)
    nrma = la.norm(A)
    nrmb = la.norm(B)
    for i in range(len(alphas)):
        residuals[:, i] = (betas[i]*A - alphas[i]*B)@V[:, i] / (
            abs(betas[i]) * nrma + abs(alphas[i]) * nrmb) / la.norm(V[:,i])
    return residuals

def compute_best_v_naive(A, B, alphas, betas):
    """ Compute best residuals naive"""
    nrmA = la.norm(A)
    nrmB = la.norm(B)
    res = np.zeros_like(alphas, dtype=float)
    for i in range(len(alphas)):
        alpha, beta = alphas[i], betas[i]
        M = beta*A - alpha*B
        sigma_min = la.svd(M, compute_uv=False, hermitian=False)[-1]
        den = (abs(beta) * nrmA) + (abs(alpha) *nrmB)
        res[i] = sigma_min / den if den !=0 else np.inf
    return res

def compute_ritz_residuals(A, B, L, T, Q, shift, tol):
    """ 
    Compute relative ritz residuals for the spectral transformation problem 
    to obtain converged Ritz pairs based on tolerance
    """
    eigvals_T, eigvecs_T = la.eigh(T)
    theta = eigvals_T
    U = Q @ eigvecs_T
    ritz_residuals = []
    U_converged = []
    theta_converged = []

    # Precompute the factorization of (A - shift * B)
    A_shift_B = A - shift * B
    lu = lu_factor(A_shift_B)
    
    # Compute the norm of L.T @ (inv(A - shift * B) @ L)
    norm_matrix = la.norm(L.T @ lu_solve(lu, L))

    for i in range(U.shape[1]):
        q = L @ U[:, i]
        v = lu_solve(lu, q)
        v = L.T @ v
        num = la.norm(v - theta[i] * U[:, i])
        den = (norm_matrix + abs(theta[i])) * la.norm(U[:, i])
        residual = num / den if den != 0 else np.inf

        #Check if the residual is within the tolerance
        if residual <= tol:
            ritz_residuals.append(residual)
            U_converged.append(U[:, i])
            theta_converged.append(theta[i])

    ritz_residuals = np.array(ritz_residuals)
    U_converged = np.array(U_converged).T
    theta_converged = np.array(theta_converged)
    return U_converged, theta_converged, ritz_residuals

def is_triu(T):
    """Check if T is upper triangular """
    return np.allclose(T, np.triu(T))

def inverse_iteration(T, tol=1e-10, max_iter=3000):
    """ Compute the smallest singular values using inverse iteration"""
    m, n = T.shape
    x = np.random.randn(n)
    x = x/np.linalg.norm(x)
    
    for k in range(max_iter):
        y = solve_triangular(T, x, trans='T', lower=True)
        z = solve_triangular(T, y, lower=False)
        z_norm = la.norm(z)
        x_new = z/z_norm

        if la.norm(x_new - x) < tol:
            break
        x = x_new
    TTx = T.T @ (T@x)
    sigma_min = 1/np.sqrt(la.norm(TTx))
    return sigma_min, x

def compute_best_v(A, B, alphas, betas, tol):
    """ Compute the best residual using the smallest singular values """
    res = np.zeros_like(alphas, dtype=float)
    Ta, Tb, Q, Z = qz(A, B)
    for i in range(len(alphas)):
        T = betas[i] * Ta - alphas[i] * Tb
        n = T.shape[0]

        # Check if subdiagonal element is zero based on tol
        for j in range(n - 1): 
            if T[j + 1, j] != 0:
                if np.abs(T[j + 1, j]) <= tol:
                    T[j + 1, j] = 0
                else:
                    blk = T[j:j + 2, j:j + 2]
                    q, r = qr(blk)
                    T[j:j + 2, j:n] = q.T @ T[j:j + 2, j:n]

        # Ensure T is upper triangular
        if not is_triu(T):
            print("T not upper triangular!")
            sys.exit(1)

        # Check for zero diagonal elements and handle them
        diag_indices = np.diag_indices_from(T)
        zero_diag = np.abs(T[diag_indices]) <= tol
        if np.any(zero_diag):
            print(f"Diagonal element(s) too small or zero in T for index {i}. Skipping...")
            res[i] = np.inf
            continue

        # Perform inverse iteration
        sigma_min, x = inverse_iteration(T, tol=tol)
        den = np.abs(betas[i]) * la.norm(A) + np.abs(alphas[i]) * la.norm(B)
        res[i] = sigma_min / den if den != 0 else np.inf

    return res

def compute_generalized_residuals(A, B, L, U_converged, theta_converged, shift):
    """
    Compute the residuals, generalized eigvalues and eigvectors 
    for the converged Ritz pairs
    """
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


def plot_residuals(eigenvalues, residuals, label=None, save_path=None):
    """ Plot the residuals for both Generalized Eigenvalues and Ritz values in a single plot """
    fig, ax = plt.subplots(figsize=(8, 6))

    label_dict = {
        'g': 'λ (Generalized)',
        'r': 'θ (Ritz)',
        'b': 'Best $v_{i}$'
    }
    title_dict = {
        'g': 'Residual vs Generalized Eigenvalues',
        'r': 'Residual vs Ritz values',
        'b': 'Best Residuals vs Generalized Eigenvalues'
    }

    color_dict = {
        'g': "blue",
        'r': 'green',
        'b': 'black'
    }
    x_label = r'$\theta$' if label=="r" else r'$\lambda$'

    img_path = f"residual_{save_path}"
    scatter_label = label_dict.get(label, 'Residuals')
    scatter_title = title_dict.get(label, "Residual vs Eigenvalues")
    color = color_dict.get(label, 'blue')

    # Plot for the residuals
    ax.scatter(eigenvalues, residuals, color=color, label=scatter_label, s=10)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(x_label, fontsize=12) 
    ax.set_ylabel('Residual', fontsize=12)
    ax.set_title(scatter_title, fontsize=12)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {img_path}")

