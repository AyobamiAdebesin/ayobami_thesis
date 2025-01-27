#!/usr/bin/python3
""" Main function """
import numpy as np
from shift_invert import lanczos, compute_ritz_pairs, make_mul_inv, construct_A
from utils import generate_matrix, relative_ritz_pair_residual, orthogonality_check, absolute_ritz_pair_residual

if __name__ == "__main__":
    sigma = 6.001
    A, D = generate_matrix(5000)
    print(np.diag(D))
    m = A.shape[0]
    n = 20
    mul_invA = make_mul_inv(A, sigma, m)
    T_n, _, Q_n, _ = lanczos(mul_invA=mul_invA, m=m, n=n)
    eigvecs, eigvals = compute_ritz_pairs(T=T_n, Q=Q_n, sigma=sigma)
    error = orthogonality_check(eigenvectors=eigvecs)
    print(eigvals)
    print(f"\nCondition number of Y: {np.linalg.cond(eigvecs)}")
    print(f"\nOrthogonality error: {error}")
    print(f"\nRitz pair residuals: {absolute_ritz_pair_residual(A=A, eigenvalues=eigvals, eigenvectors=eigvecs)}")





