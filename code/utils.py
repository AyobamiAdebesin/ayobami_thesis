#!/usr/bin/python3
""" Utility script """
import numpy as np
import numpy.linalg as la
np.random.seed(256)

def generate_matrix(size):
    """ Generate a random matrix similar to a diagonal matrix with predetermined eigenvalues """
    # create a diagonal matrix
    diagonal_values = np.random.uniform(1, 10, size)
    D = np.diag(diagonal_values)

    # random orthogonal matrix
    Q, _ = la.qr(np.random.rand(size, size))
    A = Q @ D @ Q.T
    return A, D

def relative_ritz_pair_residual(A, eigenvalues, eigenvectors):
    """Calculate the relative residuals for the Ritz pairs."""
    norm_A = np.linalg.norm(A)
    residual_vectors = A @ eigenvectors - eigenvectors * eigenvalues
    num = np.linalg.norm(residual_vectors, axis=0)
    eigvec_norms = np.linalg.norm(eigenvectors, axis=0)
    den = (norm_A + np.abs(eigenvalues)) * eigvec_norms
    residuals = num / den
    return residuals

def absolute_ritz_pair_residual(A, eigenvalues, eigenvectors):
    """ Compute the ritz pair residuals """
    return np.linalg.norm(A @ eigenvectors - eigenvectors * eigenvalues, axis=0)

def orthogonality_check(eigenvectors):
    """ Check orthogonality of Ritz vectors """
    return np.linalg.norm(eigenvectors.T@eigenvectors - np.eye(eigenvectors.shape[1]))