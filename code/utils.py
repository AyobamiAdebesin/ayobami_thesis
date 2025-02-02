#!/usr/bin/python3
""" Utility script """
import numpy as np
import numpy.linalg as la
np.random.seed(256)

def is_symmetric(matrix, tol=1e-8):
    return np.allclose(matrix, matrix.T, atol=tol)

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