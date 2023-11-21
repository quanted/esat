import numpy as np
import copy

EPSILON = 1e-15


def calculate_Q(residuals, uncertainty):
    return np.sum(np.sum(np.square(np.divide(residuals, uncertainty))))


def q_loss(V, U, W, H, uncertainty: bool = True):
    _wh = np.matmul(W, H)
    residuals = np.subtract(V, _wh)
    if uncertainty:
        residuals_u = np.divide(residuals, U)
        r2 = np.multiply(residuals_u, residuals_u)
        _q = np.sum(r2)
    else:
        _q = np.sum(np.multiply(residuals, residuals))
    return _q


def qr_loss(V, U, W, H, alpha=4.0):
    _wh = np.matmul(W, H)
    residuals = np.subtract(V, _wh)
    scaled_residuals = np.abs(residuals/U)
    robust_uncertainty = np.sqrt(scaled_residuals/alpha) * U
    robust_residuals = np.abs(residuals / robust_uncertainty)
    scaled_residuals[scaled_residuals > alpha] = robust_residuals[scaled_residuals > alpha]
    _q = np.sum(np.square(scaled_residuals))

    updated_uncertainty = copy.copy(U)
    updated_uncertainty[scaled_residuals > alpha] = robust_uncertainty[scaled_residuals > alpha]
    return _q, updated_uncertainty


def calculate_factor_correlation(factor1, factor2):
    factor1 = factor1.astype(float)
    factor2 = factor2.astype(float)
    corr_matrix = np.corrcoef(factor1, factor2)
    corr = corr_matrix[0, 1]
    r_sq = corr ** 2
    return r_sq


def compare_all_factors(matrix1, matrix2):
    matrix1 = matrix1.astype(float)
    matrix2 = matrix2.astype(float)
    swap = False
    for i in range(matrix1.shape[0]):
        m1_i = matrix1[i]
        i_r2 = calculate_factor_correlation(m1_i, matrix2[i])
        for j in range(matrix2.shape[0]):
            if j == i:
                pass
            j_r2 = calculate_factor_correlation(m1_i, matrix2[j])
            if j_r2 > i_r2:
                swap = True
    return swap
