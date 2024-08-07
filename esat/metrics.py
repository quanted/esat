"""
Collection of metric functions which are used throughout the code base.
"""
import numpy as np
import copy

EPSILON = 1e-12


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
    robust_uncertainty[robust_uncertainty <= 0.0] = 1e-12
    robust_residuals = np.abs(residuals / robust_uncertainty)
    scaled_residuals[scaled_residuals > alpha] = robust_residuals[scaled_residuals > alpha]
    _q = np.sum(np.square(scaled_residuals))

    updated_uncertainty = copy.copy(U)
    updated_uncertainty[scaled_residuals > alpha] = robust_uncertainty[scaled_residuals > alpha]
    return _q, updated_uncertainty


def q_factor(V, U, W, H):
    wh = np.matmul(W, H)
    wh[wh <= 0.0] = 1e-12
    residuals = np.subtract(V, wh)
    u_residuals = np.divide(residuals, U)
    u2_residuals = u_residuals**2
    factor_q = []
    for f_i in range(H.shape[0]):
        w_i = W[:, f_i]
        h_i = H[f_i]
        v_i = np.matmul(w_i.reshape(len(w_i), 1), [h_i])
        v_ip = v_i / wh
        q_i = np.round(np.sum(u2_residuals * v_ip), 4)
        factor_q.append(q_i)
    return factor_q
