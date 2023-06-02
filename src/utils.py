import numpy as np

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
