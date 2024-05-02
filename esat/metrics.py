"""
Collection of metric functions which are used throughout the code base.
"""
import numpy as np
import copy
import fastcluster as fc
from scipy.cluster.hierarchy import cophenet
import itertools

EPSILON = 1e-12


#  functions for connectivity, consensus, dispersion from https://github.com/yal054/snATAC/blob/master/snATAC.nmf.py
def cal_connectivity(H, idx):
    """ calculate connectivity matrix """
    # logger.info("=== calculate connectivity matrix ===")
    connectivity_mat = np.zeros((H.shape[1], H.shape[1]))
    classN = H.shape[0]
    for i in range(classN):
        xidx = list(np.concatenate(np.where(idx == i)))
        iterables = [ xidx, xidx ]
        for t in itertools.product(*iterables):
            connectivity_mat[t[0],t[1]] = 1
    return connectivity_mat


def cal_cophenetic(C):
    """ calculate cophenetic correlation coefficient """
    # logger.info("=== calculate cophenetic correlation coefficient ===")
    X = C
    Z = fc.linkage_vector(X)   # Clustering
    orign_dists = fc.pdist(X)  # Matrix of original distances between observations
    cophe_dists = cophenet(Z)  # Matrix of cophenetic distances between observations
    corr_coef = np.corrcoef(orign_dists, cophe_dists)[0,1]
    return corr_coef


def cal_dispersion(C):
    """ calculate dispersion coefficient """
    # logger.info("=== calculate dispersion coefficient ===")
    n = C.shape[1]
    corr_disp = np.sum(4 * np.square(np.concatenate(C - 1/2)))/(np.square(n))
    return corr_disp


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
