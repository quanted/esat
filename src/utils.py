import numpy as np
import copy
import fastcluster as fc
from scipy.cluster.hierarchy import cophenet
import itertools

EPSILON = 1e-15

#  functions for connectivity, consensus, dispersion from https://github.com/yal054/snATAC/blob/master/snATAC.nmf.py

def cal_connectivity(H, idx):
	""" calculate connectivity matrix """
	print("=== calculate connectivity matrix ===")
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
  print("=== calculate cophenetic correlation coefficient ===")
  X = C
  Z = fc.linkage_vector(X)   # Clustering
  orign_dists = fc.pdist(X)  # Matrix of original distances between observations
  cophe_dists = cophenet(Z)  # Matrix of cophenetic distances between observations
  corr_coef = np.corrcoef(orign_dists, cophe_dists)[0,1]
  return corr_coef

def cal_dispersion(C):
  """ calculate dispersion coefficient """
  print("=== calculate dispersion coefficient ===")
  n = C.shape[1]
  corr_disp = np.sum(4 * np.square(np.concatenate(C - 1/2)))/(np.square(n))
  return corr_disp

def np_encoder(object):
    """
    Convert any numpy type to a generic type for json serialization.

    Parameters
    ----------
    object
       Object to be converted.
    Returns
    -------
    object
        Generic object or an unchanged object if not a numpy type
    """
    if isinstance(object, np.generic):
        return object.item()


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
