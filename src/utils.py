"""
Collection of utility functions used throughout the code base.
"""

import numpy as np


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
