"""
Collection of utility functions used throughout the code base.
"""

import numpy as np
import pandas as pd


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


def min_timestep(data: pd.DataFrame):
    """
    Find the minimum timestep in a dataframe.

    Parameters
    ----------
    data
        Dataframe to be searched.

    Returns
    -------
    int
        Minimum timestep.
    """
    time_delta = data.index[1: -1] - data.index[0:-2]
    if time_delta.min().seconds < 60:
        resample = f"{time_delta.min().seconds}s"
    elif time_delta.min().seconds < 60 * 60:
        resample = f"{int(time_delta.min().seconds / 60)}min"
    elif time_delta.min().seconds > 60 * 60 and time_delta.min().days <= 0:
        resample = f"{int(time_delta.min().seconds / (60 * 60))}h"
    else:
        resample = "D"
    return resample


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


def solution_bump(profile: np.ndarray, contribution: np.ndarray, bump_range: tuple = (0.9, 1.1), seed: int = 42):
    rng = np.random.default_rng(seed)
    profile = np.copy(profile)
    for i in range(profile.shape[0]):
        for j in range(profile.shape[1]):
            # profile[i, j] = rng.normal(profile[i, j],
            #                            (profile[i, j] * bump_range[1]) - (profile[i, j] * bump_range[0]),
            #                            1)
            profile[i, j] = rng.uniform(profile[i, j] * bump_range[0], profile[i, j] * bump_range[1], 1)
    contribution = np.copy(contribution)
    for i in range(contribution.shape[0]):
        for j in range(contribution.shape[1]):
            # contribution[i, j] = rng.normal(contribution[i, j],
            #                                 np.abs((contribution[i, j] * bump_range[1]) - (contribution[i, j] * bump_range[0])),
            #                                 1)
            value_range = (contribution[i, j] * bump_range[0]), (contribution[i, j] * bump_range[1])
            contribution[i, j] = rng.uniform(np.min(value_range), np.max(value_range), 1)

    return profile, contribution
