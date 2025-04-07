"""
Collection of utility functions used throughout the code base.
"""

import numpy as np
import pandas as pd
import logging
import psutil
import os

logger = logging.getLogger(__name__)


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
    logger.info(f"Minimum timestep: {resample}")
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

def memory_estimate(n_features, n_samples, factors, cores: int = None):
    """
    Estimate the memory usage of the algorithm.

    Parameters
    ----------
    n_features
        Number of features.
    n_samples
        Number of samples.
    factors
        Number of factors.

    Returns
    -------
    int
        Estimated memory usage in bytes.
    """
    vm = psutil.virtual_memory()
    available_memory_bytes = np.round(vm.available, 4)
    cores = os.cpu_count() if cores is None else cores

    max_bytes = np.round(32 * ((n_features * n_samples)*3 + (n_features * factors) + (n_samples * factors)),4) * 10

    if max_bytes > available_memory_bytes:
        logger.warning(f"Estimated memory usage ({max_bytes:4f} bytes) exceeds available memory ({available_memory_bytes:4f} bytes).")

    max_parallel = int(available_memory_bytes // max_bytes)
    max_cores = min(max_parallel, cores)

    if max_bytes / (1024 ** 3) > 1.0:
        byte_string = f"{max_bytes / (1024 ** 3)} GB"
        available_string = f"{available_memory_bytes / (1024 ** 3)} GB"
    elif max_bytes / (1024 ** 2) > 1.0:
        byte_string =  f"{max_bytes / (1024 ** 2)} MB"
        available_string = f"{available_memory_bytes / (1024 ** 2)} GB"
    elif max_bytes / (1024) > 1.0:
        byte_string =  f"{max_bytes / (1024)} KB"
        available_string = f"{available_memory_bytes / (1024)} GB"
    else:
        byte_string =  f"{max_bytes} Bytes"
        available_string = f"{available_memory_bytes} GB"

    return {
        "max_cores": max_cores,
        "max_bytes": max_bytes,
        "available_memory_bytes": available_memory_bytes/(1024 ** 3),
        "available_string":available_string,
        "estimate": byte_string
    }
