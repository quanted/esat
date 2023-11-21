import os
import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.model.batch_nmf import BatchNMF

if __name__ == "__main__":

    k = 6  # Number of factors
    m = 40  # Number of features
    n = 300  # Number of samples
    min_noise = 0.0  # Minimum noise applied to combined input dataset cell
    max_noise = 1.0  # Maximum noise applied to combined input dataset cell
    min_unc = 5.0  # Minimum uncertainty of a cell value
    max_unc = 6.0  # Maximum uncertainty of a cell value
    seed = 42  # Random seed

    min_value = 1e-2
    rng = np.random.default_rng(seed)

    true_H = rng.uniform(1e-3, 1.0, size=(k, m))
    true_H = np.divide(true_H, np.sum(true_H, axis=0))
    true_W = rng.uniform(min_value, 100.0, size=(n, k))
    base_input = np.matmul(true_W, true_H)
    noise_matrix = (rng.uniform(min_noise, max_noise, size=(n, m)) / 100) + 1  # Generate uniform sampling of noise to be applied to the base input dataset
    data_matrix = np.multiply(base_input, noise_matrix)  # Add noise to the base input dataset
    uncertainty_values = (rng.uniform(min_unc, max_unc, size=(n, m)) / 100)  # Generate uniform sampling of uncertainty used to create the uncertainty matrix
    uncertainty_matrix = np.multiply(data_matrix, uncertainty_values)  # Create uncertainty dataset

    factors = 6
    models = 10

    method = "ws-nmf"
    batch_br = BatchNMF(V=data_matrix, U=uncertainty_matrix, max_iter=20000, converge_delta=0.01, converge_n=10,
                        factors=factors, models=models, method=method, seed=seed, verbose=True)
    batch_br.train()
    wh = batch_br.results[batch_br.best_epoch]['wh']
    res = np.corrcoef(wh, base_input)
    res_cor = res[0, 1]
    r2 = res_cor ** 2

    method2 = "ls-nmf"

    batch_br2 = BatchNMF(V=data_matrix, U=uncertainty_matrix, max_iter=100000, converge_delta=0.001, converge_n=10,
                         factors=factors, models=models, method=method2, seed=seed, verbose=True)
    batch_br2.train()
    wh2 = batch_br2.results[batch_br2.best_epoch]['wh']
    res2 = np.corrcoef(wh2, base_input)
    res_cor2 = res2[0, 1]
    r2_2 = res_cor2**2
    print(f"Factors: {factors}, Method: {method}, Q: {batch_br.results[batch_br.best_epoch]['Q']}, R2: {r2}")
    print(f"Factors: {factors}, Method: {method2}, Q: {batch_br2.results[batch_br2.best_epoch]['Q']}, R2: {r2_2}")

