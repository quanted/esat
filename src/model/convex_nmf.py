import logging
import time
import datetime
import json
import copy
import os
from tqdm import trange, tqdm
import numpy as np
from scipy.cluster.vq import kmeans
import multiprocessing as mp
from src.utils import nonzeros, calculate_Q
from nmf_pyr import nmf_pyr


logger = logging.getLogger("baseNMF")
logger.setLevel(logging.DEBUG)

EPSILON = 1e-15


class GradientDescentNMF:

    def __init__(self,
                 n_components: int,
                 V: np.ndarray,
                 U: np.ndarray,
                 H: np.ndarray = None,
                 W: np.ndarray = None,

                 seed: int = None,
                 init_kmeans: str = "W"
                 ):
        self.n_components = n_components

        self.V = V      # Data matrix
        self.U = U      # Uncertainty matrix

        self.H = H
        self.W = W

        self.WH = None
        self.residuals = None
        self.Qtrue = None
        self.converge_steps = 0
        self.converged = False

        if self.V.shape != self.U.shape:
            logger.warn(f"V and U matrix shapes are not equal, V: {V.shape}, U: {U.shape}")
        self.m, self.n = self.V.shape

        self.Ur = np.divide(1, self.U)     # Convert uncertainty to weight for multiplication operations

        if self.H is not None:
            if self.H.shape != (self.n_components, self.n):
                logger.warn(f"The provided H matrix is not the correct shape, "
                            f"H: {self.H.shape}, expected: {(self.n_components, self.n)}")
                self.H = None
        if self.W is not None:
            if self.W.shape != (self.m, self.n_components):
                logger.warn(f"The provided W matrix is not the correct shape, "
                            f"W: {self.W.shape}, expected: {(self.m, self.n_components)}")
                self.W = None

        self.init_kmeans = init_kmeans if init_kmeans in ("W", "H", "both") else None
        self.seed = 42 if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.verbose = True
        self.__build()

    def __build(self):
        if self.W is None:
            if self.init_kmeans in ("W", "both"):
                _W, dist = kmeans(obs=self.V.T, k_or_guess=self.n_components, seed=self.seed)
                self.W = _W.T
            else:
                V_avg = np.sqrt(np.mean(self.V, axis=1) / self.n_components)
                V_avg = V_avg.reshape(len(V_avg), 1)
                self.W = np.multiply(V_avg, self.rng.standard_normal(size=(self.m, self.n_components)).astype(self.V.dtype, copy=False))
                self.W = np.abs(self.W)
        if self.H is None:
            if self.init_kmeans in ("H", "both"):
                _H, dist = kmeans(obs=self.V, k_or_guess=self.n_components, seed=self.seed)
                self.H = _H
            else:
                V_avg = np.sqrt(np.mean(self.V, axis=0) / self.n_components)
                self.H = V_avg * self.rng.standard_normal(size=(self.n_components, self.n)).astype(self.V.dtype, copy=False)
                self.H = np.abs(self.H)
        self.W = self.W.astype(self.V.dtype, copy=False)
        self.H = self.H.astype(self.V.dtype, copy=False)
    def __update(self, W, H, lam: float = 1e-2, lr: float = 1e-3):
        wh = np.matmul(W, H)
        residual = (self.V - wh) / self.U
        delta_w = np.matmul(residual, H.T) - (W * lam)
        delta_h = np.matmul(W.T, residual) - (H * lam)
        _W = W + (lr * delta_w)
        _H = H + (lr * delta_h)
        _H[_H < 0] = 0
        return _W, _H

    def __q_loss(self, W=None, H=None, update: bool = True, uncertainty: bool = True):
        if W is not None and H is not None:
            _wh = np.matmul(W, H)
        else:
            _wh = np.matmul(self.W, self.H)
        residuals = np.subtract(self.V, _wh)
        if update:
            self.WH = _wh
            self.residuals = residuals
        if uncertainty:
            residuals_u = np.multiply(residuals, self.Ur)
            _q = np.sum(np.multiply(residuals_u, residuals_u))
        else:
            _q = np.sum(np.multiply(residuals, residuals))
        return _q

    def train(self, epoch: int = 0, max_iterations: int = 10000, converge_delta: float = 0.01, converge_n: int = 20, min_steps: int = 100):
        converge_delta = converge_delta
        converge_n = converge_n
        converged = False

        prior_q = []
        _q = None

        best_q = float("inf")
        best_W = None
        best_H = None
        logger.info(f"")
        t_iter = trange(max_iterations, desc=f"Epoch: {epoch}, Seed: {self.seed} Q(true): NA", position=0, leave=True)
        H = self.H
        W = self.W
        converge_i = 0
        for i in t_iter:
            W, H = self.__update(W=W, H=H)
            _q = self.__q_loss(W=W, H=H, update=False)

            if _q < best_q:
                best_W = W
                best_H = H

            if i > min_steps:
                prior_q.append(_q)
                if len(prior_q) == converge_n + 1:
                    prior_q.pop(0)
                    delta_q_min = min(prior_q)
                    delta_q_max = max(prior_q)
                    delta_q = delta_q_max - delta_q_min
                    if delta_q < converge_delta:
                        converge_i = i
                        converged = True
                        break
            t_iter.set_description(f"Epoch: {epoch}, Seed: {self.seed}, Best Q(true): {best_q}, Q(true): {round(_q, 2)}")
            t_iter.refresh()
            self.converge_steps += 1


        self.H = best_H
        self.W = best_W
        self.WH = np.matmul(best_W, best_H)
        self.converged = converged
        self.converge_steps = converge_i
        self.residuals = self.V - self.WH
        self.Qtrue = best_q




if __name__ == "__main__":

    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from src.data.datahandler import DataHandler
    from tests.factor_comparison import FactorComp

    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)

    t0 = time.time()
    input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-con.csv")
    uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-unc.csv")
    output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "BatonRouge")

    # input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-StLouis-con.csv")
    # uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-StLouis-unc.csv")
    # output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "StLouis")

    # input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-Baltimore_con.txt")
    # uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-Baltimore_unc.txt")
    # output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "Baltimore")

    index_col = "Date"
    sn_threshold = 2.0

    dh = DataHandler(
        input_path=input_file,
        uncertainty_path=uncertainty_file,
        output_path=output_path,
        index_col=index_col,
        sn_threshold=sn_threshold
    )
    # dh.scale()
    # dh.remove_outliers(quantile=0.9, drop_min=False, drop_max=True)

    n_components = 4
    V = dh.input_data_processed
    U = dh.uncertainty_data_processed
    seed = 4
    epochs = 10
    max_iterations = 40000
    converge_delta = 0.01
    converge_n = 100
    rng = np.random.default_rng(seed)
    init_kmeans = None

    for epoch in range(epochs):
        _seed = rng.integers(low=0, high=50000, size=1)[0]
        gd = GradientDescentNMF(n_components=n_components, V=V, U=U, seed=_seed, init_kmeans=init_kmeans)
        gd.train(epoch=epoch, max_iterations=max_iterations, converge_n=converge_n, converge_delta=converge_delta, min_steps=1)

    t1 = time.time()
    print(f"Runtime: {round((t1-t0)/60, 2)} min(s)")