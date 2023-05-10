import logging
import time
import datetime
import json
import copy
import os
from tqdm import trange, tqdm
import numpy as np
import multiprocessing as mp
from sklearn.decomposition import NMF

logger = logging.getLogger("baseNMF")
logger.setLevel(logging.DEBUG)

EPSILON = 1e-15


class MCNMF:
    def __init__(self,
                 n_components: int,
                 V: np.ndarray,
                 U: np.ndarray,
                 H: np.ndarray = None,
                 W: np.ndarray = None,
                 seed: int = None,
                 ):
        self.n_components = n_components

        self.V = V      # Data matrix
        self.U = U      # Uncertainty matrix
        self.Ur = 1.0/U
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

        self.V = self.V
        self.U = self.U

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

        self.seed = 42 if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.verbose = True
        self.__build()

    def __resample(self):
        # Resampled data from a normal distribution using the uncertainty data
        scale = self.U * 0.5
        V = self.rng.normal(loc=self.V, scale=scale, size=self.V.shape)
        V[V < 0] = EPSILON
        return V.astype(self.V.dtype, copy=False)

    def __build(self):
        if self.W is None:
            V_avg = np.sqrt(np.mean(self.V, axis=1) / self.n_components)
            V_avg = V_avg.reshape(len(V_avg), 1)
            self.W = np.multiply(V_avg, self.rng.standard_normal(size=(self.m, self.n_components)).astype(self.V.dtype, copy=False))
            self.W = np.abs(self.W)
        if self.H is None:
            V_avg = np.sqrt(np.mean(self.V, axis=0) / self.n_components)
            self.H = V_avg * self.rng.standard_normal(size=(self.n_components, self.n)).astype(self.V.dtype, copy=False)
            self.H = np.abs(self.H)
        self.W = self.W.astype(self.V.dtype, copy=False)
        self.H = self.H.astype(self.V.dtype, copy=False)

    def train(self, epoch, n_resamples: int = 100, max_iterations: int = 20000, converge_delta: float = 0.01, converge_n: int = 200):
        # test without mapping of factors
        H_distribution = []
        W_distribution = []
        q_distribution_list = []

        q_list_n = converge_n

        t_iter = trange(n_resamples, desc=f"Epoch: {epoch}, Resample: {0}, Seed: {self.seed} Best Q(true): NA", position=0, leave=True)
        for n in t_iter:
            V = self.__resample()
            # best_q = float("inf")
            # best_W = self.W
            # best_H = self.H
            W = copy.copy(self.W)
            H = copy.copy(self.H)
            q_list = []
            # model = NMF(n_components=self.n_components, max_iter=max_iterations, solver='mu', beta_loss='frobenius', init='custom', random_state=self.seed)
            # W = model.fit_transform(X=V, W=W, H=H)
            # H = model.components_
            # _q = self.__q_loss(W=W, H=H)
            # best_q = _q
            # best_H = H
            # best_W = W

            best_q = float("inf")
            best_W = self.W
            best_H = self.H
            i = 0
            converged = False
            for i in range(max_iterations):
                W, H = self.__multiplicative_update_kl_divergence(V=V, W=W, H=H)
                _q = self.__q_loss(W=W, H=H)
                if _q < best_q:
                    best_q = _q
                    best_W = W
                    best_H = H
                q_list.append(_q)
                if len(q_list) == q_list_n:
                    q_min = min(q_list)
                    q_max = max(q_list)
                    if q_max - q_min <= converge_delta:
                        converged = True
                        break
            q_distribution_list.append(best_q)
            H_distribution.append(best_H)
            W_distribution.append(best_W)
            t_iter.set_description(f"Epoch: {epoch}, Resample:{n}, Seed: {self.seed}, Best Q(true): {best_q}")
            t_iter.refresh()
        H_distribution = np.array(H_distribution)
        W_distribution = np.array(W_distribution)
        H_mean = np.mean(H_distribution, axis=0)
        W_mean = np.mean(W_distribution, axis=0)
        final_Q = self.__q_loss(W=W_mean, H=H_mean)
        self.H = H_mean
        self.W = W_mean

        best_q = float("inf")
        best_W = self.W
        best_H = self.H
        i = 0
        converged = False
        for i in range(max_iterations):
            W, H = self.__multiplicative_update_kl_divergence_un()
            _q = self.__q_loss(W=W, H=H)
            if _q < best_q:
                best_q = _q
                best_W = W
                best_H = H
            q_list.append(_q)
            if len(q_list) == q_list_n:
                q_min = min(q_list)
                q_max = max(q_list)
                if q_max - q_min <= converge_delta:
                    converged = True
                    break

        self.W = best_W
        self.H = best_H
        self.WH = np.matmul(self.W, self.H)
        self.Qtrue = best_q
        logger.info(f"Results - N Factors: {self.n_components}, Q: {self.Qtrue}, "
                    f"Steps: {i}/{max_iterations}, Converged: {converged}")

    def __multiplicative_update_kl_divergence_un(self, update_weight: float = 1.0):
        # Multiplicative Update (Kullback-Leibler)
        # https://perso.uclouvain.be/paul.vandooren/publications/BlondelHV07.pdf Theorem 5

        wV = np.multiply(self.Ur, self.V)
        WH = np.matmul(self.W, self.H)
        H1 = np.matmul(self.W.T, np.divide(wV, WH))
        H2 = 1.0 / (np.matmul(self.W.T, self.Ur))
        H_delta = np.multiply(update_weight, np.multiply(H2, H1))
        H = np.multiply(self.H, H_delta)

        # H = (H - H.min(axis=0)) / (H.max(axis=0) - H.min(axis=0))
        # H[H < 0] = 0

        WH = np.matmul(self.W, H)
        W1 = np.matmul(np.divide(wV, WH), H.T)
        W2 = 1.0 / (np.matmul(self.Ur, H.T))
        W_delta = np.multiply(update_weight, np.multiply(W2, W1))
        W = np.multiply(self.W, W_delta)

        return W, H

    def __multiplicative_update_kl_divergence(self, V, W, H):
        # Multiplicative Update (Kullback-Leibler)
        # https://perso.uclouvain.be/paul.vandooren/publications/BlondelHV07.pdf Theorem 5

        wV = np.multiply(self.Ur, V)
        WH = np.matmul(W, H)
        H1 = np.matmul(W.T, np.divide(wV, WH))
        H2 = 1.0 / (np.matmul(W.T, self.Ur))
        H_delta = np.multiply(H2, H1)
        H = np.multiply(H, H_delta)

        # H = (H - H.min(axis=0)) / (H.max(axis=0) - H.min(axis=0))
        # H[H < 0] = 0

        WH = np.matmul(W, H)
        W1 = np.matmul(np.divide(wV, WH), H.T)
        W2 = 1.0 / (np.matmul(self.Ur, H.T))
        W_delta = np.multiply(W2, W1)
        W = np.multiply(W, W_delta)

        return W, H

    def __q_loss(self, W=None, H=None):
        if W is not None and H is not None:
            _wh = np.matmul(W, H)
        else:
            _wh = np.matmul(self.W, self.H)
        residuals = np.subtract(self.V, _wh)
        residuals_u = np.divide(residuals, self.U)
        _q = np.sum(np.multiply(residuals_u, residuals_u))
        return _q

    @staticmethod
    def __multiplicative_update_kl_divergence(V, W, H):
        # Multiplicative Update (Kullback-Leibler)
        # https://perso.uclouvain.be/paul.vandooren/publications/BlondelHV07.pdf Theorem 5

        wh = np.matmul(W, H)
        vwh = np.divide(V, wh)
        vwhw = np.matmul(vwh.T, W)
        h2 = np.sum(vwhw, axis=1)
        h1b = np.sum(W, axis=0)
        h1 = np.divide(H.T, h1b).T
        H = np.multiply(h1, h2)

        wh = np.matmul(W, H)
        vwh = np.divide(V, wh)
        vwhw = np.matmul(vwh, H.T)
        w2 = np.sum(vwhw, axis=1)
        w1b = np.sum(H, axis=1)
        w1 = np.divide(W, w1b)
        W = np.multiply(w1.T, w2).T
        return W, H


if __name__ == "__main__":

    from src.data.datahandler import DataHandler
    from src.utils import calculate_Q
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

    n_components = 4
    V = dh.input_data_processed
    U = dh.uncertainty_data_processed
    seed = 4
    epochs = 10
    max_iterations = 10000
    converge_delta = 0.01
    converge_n = 500

    results = []
    _rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        _seed = _rng.integers(low=0, high=1e5)
        mc_nmf = MCNMF(n_components=n_components, V=V, U=U, seed=_seed)
        mc_nmf.train(epoch=epoch, n_resamples=100, max_iterations=max_iterations, converge_delta=converge_delta, converge_n=converge_n)
        results.append({
            "epoch": epoch,
            "Q": float(mc_nmf.Qtrue),
            "steps": 0,
            "converged": True,
            "H": mc_nmf.H.tolist(),
            "W": mc_nmf.W.tolist(),
            "wh": mc_nmf.WH.tolist(),
            "seed": int(mc_nmf.seed)
        })

    full_output_path = f"mc_nmf-output-f{n_components}.json"
    with open(full_output_path, 'w') as json_file:
        json.dump(results, json_file)

    # pmf_file = os.path.join("D:\\", "projects", "nmf_py", "data", f"baltimore_{n_components}f_profiles.txt")
    # pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", f"baltimore_{n_components}f_residuals.txt")
    # pmf_file = os.path.join("D:\\", "projects", "nmf_py", "data", f"baton-rouge_{n_components}f_profiles.txt")
    # pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", f"baton-rouge_{n_components}f_residuals.txt")
    pmf_profile_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"br{n_components}f_profiles.txt")
    pmf_contribution_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"br{n_components}f_contributions.txt")
    pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                      f"br{n_components}f_residuals.txt")
    # pmf_file = os.path.join("D:\\", "projects", "nmf_py", "data", f"stlouis_{n_components}f_profiles.txt")
    # pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", f"stlouis_{n_components}f_residuals.txt")
    profile_comparison = FactorComp(nmf_output_file=full_output_path, pmf_profile_file=pmf_profile_file,
                                    pmf_contribution_file=pmf_contribution_file, factors=n_components,
                                    species=len(dh.features), residuals_path=pmf_residuals_file)
    pmf_q = calculate_Q(profile_comparison.pmf_residuals.values, dh.uncertainty_data_processed)
    profile_comparison.compare(PMF_Q=pmf_q)

    t1 = time.time()
    print(f"Runtime: {round((t1-t0)/60, 2)} min(s)")