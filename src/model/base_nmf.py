import logging
import time
import datetime
import json
import os
import copy
from tqdm import trange, tqdm
import numpy as np
import multiprocessing as mp
from scipy.sparse import csr_matrix, csc_matrix
from src.utils import nonzeros, calculate_Q


logger = logging.getLogger("baseNMF")
logger.setLevel(logging.DEBUG)

EPSILON = 1e-15


class BaseNMF:

    def __init__(self,
                 n_components: int,
                 V: np.ndarray,
                 U: np.ndarray,
                 H: np.ndarray = None,
                 W: np.ndarray = None,
                 seed: int = None,
                 method: str = "mu"
                 ):
        self.n_components = n_components
        self.method = method

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

        self.V = self.V + EPSILON
        self.U = self.U + EPSILON

        self.Ur = np.divide(1, self.U)     # Convert uncertainty to weight for multiplication operations

        if self.H is not None:
            if self.H.shape != (self.n_components, self.m):
                logger.warn(f"The provided H matrix is not the correct shape, "
                            f"H: {self.H.shape}, expected: {(self.n_components, self.m)}")
                self.H = None
        if self.W is not None:
            if self.W.shape != (self.n, self.n_components):
                logger.warn(f"The provided W matrix is not the correct shape, "
                            f"W: {self.W.shape}, expected: {(self.n, self.n_components)}")
                self.W = None

        self.seed = 42 if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.verbose = True
        self.__build()

    def __build(self):
        V_avg = np.sqrt(self.V.mean() / self.n_components)
        if self.W is None:
            self.W = V_avg * self.rng.standard_normal(size=(self.m, self.n_components)).astype(self.V.dtype, copy=False)
            self.W = np.abs(self.W)
        if self.H is None:
            self.H = V_avg * self.rng.standard_normal(size=(self.n_components, self.n)).astype(self.V.dtype, copy=False)
            self.H = np.abs(self.H)

    def __update(self, update_weight: float = 1.0):
        if "gd" in self.method:
            _V = csr_matrix(self.V)
            _U = csr_matrix(self.Ur)

            X = self.W.copy()
            Y = self.H.copy().T

            Cui, Ciu = _V.tocsr(), _V.T.tocsr()
            Uui, Uiu = _U.toarray(), _U.T.toarray()

            X, Y = self.__gradient_descent(R=Cui.toarray(), U=Uui, P=X, Q=Y, K=self.n_components, steps=20)
            self.H = Y.T
            self.W = X

        elif "cg" in self.method:
            self.__conjugate_gradient_update()
        elif "euc" in self.method:
            # LS-NMF Uncertainty Multiplicative Update
            self.__multiplicative_update_euclidean(update_weight=update_weight)
        elif "is" in self.method:
            self.__multiplicative_update_is2_divergence()
        else:
            self.__multiplicative_update_kl_divergence(update_weight=update_weight)

    def __multiplicative_update_euclidean(self, update_weight: float = 1.0):
        # Multiplicative Update (Lee and Seung) ls-nmf

        wV = np.multiply(self.V, self.Ur)

        WH = np.matmul(self.W, self.H)
        wWH = np.multiply(WH, self.Ur)
        W_num = np.matmul(wV, self.H.T)
        W_den = np.matmul(wWH, self.H.T)
        W_delta = np.multiply(update_weight, np.divide(W_num, W_den + EPSILON))
        W = np.multiply(self.W, W_delta)

        W_columns = np.sum(W, axis=0)
        W = np.divide(W, W_columns)

        WH = np.matmul(W, self.H)
        wWH = np.multiply(WH, self.Ur)
        H_num = np.matmul(W.T, wV)
        H_den = np.matmul(W.T, wWH)
        H_delta = np.multiply(update_weight, np.divide(H_num, H_den + EPSILON))
        H = np.multiply(self.H, H_delta)

        self.H = np.abs(H)
        self.W = np.abs(W)

    def __multiplicative_update_kl_divergence(self, update_weight: float = 1.0):
        # Multiplicative Update (Kullback-Leibler)
        W_1 = np.matmul(np.divide(np.multiply(self.Ur, self.V), np.matmul(self.W, self.H)), self.H.T) + EPSILON
        W = np.multiply(np.divide(self.W, np.matmul(self.Ur, self.H.T)), np.multiply(update_weight, W_1))

        W_columns = np.sum(W, axis=0)
        W = np.divide(W, W_columns)

        H_1 = np.matmul(W.T, np.divide(np.multiply(self.Ur, self.V), np.matmul(W, self.H)) + EPSILON)
        H = np.multiply(np.divide(self.H, np.matmul(W.T, self.Ur)), np.multiply(update_weight, H_1))

        self.H = np.abs(H)
        self.W = np.abs(W)

    def __multiplicative_update_is_divergence(self, update_weight: float = 1.0):

        wh = np.matmul(self.W, self.H)
        _wh = copy.deepcopy(wh)
        _wh = 1 / _wh
        _wh = _wh ** 2
        _wh *= self.V
        numerator = np.matmul(self.W.T, _wh)
        wh = wh ** (-1)
        # denominator = np.matmul(W.T, wh)
        denominator = np.matmul(self.W.T, wh)
        denominator[denominator == 0] = EPSILON
        delta_H = numerator / denominator
        H = self.H * np.multiply(update_weight, delta_H)
        H[H < 0] = 0

        wh = np.matmul(self.W, H)
        _wh = copy.deepcopy(wh)
        _wh = 1 / _wh
        _wh = _wh ** 2
        _wh *= self.V
        numerator = np.matmul(_wh, H.T)
        wh = wh ** (-1)
        # denominator = np.matmul(wh, self.H.T)
        denominator = np.matmul(wh, H.T)
        denominator[denominator == 0] = EPSILON
        delta_W = numerator / denominator
        W = self.W * np.multiply(update_weight, delta_W)
        W[W < 0.0] = 0.0

        # W_columns = np.sum(W, axis=0)
        # W = np.divide(W, W_columns)

        self.H = H
        self.W = W

    def __multiplicative_update_is2_divergence(self):
        wh = np.matmul(self.W, self.H)
        wh2 = (1/wh)**2
        wh2vh = np.matmul(np.multiply(wh2, self.V), self.H.T)
        wtwh = np.matmul(1/wh, self.H.T)
        w_delta = np.divide(wh2vh, wtwh)
        W = np.multiply(self.W, w_delta)
        W[W <= 0] = EPSILON

        # W_columns = np.sum(W, axis=0)
        # W = np.divide(W, W_columns)

        wh = np.matmul(W, self.H)
        wh2 = (1/wh)**2
        wtwhv = np.matmul(W.T, np.multiply(wh2, self.V))
        wtwh = np.matmul(W.T, (1/wh))
        h_delta = np.divide(wtwhv, wtwh)
        H = np.multiply(self.H, h_delta)
        H[H <= 0] = EPSILON

        # H_columns = np.sum(H, axis=1).reshape(H.shape[0], 1)
        # H = np.divide(H, H_columns)

        self.H = H
        self.W = W

    def __conjugate_gradient_update(self, iterations: int = 20, regularization: float = 1e+0):
        _V = csr_matrix(self.V)
        _U = csr_matrix(self.Ur)

        X = self.W.copy()
        Y = self.H.copy().T

        Cui, Ciu = _V.tocsr(), _V.T.tocsr()
        Uui, Uiu = _U.toarray(), _U.T.toarray()

        _Q = []
        _Q_delta = [0]

        for iteration in tqdm(range(iterations),  desc="Running conjugate gradient iterations", position=0, leave=True):
            X = self.__cgm(Cui, Uui, X, Y, regularization=1e-4)
            Y = self.__cgm(Ciu, Uiu, Y, X, regularization=1e-4)
            # X = self.__cgm2(Cui, Uui, X, Y)
            # X[X < 0] = 0.0
            # Y = self.__cgm2(Ciu, Uiu, Y, X)
            self.H = Y.T
            self.W = X
            _q = self.__q_loss()
            _Q.append(_q)
            if iteration > 0:
                _Q_delta.append(_Q[iteration] - _Q[iteration - 1])
        self.H = Y.T
        self.W = X

    def __cgm(self, Cui, Uui, X, Y, regularization, cg_steps=3):
        users, factors = X.shape
        YtY = Y.T.dot(Y) + regularization * np.eye(factors)

        for u in range(users):
            # start from previous iteration
            x = X[u]

            # calculate residual r = (YtCuPu - (YtCuY.dot(Xu), without computing YtCuY
            r = -YtY.dot(x)
            for i, confidence in nonzeros(Cui, u):
                r += (confidence - (confidence - 1) * (Y[i].dot(x)) * Y[i]) * Uui[u][i]

            p = r.copy()
            rsold = r.dot(r)

            # for it in range(cg_steps):
            it = 0
            rsnew = 0.0
            _x = x.copy()
            for it in range(cg_steps):
            # while it < cg_steps or rsold > rsnew:
                # calculate Ap = YtCuYp - without actually calculating YtCuY
                Ap = YtY.dot(p)
                for i, confidence in nonzeros(Cui, u):
                    Ap += ((confidence - 1) * Y[i].dot(p) * Y[i]) * Uui[u][i]

                # standard CG update
                alpha = rsold / p.dot(Ap)
                r -= alpha * Ap
                rsnew = r.dot(r)
                drs = rsnew / rsold
                p = r + drs * p
                x += alpha * p
                rsold = rsnew
                # if rsnew < rsold:
                #     x = _x
                #     rsold = rsnew
                # else:
                #     break
                # it += 1
            X[u] = x
        return X

    def __cgm2(self, Cui, Uui, X, Y, regularization: float = 1e-12, max_k: int = 20, tol: float = 1e-3):
        iterations, factors = X.shape
        A = Y.T.dot(Y) + regularization * np.eye(factors)
        Cui = Cui.toarray()

        for u in range(iterations):
            _q = [float("inf")]
            x = X[u]
            xu = x.copy()
            # calculate residual (r0)
            rk = -A.dot(x)
            for i in range(Cui.shape[1]):
                data = Cui[u][i]
                rk += (data - (data - 1) * Y[i].dot(x)) * Y[i]
            pk = rk.copy()
            for k in range(max_k):
                p0tA = pk.T.dot(A)
                alpha_k = rk.T.dot(rk) / p0tA.dot(pk)
                x += alpha_k * pk
                rk = A.dot(x)
                for i in range(Cui.shape[1]):
                    data = Cui[u][i]
                    rk += ((data - 1) * Y[i].dot(x)) * Y[i]
                rk_sum = np.abs(np.sum(rk))
                _q.append(rk_sum)
                if _q[-2] < _q[-1] and k > 5:
                    x = xu
                    break
                if rk_sum < tol:
                    break
                beta_k = rk.T.dot(rk) / p0tA
                pk = rk + beta_k * pk
                xu = x.copy()
            X[u] = x
        return X

    def __gradient_descent(self, R, U, P, Q, K, steps=20, alpha=0.001, beta=1e-2):
        '''
        R: rating matrix
        P: |U| * K (User features matrix)
        Q: |D| * K (Item features matrix)
        K: latent features
        steps: iterations
        alpha: learning rate
        beta: regularization parameter'''
        Q = Q.T

        max_e = 1e12
        min_e = 1e-12

        _qs = []
        for step in range(steps):
            for i in range(len(R)):
                for j in range(len(R[i])):
                    if R[i][j] > 0:
                        # calculate error
                        # eij = (R[i][j] - np.dot(P[i, :], Q[:, j]))
                        eij = (R[i][j] - np.dot(P[i, :], Q[:, j])) * U[i][j]
                        if eij > 0:
                            eij = min(eij, max_e)
                        else:
                            eij = max(eij, min_e)
                        for k in range(K):
                            # calculate gradient with a and beta parameter
                            P[i][k] = max(0.0, P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k]))
                            Q[k][j] = max(0.0, Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j]))

            q = self.__q_loss(W=P, H=Q)
            _qs.append(q)
            # 0.001: local minimum
            if q < 0.001:
                break
        return P, Q.T

    def __nonlinear_conjugate_descent(self, V, U, T, O, i_max: int=2000, eps: float=1e-12):
        '''

        :param V:
        :param U:
        :param T: The initial state of the target matrix to be calculated
        :param O: The constant matrix which forms the computed values y = TO
        :param m_max:
        :param eps:
        :return:
        '''
        pass

        h_max, factors = T.shape
        X = V.copy()
        Y = np.matmul(T, O)
        weights = np.divide(1, U)

        f = T.copy()

        J1 = np.ones(shape=V.shape)             # Calculated as the partial derivative of dy/df: t_i - t_0
        J2 = np.ones(shape=V.shape)             # Calculated as the patial derivative of dy/df: t_i - t_i-1
        J3 = np.ones(shape=V.shape)             # Calculated as the partial derivative of df_h/df_n: t_i - t_i-1
        I = np.eye(V.shape[0])
        J = np.matmul([J1, J2], [I, J3])

        y0 = Y.copy()
        yi1 = Y.copy()
        fi1 = f.copy()

        precondition = np.divide(1, np.sum(weights, axis=0))
        p = 0

        # for i in range(i_max):
        #     f = f + t           # Step 3.1 - compute SE factors f_n
        #     Q = self.__q_loss(f, O)
        #     J1 = (Y - yi1) / (f - )

    def __q_loss(self, W=None, H=None, update: bool = True):
        if W is not None and H is not None:
            _wh = np.matmul(W, H)
        else:
            _wh = np.matmul(self.W, self.H)
        residuals = np.subtract(self.V, _wh)
        if update:
            self.WH = _wh
            self.residuals = residuals
        residuals_u = np.multiply(residuals, self.Ur)
        # _q = np.sum(np.multiply(residuals, residuals))
        _q = np.sum(np.multiply(residuals_u, residuals_u))
        return _q

    def train(self, epoch: int = 0, max_iterations: int = 10000, converge_delta: float = 0.01, converge_n: int = 20):
        converge_delta = converge_delta
        converge_n = converge_n
        converged = False

        prior_q = []
        _q = None

        best_q = float("inf")
        best_results = self
        reset_i = 0
        reset_max_i = 500
        update_weight = 1.0
        step_change_max = 20
        step_changes = 0

        t_iter = trange(max_iterations, desc=f"Epoch: {epoch}, Seed: {self.seed} Q(true): NA", leave=True)
        for i in t_iter:
            self.__update(update_weight=update_weight)
            _q = self.__q_loss()

            prior_q.append(_q)
            if len(prior_q) == converge_n + 1:
                prior_q.pop(0)
                delta_q_min = min(prior_q)
                delta_q_max = max(prior_q)
                if (delta_q_max - delta_q_min) < converge_delta:
                    converged = True
            t_iter.set_description(f"Epoch: {epoch}, Seed: {self.seed}, Best Q(true): {best_q}, Q(true): {round(_q, 2)}")
            t_iter.refresh()
            self.converge_steps += 1

            if best_q > _q:
                reset_i = 0
                best_q = _q
                best_results = copy.copy(self)
            else:
                reset_i += 1

            if converged or step_changes == step_change_max:
                self.converged = True
                break
            if reset_i >= reset_max_i:
                self.H = best_results.H
                self.W = best_results.W
                update_weight *= 0.98
                reset_i = 0
                step_changes += 1

        self.H = best_results.H
        self.W = best_results.W
        self.WH = best_results.WH
        self.residuals = best_results.residuals
        self.Qtrue = best_q


class BaseSearch:
    def __init__(self,
                 n_components: int,
                 V: np.ndarray,
                 U: np.ndarray,
                 H: np.ndarray = None,
                 W: np.ndarray = None,
                 seed: int = 42,
                 epochs: int = 20,
                 max_iterations: int = 10000,
                 converge_delta: float = 0.01,
                 converge_n: int = 20,
                 method: str = "mu"
                 ):
        self.n_components = n_components
        self.method = method

        self.V = V
        self.U = U

        self.H = H
        self.W = W

        self.seed = seed

        self.epochs = epochs
        self.max_iterations = max_iterations
        self.converge_delta = converge_delta
        self.converge_n = converge_n

        self.seed = 42 if seed is None else seed
        self.rng = np.random.default_rng(self.seed)

        self.results = []
        self.best_epoch = None

    def parallel_train(self):
        t0 = time.time()
        pool = mp.Pool()

        input_parameters = []
        for i in range(epochs):
            _seed = self.rng.integers(low=0, high=1e5)
            input_parameters.append((i, _seed))

        results = []
        for result in pool.starmap(self.p_train_task, input_parameters):
            results.append(result)

        best_epoch = -1
        best_q = float("inf")
        ordered_results = [None for i in range(len(results))]
        for result in results:
            epoch = int(result["epoch"])
            ordered_results[epoch] = result
            if result["Q"] < best_q:
                best_q = result["Q"]
                best_epoch = epoch
        self.results = ordered_results
        pool.close()
        t1 = time.time()
        logger.info(f"Results - Best Model: {best_epoch}, Converged: {self.results[best_epoch]['converged']}, "
                     f"Q: {self.results[best_epoch]['Q']}")
        logger.info(f"Runtime: {round((t1-t0)/60, 2)} min(s)")
        self.best_epoch = best_epoch

    def p_train_task(self, epoch, seed):
        _nmf = BaseNMF(
            n_components=self.n_components,
            method=self.method,
            V=self.V,
            U=self.U,
            H=self.H,
            W=self.W,
            seed=seed
        )
        _nmf.train(epoch=epoch, max_iterations=self.max_iterations, converge_delta=self.converge_delta,
                   converge_n=self.converge_n)
        return {
            "epoch": epoch,
            "Q": float(_nmf.Qtrue),
            "steps": _nmf.converge_steps,
            "converged": _nmf.converged,
            "H": _nmf.H,
            "W": _nmf.W,
            "wh": _nmf.WH,
            "seed": int(seed)
        }

    def train(self):
        best_Q = float("inf")
        best_epoch = None

        t0 = time.time()
        for i in range(self.epochs):
            _seed = self.rng.integers(low=0, high=1e5)
            _nmf = BaseNMF(
                n_components=self.n_components,
                method=self.method,
                V=self.V,
                U=self.U,
                H=self.H,
                W=self.W,
                seed=_seed
            )
            _nmf.train(epoch=i, max_iterations=self.max_iterations,
                       converge_delta=self.converge_delta, converge_n=self.converge_n
                       )
            if _nmf.Qtrue < best_Q:
                best_Q = _nmf.Qtrue
                best_epoch = i
            self.results.append({
                    "epoch": i,
                    "Q": float(_nmf.Qtrue),
                    "steps": _nmf.converge_steps,
                    "converged": _nmf.converged,
                    "H": _nmf.H,
                    "W": _nmf.W,
                    "wh": _nmf.WH,
                    "seed": int(_seed)
                })
        t1 = time.time()
        logger.info(f"Results - Best Model: {best_epoch}, Converged: {self.results[best_epoch]['converged']}, "
                     f"Q: {self.results[best_epoch]['Q']}")
        logger.info(f"Runtime: {round((t1-t0)/60, 2)} min(s)")
        self.best_epoch = best_epoch

    def save(self, output_name: str = None, output_path: str = None):
        if output_name is None:
            output_name = f"results_{datetime.datetime.now().strftime('%d-%m-%Y_%H%M%S')}.json"
        if output_path is None:
            output_path = "."
        elif not os.path.exists(output_path):
            os.mkdir(output_path)
        full_output_path = os.path.join(output_path, output_name)
        processed_results = []
        for result in self.results:
            processed_result = {}
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    v = v.astype(float).tolist()
                processed_result[k] = v
            processed_results.append(processed_result)
        with open(full_output_path, 'w') as json_file:
            json.dump(processed_results, json_file)
            logger.info(f"Results saved to: {full_output_path}")


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

    dh = DataHandler(
        input_path=input_file,
        uncertainty_path=uncertainty_file,
        output_path=output_path,
        index_col=index_col
    )
    # dh.scale()
    # dh.remove_outliers(quantile=0.9, drop_min=False, drop_max=True)

    n_components = 9
    method = "mu"
    V = dh.input_data_processed
    U = dh.uncertainty_data_processed
    seed = 42
    epochs = 10
    max_iterations = 40000
    converge_delta = 0.1
    converge_n = 100

    bs = BaseSearch(n_components=n_components, method=method, V=V, U=U, seed=seed, epochs=epochs, max_iterations=max_iterations,
                    converge_delta=converge_delta, converge_n=converge_n)
    bs.train()
    # bs.parallel_train()


    full_output_path = "test-base-save-01.json"
    bs.save(output_name=full_output_path)

    # pmf_file = os.path.join("D:\\", "projects", "nmf_py", "data", f"baltimore_{n_components}f_profiles.txt")
    # pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", f"baltimore_{n_components}f_residuals.txt")
    # pmf_file = os.path.join("D:\\", "projects", "nmf_py", "data", f"baton-rouge_{n_components}f_profiles.txt")
    # pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", f"baton-rouge_{n_components}f_residuals.txt")
    pmf_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test", f"br{n_components}f_profiles.txt")
    pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", "factor_test",
                                      f"br{n_components}f_residuals.txt")
    # pmf_file = os.path.join("D:\\", "projects", "nmf_py", "data", f"stlouis_{n_components}f_profiles.txt")
    # pmf_residuals_file = os.path.join("D:\\", "projects", "nmf_py", "data", f"stlouis_{n_components}f_residuals.txt")
    profile_comparison = FactorComp(nmf_output=full_output_path, pmf_output=pmf_file, factors=n_components,
                                    species=len(dh.features), residuals_path=pmf_residuals_file)
    pmf_q = calculate_Q(profile_comparison.pmf_residuals.values, dh.uncertainty_data_processed)
    profile_comparison.compare(PMF_Q=pmf_q)

    t1 = time.time()
    print(f"Runtime: {round((t1-t0)/60, 2)} min(s)")

    # df = pd.DataFrame(bs.results)
    # df.plot(kind='scatter', x='Q', y='delta_coef', logx=True, title="Q vs Delta Coef")
    # df.plot(kind='scatter', x='Q', y='delta_n', logx=True, title="Q vs Delta N")
    # df.plot(kind='scatter', x='Q', y='delta_decay', logx=True, title="Q vs Delta Decay")
    # plt.show()
