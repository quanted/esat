import logging
import time
import datetime
import json
import copy
import os
from tqdm import trange, tqdm
import numpy as np
import multiprocessing as mp
from scipy.sparse import csr_matrix, csc_matrix
from src.utils import nonzeros, calculate_Q
from nmf_pyr import nmf_pyr


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
                 data_mask: np.ndarray = None,
                 seed: int = None,
                 method: str = "mu"
                 ):
        self.n_components = n_components
        self.method = method

        self.V = V      # Data matrix
        self.U = U      # Uncertainty matrix

        self.H = H
        self.W = W

        self.data_mask = data_mask

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
            if self.H.shape != (self.n_components, self.n):
                logger.warn(f"The provided H matrix is not the correct shape, "
                            f"H: {self.H.shape}, expected: {(self.n_components, self.n)}")
                self.H = None
        if self.W is not None:
            if self.W.shape != (self.m, self.n_components):
                logger.warn(f"The provided W matrix is not the correct shape, "
                            f"W: {self.W.shape}, expected: {(self.m, self.n_components)}")
                self.W = None

        if self.data_mask is not None:
            if self.data_mask.shape != self.V.shape:
                logger.warn(f"The provided data mask is not the correct shape")
                self.data_mask = None

        self.seed = 42 if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.verbose = True
        self.__build()

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
        if self.data_mask is None:
            self.data_mask = np.ones(shape=self.V.shape)
            self.V = np.multiply(self.data_mask, self.V)
            self.U = np.multiply(self.data_mask, self.U)
            self.Ur = 1.0 / self.U
        self.W = self.W.astype(self.V.dtype, copy=False)
        self.H = self.H.astype(self.V.dtype, copy=False)

    def __update(self, max_iterations: int = 10000, converge_delta: float = 0.01, converge_n: int = 200,
                 update_weight: float = 1.0, update_decay: float = 0.98, retries: int = 20):
        lowest_q = float("inf")
        best_W = self.W
        best_H = self.H

        if "gd" in self.method:
            _W = self.W.copy()
            _H = self.H.copy()
            max_iterations = 20000
            q0 = self.__q_loss(W=_W, H=_H, update=False)
            qs = [q0]

            max_resets = 100
            i_resets = max_resets
            gamma = 1e-1
            t_iter = trange(max_iterations, desc=f"PGD - Seed: {self.seed} Q(true): NA", position=0, leave=True)
            for i in t_iter:
                _H, qh = self.__optimal_gradient_descent(Wt=_W, Ht=_H, gamma=gamma)
                _W, qi = self.__optimal_gradient_descent(Wt=_H.T, Ht=_W.T, gamma=gamma, switched=True)
                _W = _W.T
                if qi < lowest_q:
                    best_W = _W
                    best_H = _H
                    lowest_q = qi
                    i_resets = max_resets
                else:
                    i_resets = i_resets - 1
                    gamma *= 0.1
                if i_resets <= 0:
                    break
                qs.append(qi)
                t_iter.set_description(f"PGD - Seed: {self.seed}, Best Q(true): {lowest_q}, Q(true): {round(qi, 2)}")
                t_iter.refresh()
            W = best_W
            H = best_H
        elif "cg" in self.method:
            W, H = self.__conjugate_gradient_update()
        elif "ls-nmf" in self.method:
            # LS-NMF Uncertainty Multiplicative Update
            update_weights = [
                1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
                2, 2.5, 3, 5, 10,
                0.05, 0.01, 0.001, 0.0001
            ]
            for i in range(len(update_weights)):
                _update_weight = update_weights[i]
                W, H = self.__ls_nmf(update_weight=_update_weight)
                _q = self.__q_loss(W=W, H=H, update=False)
                if _q < lowest_q:
                    best_W = W
                    best_H = H
                    lowest_q = _q
            W = best_W
            H = best_H
        elif "euc" in self.method or "fl" in self.method:
            for i in range(retries):
                W, H = self.__multiplicative_update_euclidean(update_weight=update_weight)
                _q = self.__q_loss(W=W, H=H, update=False)
                if _q < lowest_q:
                    best_W = W
                    best_H = H
                    lowest_q = _q
                update_weight *= update_decay
            W = best_W
            H = best_H
        elif "is" in self.method:
            update_weights = [
                1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
                2, 2.5, 3, 5, 10,
                0.05, 0.01, 0.001, 0.0001
            ]
            for i in range(len(update_weights)):
                _update_weight = update_weights[i]
                W, H = self.__multiplicative_update_is_divergence(update_weight=_update_weight)
                _q = self.__q_loss(W=W, H=H, update=False)
                if _q < lowest_q:
                    best_W = W
                    best_H = H
                    lowest_q = _q
            W = best_W
            H = best_H
        else:
            update_weights = [
                1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 1.1, 1.2, 1.3, 1.4, 1.5,
            ]
            for i in range(len(update_weights)):
                _update_weight = update_weights[i]
                W, H = self.__multiplicative_update_kl_divergence(update_weight=_update_weight)
                _q = self.__q_loss(W=W, H=H, update=False)
                if _q < lowest_q:
                    best_W = W
                    best_H = H
                    lowest_q = _q
            W = best_W
            H = best_H
        self.W = W
        self.H = H

    def __ls_nmf(self, update_weight: float = 1.0):
        # Multiplicative Update (Lee and Seung) ls-nmf
        # https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-7-175

        Ur = self.Ur

        UV = np.multiply(Ur, self.V)
        WH = np.matmul(self.W, self.H)
        H_num = np.matmul(self.W.T, UV)
        H_den = np.matmul(self.W.T, np.multiply(Ur, WH))
        H = np.multiply(self.H, np.multiply(update_weight, np.divide(H_num, H_den)))

        WH = np.matmul(self.W, H)
        W_num = np.matmul(UV, H.T)
        W_den = np.matmul(np.multiply(Ur, WH), H.T)
        W = np.multiply(self.W, np.multiply(update_weight, np.divide(W_num, W_den)))

        return W, H

    def __multiplicative_update_euclidean(self, update_weight: float = 1.0):
        # https://perso.uclouvain.be/paul.vandooren/publications/BlondelHV07.pdf Theorem 4
        # https://arxiv.org/pdf/1612.06037.pdf
        # V = WH (UV)

        Ur = self.Ur
        wV = np.multiply(Ur, self.V)
        WH = np.matmul(self.W, self.H)
        H_delta = np.multiply(update_weight, np.divide(np.matmul(self.W.T, wV), np.matmul(self.W.T, np.multiply(Ur, WH))))
        H = np.multiply(self.H, H_delta)

        WH = np.matmul(self.W, H)
        W_delta = np.multiply(update_weight, np.divide(np.matmul(wV, H.T), np.matmul(np.multiply(Ur, WH), H.T)))
        W = np.multiply(self.W, W_delta)

        return W, H

    def __multiplicative_update_kl_divergence(self, update_weight: float = 1.0):
        # Multiplicative Update (Kullback-Leibler)
        # https://perso.uclouvain.be/paul.vandooren/publications/BlondelHV07.pdf Theorem 5

        # wV = np.multiply(self.Ur, self.V)

        # WH = np.matmul(self.W, self.H)
        # H1 = np.multiply(update_weight, np.matmul(self.W.T, np.divide(wV, WH)))
        # H = np.multiply(np.divide(self.H, np.matmul(self.W.T, self.Ur)), H1)
        #
        # WH = np.matmul(self.W, H)
        # W1 = np.multiply(update_weight, np.matmul(np.divide(wV, WH), H.T))
        # W = np.multiply(np.divide(self.W, np.matmul(self.Ur, H.T)), W1)

        wV = np.multiply(self.Ur, self.V)
        WH = np.matmul(self.W, self.H)
        H1 = np.matmul(self.W.T, np.divide(wV, WH))
        H2 = 1.0 / (np.matmul(self.W.T, self.Ur))
        H_delta = np.multiply(update_weight, np.multiply(H2, H1))
        H = np.multiply(self.H, H_delta)

        WH = np.matmul(self.W, H)
        W1 = np.matmul(np.divide(wV, WH), H.T)
        W2 = 1.0 / (np.matmul(self.Ur, H.T))
        W_delta = np.multiply(update_weight, np.multiply(W2, W1))
        W = np.multiply(self.W, W_delta)

        return W, H

    def __multiplicative_update_is_divergence(self, update_weight: float = 1.0):

        wh = np.matmul(self.W, self.H)
        _wh = copy.deepcopy(wh)
        _wh = 1 / _wh
        _wh = _wh ** 2
        _wh *= self.V
        _wh *= self.Ur
        numerator = np.matmul(self.W.T, _wh)
        wh = wh ** (-1)
        denominator = np.matmul(self.W.T, self.Ur * wh)
        denominator[denominator == 0] = EPSILON
        delta_H = numerator / denominator
        H = self.H * np.multiply(update_weight, delta_H)
        H[H <= 0] = EPSILON

        wh = np.matmul(self.W, H)
        _wh = copy.deepcopy(wh)
        _wh = 1 / _wh
        _wh = _wh ** 2
        _wh *= self.V
        _wh *= self.Ur
        numerator = np.matmul(_wh, H.T)
        wh = wh ** (-1)
        denominator = np.matmul(self.Ur * wh, H.T)
        denominator[denominator == 0] = EPSILON
        delta_W = numerator / denominator
        W = self.W * np.multiply(update_weight, delta_W)
        W[W <= 0.0] = EPSILON

        return W, H

    def __multiplicative_update_is2_divergence(self, update_weight: float = 1.0):

        Ur = self.Ur

        wh = np.matmul(self.W, self.H)
        H_num = np.matmul(self.W.T, np.multiply((Ur*wh)**-2, Ur*self.V))
        H_den = np.matmul(self.W.T, wh**-1)
        delta_H = update_weight * np.divide(H_num, H_den)
        H = np.multiply(self.H, delta_H)
        H[H <= 0] = EPSILON

        # H_columns = np.sum(H, axis=0)
        # H = np.divide(H, H_columns)

        wh2 = np.matmul(self.W, H)
        W_num = np.matmul(np.multiply((Ur*wh2)**-2, Ur*self.V), H.T)
        W_den = np.matmul(wh2**-1, H.T)
        delta_W = update_weight * np.divide(W_num, W_den)
        W = np.multiply(self.W, delta_W)
        W[W <= 0] = EPSILON

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
            _q = self.__q_loss(H=Y.T, W=X)
            _Q.append(_q)
            if iteration > 0:
                _Q_delta.append(_Q[iteration] - _Q[iteration - 1])
        return X, Y.T

    def __cgm(self, Cui, Uui, X, Y, regularization, cg_steps=3):
        users, factors = X.shape
        YtY = Y.T.dot(Y) + regularization * np.eye(factors)

        for u in range(users):
            # start from previous iteration
            x = X[u]

            # calculate residual r = (YtCuPu - (YtCuY.dot(Xu), without computing YtCuY
            r = -YtY.dot(x)
            for i, confidence in nonzeros(Cui, u):
                r += (confidence - (confidence - 1) * (Y[i].dot(x)) * Y[i])

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
                    Ap += ((confidence - 1) * Y[i].dot(p) * Y[i])

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

    def __projected_conjugate_gradient(self, iterations: int = 2000):
        pass
        x = self.V
        y = np.matmul(self.W, self.H)

        c = np.ones(shape=x.shape)
        W = self.Ur                     # compute weights   (2.1)
        p = np.zeros(shape=x.shape)       # compute p_n       (2.2)
        rho = 0                         # (2.3)

        beta = np.zeros(shape=x.shape)
        t = np.zeroes(shape=x.shape)

        for step in range(iterations):
                                            # compute SE factors f_h (E9.4)     (3.1)
            Q1 = self.__q_loss()            # compute current fit Q1 (E2.1)    (3.2)
            J = None                                # compute J1, J2, J3 (E4.3, E9.5)   (3.3)
            g = J.T * W * (x - y)                        # compute gradient g (E9.7)         (3.4)
            z = c * p * g                   # compute transformed gradient z: z_n = c_n * p_n * g_n
            if rho == 0:                    # (3.6)
                beta = 0
                rho = np.matmul(g.T, z)
            else:
                beta = np.matmul(g.T, z) / rho
                rho = np.matmul(g.T, z)
            t = beta * t + z                    # (3.7)
            tau = t.T * J.T * W * J * t         # (3.8)
            omega = t.T * J.T * W * (x - y)     # -- (E9.7)
            alpha = omega / tau                 # (3.9) compute initial approximation for the step length
            Q2 = Q(max(f + alpha * t, 1))       # (3.10)
            satisfied_condition = False
            if Q2 < Q1:                         # (3.11)
                alpha = alpha / (2 - (Q1 - Q2) * tau/omega**2)
                satisfied_condition = True
            else:
                max_tries = 100
                i_tries = 0
                while i_tries < max_tries:
                    alpha = alpha * 0.99
                    Q2 = Q(max(f + alpha * t, 1))
                    if Q2 < Q1:
                        satisfied_condition = True
                        break
            if satisfied_condition:
                f = max(f + alpha * t, 1)       # (3.14)
                # Step 4...
                c

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

    def __optimal_gradient_descent(self, Wt, Ht, switched: bool = False, gamma: float = 1e-3, max_iterations: int = 3):
        """

        :param Wt:
        :param Ht:
        :return:
        """
        Y = copy.copy(Ht)
        alpha = 1.0
        L = np.matmul(Wt.T, Wt)
        L[L <= 0] = EPSILON

        stop_condition = False
        i = 0
        if switched:
            q_1 = self.__q_loss(W=Ht.T, H=Wt.T, update=False)
        else:
            q_1 = self.__q_loss(W=Wt, H=Ht, update=False)
        q = [q_1]
        # Hk_1 = np.zeros(shape=Ht.shape)
        Hk_1 = copy.copy(Ht)
        l1 = 1 / L
        Hk_p = Y
        best_q = q_1
        best_Hk = copy.copy(Ht)

        grad_coef = [
            1, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 2.5, 3, 4, 5,
            10, 25, 50, 100, 150, 200, 250, 500, 1000, 2500, 5000, 1e4, 1e5, 1e6,
            0.99, 0.98, 0.97, 0.96, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.1,
            0.05, 0.01, 0.005, 0.001, 0.0005, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8
        ]

        while not stop_condition:
            if switched:
                grad = self.__calculate_gradient(W=Y.T, H=Wt.T, switched=switched).T
            else:
                grad = self.__calculate_gradient(W=Wt, H=Y, switched=switched)
            pgrad = np.matmul(l1, grad)
            # pgrad = gamma * pgrad
            #
            # Hk = Y - pgrad
            # Hk[Hk < 0.0] = 0.0

            best_grad = None
            best_grad_q = float("inf")
            best_coef = None
            for j in range(len(grad_coef) * 2):
                coef = grad_coef[int(j/2)]
                if not j % 2 == 0:
                    coef *= -1.
                _Hk = Y - gamma * (coef * pgrad)
                _Hk[_Hk < 0.0] = 0.0
                if switched:
                    _qk = self.__q_loss(W=_Hk.T, H=Wt.T, update=False)
                else:
                    _qk = self.__q_loss(W=Wt, H=_Hk, update=False)
                if _qk < best_grad_q:
                    best_grad_q = _qk
                    best_grad = _Hk
                    best_coef = coef
            Hk = best_grad

            alpha_k = (1 + np.sqrt(4 * alpha ** 2 + 1)) / 2
            Y = Hk + ((alpha-1)/alpha_k) * (Hk - Hk_1)
            alpha = alpha_k
            if switched:
                qk = self.__q_loss(W=Hk.T, H=Wt.T, update=False)
            else:
                qk = self.__q_loss(W=Wt, H=Hk, update=False)
            i += 1
            if qk < best_q:
                best_q = qk
                best_Hk = Hk
            q.append(qk)

            if i >= max_iterations:
                stop_condition = True
        return best_Hk, best_q

    def __calculate_gradient(self, W, H, switched: bool = False):
        weighted_residuals = 1/self.n_components * np.multiply(self.Ur, np.matmul(W, H) - self.V)

        if not switched:
            gradient = np.matmul(W.T, weighted_residuals)
        else:
            gradient = np.matmul(weighted_residuals, H.T)
        return gradient

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
        best_results = self
        reset_i = 0
        reset_max_i = 100
        update_weight = 1.0
        step_change_max = 10
        step_changes = 0
        update_decay = 0.99

        t_iter = trange(max_iterations, desc=f"Epoch: {epoch}, Seed: {self.seed} Q(true): NA", position=0, leave=True)
        for i in t_iter:
            self.__update(update_weight=update_weight, update_decay=update_decay)
            _q = self.__q_loss()

            if i > min_steps:
                prior_q.append(_q)
                if len(prior_q) == converge_n + 1:
                    prior_q.pop(0)
                    delta_q_min = min(prior_q)
                    delta_q_max = max(prior_q)
                    delta_q = delta_q_max - delta_q_min
                    delta_best_q = delta_q_min - best_q
                    if delta_q < converge_delta or delta_best_q > 5.0:
                        converged = True
            t_iter.set_description(f"Epoch: {epoch}, Seed: {self.seed}, Best Q(true): {best_q}, Q(true): {round(_q, 2)}")
            t_iter.refresh()
            self.converge_steps += 1

            if best_q > _q:
                reset_i = 0
                best_q = _q
                best_results = copy.copy(self)
                update_weight = 1.0
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
                 data_mask: np.ndarray = None,
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

        self.data_mask = data_mask

        self.seed = seed

        self.epochs = epochs
        self.max_iterations = max_iterations if method != "gd" else 1
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
        for i in range(self.epochs):
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
        logger.info(f"Results - N Factors: {self.n_components}, Best Model: {best_epoch}, "
                    f"Q: {self.results[best_epoch]['Q']}, Converged: {self.results[best_epoch]['converged']}")
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

    def optimized_train(self):
        best_Q = float("inf")
        best_epoch = None

        t0 = time.time()
        t_iter = trange(self.epochs, desc=f"Epoch: 0, Seed: {self.seed} Q(true): NA")
        V = self.V.astype(np.float64)
        U = self.U.astype(np.float64)
        update_weight = 2.5

        for i in t_iter:
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
            _results = nmf_pyr.nmf_kl(V, U, _nmf.W, _nmf.H, update_weight, self.max_iterations, self.converge_delta,
                                      self.converge_n)[0]
            W, H, q, converged, i_steps, q_list = _results
            # test_q1 = calculate_Q((_nmf.V - np.matmul(W, H)), _nmf.U)
            t_iter.set_description(f"Epoch: {i}, Seed: {self.seed}, Q(true): {round(q, 2)}")
            t_iter.refresh()
            if q < best_Q:
                best_Q = q
                best_epoch = i
            WH = np.matmul(W, H)
            self.results.append({
                "epoch": i,
                "Q": float(q),
                "steps": i_steps,
                "converged": converged,
                "H": H,
                "W": W,
                "wh": WH,
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

    n_components = 7
    method = "mu"                   # "kl", "ls-nmf", "is", "euc", "gd"
    V = dh.input_data_processed
    U = dh.uncertainty_data_processed
    seed = 42
    epochs = 10
    max_iterations = 20000
    converge_delta = 0.01
    converge_n = 500

    data_mask = dh.sn_mask

    bs = BaseSearch(n_components=n_components, method=method, V=V, U=U, seed=seed, epochs=epochs, max_iterations=max_iterations,
                    converge_delta=converge_delta, converge_n=converge_n, data_mask=data_mask)
    # bs.train()
    bs.parallel_train()

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
