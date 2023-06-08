import numpy as np
from src.utils import EPSILON
from numpy import linalg as npl


class WSNMF:

    @staticmethod
    def update(
            V: np.ndarray,
            We: np.ndarray,
            W: np.ndarray,
            H: np.ndarray
    ):
        """
        Semi-NMF algorithm as described in https://people.eecs.berkeley.edu/~jordan/papers/ding-li-jordan-pami.pdf
        and https://www2.dc.ufscar.br/~marcela/anaisKDMiLe2014/artigos/FULL/kdmile2014_Artigo15.pdf
        Weighted semi-NMF algorithm.
        :param V: The input data matrix.
        :param We: The weights matrix, derived from the uncertainty matrix.
        :param W: The factor contribution matrix.
        :param H: The factor profile matrix.
        :return: The updated factor contribution and factor profile matrices. (W, H)
        """
        uV = np.multiply(We, V)

        _W = []
        for i in range(V.shape[0]):
            wei = We[i]
            wei_d = np.diagflat(wei)

            uv_i = uV[i]
            uvi = uv_i.reshape(len(uv_i), 1)

            _w_n = np.matmul(H, uvi).flatten()

            uh = np.matmul(wei_d, H.T)
            _w_d = np.matmul(H, uh)
            _w_dm = np.matrix(_w_d)
            if npl.det(_w_dm) == 0:
                _w_di = np.array(npl.pinv(_w_dm))
            else:
                _w_di = np.array(npl.inv(_w_dm))
            _w = np.matmul(_w_n, _w_di)
            _W.append(_w)
        W = np.array(_W)

        # (S2)
        W_n = (np.abs(W) - W) / 2.0
        W_p = (np.abs(W) + W) / 2.0

        _H = []
        for j in range(V.shape[1]):
            wej = We[:, j]
            wej_d = np.diagflat(wej)

            uv_j = uV[:, j]
            uv_j = uv_j.reshape(len(uv_j), 1)

            n1 = np.matmul(uv_j.T, W_p)[0]
            d1 = np.matmul(uv_j.T, W_n)[0]

            n2a = np.matmul(W_n.T, wej_d)
            n2b = np.matmul(n2a, W_n)
            d2a = np.matmul(W_p.T, wej_d)
            d2b = np.matmul(d2a, W_p)

            hj = H.T[j]
            n2 = np.matmul(hj, n2b)
            d2 = np.matmul(hj, d2b)

            _n = (n1 + n2) + EPSILON
            _d = (d1 + d2) + EPSILON
            h_delta = np.sqrt(_n/_d)
            _h = np.multiply(hj, h_delta)
            _H.append(_h)

        H = np.array(_H).T
        return W, H
