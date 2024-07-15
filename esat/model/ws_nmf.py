import numpy as np
from esat.metrics import EPSILON
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
        Weighted Semi-NMF algorithm.

        The details of the semi-nmf algorithm are described in 'Convex and Semi-Nonnegative Matrix Factorizations'
        (https://doi.org/10.1109/TPAMI.2008.277). The algorithm described here does not include the use of uncertainty
        or weights. The paper 'Semi-NMF and Weighted Semi-NMF Algorithms Comparison' by Eric Velten de Melo and
        Jacques Wainer provides some additional details for part of the weighted semi-NMF algorithm as defined in this
        function.

        The update procedure defined in this function was created by merging the main concepts of these two papers.

        Parameters
        ----------
        V : np.ndarray
           The input dataset.
        We : np.ndarray
           The weights calculated from the input uncertainty dataset.
        W : np.ndarray
           The factor contribution matrix, prior W is not used by this algorithm but provided here for testing.
        H : np.ndarray
           The factor profile matrix.

        Returns
        -------
        np.ndarray, np.ndarray
           The updated W and H matrices.
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
            _w_dm = np.array(_w_d)
            # _w_dm = np.matrix(_w_d)
            if npl.det(_w_dm) == 0:
                _w_di = npl.pinv(_w_dm)
            #     _w_di = np.array(npl.pinv(_w_dm))
            else:
                _w_di = npl.inv(_w_dm)
            #     _w_di = np.array(npl.inv(_w_dm))
            _w = np.matmul(_w_n, _w_di)
            _W.append(_w)
        W = np.array(_W)

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
