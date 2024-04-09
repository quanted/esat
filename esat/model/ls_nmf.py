import numpy as np


class LSNMF:

    @staticmethod
    def update(
            V: np.ndarray,
            We: np.ndarray,
            W: np.ndarray,
            H: np.ndarray
    ):
        """
        The update procedure for the least-squares nmf (ls-nmf) algorithm.

        The ls-nmf algorithm is described in the publication 'LS-NMF: A modified non-negative matrix factorization
        algorithm utilizing uncertainty estimates' (https://doi.org/10.1186/1471-2105-7-175).

        Parameters
        ----------
        V : np.ndarray
           The input dataset.
        We : np.ndarray
           The weights calculated from the input uncertainty dataset.
        W : np.ndarray
           The factor contribution matrix.
        H : np.ndarray
           The factor profile matrix.

        Returns
        -------
        np.ndarray, np.ndarray
           The updated W and H matrices.

        """
        WeV = np.multiply(We, V)
        WH = np.matmul(W, H)
        H_num = np.matmul(W.T, WeV)
        H_den = np.matmul(W.T, np.multiply(We, WH))
        H = np.multiply(H, np.divide(H_num, H_den))

        WH = np.matmul(W, H)
        W_num = np.matmul(WeV, H.T)
        W_den = np.matmul(np.multiply(We, WH), H.T)
        W = np.multiply(W, np.divide(W_num, W_den))

        return W, H
