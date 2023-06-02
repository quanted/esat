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
        Multiplicative Update (Lee and Seung) ls-nmf as described in
        https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-7-175
        :param V: The input data matrix.
        :param We: The weight matrix, calculated from the uncertainty matrix.
        :param W: The factor contribution matrix.
        :param H: The factor profile matrix.
        :return: The updated factor contribution and factor profile matrices. (W, H)
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
