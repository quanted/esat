import numpy as np
from numpy import linalg as npl


class MultiLinearEngine:
    def __init__(self,
                 factors: int,
                 V: np.ndarray,
                 U: np.ndarray,
                 H: np.ndarray,
                 W: np.ndarray
                 ):
        self.factors = factors
        self.V = V              # input data matrix (m x n)
        self.U = U              # uncertainty data matrix (m x n)
        self.H = H              # factor profiles (factors x n)
        self.W = W              # factor contributions (m x factors)

        self.m, self.n = self.V.shape

        self.C = np.ones(shape=(self.m, self.n))    # inverse preconditioning matrix
        self.weights = 1 / np.sqrt(self.U)          # calculate the weights from the uncertainty matrix (Step 2.1)


    def compute_jacobian(self):

        pass
    def run(self):
        pass
