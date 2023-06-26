import sys
import os
module_path = os.path.abspath(os.path.join('..', "nmf_py"))
sys.path.append(module_path)

from src.model.ls_nmf import LSNMF
from src.model.ws_nmf import WSNMF
from src.utils import q_loss
from scipy.cluster.vq import kmeans2, whiten
from fcmeans import FCM
from tqdm import trange
from datetime import datetime
import numpy as np
import logging
import time


logger = logging.getLogger("NMF")
logger.setLevel(logging.INFO)


class NMF:
    """
    The Non-negative matrix factorization python package primary class for creating new models.
    """
    def __init__(self,
                 V: np.ndarray,
                 U: np.ndarray,
                 factors: int,
                 method: str = "ls-nmf",
                 seed: int = 42,
                 optimized: bool = False,
                 parallelized: bool = True,
                 verbose: bool = False
                 ):

        self.V = V.astype(np.float64)
        self.U = U.astype(np.float64) + 1e-15
        self.We = np.divide(1, self.U**2).astype(np.float64)

        self.m, self.n = self.V.shape

        self.factors = factors

        self.H = None
        self.W = None

        self.method = method.lower()

        self.seed = 42 if seed is None else seed
        self.rng = np.random.default_rng(self.seed)

        self.Qrobust = None
        self.Qtrue = None
        self.WH = None

        self.epoch = -1
        self.metadata = {
            "creation_date": datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S %Z")
        }
        self.converged = False
        self.converge_steps = 0

        self.parallelized = parallelized
        self.verbose = verbose
        self.__has_neg = False

        self.__validate()
        self.__validated = False
        self.__initialized = False

        self.update_step = WSNMF.update
        if self.method == "ls-nmf" and not self.__has_neg:
            self.update_step = LSNMF.update

        self.optimized = optimized
        if self.optimized:
            # Attempt to load rust code for optimized model train
            from nmf_pyr import nmf_pyr
            if self.method == "ls-nmf"and not self.__has_neg:
                self.optimized_update = nmf_pyr.ls_nmf
            else:
                if self.parallelized:
                    self.optimized_update = nmf_pyr.ws_nmf_p
                else:
                    self.optimized_update = nmf_pyr.ws_nmf

    def __validate(self):
        """
        Validate the matrices used in NMF.
        Validation Criteria:
        V - Must be a numpy array, containing all numeric and no missing/NAN values.
        U - Must be a numpy array, containing all positive numeric and no missing/NAN values,
        and have the same dimensions of V.
        H - Must be a numpy array, containing all positive numeric and no missing/NAN values,
        and have dimensions (factors, N)
        W - Must be a numpy array, containing all numeric and no missing/NAN values, and have dimensions (M, factors)
        :return:
        """
        has_neg = False
        validated = True
        if type(self.V) != np.ndarray:
            logger.error(f"Input dataset V is not a numpy array. Current type: {type(self.V)}")
            validated = False
        else:
            if np.any(np.isnan(self.V)):
                logger.error("Input dataset V contains missing or invalid values.")
                validated = False
            if self.V.min() < 0.0:
                has_neg = True
        if type(self.U) != np.ndarray:
            logger.error(f"Uncertainty dataset U is not a numpy array. Current type: {type(self.U)}")
            validated = False
        else:
            if np.any(np.isnan(self.U)):
                logger.error("Uncertainty dataset U contains missing or invalid values.")
                validated = False
            if self.U.min() < 0.0:
                logger.error("Uncertainty dataset U contains negative values, matrix can only contain positive values.")
                validated = False
        if type(self.V) == np.ndarray and type(self.U) == np.ndarray:
            if self.V.shape != self.U.shape:
                logger.error(f"The input and uncertainty datasets must have the same dimensions. "
                             f"Current V: {self.V.shape}, U: {self.U.shape}.")
                validated = False
        if self.H is not None:
            if type(self.H) != np.ndarray:
                logger.error(f"Factor profile matrix H is not a numpy array. Current type: {type(self.H)}")
                validated = False
            else:
                if self.H.shape != (self.factors, self.n):
                    logger.error(f"Factor profile matrix H must have dimensions of ({self.factors}, {self.n})."
                                 f" Current dimensions {self.H.shape}")
                    validated = False
                if np.any(np.isnan(self.H)):
                    logger.error("Factor profile matrix H contains missing or invalid values.")
                    validated = False
                if self.H.min() < 0.0:
                    logger.error(
                        "Factor profile matrix H contains negative values, matrix can only contain positive values.")
                    validated = False
        else:
            validated = False
        if self.W is not None:
            if type(self.W) != np.ndarray:
                logger.error(f"Factor contribution matrix W is not a numpy array. Current type: {type(self.W)}")
                validated = False
            else:
                if self.W.shape != (self.m, self.factors):
                    logger.error(f"Factor contribution matrix W must have dimensions of ({self.m}, "
                                 f"{self.factors}). Current dimensions {self.W.shape}")
                    validated = False
                if np.any(np.isnan(self.W)):
                    logger.error("Factor contribution matrix W contains missing or invalid values.")
                    validated = False
                if self.W.min() < 0.0:
                    has_neg = True
        else:
            validated = False

        if validated and self.verbose:
            logger.debug("All inputs and initialized matrices have been validated.")
        self.__validated = validated
        self.__has_neg = has_neg

    def initialize(self,
                   H: np.ndarray = None,
                   W: np.ndarray = None,
                   init_method: str = "column_mean",
                   init_norm: bool = True,
                   fuzziness: float = 5.0
                   ):

        self.metadata["init_method"] = init_method
        obs = self.V
        if init_norm:
            obs = whiten(obs=self.V)

        if "kmeans" in init_method.lower():
            self.metadata["init_norm"] = init_norm
            centroids, clusters = kmeans2(data=obs, k=self.factors, seed=self.seed)
            contributions = np.zeros(shape=(len(clusters), self.factors)) + (1.0 / self.factors)
            for i, c in enumerate(clusters):
                contributions[i, c] = 1.0
            W = contributions
            H = np.abs(centroids)
            if self.verbose:
                logger.debug(f"Factor profile and contribution matrices initialized using k-means clustering. "
                             f"The observations were {'not' if not init_norm else ''} normalized.")
        elif "cmeans" in init_method.lower():
            self.metadata["init_norm"] = init_norm
            self.metadata["init_cmeans_fuzziness"] = fuzziness
            fcm = FCM(n_clusters=self.factors, m=fuzziness, random_state=self.seed)
            fcm.fit(obs)
            H = np.abs(fcm.centers)
            W = fcm.u
            if self.verbose:
                logger.debug(f"Factor profile and contribution matrices initialized using fuzzy c-means clustering. "
                             f"The observations were {'not' if not init_norm else ''} normalized.")
        else:
            if H is None:
                V_avg = np.sqrt(np.mean(self.V, axis=0) / self.factors)
                H = V_avg * self.rng.standard_normal(size=(self.factors, self.n)).astype(self.V.dtype, copy=False)
                H = np.abs(H)
            if W is None:
                V_avg = np.sqrt(np.mean(self.V, axis=1) / self.factors)
                V_avg = V_avg.reshape(len(V_avg), 1)
                W = np.multiply(V_avg, self.rng.standard_normal(size=(self.m, self.factors)).astype(self.V.dtype,
                                                                                                    copy=False))
                if self.method == "ls-nmf":
                    W = np.abs(W)
            if self.verbose:
                logger.debug(f"Factor profile and contribution matrices initialized using random selection from a "
                             f"normal distribution with a mean determined from the column mean divided by the number "
                             f"of factors.")
        self.H = H
        self.W = W
        self.init_method = init_method
        self.__initialized = True
        if self.verbose:
            logger.debug("Completed initializing the factor profile and contribution matrices.")
        self.__validate()

    def summary(self):
        logger.info("-------------------------------- NMF-EPA Model Details -----------------------------------------")
        # TODO: Add detailed summary of the model parameters
        logger.info("------------------------------------------------------------------------------------------------")

    def train(self,
              max_iter: int = 2000,
              converge_delta: float = 0.1,
              converge_n: int = 100,
              epoch: int = -1
              ):
        if not self.__initialized:
            logger.warn("Model is not initialized, initializing with default parameters")
            self.initialize()
        if not self.__validated:
            logger.error("Current model inputs and parameters are not valid.")
            return -1

        V = self.V
        U = self.U
        W = self.W
        H = self.H
        We = self.We

        if self.optimized:
            t0 = time.time()
            _results = self.optimized_update(V, U, We, W, H, max_iter, converge_delta, converge_n)[0]
            W, H, q, self.converged, self.converge_steps, q_list = _results
            t1 = time.time()
            if self.verbose:
                logger.info(f"Model: {epoch}, Seed: {self.seed}, Q(true): {round(q, 4)}, "
                      f"Steps: {self.converge_steps}/{max_iter}, Converged: {self.converged}, "
                      f"Runtime: {round(t1-t0, 2)} sec")
        else:
            q = None
            converged = False
            prior_q = []
            t_iter = trange(max_iter, desc=f"Model: {epoch}, Seed: {self.seed}, Q(true): NA", position=0, leave=True)
            for i in t_iter:
                W, H = self.update_step(V=V, We=We, W=W, H=H)
                q = q_loss(V=V, U=U, W=W, H=H)
                prior_q.append(q)
                if len(prior_q) > converge_n:
                    prior_q.pop(0)
                    delta_q_first = prior_q[0]
                    delta_q_last = prior_q[-1]
                    delta_q = delta_q_first - delta_q_last
                    if delta_q < converge_delta:
                        converged = True
                t_iter.set_description(f"Model: {epoch}, Seed: {self.seed}, Q(true): {round(q, 2)}")
                t_iter.refresh()
                self.converge_steps += 1

                if converged:
                    self.converged = True
                    break

        self.epoch = epoch
        self.H = H
        self.W = W
        self.WH = np.matmul(W, H)
        self.Qtrue = q
        self.metadata["completion_date"] = datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S %Z")

    def results(self):
        return {
            "epoch": self.epoch,
            "Q": float(self.Qtrue),
            "steps": self.converge_steps,
            "converged": self.converged,
            "H": self.H,
            "W": self.W,
            "wh": self.WH,
            "seed": int(self.seed),
            "metadata": self.metadata
        }

if __name__ == "__main__":
    import time
    import os
    from src.data.datahandler import DataHandler

    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)

    factors = 4
    method = "ws-nmf"                   # "ls-nmf", "ws-nmf"
    init_method = "col_means"           # default is column means, "kmeans", "cmeans"
    init_norm = True
    seed = 42
    # seed = 26586        #most comparable model to PMF5
    max_iterations = 20000
    converge_delta = 0.1
    converge_n = 10
    dataset = "br"          # "br": Baton Rouge, "b": Baltimore, "sl": St Louis
    verbose = True

    if dataset == "br":
        input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-con.csv")
        uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-BatonRouge-unc.csv")
        output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "BatonRouge")
    elif dataset == "b":
        input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-Baltimore_con.txt")
        uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-Baltimore_unc.txt")
        output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "Baltimore")
    elif dataset == "sl":
        input_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-StLouis-con.csv")
        uncertainty_file = os.path.join("D:\\", "projects", "nmf_py", "data", "Dataset-StLouis-unc.csv")
        output_path = os.path.join("D:\\", "projects", "nmf_py", "output", "StLouis")

    index_col = "Date"
    sn_threshold = 2.0

    dh = DataHandler(
        input_path=input_file,
        uncertainty_path=uncertainty_file,
        index_col=index_col,
        sn_threshold=sn_threshold
    )
    V = dh.input_data_processed
    U = dh.uncertainty_data_processed

    t0 = time.time()
    print("Running python code")
    nmf = NMF(V=V, U=U, factors=factors, method=method, seed=seed, optimized=False, verbose=verbose)
    nmf.initialize(init_method=init_method, init_norm=init_norm, fuzziness=5.0)
    nmf.train(max_iter=max_iterations, converge_delta=converge_delta, converge_n=converge_n)
    t1 = time.time()
    print(f"Runtime: {round((t1-t0)/60, 2)} min(s)")

    print("Running rust code")
    nmf2 = NMF(V=V, U=U, factors=factors, method=method, seed=seed, optimized=True, verbose=verbose)
    nmf2.initialize(init_method=init_method, init_norm=init_norm, fuzziness=5.0)
    nmf2.train(max_iter=max_iterations, converge_delta=converge_delta, converge_n=converge_n)

    t2 = time.time()
    print(f"Runtime: {round((t2-t1)/60, 2)} min(s)")


