import sys
import os

module_path = os.path.abspath(os.path.join('..', "nmf_py"))
sys.path.append(module_path)

from src.model.ls_nmf import LSNMF
from src.model.ws_nmf import WSNMF
from src.utils import q_loss, qr_loss
from scipy.cluster.vq import kmeans2, whiten
from fcmeans import FCM
from tqdm import trange
from datetime import datetime
from pathlib import Path
import numpy as np
import logging
import copy
import pickle
import json
import time

logger = logging.getLogger("NMF")
logger.setLevel(logging.INFO)


class NMF:
    """
    The primary Non-negative Matrix Factorization (NMF) model object which holds and manages the configuration, data,
    and meta-data for executing and analyzing NMF output.
    """

    def __init__(self,
                 V: np.ndarray,
                 U: np.ndarray,
                 factors: int,
                 method: str = "ws-nmf",
                 seed: int = 42,
                 optimized: bool = False,
                 parallelized: bool = True,
                 verbose: bool = False
                 ):
        """
        The NMF class object contains all the parameters and data required for executing one of the implemented NMF
        algorithms.

        The NMF class contains the core logic for managing all the steps in the NMF workflow. These include:

        1) The initialization of the factor profile (H) and factor contribution matrices (W) where these matrices can
        be set using passed in values, or randomly determined based upon the input data through mean distributions,
        k-means, or fuzzy c-means clustering.

        2) The executing of the specified NMF algorithm for updating the W and H matrices. The two currently implemented
        algorithms are least-squares nmf (LS-NMF) and weighted-semi nmf (WS-NMF).

        Parameters
        ----------
        V : np.ndarray
            The input data matrix containing M samples (rows) by N features (columns).
        U : np.ndarray
            The uncertainty associated with the data points in the V matrix, of shape M x N.
        factors : int
            The number of factors, sources, NMF will create through the W and H matrices.
        method : str
            The NMF algorithm to be used for updating the W and H matrices. Options are: 'ls-nmf' and 'ws-nmf'.
        seed : int
            The random seed used for initializing the W and H matrices. Default is 42.
        optimized : bool
            The two update algorithms have also been written in Rust, which can be compiled with maturin, providing
            an optimized implementation for rapid training of NMF models. Setting optimized to True will run the
            compiled Rust functions.
        parallelized : bool
            The Rust implementation of 'ws-nmf' has a parallelized version for increased optimization. This parameter is
            only used when method='ws-nmf' and optimized=True, then setting parallelized=True will run the parallel
            version of the function.
        verbose : bool
            Allows for increased verbosity of the initialization and model training steps.
        """

        self.V = V.astype(np.float64)
        self.U = U.astype(np.float64)

        # The uncertainty matrix must not contain any zeros
        self.U[self.U < 1e-12] = 1e-12

        # Weights are calculated from the uncertainty matrix as 1/U^{2}
        self.We = np.divide(1, self.U ** 2).astype(np.float64)

        self.m, self.n = self.V.shape

        self.factors = factors

        self.H = None
        self.W = None

        # Default to ws-nmf if the method is not valid.
        self.method = method.lower() if method.lower() in ('ls-nmf', 'ws-nmf') else 'ws-nmf'

        self.seed = 42 if seed is None else seed
        self.rng = np.random.default_rng(self.seed)

        self.Qrobust = None
        self.Qtrue = None
        self.WH = None

        self.model_i = -1
        self.metadata = {
            "creation_date": datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S %Z"),
            "method": self.method,
            "seed": self.seed,
            "samples": int(self.m),
            "features": int(self.n),
            "factors": self.factors
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
        # The ls-nmf algorithm is for matrices that only contain positive values, while ws-nmf allows for negative
        # values in the V and W matices.
        if self.method == "ls-nmf" and not self.__has_neg:
            self.update_step = LSNMF.update

        self.optimized = optimized
        if self.optimized:
            # Attempt to load rust code for optimized model training
            from nmf_pyr import nmf_pyr
            if self.method == "ls-nmf" and not self.__has_neg:
                self.optimized_update = nmf_pyr.ls_nmf
            else:
                if self.parallelized:
                    self.optimized_update = nmf_pyr.ws_nmf_p
                else:
                    self.optimized_update = nmf_pyr.ws_nmf

    def __validate(self):
        """
        Validates the input data and uncertainty matrices, as well as W and H matrices when they are provided by the
        user.

        Validation Criteria:
        V - Must be a numpy array, containing all numeric and no missing/NAN values.
        U - Must be a numpy array, containing all positive numeric and no missing/NAN values,
        and have the same shape as V.
        H - Must be a numpy array, containing all positive numeric and no missing/NAN values,
        and have the shape (factors, N)
        W - Must be a numpy array, containing all numeric and no missing/NAN values, and have the shape (M, factors)
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
        """
        Initialize the factor profile (H) and factor contribution matrices (W).

        The W and H matrices can be created using several methods or be passed in by the user. The shapes of these
        matrices are W: (M, factors) and H: (factors: N). There are three methods for initializing the W and H matrices:
        1) K Means Clustering ('kmeans'), which will cluster the input dataset into the number of factors set, then assign
        the contributions of to those factors, the H matrix is calculated from the centroids of those clusters.
        2) Fuzzy C-Means Clustering ('ceamns'), which will cluster the input dataset in the same way as kmeans but sets
        the contributions based upon the ratio of the distance to the clusters.
        3) A random sampling based upon the square root of the mean of the features (columns), the default method.

        Parameters
        ----------
        H : np.ndarray
           The factor profile matrix of shape (factors, N), provided by the user when not using one of the three
           initialization methods. H is always a non-negative matrix.
        W : np.ndarray
           The factor contribution matrix of shape (M, factors), provided by the user when not using one of the three
           initialization methods. When using method=ws-nmf, the W matrix can contain negative values.
        init_method : str
           The default option is column means, though any option other than 'kmeans' or 'cmeans' will use the column
           means initialization when W and/or H is not provided.
        init_norm : bool
           When using init_method either 'kmeans' or 'cmeans', this option allows for normalizing the input dataset
           prior to clustering.
        fuzziness : float
           The amount of fuzziness to apply to fuzzy c-means clustering. Default is 5.
        """

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
            if self.verbose and (H is None and W is None):
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
        """
        Provides a summary of the model configuration and results if completed.
        """
        logger.info("------------\t\tNMF Model Details\t\t-----------")
        logger.info(f"\tMethod: {self.method}\t\t\t\tFactors: {self.factors}")
        logger.info(f"\tNumber of Features: {self.n}\t\tNumber of Samples: {self.m}")
        logger.info(f"\tRandom Seed: {self.seed}\t\t\t\tOptimized: {self.optimized}")
        if self.WH is not None:
            logger.info("---------------\t\tModel Results\t\t--------------")
            logger.info(f"\tQ(true): {round(self.Qtrue, 2)}\t\t\tQ(robust): {round(self.Qrobust, 2)}")
            logger.info(f"\tConverged: {self.converged}\t\t\t\tConverge Steps: {self.converge_steps}")
            logger.info(f"\tRobust Mode: {'Yes' if self.metadata['robust_mode'] else 'No'}")
        logger.info("------------------------------------------------------")

    def train(self,
              max_iter: int = 20000,
              converge_delta: float = 0.1,
              converge_n: int = 100,
              model_i: int = -1,
              robust_mode: bool = False,
              robust_n: int = 200,
              robust_alpha: float = 4
              ):
        """
        Train the NMF model by iteratively updating the W and H matrices reducing the loss value Q until convergence.

        The train method runs the update algorithm until the convergence criteria is met or the maximum number
        of iterations is reached. The stopping conditions are specified by the input parameters to the train method. The
        maximum number of iterations is set by the max_iter parameter, default is 2000, and the convergence criteria is
        defined as a change in the loss value Q less than converge_delta, default is 0.1, over converge_n steps, default
        is 100.

        The loss function has an alternative mode, where the weights are modified to decrease the impact of data points
        that have a high uncertainty-scaled residual, greater than 4. This is the same loss function that calculates the
        Q(robust) value, turning robust_mode=True will switch to using the robust value for updating W and H. Robust_n
        is the number of iterations to run in the default mode before switching to the robust mode, waiting for a
        partial complete solution to be found before reducing the impact of those outlier residuals. Robust_alpha is
        both the cut off value of the uncertainty scaled residuals and the square root of the scaled residuals over
        robust_alpha is the adjustment made to the weights.

        Parameters
        ----------
        max_iter : int
           The maximum number of iterations to update W and H matrices. Default: 20000
        converge_delta : float
           The change in the loss value where the model will be considered converged. Default: 0.1
        converge_n : int
           The number of iterations where the change in the loss value is less than converge_delta for the model to be
           considered converged. Default: 100
        model_i : int
           The model index, used for identifying models for parallelized processing.
        robust_mode : bool
           Used to turn on the robust mode, use the robust loss value in the update algorithm. Default: False
        robust_n : int
           When robust_mode=True, the number of iterations to use the default mode before turning on the robust mode to
           prevent reducing the impact of non-outliers. Default: 200
        robust_alpha : float
           When robust_mode=True, the cutoff of the uncertainty scaled residuals to decrease the weights. Robust weights
            are calculated as the uncertainty multiplied by the square root of the scaled residuals over robust_alpha.
            Default: 4.0
        """
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
        converged = False

        if self.optimized:
            t0 = time.time()
            _results = self.optimized_update(V, U, We, W, H, max_iter, converge_delta, converge_n,
                                             robust_mode, robust_n, robust_alpha)[0]
            W, H, q, self.converged, self.converge_steps, q_list = _results
            q_true = q_loss(V=V, U=U, W=W, H=H)
            q_robust, U_robust = qr_loss(V=V, U=U, W=W, H=H, alpha=robust_alpha)
            t1 = time.time()
            if self.verbose:
                logger.info(f"Model: {model_i}, Seed: {self.seed}, Q(true): {round(q_true, 4)}, "
                            f"Q(robust): {round(q_robust, 4)}, Steps: {self.converge_steps}/{max_iter}, "
                            f"Converged: {self.converged}, Runtime: {round(t1 - t0, 2)} sec")
        else:
            prior_q = []
            We_prime = copy.copy(self.We)
            t_iter = trange(max_iter, desc=f"Model: {model_i}, Seed: {self.seed}, Q(true): NA, Q(robust): NA",
                            position=0, leave=True, disable=not self.verbose)
            for i in t_iter:
                W, H = self.update_step(V=V, We=We_prime, W=W, H=H)
                q_true = q_loss(V=V, U=U, W=W, H=H)
                q_robust, U_robust = qr_loss(V=V, U=U, W=W, H=H, alpha=robust_alpha)
                q = q_true
                if robust_mode:
                    if i > robust_n:
                        q = q_robust
                        We_prime = np.divide(1, U_robust ** 2).astype(np.float64)
                prior_q.append(q)
                if len(prior_q) > converge_n:
                    prior_q.pop(0)
                    delta_q_first = prior_q[0]
                    delta_q_last = prior_q[-1]
                    delta_q = delta_q_first - delta_q_last
                    if delta_q < converge_delta:
                        converged = True
                t_iter.set_description(f"Model: {model_i}, Seed: {self.seed}, Q(true): {round(q_true, 2)}, "
                                       f"Q(robust): {round(q_robust, 2)}")
                t_iter.refresh()
                self.converge_steps += 1

                if converged:
                    self.converged = True
                    break

        self.model_i = model_i
        self.H = H
        self.W = W
        self.WH = np.matmul(W, H)
        self.Qtrue = q_loss(V=V, U=U, W=W, H=H)
        self.Qrobust, _ = qr_loss(V=V, U=U, W=W, H=H, alpha=robust_alpha)
        self.metadata["completion_date"] = datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S %Z")
        self.metadata["max_iterations"] = int(max_iter)
        self.metadata["converge_delta"] = float(converge_delta)
        self.metadata["converge_n"] = int(converge_n)
        self.metadata["model_i"] = int(model_i)
        self.metadata["robust_mode"] = robust_mode
        if robust_mode:
            self.metadata["robust_n"] = int(robust_n)
            self.metadata["robust_alpha"] = float(robust_alpha)

    @staticmethod
    def __np_encoder(object):
        """
        Convert any numpy type to a generic type for json serialization.

        Parameters
        ----------
        object
           Object to be converted.
        Returns
        -------
        object
            Generic object or an unchanged object if not a numpy type
        """
        if isinstance(object, np.generic):
            return object.item()

    def save(self,
             model_name: str,
             output_directory: str,
             pickle_model: bool = False,
             header: list = None):
        """
        Save the NMF model to file.

        Two options are provided for saving the output of NMF to file, 1) saving the NMF to separate files (csv and
        json) and 2) saving the NMF model to a binary pickle object. The files are written to the provided output_directory
        path, if it exists, using the model_name for the file names.

        Parameters
        ----------
        model_name : str
           The name for the model save files.
        output_directory : str
           The path to save the files to, path must exist.
        pickle_model : bool
           Saving the model to a pickle file, default = False.
        header : list
           A list of headers, feature names, to add to the top of the csv files. Default: None

        Returns
        -------
        str
           The path to the output directory, if pickle=False or the path to the pickle file. If save fails returns None

        """
        factor_header = None
        if header is not None:
            factor_header = [f"Factor {i + 1}" for i in range(self.factors)]
            factor_header = ",".join(factor_header)
            header = ",".join(header)
        output_directory = Path(output_directory)
        if not output_directory.is_absolute():
            current_directory = os.path.abspath(__file__)
            output_directory = Path(os.path.join(current_directory, output_directory)).resolve()
        if os.path.exists(output_directory):
            if pickle_model:
                file_path = os.path.join(output_directory, f"{model_name}.pkl")
                with open(file_path, "wb") as save_file:
                    pickle.dump(self, save_file)
                    logger.info(f"NMF model saved to pickle file: {file_path}")
            else:
                file_path = output_directory
                meta_file = os.path.join(output_directory, f"{model_name}-metadata.json")
                with open(meta_file, "w") as mfile:
                    json.dump(self.metadata, mfile, default=self.__np_encoder)
                    logger.info(f"NMF model metadata saved to file: {meta_file}")
                profile_file = os.path.join(output_directory, f"{model_name}-profile.csv")
                with open(profile_file, "w") as pfile:
                    profile_comment = f"Factor Profile (H) Matrix\nMetadata File: {meta_file}\n\n"
                    np.savetxt(pfile, self.H, delimiter=',', header=header, comments=profile_comment)
                    logger.info(f"NMF model factor profile saved to file: {profile_file}")
                contribution_file = os.path.join(output_directory, f"{model_name}-contribution.csv")
                with open(contribution_file, "w") as cfile:
                    contribution_comment = f"Factor Contribution (W) Matrix\nMetadata File: {meta_file}\n\n"
                    np.savetxt(cfile, self.W, delimiter=',', header=factor_header, comments=contribution_comment)
                    logger.info(f"NMF model factor contribution saved to file: {contribution_file}")
                v_prime_file = os.path.join(output_directory, f"{model_name}-vprime.csv")
                with open(v_prime_file, 'w') as vpfile:
                    vp_comment = f"Estimated Data (WH=V') Matrix\nMetadata File: {meta_file}\n\n"
                    np.savetxt(vpfile, self.WH, delimiter=',', header=header, comments=vp_comment)
                    logger.info(f"NMF model V' saved to file: {v_prime_file}")
                residual_file = os.path.join(output_directory, f"{model_name}-residuals.csv")
                with open(residual_file, 'w') as rfile:
                    residual_comment = f"Residual Matrix (V-V')\nMetadata File: {meta_file}\n\n"
                    residuals = self.V - self.WH
                    np.savetxt(rfile, residuals, delimiter=',', header=header, comments=residual_comment)
                    logger.info(f"NMF model residuals saved to file: {residual_file}")
            return file_path

        else:
            logger.error(f"Output directory does not exist. Specified directory: {output_directory}")
            return None

    @staticmethod
    def load(file_path: str):
        """
        Load a previously saved NMF pickle file.

        Parameters
        ----------
        file_path : str
           File path to a previously saved NMF pickle file

        Returns
        -------
        NMF
           On successful load, will return a previously saved NMF object. Will return None on load fail.
        """
        file_path = Path(file_path)
        if not file_path.is_absolute():
            current_directory = os.path.abspath(__file__)
            file_path = Path(os.path.join(current_directory, file_path)).resolve()
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as pfile:
                    nmf = pickle.load(pfile)
                    return nmf
            except pickle.PickleError as p_error:
                logger.error(f"Failed to load NMF pickle file {file_path}. \nError: {p_error}")
                return None
        else:
            logger.error(f"NMF load file failed, specified pickle file does not exist. File Path: {file_path}")
            return None


if __name__ == "__main__":

    # Test code for running a single NMF model, using both the python and Rust functions, includes model save and loads.
    import time
    import os
    from src.data.datahandler import DataHandler

    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)

    factors = 6
    method = "ws-nmf"                   # "ls-nmf", "ws-nmf"
    init_method = "col_means"           # default is column means, "kmeans", "cmeans"
    init_norm = True
    seed = 42
    max_iterations = 20000
    converge_delta = 0.1
    converge_n = 10
    dataset = "br"                      # "br": Baton Rouge, "b": Baltimore, "sl": St Louis
    verbose = True
    robust_mode = True
    robust_n = 100
    robust_alpha = 4

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
    nmf.train(max_iter=max_iterations, converge_delta=converge_delta, converge_n=converge_n,
              robust_alpha=robust_alpha, robust_n=robust_n, robust_mode=robust_mode)
    t1 = time.time()
    print(f"Runtime: {round((t1 - t0) / 60, 2)} min(s)")

    print("Running rust code")
    nmf2 = NMF(V=V, U=U, factors=factors, method=method, seed=seed, optimized=True, verbose=verbose)
    nmf2.initialize(init_method=init_method, init_norm=init_norm, fuzziness=5.0)
    nmf2.train(max_iter=max_iterations, converge_delta=converge_delta, converge_n=converge_n,
               robust_alpha=robust_alpha, robust_n=robust_n, robust_mode=robust_mode)
    t2 = time.time()
    print(f"Runtime: {round((t2 - t1) / 60, 2)} min(s)")

    nmf2.save(model_name="test", output_directory="..\\..\\..\\data\\output\\", pickle_model=False,
              header=list(dh.features))
    nmf2.summary()

    _nmf = NMF.load(file_path="..\\..\\..\\data\\output\\test.pkl")
