import sys
import os
module_path = os.path.abspath(os.path.join('..', "nmf_py"))
sys.path.append(module_path)

import logging
import time
import pickle
import numpy as np
from pathlib import Path
import multiprocessing as mp
from src.model.nmf import NMF


logger = logging.getLogger("NMF")
logger.setLevel(logging.DEBUG)


class BatchNMF:
    """
    The batch NMF class is used to create multiple NMF models, using the same input configuration and different
    random seeds for initialization of W and H matrices.
    """
    def __init__(self,
                 V: np.ndarray,
                 U: np.ndarray,
                 factors: int,
                 models: int = 20,
                 method: str = "ls-nmf",
                 seed: int = 42,
                 init_method: str = "column_mean",
                 init_norm: bool = True,
                 fuzziness: float = 5.0,
                 max_iter: int = 20000,
                 converge_delta: float = 0.1,
                 converge_n: int = 100,
                 best_robust: bool = True,
                 robust_mode: bool = False,
                 robust_n: int = 200,
                 robust_alpha: float = 4.0,
                 parallel: bool = True,
                 optimized: bool = False,
                 verbose: bool = True
                 ):
        """
        The batch NMF class allows for the parallel execution of multiple NMF models.

        The set of parameters for the batch include both the initialization parameters and the model run parameters.

        Parameters
        ----------
        V : np.ndarray
            The input data matrix containing M samples (rows) by N features (columns).
        U : np.ndarray
            The uncertainty associated with the data points in the V matrix, of shape M x N.
        factors : int
            The number of factors, sources, NMF will create through the W and H matrices.
        models : int
            The number of NMF models to create. Default = 20.
        method : str
            The NMF algorithm to be used for updating the W and H matrices. Options are: 'ls-nmf' and 'ws-nmf'.
        seed : int
            The random seed used for initializing the W and H matrices. Default is 42.
        init_method : str
           The default option is column means, though any option other than 'kmeans' or 'cmeans' will use the column
           means initialization when W and/or H is not provided.
        init_norm : bool
           When using init_method either 'kmeans' or 'cmeans', this option allows for normalizing the input dataset
           prior to clustering.
        fuzziness : float
           The amount of fuzziness to apply to fuzzy c-means clustering. Default is 5.
        max_iter : int
           The maximum number of iterations to update W and H matrices. Default: 20000
        converge_delta:  float
           The change in the loss value where the model will be considered converged. Default: 0.1
        converge_n : int
           The number of iterations where the change in the loss value is less than converge_delta for the model to be
           considered converged. Default: 100
        best_robust: bool
           Use the Q(robust) loss value to determine which model is the best, instead of Q(true). Default = True.
        robust_mode : bool
           Used to turn on the robust mode, use the robust loss value in the update algorithm. Default: False
        robust_n : int
           When robust_mode=True, the number of iterations to use the default mode before turning on the robust mode to
           prevent reducing the impact of non-outliers. Default: 200
        robust_alpha : int
           When robust_mode=True, the cutoff of the uncertainty scaled residuals to decrease the weights. Robust weights
           are calculated as the uncertainty multiplied by the square root of the scaled residuals over robust_alpha.
           Default: 4
        parallel : bool
            Run the individual models in parallel, not the same as the optimized parallelized option for an NMF ws-nmf
            model. Default = True.
        optimized: bool
            The two update algorithms have also been written in Rust, which can be compiled with maturin, providing
            an optimized implementation for rapid training of NMF models. Setting optimized to True will run the
            compiled Rust functions.
        verbose : bool
            Allows for increased verbosity of the initialization and model training steps.
        """
        self.factors = factors
        self.method = method

        self.V = V
        self.U = U

        self.seed = seed

        self.models = models
        self.max_iter = max_iter
        self.converge_delta = converge_delta
        self.converge_n = converge_n
        self.best_robust = best_robust

        self.seed = 42 if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.init_method = init_method
        self.init_norm = init_norm
        self.fuzziness = fuzziness

        self.robust_mode = robust_mode
        self.robust_n = robust_n
        self.robust_alpha = robust_alpha

        self.parallel = parallel
        self.optimized = optimized
        self.verbose = verbose
        self.results = []
        self.best_model = None

    def train(self):
        """
        Execute the training sequence for the batch of NMF models using the shared configuration parameters.
        """
        t0 = time.time()
        if self.parallel:
            # TODO: Add batch processing for large datasets and large number of epochs to reduce memory requirements.
            pool = mp.Pool()
            input_parameters = []
            for i in range(1, self.models+1):
                _seed = self.rng.integers(low=0, high=1e5)
                _nmf = NMF(
                    factors=self.factors,
                    method=self.method,
                    V=self.V,
                    U=self.U,
                    seed=_seed,
                    optimized=self.optimized
                )
                _nmf.initialize(init_method=self.init_method, init_norm=self.init_norm, fuzziness=self.fuzziness)
                input_parameters.append((_nmf, i))

            results = pool.starmap(self._train_task, input_parameters)
            best_model = -1
            best_q = float("inf")
            ordered_results = [None for i in range(0, len(results)+1)]
            for result in results:
                model_i, _nmf = result
                ordered_results[model_i] = _nmf
                _nmf_q = _nmf.Qrobust if self.best_robust else _nmf.Qtrue
                if _nmf_q < best_q:
                    best_q = _nmf_q
                    best_model = model_i
            if self.verbose:
                for i, result in enumerate(ordered_results):
                    if result is None:
                        continue
                    logger.info(f"Model: {i}, Q(true): {round(result.Qtrue, 4)}, "
                                f"Q(robust): {round(result.Qrobust, 4)}, Seed: {result.seed}, "
                                f"Converged: {result.converged}, Steps: {result.converge_steps}/{self.max_iter}")
            self.results = ordered_results
            pool.close()
        else:
            self.results = []
            best_Q = float("inf")
            best_model = -1
            for model_i in range(1, self.models+1):
                _seed = self.rng.integers(low=0, high=1e5)
                _nmf = NMF(
                    factors=self.factors,
                    method=self.method,
                    V=self.V,
                    U=self.U,
                    seed=_seed
                )
                _nmf.initialize(init_method=self.init_method, init_norm=self.init_norm, fuzziness=self.fuzziness)
                run = _nmf.train(max_iter=self.max_iter, converge_delta=self.converge_delta, converge_n=self.converge_n,
                                 model_i=model_i, robust_mode=self.robust_mode, robust_n=self.robust_n,
                                 robust_alpha=self.robust_alpha)
                if run == -1:
                    logger.error(f"Unable to execute batch run of NMF models. Model: {model_i}")
                    pass
                _nmf_q = _nmf.Qrobust if self.best_robust else _nmf.Qtrue
                if _nmf_q < best_Q:
                    best_Q = _nmf_q
                    best_model = model_i
                self.results.append(_nmf)
        t1 = time.time()
        logger.info(f"Results - Best Model: {best_model}, Q(true): {self.results[best_model].Qtrue}, "
                    f"Q(robust): {self.results[best_model].Qrobust}, Converged: {self.results[best_model].converged}")
        logger.info(f"Runtime: {round((t1 - t0) / 60, 2)} min(s)")
        self.best_model = best_model

    def _train_task(self, nmf, model_i) -> (int, NMF):
        """
        Parallelized train task.

        Parameters
        ----------
        nmf : NMF
            Initialized NMF object.
        model_i : int
            Model id used for batch model referencing.

        Returns
        -------
        int, NMF
            The model id and the trained NMF object.

        """
        t0 = time.time()
        nmf.train(max_iter=self.max_iter, converge_delta=self.converge_delta, converge_n=self.converge_n, model_i=model_i,
                  robust_mode=self.robust_mode, robust_n=self.robust_n, robust_alpha=self.robust_alpha)
        t1 = time.time()
        if self.verbose:
            logger.info(f"Model: {model_i}, Seed: {nmf.seed}, "
                        f"Q(true): {round(nmf.Qtrue, 4)}, Q(robust): {round(nmf.Qrobust, 4)}, "
                        f"Steps: {nmf.converge_steps}/{self.max_iter}, Converged: {nmf.converged}, "
                        f"Runtime: {round(t1 - t0, 2)} sec")
        return model_i, nmf

    def save(self, batch_name: str,
             output_directory: str,
             pickle_model: bool = False,
             pickle_batch: bool = True,
             header: list = None):
        """
        Save the collection of NMF models. They can be saved as individual files (csv and json files),
        as individual pickle models (each NMF model), or as a single NMF model of the batch NMF object.

        Parameters
        ----------
        batch_name : str
            The name to use for the batch save files.
        output_directory :
            The output directory to save the batch nmf files to.
        pickle_model : bool
            Pickle the individual models, creating a separate pickle file for each NMF model. Default = False.
        pickle_batch : bool
            Pickle the batch NMF object, which will contain all the NMF objects. Default = True.
        header : list
           A list of headers, feature names, to add to the top of the csv files. Default: None

        Returns
        -------
        str
           The path to the output directory, if pickle=False or the path to the pickle file. If save fails returns None

        """
        output_directory = Path(output_directory)
        if not output_directory.is_absolute():
            current_directory = os.path.abspath(__file__)
            output_directory = Path(os.path.join(current_directory, output_directory)).resolve()
        if os.path.exists(output_directory):
            if pickle_batch:
                file_path = os.path.join(output_directory, f"{batch_name}.pkl")
                with open(file_path, "wb") as save_file:
                    pickle.dump(self, save_file)
                    logger.info(f"Batch NMF models saved to pickle file: {file_path}")
            else:
                file_path = output_directory
                for i, _nmf in enumerate(self.results):
                    file_name = f"{batch_name}-model-{i}"
                    _nmf.save(model_name=file_name, output_directory=output_directory,
                              pickle_model=pickle_model, header=header)
            logger.info(f"All batch NMF models saved. Name: {batch_name}, Directory: {output_directory}")
            return file_path
        else:
            logger.error(f"Output directory does not exist. Specified directory: {output_directory}")
            return None

    @staticmethod
    def load(file_path: str):
        """
        Load a previously saved Batch NMF pickle file.

        Parameters
        ----------
        file_path : str
           File path to a previously saved Batch NMF pickle file

        Returns
        -------
        BatchNMF
           On successful load, will return a previously saved Batch NMF object. Will return None on load fail.
        """
        file_path = Path(file_path)
        if not file_path.is_absolute():
            current_directory = os.path.abspath(__file__)
            file_path = Path(os.path.join(current_directory, file_path)).resolve()
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as pfile:
                    bnmf = pickle.load(pfile)
                    return bnmf
            except pickle.PickleError as p_error:
                logger.error(f"Failed to load BatchNMF pickle file {file_path}. \nError: {p_error}")
                return None
        else:
            logger.error(f"BatchNMF load file failed, specified pickle file does not exist. File Path: {file_path}")
            return None