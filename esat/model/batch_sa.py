import datetime
import os
import sys
import logging
import time
import pickle
import threading
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from logging.handlers import QueueHandler, QueueListener
from esat.model.sa import SA
from esat.utils import memory_estimate
from esat_rust import clear_screen

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchSA:
    """
    The batch SA class is used to create multiple SA models, using the same input configuration and different
    random seeds for initialization of W and H matrices.

    The batch SA class allows for the parallel execution of multiple NMF models.

    The set of parameters for the batch include both the initialization parameters and the model run parameters.

    Parameters
    ----------
    V : np.ndarray
        The input data matrix containing M samples (rows) by N features (columns).
    U : np.ndarray
        The uncertainty associated with the data points in the V matrix, of shape M x N.
    factors : int
        The number of factors, sources, SA will create through the W and H matrices.
    models : int
        The number of SA models to create. Default = 20.
    method : str
        The NMF algorithm to be used for updating the W and H matrices. Options are: 'ls-nmf' and 'ws-nmf'.
    seed : int
        The random seed used for initializing the W and H matrices. Default is 42.
    H : np.ndarray
        Optional, predefined factor profile matrix. Accepts profiles of size one to 'factors'.
    W : np.ndarray
        Optional, predefined factor contribution matrix.
    init_method : str
       The default option is column means 'col_means' or 'kmeans' can be specified.
    init_norm : bool
       When using init_method 'kmeans', this option allows for normalizing the input dataset
       prior to clustering.
    max_iter : int
       The maximum number of iterations to update W and H matrices. Default: 20000
    converge_delta:  float
       The change in the loss value where the model will be considered converged. Default: 0.1
    converge_n : int
       The number of iterations where the change in the loss value is less than converge_delta for the model to be
       considered converged. Default: 100
    best_robust: bool
       Use the Q(robust) loss value to determine which model is the best, instead of Q(true). Default = True.
    parallel : bool
        Run the individual models in parallel, not the same as the optimized parallelized option for an SA ws-nmf
        model. Default = True.
    cores : int
        The number of cores to use for parallel processing. Default is the number of cores - 1.
    hold_h : bool
        Hold the H matrix constant during the model training process. Default is False.
    verbose : bool
        Allows for increased verbosity of the initialization and model training steps.
    """
    def __init__(self,
                 V: np.ndarray,
                 U: np.ndarray,
                 factors: int,
                 models: int = 20,
                 method: str = "ls-nmf",
                 seed: int = 42,
                 H: np.ndarray = None,
                 W: np.ndarray = None,
                 init_method: str = "column_mean",
                 init_norm: bool = True,
                 max_iter: int = 20000,
                 converge_delta: float = 0.1,
                 converge_n: int = 100,
                 best_robust: bool = True,
                 parallel: bool = True,
                 cores: int = None,
                 hold_h: bool = False,
                 delay_h: int = -1,
                 verbose: bool = True,
                 progress_callback: callable = None
                 ):
        """
        Constructor method.
        """
        self.factors = int(factors)
        self.method = str(method)

        self.V = V
        self.U = U
        self.H = H
        self.W = W

        self.models = int(models)
        self.max_iter = int(max_iter)
        self.converge_delta = float(converge_delta)
        self.converge_n = int(converge_n)
        self.best_robust = best_robust

        self.seed = 42 if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.init_method = str(init_method)
        self.init_norm = bool(init_norm)
        self.hold_h = hold_h
        self.delay_h = delay_h

        self.progress_callback = progress_callback if callable(progress_callback) else None

        system_options = memory_estimate(self.V.shape[1], self.V.shape[0], self.factors, cores=cores)
        cores = -1 if cores is None else cores

        self.runtime = None
        self.parallel = parallel if isinstance(parallel, bool) else str(parallel).lower() == "true"
        self.cores = cores if cores > 0 else max(int(system_options["max_cores"] * 0.75), 1)
        self.optimized = True
        self.verbose = verbose if isinstance(verbose, bool) else str(verbose).lower() == "true"
        self.results = []
        self.best_model = None
        self.update_step = None

        if self.verbose:
            self.details()
            logger.info(f"Estimated memory available: {np.round(system_options['available_memory_bytes'], 4)} Gb")
            logger.info(f"Estimated memory per model: {system_options['estimate']}")
            logger.info(f"Estimated maximum number of cores: {system_options['max_cores']}")
            logger.info(f"Using {self.cores} cores for parallel processing.")
            logger.info("-------------------------------------------------")

    def details(self):
        logger.info(f"Batch Source Apportionment Instance Configuration")
        logger.info("-------------------------------------------------")
        logger.info(f"Factors: {self.factors}, Method: {self.method}, Models: {self.models}")
        logger.info(f"Max Iterations: {self.max_iter}, Converge Delta: {self.converge_delta}, "
                    f"Converge N: {self.converge_n}")
        logger.info(f"Random Seed: {self.seed}, Init Method: {self.init_method}")
        logger.info(f"Parallel: {self.parallel}, Verbose: {self.verbose}")
        if len(self.results) > 0:
            best_model = self.best_model
            logger.info("------------------------------------------------ Batch Results ------------------------------------------------")
            for i, result in enumerate(self.results):
                if result is None:
                    continue

                logger.info(f"Model: {i + 1}, "
                            f"Q(true): {result.Qtrue:.4f}, "
                            f"MSE(true): {float(result.Qtrue / self.V.size):.4f}, "
                            f"Q(robust): {float(result.Qrobust):.4f}, "
                            f"MSE(robust): {float(result.Qrobust / self.V.size):.4f}, Seed: {result.seed}, "
                            f"Converged: {result.converged}, Steps: {result.converge_steps}/{self.max_iter}")
            logger.info(f"Results - Best Model: {best_model+1}, "
                        f"Q(true): {float(self.results[best_model].Qtrue):.4f}, "
                        f"MSE(true): {float(self.results[best_model].Qtrue/self.V.size):.4f}, "
                        f"Q(robust): {float(self.results[best_model].Qrobust):.4f}, "
                        f"MSE(robust): {float(self.results[best_model].Qrobust/self.V.size):.4f}, "
                        f"Converged: {self.results[best_model].converged}")

    def train(self, min_limit: int = None):
        """
        Execute the training sequence for the batch of SA models using the shared configuration parameters.

        Parameters
        ----------
        min_limit : int
            The maximum allowed time limit for training, in minutes. Default is None and specifying this parameter will
            not enforce the time limit on the current iteration but will halt the training process if a single model
            exceeds the limit.

        Returns
        -------
        bool, str
            True and "" if the model train is successful, if training fails then the function will return False with
            an error message explaining the reason training ended.
        """
        t0 = time.time()

        with mp.Manager() as manager:  # Use Manager to create shared objects
            log_queue = manager.Queue()
            progress_queue = manager.Queue() if self.progress_callback else None

            listener = logging_listener(log_queue)  # Start the logging listener in the main process

            try:
                logger.info("Starting batch training of SA models.")
                if self.verbose:
                    logger.info(
                        f"Running batch SA models in {f'parallel using {self.cores} cores.' if self.parallel else 'single-core mode.'}")
                clear_screen()

                if not self.progress_callback:
                    print("\n" * (max(self.models, self.cores) + 1))
                    print("\033[H")
                    sys.stdout.flush()

                if self.parallel:
                    logger.info("Initializing multiprocessing setup.")
                    progress_listener = None
                    if self.progress_callback:
                        progress_listener = threading.Thread(
                            target=_progress_listener,
                            args=(progress_queue, self.progress_callback),
                            daemon=True
                        )
                        progress_listener.start()
                        logger.info("Progress listener thread started.")

                    input_parameters = []
                    for i in range(1, self.models + 1):
                        _seed = self.rng.integers(low=0, high=1e5)
                        _sa = SA(
                            factors=self.factors,
                            method=self.method,
                            V=self.V,
                            U=self.U,
                            seed=_seed,
                            verbose=False
                        )
                        i_H = self.H
                        if self.H is not None:
                            self.H = np.array(self.H)
                            if len(self.H.shape) == 3:
                                i_H = self.H[i - 1]
                            elif len(self.H.shape) == 2:
                                i_H = self.H
                        i_W = self.W
                        if self.W is not None:
                            self.W = np.array(self.W)
                            if len(self.W.shape) == 3:
                                i_W = self.W[i - 1]
                            elif len(self.W.shape) == 2:
                                i_W = self.W
                        _sa.initialize(H=i_H, W=i_W,
                                       init_method=self.init_method,
                                       init_norm=self.init_norm)
                        input_parameters.append((
                            _sa, i, progress_queue, log_queue,
                            self.max_iter, self.converge_delta, self.converge_n,
                            self.update_step, self.hold_h, self.delay_h, self.progress_callback
                        ))
                    logger.info("Input parameters for multiprocessing prepared.")

                    pool = mp.Pool(processes=self.cores)
                    logger.info("Multiprocessing pool created.")
                    results = pool.starmap(_train_task, input_parameters)
                    logger.info("Training tasks completed in multiprocessing pool.")
                    pool.close()
                    pool.join()

                    if self.progress_callback:
                        progress_queue.put("DONE")
                        progress_listener.join()
                        logger.info("Progress listener thread joined.")

                    best_model = -1
                    best_q = float("inf")
                    ordered_results = [None for _ in range(len(results) + 1)]
                    for result in results:
                        model_i, _sa = result
                        ordered_results[model_i - 1] = _sa
                        _nmf_q = _sa.Qrobust if self.best_robust else _sa.Qtrue
                        if _nmf_q < best_q:
                            best_q = _nmf_q
                            best_model = model_i
                    self.results = ordered_results
                else:
                    logger.info("Running models sequentially.")
                    self.results = []
                    best_Q = float("inf")
                    best_model = -1
                    for model_i in range(1, self.models + 1):
                        t3 = time.time()
                        _seed = self.rng.integers(low=0, high=1e5)
                        _sa = SA(
                            factors=self.factors,
                            method=self.method,
                            V=self.V,
                            U=self.U,
                            seed=_seed,
                            verbose=self.verbose,
                        )
                        _sa.initialize(H=self.H, W=self.W,
                                       init_method=self.init_method,
                                       init_norm=self.init_norm)
                        logger.info(f"Training model {model_i} with seed {_seed}.")
                        run = _sa.train(max_iter=self.max_iter, converge_delta=self.converge_delta,
                                        converge_n=self.converge_n,
                                        model_i=model_i, update_step=self.update_step, hold_h=self.hold_h,
                                        progress_callback=self.progress_callback)
                        t4 = time.time()
                        t_delta = datetime.timedelta(seconds=t4 - t3)
                        logger.info(f"Model {model_i} training completed in {t_delta}.")
                        if min_limit and t_delta.seconds / 60 > min_limit:
                            logger.warning(f"SA model training time exceeded specified runtime limit.")
                            return False, f"Error: Model train time: {t_delta} exceeded runtime limit: {min_limit} min(s)"
                        if run == -1:
                            logger.error(f"Unable to execute batch run of SA models. Model: {model_i}")
                            pass
                        _nmf_q = _sa.Qrobust if self.best_robust else _sa.Qtrue
                        if _nmf_q < best_Q:
                            best_Q = _nmf_q
                            best_model = model_i
                        self.results.append(_sa)

                t1 = time.time()
                self.runtime = round(t1 - t0, 2)
                logger.info(f"Batch training completed in {self.runtime} seconds.")

                best_model = best_model - 1
                if self.verbose:
                    logger.info(
                        "------------------------------------------------ Batch Results ------------------------------------------------")
                    for i, result in enumerate(self.results):
                        if result is None:
                            continue
                        logger.info(f"Model: {i + 1}, "
                                    f"Q(true): {result.Qtrue:.4f}, "
                                    f"MSE(true): {float(result.Qtrue / self.V.size):.4f}, "
                                    f"Q(robust): {float(result.Qrobust):.4f}, "
                                    f"MSE(robust): {float(result.Qrobust / self.V.size):.4f}, Seed: {result.seed}, "
                                    f"Converged: {result.converged}, Steps: {result.converge_steps}/{self.max_iter}")

                    logger.info(f"Results - Best Model: {best_model + 1}, "
                                f"Q(true): {float(self.results[best_model].Qtrue):.4f}, "
                                f"MSE(true): {float(self.results[best_model].Qtrue / self.V.size):.4f}, "
                                f"Q(robust): {float(self.results[best_model].Qrobust):.4f}, "
                                f"MSE(robust): {float(self.results[best_model].Qrobust / self.V.size):.4f}, "
                                f"Converged: {self.results[best_model].converged}")
                    logger.info(f"Runtime: {round((t1 - t0) / 60, 2)} min(s)")

                self.best_model = best_model
                if self.results[-1] is None:
                    self.results.pop(-1)
                return True, ""
            except Exception as e:
                logger.error(f"An error occurred during training: {e}")
                return False, str(e)
            finally:
                listener.stop()

    def _train_task(self, sa, model_i, progress_queue, log_queue):
        """
        Parallelized train task with logging.
        """
        configure_logging(log_queue)  # Configure logging for the child process
        logger = logging.getLogger()
        logger.info(f"Starting SA model {model_i} with seed {sa.seed}")

        def queue_callback(*args):
            if progress_queue is not None:
                progress_queue.put(args)

        cb = queue_callback if progress_queue is not None else self.progress_callback

        if progress_queue is not None and not is_picklable(queue_callback):
            logger.error("queue_callback is not picklable. Progress reporting will not work in multiprocessing.")
            cb = None

        if progress_queue is None and self.progress_callback and not is_picklable(self.progress_callback):
            logger.error("progress_callback is not picklable. It will not be used.")
            cb = None

        if not callable(cb):
            logger.error("Provided progress callback is not callable. Using default progress callback.")
            cb = None

        sa.train(
            max_iter=self.max_iter,
            converge_delta=self.converge_delta,
            converge_n=self.converge_n,
            model_i=model_i,
            update_step=self.update_step,
            hold_h=self.hold_h,
            delay_h=self.delay_h,
            progress_callback=cb
        )
        return model_i, sa

    def save(self, batch_name: str,
             output_directory: str,
             pickle_model: bool = False,
             pickle_batch: bool = True,
             header: list = None):
        """
        Save the collection of SA models. They can be saved as individual files (csv and json files),
        as individual pickle models (each SA model), or as a single SA model of the batch SA object.

        Parameters
        ----------
        batch_name : str
            The name to use for the batch save files.
        output_directory :
            The output directory to save the batch sa files to.
        pickle_model : bool
            Pickle the individual models, creating a separate pickle file for each SA model. Default = False.
        pickle_batch : bool
            Pickle the batch SA object, which will contain all the SA objects. Default = True.
        header : list
           A list of headers, feature names, to add to the top of the csv files. Default: None

        Returns
        -------
        str
           The path to the output directory, if pickle=False or the path to the pickle file. If save fails returns None

        """
        output_directory = Path(output_directory)
        if not output_directory.is_absolute():
            logger.error("Provided output directory is not an absolute path. Must provide an absolute path.")
            return None
        if os.path.exists(output_directory):
            if pickle_batch:
                file_path = os.path.join(output_directory, f"{batch_name}.pkl")
                with open(file_path, "wb") as save_file:
                    pickle.dump(self, save_file)
                    logger.info(f"Batch SA models saved to pickle file: {file_path}")
            else:
                file_path = output_directory
                for i, _nmf in enumerate(self.results):
                    if _nmf is not None:
                        file_name = f"{batch_name}-model-{i}"
                        _nmf.save(model_name=file_name, output_directory=output_directory,
                                  pickle_model=pickle_model, header=header)
            logger.info(f"All batch SA models saved. Name: {batch_name}, Directory: {output_directory}")
            return file_path
        else:
            logger.error(f"Output directory does not exist. Specified directory: {output_directory}")
            return None

    @staticmethod
    def load(file_path: str):
        """
        Load a previously saved Batch SA pickle file.

        Parameters
        ----------
        file_path : str
           File path to a previously saved Batch SA pickle file

        Returns
        -------
        BatchSA
           On successful load, will return a previously saved Batch SA object. Will return None on load fail.
        """
        file_path = Path(file_path)
        if not file_path.is_absolute():
            logger.error("Provided path is not an absolute path. Must provide an absolute path.")
            return None
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as pfile:
                    bsa = pickle.load(pfile)
                    return bsa
            except pickle.PickleError as p_error:
                logger.error(f"Failed to load BatchSA pickle file {file_path}. \nError: {p_error}")
                return None
        else:
            logger.error(f"BatchSA load file failed, specified pickle file does not exist. File Path: {file_path}")
            return None


def _train_task(sa, model_i, progress_queue, log_queue, max_iter, converge_delta, converge_n, update_step, hold_h, delay_h, progress_callback):
    configure_logging(log_queue)
    logger = logging.getLogger()
    logger.info(f"Starting SA model {model_i} with seed {sa.seed}")

    def queue_callback(*args):
        if progress_queue is not None:
            progress_queue.put(args)

    cb = queue_callback if progress_queue is not None else progress_callback

    sa.train(
        max_iter=max_iter,
        converge_delta=converge_delta,
        converge_n=converge_n,
        model_i=model_i,
        update_step=update_step,
        hold_h=hold_h,
        delay_h=delay_h,
        progress_callback=cb
    )
    return model_i, sa


def configure_logging(log_queue):
    """
    Configures logging for a child process to send log messages to the log queue.

    Parameters
    ----------
    log_queue : multiprocessing.Queue
        The queue to send log messages to.
    """
    queue_handler = QueueHandler(log_queue)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # Set the desired logging level
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(queue_handler)

def logging_listener(log_queue):
    """
    Sets up a logging listener to handle log messages from a multiprocessing.Queue.

    Parameters
    ----------
    log_queue : multiprocessing.Queue
        The queue to receive log messages from child processes.

    Returns
    -------
    QueueListener
        The logging listener that listens for log messages.
    """
    handler = logging.StreamHandler()  # Logs to the console
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    handler.setFormatter(formatter)

    listener = QueueListener(log_queue, handler)
    listener.start()
    return listener

def _progress_listener(queue, callback):
    while True:
        msg = queue.get()
        if msg == "DONE":
            break
        callback(*msg)

def is_picklable(obj):
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False