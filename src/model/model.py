import os
import logging
import datetime
import json
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
from tqdm import trange
from src.model.nmf import NMF
from src.data.datahandler import DataHandler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger()


class NMFModel:
    def __init__(self,
                 dh: DataHandler,
                 epochs: int = 50,
                 n_components: int = 4,
                 max_iterations: int = 10000,
                 seed: int = 42,
                 use_original_convergence: bool = False,
                 lr_initial: float = 1e-0,
                 lr_decay_steps: int = 100,
                 lr_decay_rate: float = 0.98,
                 converge_diff: float = 100,
                 converge_iter: int = 100,
                 initial_H: list = None,
                 initial_W: list = None,
                 quiet: bool = False,
                 ):

        self.dh = dh
        self.epochs = epochs
        self.n_components = n_components
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.max_iterations = max_iterations
        self.use_original_convergence = use_original_convergence

        self.lr_initial = lr_initial
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate

        self.initial_H = initial_H
        self.initial_W = initial_W

        self.learning_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr_initial,
            decay_steps=lr_decay_steps,
            decay_rate=lr_decay_rate
        )

        self.loss_metric = tf.keras.metrics.MeanAbsoluteError()
        # self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_schedule)        # Avg R2 = 0.986
        # self.optimizer = keras.optimizers.Adam(learning_rate=1.0)
        self.optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_schedule)   # Avg R2 = 0.988 (best of 100)

        self.convergence_critera = {
            "difference_threshold": converge_diff,
            "iterations_threshold": converge_iter
        }
        if self.initial_H is not None:
            self.initial_H = np.array(self.initial_H)
            i_H = self.initial_H[0]
        else:
            i_H = None
        if self.initial_W is not None:
            self.initial_W = np.array(self.initial_W)
            i_W = self.initial_W[0]
        else:
            i_W = None

        self.nmf = NMF(seed=seed, H=i_H, W=i_W, n_components=self.n_components)
        self.results = None
        self.quiet = quiet
        if not quiet:
            self.print()

    def print(self):
        # logger.info("")
        logger.info("--------------------------------NMF-EPA Model Details ------------------------------------------")
        logger.info(f"Epochs: {self.epochs}, N Components: {self.n_components}, Seed: {self.seed}")
        logger.info(f"Max Iterations: {self.max_iterations}, Feature Count: {len(self.dh.features)}, "
                    f"Sample Count: {self.dh.input_dataset[0].shape[0]}")
        logger.info(f"Learning Rate, Initial: {self.lr_initial}, Decay Rate: {self.lr_decay_rate}, "
                    f"Decay Step: {self.lr_decay_steps}")
        logger.info(f"Convergence Difference Threshold {self.convergence_critera['difference_threshold']}, "
                    f"Convergence Iterations Threshold: {self.convergence_critera['iterations_threshold']}")
        logger.info(f"Number of GPU's available: {len(tf.config.list_physical_devices('GPU'))}")
        logger.info("------------------------------------------------------------------------------------------------")

    def fit(self):
        results = []
        silent = self.quiet

        for epoch in range(self.epochs):
            e_seed = self.rng.integers(low=0, high=1e5)
            tf.random.set_seed(e_seed)

            i_step = 0
            if epoch > 0:
                keras.backend.clear_session()
                self.loss_metric.reset_state()
                if self.initial_H is not None:
                    _i_H = epoch % len(self.initial_H)
                    i_H = self.initial_H[_i_H]
                else:
                    i_H = None
                if self.initial_W is not None:
                    _i_W = epoch % len(self.initial_W)
                    i_W = self.initial_W[_i_W]
                else:
                    i_W = None
                self.nmf = NMF(seed=e_seed, H=i_H, W=i_W, n_components=self.n_components)
                for var in self.optimizer.variables():
                    var.assign(tf.zeros_like(var))

            converged = False
            best_q = float("inf")
            best_H = None
            best_W = None
            best_WH = None

            p_loss = []
            q_loss_n = self.convergence_critera["iterations_threshold"]
            q_diff = self.convergence_critera["difference_threshold"]

            # Epoch training loop
            t_iter = trange(self.max_iterations, desc=f"Epoch {epoch + 1} fit: Q = NA", leave=True, disable=silent)
            for i_step in t_iter:

                with tf.GradientTape() as tape:
                    wh, h, w = self.nmf(self.dh.input_dataset)
                    q_loss = tf.math.reduce_sum(
                        tf.math.square(
                            tf.math.divide(tf.math.subtract(self.dh.input_dataset[0], wh),
                                           self.dh.input_dataset[1])
                        )
                    )
                grads = tape.gradient(q_loss, self.nmf.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.nmf.trainable_weights))
                self.loss_metric(self.dh.input_dataset[0], wh)

                t_iter.set_description(f"Epoch {epoch + 1} fit: Q = {round(q_loss.numpy(), 2)}")
                t_iter.refresh()

                # Convergence check
                p_loss.append(q_loss.numpy())
                if len(p_loss) > q_loss_n:
                    p_loss.pop(0)
                if int(len(p_loss)) == q_loss_n:
                    q_min = min(p_loss)
                    q_max = max(p_loss)
                    if q_max - q_min <= q_diff:
                        converged = True

                if best_q > q_loss:
                    best_q = q_loss.numpy()
                    best_H = h.numpy()
                    best_W = w.numpy()
                    best_WH = wh
                if converged:
                    break

            if not self.quiet:
                logger.info(f"\rEpoch: {epoch + 1}, Best SUM(Q): {round(best_q, 2)}, "
                            f"Steps Run: {i_step + 1}, Converged: {converged}, Seed: {e_seed}")
            results.append(
                {
                    "epoch": epoch,
                    "Q": best_q.astype(float),
                    "steps": i_step,
                    "converged": converged,
                    "H": best_H.astype(float),
                    "W": best_W.astype(float),
                    "wh": best_WH.numpy().astype(float)
                }
            )
        self.results = results

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

    def print_results(self):
        logger.info("Results")
        best_q = float("inf")
        best_epoch = -1
        for i, result in enumerate(self.results):
            if result['Q'] < best_q:
                best_q = result['Q']
                best_epoch = i
        for i, result in enumerate(self.results):
            if i == best_epoch:
                result_string = f"Best Model - Epoch: {result['epoch'] + 1}, Q: {round(result['Q'], 2)}, " \
                                f"Converged: {result['converged']}"
            else:
                result_string = f"Epoch: {result['epoch'] + 1}, Q: {round(result['Q'], 2)}, " \
                                f"Converged: {result['converged']}"
            logger.info(result_string)

    @staticmethod
    def load(nmf_path):
        if not os.path.exists(nmf_path):
            logger.error(f"File to existing NMF model not found at: {nmf_path}")
            exit()
        # TODO: Implement load model from saved file

