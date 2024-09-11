import logging
import os
import copy
import numpy as np
import pandas as pd
import multiprocessing as mp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from esat.model.sa import SA
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FactorEstimator:
    """
    Factor search uses a Monte Carlo sampling approach for testing different factor counts using cross-validation
    testing. Both a train and a test MSE are calculated for each model in the search. These MSE values are averaged
    for each factor count and the change in test MSE is used to estimate the factor count for the dataset.

    Reference: http://alexhwilliams.info/itsneuronalblog/2018/02/26/crossval/

    Parameters
    ----------
    V : np.ndarray
        The input dataset to use for the factor search.
    U : np.ndarray
        The uncertainty dataset to use for the factor search.
    seed : int
        The random seed to use for the model initialization, cross-validation masking, and factor selection.
    test_percent : float
        The decimal percentage of values in the input dataset to use for the MSE test calculation.
    k_coef: float
        The K estimate metric calculation uses a coefficient that can be used for tuning.
    """
    def __init__(self, V: np.ndarray, U: np.ndarray, seed: int = 42, test_percent: float = 0.1, k_coef: float = 1.0):
        self.V = V
        self.U = U
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        self.test_percent = test_percent
        self.m, self.n = self.V.shape

        self.k_coef = k_coef
        self.min_factors = 2
        self.max_factors = 15
        self.samples = 200
        self.pbar = None
        self.train_mse = None
        self.test_mse = None
        self.q_true = None
        self.q_robust = None
        self.estimated_factor = None
        self.results_df = None

    def _get_mask(self, threshold=0.1):
        _mask = np.zeros(shape=self.V.shape)
        for feature in range(self.m):
            feature_i = self.rng.random(size=self.n) > threshold
            _mask[feature] = feature_i
        return _mask.astype(int)

    def _update_pbar(self, results):
        list_i = results[2]-self.min_factors
        self.train_mse[list_i].append(results[0])
        self.test_mse[list_i].append(results[1])
        self.q_true[list_i].append(results[3])
        self.q_robust[list_i].append(results[4])
        self.pbar.update(1)

    @staticmethod
    def _random_sample(V, U, mask, seed, factor_n, max_iter: int = 2000, converge_delta: float = 1.0, converge_n: int = 10):
        m_train = np.count_nonzero(mask)
        i_mask = copy.copy(mask)
        i_mask[i_mask == 1] = 0
        i_mask[i_mask == 0] = 1
        m_test = np.count_nonzero(i_mask)
        _sa = SA(V=V, U=U, factors=factor_n, method="ls-nmf", seed=seed, optimized=True, verbose=False)
        _sa.initialize()
        _sa.train(max_iter=max_iter, converge_delta=converge_delta, converge_n=converge_n)
        residuals = V - _sa.WH
        s_residuals = residuals**2
        train_residuals = np.multiply(mask, s_residuals)

        test_residuals = np.multiply(i_mask, s_residuals)
        train_mse = np.round(train_residuals.sum()/m_train, 5)
        test_mse = np.round(test_residuals.sum()/m_test, 5)
        return train_mse, test_mse, factor_n, _sa.Qtrue, _sa.Qrobust

    def run(self, samples: int = 200, min_factors: int = 2, max_factors: int = 15, max_iterations: int = 2000,
            converge_delta: float = 1.0, converge_n: int = 10):
        """
        Run the Monte Carlo sampling for a random set of models using factor counts between min_factors and max_factors
        a specified number of times, samples.

        When the results are inconclusive or there are several large peaks in the delta MSE line, increasing the sample
        count can help narrow the estimation.

        Parameters
        ----------
        samples : int
            The number of random samples to take for the factor estimation.
        min_factors : int
            The minimum number of factors to consider in the random sampling.
        max_factors : int
            The maximum number of factors to consider in the random sampling.
        max_iterations : int
            The maximum number of iterations to run the models.
        converge_delta : float
            The change in the loss value over a specified numbers of steps for the model to be considered converged.
        converge_n : int
            The number of steps where the loss changes by less than converge_delta, for the model to be considered
            converged.

        Returns
        -------
        pd.DataFrame
            The results of the factor search showing the metrics used to estimate the factor count.
        """
        self.min_factors = min_factors
        self.max_factors = max_factors + 1
        self.samples = samples
        self.pbar = tqdm(total=samples, desc="Rapid random sampling for factor estimation")

        self.train_mse = [[] for i in range(self.max_factors - self.min_factors)]
        self.test_mse = [[] for i in range(self.max_factors - self.min_factors)]
        self.q_true = [[] for i in range(self.max_factors - self.min_factors)]
        self.q_robust = [[] for i in range(self.max_factors - self.min_factors)]

        pool_parameters = []
        for i in range(samples):
            seed_i = self.rng.integers(low=100, high=1e6, size=1)[0]
            factor_i = self.rng.integers(low=self.min_factors, high=self.max_factors, size=1)[0]
            mask = self._get_mask(threshold=self.test_percent)
            pool_parameters.append((self.V, self.U, mask, seed_i, factor_i, max_iterations, converge_delta, converge_n))

        pool = mp.Pool(os.cpu_count()-1)
        results = []
        for p_parameter in pool_parameters:
            r = pool.apply_async(self._random_sample, p_parameter, callback=self._update_pbar)
            results.append(r)
        for r in results:
            r.wait()
        for r in results:
            r.get()
        pool.close()
        pool.join()
        self.pbar.close()

        self.train_mse = [np.mean(i) for i in self.train_mse]
        self.test_mse = [np.mean(i) for i in self.test_mse]
        self.q_true = [np.mean(i) for i in self.q_true]
        self.q_robust = [np.mean(i) for i in self.q_robust]

        return self._results()

    def _results(self):
        delta_mse_r = []
        for factor_n in range(0, len(self.test_mse) - 1):
            delta_i = self.test_mse[factor_n] - self.test_mse[factor_n + 1]
            delta_mse_r.append(delta_i)
        c = 1.01 * np.max(delta_mse_r)
        ratio_delta = [np.nan]
        for factor_n in range(0, len(self.test_mse) - 2):
            rd = c*(delta_mse_r[factor_n]/delta_mse_r[factor_n + 1])
            # rd = (delta_mse_r[factor_n]/min1) / np.abs(max1 - c * delta_mse_r[factor_n + 1])
            ratio_delta.append(rd)
        ratio_delta.append(np.nan)

        mse_min = np.min(self.test_mse)
        k_est = []
        for factor_n in range(0, self.max_factors-self.min_factors):
            rd = mse_min/(self.test_mse[factor_n]*np.power(factor_n+self.min_factors, self.k_coef))
            k_est.append(rd)
        delta_mse = [np.nan]
        for factor_n in range(0, len(self.test_mse) - 1):
            delta_i = self.test_mse[factor_n] - self.test_mse[factor_n + 1]
            delta_mse.append(delta_i)
        if np.all(np.isnan(k_est)):
            self.estimated_factor = -1
        else:
            self.estimated_factor = np.nanargmax(k_est) + self.min_factors
        # logger.info(f"Estimated factor count: {self.estimated_factor}")
        self.results_df = pd.DataFrame(data=
                                       {
                                           "Factors": list(range(self.min_factors, self.max_factors)),
                                           "Test MSE": self.test_mse,
                                           "Train MSE": self.train_mse,
                                           "Delta MSE": delta_mse,
                                           "Delta Ratio": ratio_delta,
                                           "K Estimate": k_est,
                                           "Q(True)": self.q_true,
                                           "Q(Robust)": self.q_robust
                                       })
        return self.results_df

    def plot(self, actual_count: int = None):
        """
        Plot the results of the factor search as seen by the results table. When the actual number of factors are known,
        they can be provided using the actual_count parameter. The estimated factor count will be shown as a red dashed
        vertical line, the actual factor count is shown as a black dashed vertical line when it is provided.

        Parameters
        ----------
        actual_count : int
            The known factor count value, such as when using the Simulator.

        """
        mse_fig = make_subplots(specs=[[{"secondary_y": True}]])
        x = list(range(self.min_factors, self.max_factors))
        mse_fig.add_trace(go.Scatter(x=x, y=self.results_df["Train MSE"], name="Train MSE", mode='lines+markers'),
                          secondary_y=False)
        mse_fig.add_trace(go.Scatter(x=x, y=self.results_df["Test MSE"], name="Test MSE", mode='lines+markers'),
                          secondary_y=False)
        mse_fig.add_trace(go.Scatter(x=x, y=self.results_df["Delta MSE"], name="Delta MSE", mode='lines+markers'),
                          secondary_y=False)
        mse_fig.add_trace(go.Scatter(x=x, y=self.results_df["Delta Ratio"], name="Ratio Delta", mode='lines+markers'),
                          secondary_y=False)
        mse_fig.add_trace(go.Scatter(x=x, y=self.results_df["K Estimate"], name="K Estimate", mode='lines+markers'),
                          secondary_y=False)
        mse_fig.add_trace(go.Scatter(x=x, y=self.results_df["Q(True)"], name="Q(True)", mode='lines',
                                     line=dict(width=1, dash='dash')),
                          secondary_y=True)
        if actual_count:
            mse_fig.add_vline(x=actual_count, line_width=1, line_dash="dash", line_color="black",
                              name="Actual Factor Count")
        mse_fig.add_vline(x=self.estimated_factor, line_width=1, line_dash="dash", line_color="red",
                          name="Estimated Factor Count")
        mse_fig.update_layout(width=800, height=800, title_text="Factor Estimation", hovermode='x unified')
        mse_fig.update_yaxes(title_text="Mean Squared Error", secondary_y=False)
        mse_fig.update_yaxes(title_text="Q(True)", secondary_y=True)
        mse_fig.update_xaxes(title_text="Number of Factors")
        mse_fig.show()
