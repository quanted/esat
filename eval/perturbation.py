import copy
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import multiprocessing as mp

from tqdm.notebook import tqdm
from plotly.subplots import make_subplots
from esat.model.sa import SA
from esat_eval.factor_comparison import FactorCompareV2


logger = logging.getLogger(__name__)


class Perturbation:
    """
    Class for running uncertainty perturbation analysis on a model.

    Parameters
    ----------
    V : np.ndarray
        The input data matrix.
    U : np.ndarray
        The output data matrix.
    factors : int
        The number of factors to consider.
    base_model : SA
        The base model to perturb.
    method : str
        The NMF algorithm to use for the base and perturbed models. If base_model is provided this is ignored.
    random_seed : int
        The random seed for reproducibility.
    perturb_percent : float
        The percentage of cell values to perturb.
    sigma : float
        The standard deviation of the perturbation lognormal distribution.
    models : int
        The number of perturbed models to generate.
    max_iterations : int
        The maximum number of iterations for the perturbation model training.
    converge_n : int
        The number of iterations to consider for convergence.
    converge_delta : float
        The convergence delta.
    threshold : float
        The threshold for the perturbation analysis.
    compare_method : str
        The method to use for comparing the perturbed models.
    verbose : bool
        Whether to print verbose output.
    """

    comparison = None
    perturb_mapping = None
    perturb_correlations = None
    multiplier_stats = None
    perturb_W = None
    perturb_H = None
    perturbed_H_df = None

    def __init__(self,
                 V,
                 U,
                 factors: int,
                 base_model: SA = None,
                 method: str = "ls-nmf",
                 random_seed: int = 42,
                 perturb_percent: float = 1.0,
                 sigma: float = 0.33,
                 models: int = 50,
                 max_iterations: int = 20000,
                 converge_n: int = 20,
                 converge_delta: float = 0.001,
                 threshold: float = 0.1,
                 compare_method: str = "raae",
                 verbose: bool = False
                 ):
        self.V = V
        self.U = U

        self.factors = factors
        self.base_model = base_model
        self.method = method if base_model is None else base_model.method
        self.random_seed = random_seed

        self.perturb_percent = perturb_percent
        self.sigma = sigma

        self.models = models
        self.max_iterations = max_iterations

        self.converge_n = converge_n
        self.converge_delta = converge_delta

        self.threshold = threshold
        self.compare_method = compare_method
        self.verbose = verbose

        self.rng = np.random.default_rng(seed=random_seed)
        self.perturbed_models = None
        self.perturbed_multipliers = None

    def run(self):
        """
        Run the uncertainty perturbation analysis
        """
        if self.base_model is None:
            sa_model = SA(V=self.V, U=self.U, factors=self.factors, seed=self.random_seed, verbose=self.verbose, method=self.method)
            sa_model.initialize()
            sa_model.train(max_iter=self.max_iterations, converge_delta=self.converge_delta, converge_n=self.converge_n)
            self.base_model = sa_model

        pool = mp.Pool(processes=int(mp.cpu_count() * 0.75))
        parallel_args = [(i, self.rng.integers(0, 1e5)) for i in range(self.models)]

        perturbed_models = []
        perturbed_multipliers = []
        pbar = tqdm(total=self.models, desc="Running Permutations on base model")
        for pool_results in pool.starmap(self._parallel_perturb, parallel_args):
            idx, random_seed, i_sa_model, i_m = pool_results
            perturbed_models.append(i_sa_model)
            perturbed_multipliers.append(i_m)
            pbar.update()
        pbar.close()
        pool.close()
        self.perturbed_models = perturbed_models
        self.perturbed_multipliers = perturbed_multipliers

    def _parallel_perturb(self, idx, random_seed):
        i_u, i_m = self.perturb(random_seed)
        i_sa_model = SA(self.V, U=i_u, factors=self.factors, seed=random_seed, verbose=self.verbose, method=self.method)
        i_sa_model.initialize(H=self.base_model.H, W=self.base_model.W)
        i_sa_model.train(max_iter=self.max_iterations, converge_delta=self.converge_delta, converge_n=self.converge_n)
        return idx, random_seed, i_sa_model, i_m

    def perturb(self, random_seed):
        """
        Create a perturbed instances of the uncertainty dataset using the provided random seed and perturb parameters.

        Parameters
        ----------
        random_seed : int
            The random seed for reproducibility.

        Returns
        -------
        np.ndarray
            The perturbed U matrix.
        np.ndarray
            The perturbed multiplier matrix.

        """
        rng = np.random.default_rng(seed=random_seed)
        i_u = copy.copy(self.U)
        sigma = self.sigma
        perturb_p = self.perturb_percent
        if isinstance(self.perturb_percent, float):
            perturb_p = [self.perturb_percent for i in range(self.U.shape[1])]
        elif isinstance(self.perturb_percent, list) and len(self.perturb_percent) != self.U.shape[1]:
            perturb_p = [self.perturb_percent[0] for i in range(self.U.shape[1])]
        if isinstance(self.sigma, float):
            sigma = [self.sigma for i in range(self.U.shape[1])]
        elif isinstance(self.sigma, list) and len(self.sigma) != self.U.shape[1]:
            sigma = [self.sigma[0] for i in range(self.U.shape[1])]
        i_m = np.zeros(shape=i_u.shape)
        for i, _p in enumerate(perturb_p):
            i_mask = rng.random(size=self.U[:, i].shape) > _p
            i_mean = 0.0
            i_logn = rng.lognormal(i_mean, sigma[i], size=self.U[:, i].shape)
            i_m[:, i] = i_logn
            i_m[:, i][i_mask] = i_m[:, i][i_mask]
            ij_u = i_u[:, i]
            ij_u = ij_u * i_logn
            ij_u[i_mask] = ij_u[i_mask]
            ij_u[ij_u <= 0.0] = 1e-12
            i_u[:, i] = ij_u
        return i_u, i_m

    def compare(self, compare_method: str = None, in_notebook: bool = False):
        """
        Compare the perturbed models to the base model using the provided comparison method.

        Parameters
        ----------
        compare_method : str
            The comparison method to use, options include: "raae", "emc", "corr".
        in_notebook : bool
            Whether to display the comparison results in a Jupyter notebook.
        """
        if compare_method is not None:
            if compare_method.lower() in ("raae", "emc", "corr"):
                self.compare_method = compare_method.lower()
        logger.info(f"Comparing {len(self.perturbed_models)} perturbed models using {self.compare_method}")

        self.comparison = FactorCompareV2(self.base_model, self.perturbed_models, in_notebook=in_notebook)
        self.perturb_mapping, self.perturb_correlations = self.comparison.determine_map(self.compare_method)
        logger.info(f"Perturbed factor mean values")
        logger.info(pd.DataFrame([np.mean(np.array(list(self.perturb_correlations.values())), axis=0)],
                                 index=[self.compare_method.upper()],
                                 columns=[f"Factor {i}" for i in range(1, self.factors + 1)]))

        # multiplier statistics
        m_stats = []
        for pm in self.perturbed_multipliers:
            pm_i = np.percentile(pm, [1, 25, 50, 75, 99])
            m_stats.append(pm_i)
        u_m_stats = np.mean(np.array(m_stats), axis=0)
        m_stats_dict = dict(zip(["first", "25th", "50th", "75th", "99th"], u_m_stats))
        m_stats_dict["sigma"] = self.sigma
        self.multiplier_stats = pd.Series(data=m_stats_dict)
        logger.info("\nPerturbation Multiplier Details")
        logger.info(self.multiplier_stats)

        # perturbed W and H
        base_H = self.base_model.H
        base_H = (base_H / np.sum(base_H, axis=0))

        base_W = self.base_model.W
        base_W = (base_W.T / np.sum(base_W.T, axis=0)).T

        threshold = 0.95

        p_Hs = []
        p_Ws = []
        p_Qs = []
        for p, p_model in enumerate(self.perturbed_models):
            p_H = (p_model.H / np.sum(p_model.H, axis=0))
            p_W = (p_model.W.T / np.sum(p_model.W.T, axis=0)).T
            _p_H = []
            _p_W = []
            p_mapping = self.perturb_mapping[p]
            p_map_metric = np.array(self.perturb_correlations[p]) > threshold
            for i in range(p_model.factors):
                _p_H.append(p_H[p_mapping[i]])
                _p_W.append(p_W[:, p_mapping[i]])
            _p_H = np.array(_p_H)
            _p_W = np.array(_p_W)
            p_Hs.append(_p_H)
            p_Ws.append(_p_W)
            p_Qs.append(p_model.Qtrue)
        self.perturb_H = np.dstack(p_Hs)
        self.perturb_W = np.dstack(p_Ws)

        mean_perturb_H = np.mean(self.perturb_H, axis=2)
        std_perturb_H = np.std(self.perturb_H, axis=2)
        min_perturb_H = np.min(self.perturb_H, axis=2)
        max_perturb_H = np.max(self.perturb_H, axis=2)

        factor_i = 1
        f1_dict = {
            "Base": base_H[factor_i],
            "Mean Perturb": mean_perturb_H[factor_i],
            "% diff": np.round(100 * (mean_perturb_H[factor_i] - base_H[factor_i]) / (
                        (mean_perturb_H[factor_i] + base_H[factor_i]) / 2), 4),
            "STD Perturb": std_perturb_H[factor_i],
            "Min Perturb": min_perturb_H[factor_i],
            "Max Perturb": max_perturb_H[factor_i],
        }
        f1_dict["% diff"][f1_dict["Mean Perturb"] < 1e-6] = 0.0
        feature_labels = [f"Feature {i + 1}" for i in range(base_H.shape[1])]
        self.perturbed_H_df = pd.DataFrame(f1_dict, index=feature_labels).round(10)
        logger.info("\nPerturbed H Results")
        logger.info(self.perturbed_H_df)
        logger.info(
            f"\nQTrue - Base: {np.round(self.base_model.Qtrue, 4)},"
            f" Perturb Mean: {np.round(np.mean(p_Qs), 4)}, "
            f"Perturb STD: {np.round(np.std(p_Qs), 4)}, "
            f"Perturb Min: {np.round(np.min(p_Qs), 4)}, "
            f"Perturb Max: {np.round(np.max(p_Qs), 4)}"
        )

    def plot_norm_perturb_contributions(self, return_figure: bool = False):
        """
        Plot the perturbed mean normalized contributions for each factor.

        Parameters
        ----------
        return_figure : bool
            Whether to return the plotly figure or display it.

        Returns
        -------
        go.Figure
            The plotly figure.

        """
        if self.perturb_W is None:
            logger.error("Comparison must be run before plotting perturbed results")
            return None
        base_W = self.base_model.W
        base_W = (base_W.T / np.sum(base_W.T, axis=0)).T
        factor_labels = [f"Factor {i + 1}" for i in range(base_W.shape[1])]

        p_W_mean = np.mean(self.perturb_W, axis=(1, 2))
        base_W_mean = np.mean(base_W, axis=0)
        factor_w0_fig = make_subplots()
        factor_w0_fig.add_trace(go.Bar(x=factor_labels, y=100 * p_W_mean, name="Perturbed W"))
        factor_w0_fig.add_trace(go.Bar(x=factor_labels, y=100 * base_W_mean, name="Base W"))
        factor_w0_fig.update_layout(title=f"Perturbed Mean Factor Contribution Results", width=1200, height=800,
                                    hovermode='x unified', barmode='group')
        factor_w0_fig.update_yaxes(title_text="Mean Normalized Contributions (%)")
        if return_figure:
            return factor_w0_fig
        factor_w0_fig.show()

    def plot_average_source_contributions(self, return_figure: bool = False):
        """
        Plot the perturbed average source contributions for each factor.

        Parameters
        ----------
        return_figure : bool
            Whether to return the plotly figure or display it.

        Returns
        -------
        go.Figure
            The plotly figure.

        """
        if self.perturb_W is None:
            logger.error("Comparison must be run before plotting perturbed results")
            return None
        base_W = self.base_model.W
        base_W = (base_W.T / np.sum(base_W.T, axis=0)).T
        factor_labels = [f"Factor {i + 1}" for i in range(base_W.shape[1])]

        factor_w1_fig = make_subplots()
        p_W_means = np.mean(self.perturb_W, axis=1)
        base_W_means = np.mean(base_W, axis=0)
        for i in range(len(factor_labels)):
            i_W_means = p_W_means[i]
            factor_w1_fig.add_trace(go.Box(y=i_W_means, name=factor_labels[i]))
        factor_w1_fig.add_trace(
            go.Scatter(x=factor_labels, y=base_W_means, name="Base", mode="markers", marker_color="black"))
        factor_w1_fig.update_layout(title=f"Perturbed Average Source Contributions - Sigma: {self.sigma}",
                                    width=1200, height=800, hovermode='x unified')
        factor_w1_fig.update_yaxes(title_text="Source Contributions")
        if return_figure:
            return factor_w1_fig
        factor_w1_fig.show()

    def plot_correlation_metrics(self, return_figure: bool = False):
        """
        Plot the perturbed correlation metrics for each factor.

        Parameters
        ----------
        return_figure : bool
            Whether to return the plotly figure or display it.

        Returns
        -------
        go.Figure
            The plotly figure.

        """
        if self.perturb_W is None:
            logger.error("Comparison must be run before plotting perturbed results")
            return None
        base_W = self.base_model.W
        base_W = (base_W.T / np.sum(base_W.T, axis=0)).T
        factor_labels = [f"Factor {i + 1}" for i in range(base_W.shape[1])]

        factor_corr_fig = make_subplots()
        factor_corr = np.zeros(shape=(self.models, len(factor_labels)))
        for i, v in self.perturb_mapping.items():
            factor_corr[i] = np.array(self.perturb_correlations[i])[v]
        for i in range(len(factor_labels)):
            i_corr_means = factor_corr[:, i]
            factor_corr_fig.add_trace(go.Box(y=i_corr_means, name=factor_labels[i]))
        factor_corr_fig.add_hline(y=np.mean(factor_corr))
        factor_corr_fig.update_layout(title=f"Perturbed Source Correlation Metric - Sigma: {self.sigma}", width=1200,
                                      height=800, hovermode='x unified')
        factor_corr_fig.update_yaxes(title_text=self.compare_method.upper())
        if return_figure:
            return factor_corr_fig
        factor_corr_fig.show()

    def plot_perturbed_factor(self, factor_idx, return_figure: bool = False):
        """
        Plot the perturbed factor profile results.

        Parameters
        ----------
        factor_idx : int
            The factor index to plot.
        return_figure : bool
            Whether to return the plotly figure or display it.

        Returns
        -------
        go.Figure
            The plotly figure.

        """
        if self.perturb_W is None:
            logger.error("Comparison must be run before plotting perturbed results")
            return None
        if factor_idx >= self.factors:
            logger.error(f"Factor index {factor_idx} is out of range")
            return None

        base_H = self.base_model.H
        base_H = (base_H / np.sum(base_H, axis=0))

        feature_labels = [f"Feature {i + 1}" for i in range(base_H.shape[1])]

        factor_p0_fig = make_subplots(specs=[[{"secondary_y": True}]])
        for i in range(len(feature_labels)):
            f_feature_i = self.perturb_H[factor_idx, i]
            factor_p0_fig.add_trace(go.Box(y=f_feature_i, name=feature_labels[i]), secondary_y=False)
        factor_p0_fig.add_trace(
            go.Scatter(x=feature_labels, y=base_H[factor_idx], name="Base", mode="markers", marker_color="black"),
            secondary_y=False)
        factor_p0_fig.update_layout(title=f"Perturbed Factor {factor_idx} Profile Results", width=1200, height=800,
                                    hovermode='x unified')
        factor_p0_fig.update_yaxes(title_text="Normalized Profile", secondary_y=False)
        if return_figure:
            return factor_p0_fig
        factor_p0_fig.show()



