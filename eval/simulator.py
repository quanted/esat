import logging
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from esat.model.sa import SA
from esat.metrics import q_loss, qr_loss
from esat.model.batch_sa import BatchSA
from esat_eval.factor_comparison import FactorCompare

logger = logging.getLogger(__name__)


class Simulator:
    """
    The ESAT Simulator provides methods for generating customized synthetic source profiles and datasets. These
    synthetic datasets can then be passed to SA or BatchSA instances. The results of those model runs can be evaluated
    against the known synthetic profiles using the Simulator compare function. A visualization of the comparison is
    available with the plot_comparison function.

    The synthetic profile matrix (H) is generated from a uniform distribution [0.0, 1.0).
    The synthetic contribution matrix (W) is generated from a uniform distribution [0.0, 1.0) * contribution_max.
    The synthetic dataset is the matrix multiplication product of WH + noise + outliers.
    Noise is added to the dataset from a normal distribution, scaled by the dataset.
    Outliers are added to the dataset at random, for the decimal percentage outlier_p parameters, multiplying the
    dataset value by outlier_mag.
    Uncertainty is generated from a normal distribution, scaled by the dataset.

    #TODO: Simulator save and load functions
    #TODO: Looper, batch simulator mode
    #TODO: Sum of profile residuals (synthetic - modeled) metrics

    Parameters
    ----------
    seed : int
        The seed for the random number generator.
    factors_n : int
        The number of synthetic factors to generate.
    features_n : int
        The number of synthetic features in the dataset.
    samples_n : int
        The number of synthetic samples in the dataset.
    outliers : bool
        Include outliers in the synthetic dataset.
    outlier_p : float
        The decimal percentage of outliers in the dataset.
    outlier_mag : float
        The magnitude of the outliers on the dataset elements.
    contribution_max : int
        The maximum value in the synthetic contribution matrix (W).
    noise_mean : float
        The mean decimal percentage of the synthetic dataset for noise.
    noise_scale : float
        The scale of the normal distribution for the noise.
    uncertainty_mean : float
        The mean decimal percentage of the uncertainty of the synthetic dataset.
    uncertainty_scale : float
        The scale of the normal distribution for the uncertainty.
    """
    def __init__(self,
                 seed: int,
                 factors_n: int,
                 features_n: int,
                 samples_n: int,
                 outliers: bool = True,
                 outlier_p: float = 0.10,
                 outlier_mag: float = 2.0,
                 contribution_max: int = 10,
                 noise_mean: float = 0.1,
                 noise_scale: float = 0.02,
                 uncertainty_mean: float = 0.05,
                 uncertainty_scale: float = 0.01
                 ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.factors_n = factors_n
        self.features_n = features_n
        self.samples_n = samples_n
        self.syn_columns = [f"Feature {i}" for i in range(1, self.features_n+1)]
        self.syn_factor_columns = [f"Factor {i}" for i in range(1, self.factors_n+1)]

        self.outliers = outliers
        self.outlier_p = outlier_p
        self.outlier_mag = outlier_mag
        self.noise_mean = noise_mean
        self.noise_scale = noise_scale
        self.uncertainty_mean = uncertainty_mean
        self.uncertainty_scale = uncertainty_scale

        self.syn_profiles = None
        self.syn_contributions = self.rng.random(size=(self.samples_n, self.factors_n)) * contribution_max

        self.syn_data = None
        self.syn_uncertainty = None

        self.syn_data_df = None
        self.syn_uncertainty_df = None
        self.syn_profiles_df = None
        self.syn_contributions_df = None
        self.syn_sa = None

        self.batch_sa = None
        self.factor_compare = None

        self.generate_profiles()

    def generate_profiles(self, profiles: np.ndarray = None):
        """
        Generate the synthetic profiles. Run on Simulator initialization, but customized profiles can be used inplace
        of the randomly generated synthetic profile by passing in a profile matrix

        Parameters
        ----------
        profiles : np.ndarray
            A custom profile matrix to be used in place of the random synthetic profile. Matrix must have shape
            (factors_n, features_n)

        """
        if profiles is None:
            self.syn_profiles = np.zeros(shape=(self.factors_n, self.features_n))
            factor_list = list(range(self.factors_n))
            for i in range(self.features_n):
                # the number of factors which contribute to current feature
                factor_features_n = self.rng.integers(1, self.factors_n, 1)
                # the specific factors which contribute to current feature
                factor_feature_selected = self.rng.choice(factor_list, size=factor_features_n, replace=False)
                for j in factor_feature_selected:
                    ji_value = self.rng.random(size=1)
                    self.syn_profiles[j][i] = ji_value
            self.syn_profiles[self.syn_profiles <= 0.0] = 1e-12
        else:
            self.syn_profiles = profiles
        logger.info("Synthetic profiles generated")
        self._generate_data()
        self._generate_dfs()
        self._create_sa()

    def _generate_data(self):
        """
        Generate the synthetic dataset from the previously created profile and contribution matrices.
        """
        syn_data = np.matmul(self.syn_contributions, self.syn_profiles)
        # Add noise
        noise = syn_data * self.rng.normal(loc=self.noise_mean, scale=self.noise_scale, size=syn_data.shape)
        syn_data = np.add(syn_data, noise)
        # Add outliers
        if self.outliers:
            outlier_mask = self.rng.uniform(size=syn_data.shape)
            syn_data[outlier_mask <= self.outlier_p] = syn_data[outlier_mask <= self.outlier_p] * self.outlier_mag
        # Make sure no negative or zero values are in the dataset
        syn_data[syn_data <= 0.0] = 1e-12
        self.syn_data = syn_data
        logger.info("Synthetic data generated")

        syn_unc_p = self.rng.normal(loc=self.uncertainty_mean, scale=self.uncertainty_scale, size=syn_data.shape)
        syn_uncertainty = syn_data * syn_unc_p
        syn_uncertainty[syn_uncertainty <= 0.0] = 1e-12
        self.syn_uncertainty = syn_uncertainty
        logger.info("Synthetic uncertainty data generated")

    def _generate_dfs(self):
        """
        Create data, uncertainty, profile and contribution dataframes with column labels and index.
        """
        time_steps = pd.date_range(datetime.now().strftime("%Y-%m-%d"), periods=self.samples_n, freq='d')

        self.syn_data_df = pd.DataFrame(self.syn_data, columns=self.syn_columns)
        self.syn_data_df['Date'] = time_steps
        self.syn_data_df.set_index('Date', inplace=True)
        self.syn_uncertainty_df = pd.DataFrame(self.syn_uncertainty, columns=self.syn_columns)
        self.syn_uncertainty_df['Date'] = time_steps
        self.syn_uncertainty_df.set_index('Date', inplace=True)
        self.syn_profiles_df = pd.DataFrame(self.syn_profiles.T, columns=self.syn_factor_columns)
        self.syn_contributions_df = pd.DataFrame(self.syn_contributions, columns=self.syn_factor_columns)
        logger.info("Synthetic dataframes completed")

    def _create_sa(self):
        """
        Create a SA instance using the synthetic data, uncertainty, profile and contributions. The synthetic SA instance
        can then be used for all existing analysis and plotting functions in ModelAnalysis.
        """
        self.syn_sa = SA(V=self.syn_data, U=self.syn_uncertainty, factors=self.factors_n, seed=self.seed)
        self.syn_sa.H = self.syn_profiles
        self.syn_sa.W = self.syn_contributions
        self.syn_sa.WH = np.matmul(self.syn_contributions, self.syn_profiles)
        self.syn_sa.Qrobust = qr_loss(V=self.syn_sa.V, U=self.syn_sa.U, W=self.syn_sa.W, H=self.syn_sa.H)
        self.syn_sa.Qtrue = q_loss(V=self.syn_sa.V, U=self.syn_sa.U, W=self.syn_sa.W, H=self.syn_sa.H)
        logger.info("Synthetic source apportionment instance created.")

    def get_data(self):
        """
        Get the synthetic data and uncertainty dataframes to use with the DataHandler.

        Returns
        -------
            pd.DataFrame, pd.DataFrame
        """
        return self.syn_data_df, self.syn_uncertainty_df

    def compare(self, batch_sa: BatchSA, selected_model: int = None):
        """
        Run the profile comparison, evaluating the results of each of the models in the BatchSA instance. All models are
        evaluated with the results for each model available in simulator.factor_compare.model_results

        The model with the highest average R squared value for the factor mapping is defined as the best_model, which
        can be different from the most optimal model, model with the lowest loss value. If they are different the best
        mapping for the most optimal model is also provided.

        A mapping details for a specific model can also be found by specifying the selected_model parameter, model by
        index. Requires that compare has already been completed on the instance.

        Parameters
        ----------
        batch_sa : BatchSA
            Completed instance of BatchSA to compare the output models to the known synthetic profiles.
        selected_model : int
            If specified, displays the best mapping for the specified model.
        """
        self.batch_sa = batch_sa
        if selected_model is None:
            logger.info("Searching all models in the batch to find which has the highest average correlation mapping.")
            self.factor_compare = FactorCompare(input_df=self.syn_data_df,
                                                uncertainty_df=self.syn_uncertainty_df,
                                                base_profile_df=self.syn_profiles_df,
                                                base_contribution_df=self.syn_contributions_df,
                                                factors_columns=self.syn_factor_columns,
                                                features=self.syn_columns,
                                                batch_sa=self.batch_sa
                                                )
            self.factor_compare.compare()
            logger.info("\n")
            if self.factor_compare.best_model == self.batch_sa.best_model:
                logger.info(f"The most optimal model (loss) is also the highest average correlation mapping.")
                return
            else:
                logger.info("Mappings for the most optimal model (loss) in the batch.")
                self.factor_compare.print_results(model=self.batch_sa.best_model)
        else:
            if self.factor_compare is None:
                logger.error("Factor compare must be completed prior to selecting a model.")
            logger.info(f"Mappings for the most selected model in the batch. Model: {selected_model}")
            self.factor_compare.print_results(model=selected_model)

    def plot_comparison(self, grouped=False, model_i: int = None):
        """
        Plot the results of the output comparison for the model with the highest correlated mapping, if model_i is not
        specified. Otherwise, plots the output comparison of model_i to the synthetic profiles.

        Parameters
        ----------
        grouped : bool
            Generate a single plot for all the profiles, otherwise all profiles are graphed individually.
        model_i : int
            The model index for the comparison, when not specified will default to the model with the highest
            correlation mapping.
        """
        if self.batch_sa is None:
            logger.error("A batch source apportionment must be completed and compared before plotting.")
            return
        if not model_i:
            model_i = self.factor_compare.best_model

        factor_compare = self.factor_compare.model_results[model_i]

        color_map = px.colors.sample_colorscale("plasma", [n / (self.factors_n - 1) for n in range(self.factors_n)])
        r_color_map = px.colors.sample_colorscale("jet", [n / (100 - 1) for n in range(100)])

        syn_H = self.syn_profiles
        norm_syn_H = 100 * (syn_H / syn_H.sum(axis=0))

        _H = self.batch_sa.results[model_i].H
        norm_H = 100 * (_H / _H.sum(axis=0))

        if grouped:
            subplot_titles = [f"Factor {i}" for i in range(1, self.factors_n + 1)] + factor_compare['factor_map']

            h_fig = make_subplots(rows=4, cols=self.factors_n, vertical_spacing=0.03, subplot_titles=subplot_titles,
                                  row_heights=[0.35, 0.35, 0.04, 0.3])
            for i in range(1, self.factors_n + 1):
                h_fig.add_trace(
                    go.Bar(name=f"Factor {i}", x=self.syn_columns, y=norm_syn_H[i - 1], marker_color=color_map[i - 1]),
                    row=1, col=i)
                map_i = int(factor_compare['factor_map'][i - 1].split(" ")[1]) - 1
                h_fig.add_trace(
                    go.Bar(name=f"Factor {factor_compare['factor_map'][i - 1]}", x=self.syn_columns, y=norm_H[map_i],
                           marker_color=color_map[i - 1]), row=2, col=i)
                i_r2 = factor_compare['best_factor_r'][i - 1]
                h_fig.add_trace(go.Bar(name="R2", x=(i_r2,), orientation='h', marker_color=r_color_map[int(100 * i_r2)],
                                       text=np.round(i_r2, 4), textposition="inside", hoverinfo='text',
                                       hovertemplate="R2: %{x:4f}<extra></extra>"), row=3, col=i)
                h_fig.add_trace(go.Bar(name="", x=self.syn_columns, y=norm_syn_H[i - 1] - norm_H[map_i],
                                       marker_color=color_map[i - 1]), row=4, col=i)
            h_fig.update_yaxes(title_text="Synthetic Profile", row=1, col=1, title_standoff=3)
            h_fig.update_yaxes(title_text="Model Profile", row=2, col=1, title_standoff=3)
            h_fig.update_yaxes(title_text="R2", row=3, col=1, title_standoff=25)
            h_fig.update_yaxes(title_text="Difference", row=4, col=1, title_standoff=3)
            h_fig.update_xaxes(row=1, showticklabels=False)
            h_fig.update_xaxes(row=2, showticklabels=False)
            h_fig.update_xaxes(row=3, range=[0, 1.0])
            h_fig.update_yaxes(row=3, showticklabels=False)
            h_fig.update_yaxes(row=4, range=[-50, 50])
            h_fig.update_layout(title_text=f"Factor Profile Comparison - Model: {model_i + 1}", width=1600,
                                height=1000, hovermode='x', showlegend=False)
            h_fig.show()
        else:
            subplot_titles = [f"Factor {i}: {factor_compare['factor_map'][i - 1]}" for i in range(1, self.factors_n + 1)]
            for i in range(1, self.factors_n + 1):
                map_i = int(factor_compare['factor_map'][i - 1].split(" ")[1]) - 1
                abs_res = np.round(np.abs(norm_syn_H[i - 1] - norm_H[map_i]).sum(), 2)
                label = (f"{subplot_titles[i - 1]} - Residual: {abs_res}", "", "")
                h_fig = make_subplots(rows=3, cols=1, vertical_spacing=0.05, subplot_titles=label,
                                      row_heights=[0.5, 0.08, 0.3])
                h_fig.add_trace(
                    go.Bar(name=f"Syn Factor {i}", x=self.syn_columns, y=norm_syn_H[i - 1], marker_color="black"),
                    row=1, col=1)
                h_fig.add_trace(
                    go.Bar(name=f"Modelled {factor_compare['factor_map'][i - 1]}", x=self.syn_columns, y=norm_H[map_i],
                           marker_color="green"), row=1, col=1)
                i_r2 = factor_compare['best_factor_r'][i - 1]
                h_fig.add_trace(go.Bar(name="R2", x=(i_r2,), orientation='h', marker_color=r_color_map[int(100 * i_r2)],
                                       text=np.round(i_r2, 4), textposition="inside", hoverinfo='text',
                                       hovertemplate="R2: %{x:4f}<extra></extra>", showlegend=False), row=2, col=1)
                h_fig.add_trace(
                    go.Bar(name="", x=self.syn_columns, y=norm_syn_H[i - 1] - norm_H[map_i], marker_color="blue",
                           showlegend=False), row=3, col=1)
                h_fig.update_yaxes(title_text="Synthetic Profile", row=1, col=1, title_standoff=3)
                h_fig.update_yaxes(title_text="R2", row=2, col=1, title_standoff=25)
                h_fig.update_yaxes(title_text="Difference", row=3, col=1, title_standoff=3)
                h_fig.update_xaxes(row=1, showticklabels=False)
                h_fig.update_xaxes(row=2, range=[0, 1.0])
                h_fig.update_yaxes(row=2, showticklabels=False)
                h_fig.update_yaxes(row=3, range=[-50, 50])
                h_fig.update_layout(title_text=f"Factor Profile Comparison - Model: {model_i + 1}",
                                    width=1000, height=600, hovermode='x', showlegend=True)
                h_fig.show()
