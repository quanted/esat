import logging
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from esat.model.sa import SA
from esat.metrics import q_loss, qr_loss
from esat.model.batch_sa import BatchSA
try:
    from esat_eval.factor_comparison import FactorCompare
except ModuleNotFoundError as e:
    from eval.factor_comparison import FactorCompare

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

    #TODO: Looper, batch simulator mode

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
    noise_mean_min : float
        The minimum value for the randomly selected mean decimal percentage of the synthetic dataset for noise, by feature.
    noise_mean_max : float
        The maximum value for the randomly selected mean decimal percentage of the synthetic dataset for noise, by feature.
    noise_scale : float
        The scale of the normal distribution for the noise, standard deviation of the distribution.
    uncertainty_mean_min : float
        The minimum value for the randomly selected mean decimal percentage of the uncertainty of the synthetic dataset, by feature.
    uncertainty_mean_max : float
        The maximum value for the randomly selected mean decimal percentage of the uncertainty of the synthetic dataset, by feature.
    uncertainty_scale : float
        The scale of the normal distribution for the uncertainty, standard deviation of the distribution.
    verbose: bool
        Turn on verbosity for added logging.
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
                 noise_mean_min: float = 0.1,
                 noise_mean_max: float = 0.12,
                 noise_scale: float = 0.02,
                 uncertainty_mean_min: float = 0.05,
                 uncertainty_mean_max: float = 0.05,
                 uncertainty_scale: float = 0.01,
                 verbose: bool = True,
                 ):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.verbose = verbose
        self.factors_n = int(factors_n)
        self.features_n = int(features_n)
        self.samples_n = int(samples_n)
        self.syn_columns = [f"Feature {i}" for i in range(1, self.features_n+1)]
        self.syn_factor_columns = [f"Factor {i}" for i in range(1, self.factors_n+1)]

        self.outliers = outliers
        self.outlier_p = float(outlier_p)
        self.outlier_mag = float(outlier_mag)
        self.noise_mean_min = float(noise_mean_min)
        self.noise_mean_max = float(noise_mean_max)
        self.noise_scale = float(noise_scale)
        self.uncertainty_mean_min = float(uncertainty_mean_min)
        self.uncertainty_mean_max = float(uncertainty_mean_max)
        self.uncertainty_scale = float(uncertainty_scale)

        self.syn_profiles = None
        self.syn_contributions = self.rng.random(size=(self.samples_n, self.factors_n)) * float(contribution_max)
        self.time_steps = pd.date_range(datetime.now().strftime("%Y-%m-%d"), periods=self.samples_n, freq='d')

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
        _profiles = None
        if profiles is not None:
            if profiles.shape != (self.factors_n, self.features_n):
                _profiles = profiles
                profiles = None
        if profiles is None:
            self.syn_profiles = np.zeros(shape=(self.factors_n, self.features_n))
            factor_list = list(range(self.factors_n))
            for i in range(self.features_n):
                # the number of factors which contribute to current feature
                factor_features_n = self.rng.integers(1, self.factors_n, 1)
                # the specific factors which contribute to current feature
                factor_feature_selected = self.rng.choice(factor_list, size=factor_features_n, replace=False)
                for j in factor_feature_selected:
                    ji_value = self.rng.random(size=None)
                    self.syn_profiles[j][i] = float(ji_value)
        else:
            self.syn_profiles = profiles
        if _profiles is not None:
            for i in range(_profiles.shape[0]):
                self.syn_profiles[i] = _profiles[i]
        self.syn_profiles[self.syn_profiles <= 0.0] = 1e-12
        if self.verbose:
            logger.info("Synthetic profiles generated")

    def update_contribution(self,
                            factor_i: int,
                            curve_type: str,
                            scale: float = 0.1,
                            frequency: float = 0.5,
                            maximum: float = 1.0,
                            minimum: float = 0.1):
        """
        Update the contributions for a specific factor to follow the curve type. The values are randomly sampled from
        a normal distribution around the curve type as defined by the magnitude, frequency and/or slope values. The
        input and uncertainty data are recalculated with each update.

        Parameters
        ----------
        factor_i : int
            The factor contribution to update, by index.
        curve_type : str
            The type of curve that describes the factor contribution. Options include: 'uniform', 'increasing',
            'decreasing', 'logistic', 'periodic'. Default: uniform.
        scale : float
            The scale of the normal distribution of the curve value to be resampled.
        frequency : float
            The frequency of slope change in periodic and logistic curves.
        maximum : float
            The maximum value for all curves.
        minimum : float
            The minimum value for all curves.
        """
        if minimum > maximum:
            maximum, minimum = minimum, maximum
        elif minimum == maximum:
            maximum *= 2

        if curve_type == "increasing":
            curve_y = np.linspace(minimum, maximum, num=self.samples_n)
        elif curve_type == "decreasing":
            curve_y = np.linspace(maximum, minimum, num=self.samples_n)
        elif curve_type == "logistic":
            x_periods = []
            n_periods = int(1.0/frequency) + 1
            for i in range(0, n_periods+1):
                if i == n_periods:
                    steps = self.samples_n - (2 * int(self.samples_n/n_periods))
                else:
                    steps = int(self.samples_n/n_periods)
                if i % 2 == 0:
                    x_i = np.linspace(-100, 100, steps)
                else:
                    x_i = np.linspace(100, -100, steps)
                x_periods.append(x_i)
            curve_x = np.concatenate(x_periods, axis=None)
            curve_y = minimum + (maximum - minimum)/(1.0 + np.exp(-curve_x))
        elif curve_type == "periodic":
            n_periods = int(1.0/frequency) + 1
            curve_a = np.linspace(-np.pi, np.pi, int(self.samples_n / n_periods))
            curve_x = np.tile(curve_a[0:len(curve_a) - 1], n_periods + 1)
            curve_x = curve_x[0:self.samples_n]
            curve_y = np.sin(curve_x) * ((maximum - minimum) / 2)
            curve_y = (curve_y + (np.abs(np.min(curve_y))) + minimum)
        else:
            curve_y = self.rng.random(size=(self.samples_n, 1)) * float(maximum)
        curve_y = curve_y[:self.samples_n]
        y_data = self.rng.normal(loc=curve_y, scale=scale, size=curve_y.shape).flatten()
        y_data[y_data <= 0.0] = 1e-12
        self.syn_contributions[:, factor_i] = y_data
        if self.verbose:
            logger.info(f"Synthetic factor {factor_i + 1} contribution updated as a random sampling from a normal "
                        f"distribution along a {curve_type} curve.")
            logger.info(f"Frequency: {frequency}, Shape: {curve_y.shape}")

    def _generate_data(self):
        """
        Generate the synthetic dataset from the previously created profile and contribution matrices.
        """
        syn_data = np.matmul(self.syn_contributions, self.syn_profiles)
        # Add noise
        noise_mean = self.rng.uniform(low=self.noise_mean_min, high=self.noise_mean_max, size=self.features_n)
        noise = syn_data * self.rng.normal(loc=noise_mean, scale=self.noise_scale, size=syn_data.shape)
        negative_noise_mask = self.rng.uniform(size=syn_data.shape)
        noise[negative_noise_mask < 0.5] = noise[negative_noise_mask < 0.5] * -1.0
        syn_data = np.add(syn_data, noise)
        # Add outliers
        if self.outliers:
            outlier_mask = self.rng.uniform(size=syn_data.shape)
            outlier_p = self.outlier_p * 0.5
            outlier_n = 1.0 - (self.outlier_p * 0.5)
            syn_data[outlier_mask <= outlier_n] = syn_data[outlier_mask <= outlier_n] * self.outlier_mag
            syn_data[outlier_mask >= outlier_p] = syn_data[outlier_mask >= outlier_p] / self.outlier_mag
        # Make sure no negative or zero values are in the dataset
        syn_data[syn_data <= 0.0] = 1e-12
        self.syn_data = syn_data
        if self.verbose:
            logger.info("Synthetic data generated")

        uncertainty_mean = self.rng.uniform(low=self.uncertainty_mean_min, high=self.uncertainty_mean_max,
                                            size=self.features_n)
        syn_unc_p = self.rng.normal(loc=uncertainty_mean, scale=self.uncertainty_scale, size=syn_data.shape)
        syn_uncertainty = syn_data * syn_unc_p
        syn_uncertainty[syn_uncertainty <= 0.0] = 1e-4
        self.syn_uncertainty = syn_uncertainty
        if self.verbose:
            logger.info("Synthetic uncertainty data generated")

    def _generate_dfs(self):
        """
        Create data, uncertainty, profile and contribution dataframes with column labels and index.
        """

        self.syn_data_df = pd.DataFrame(self.syn_data, columns=self.syn_columns)
        self.syn_data_df['Date'] = self.time_steps
        self.syn_data_df.set_index('Date', inplace=True)
        self.syn_uncertainty_df = pd.DataFrame(self.syn_uncertainty, columns=self.syn_columns)
        self.syn_uncertainty_df['Date'] = self.time_steps
        self.syn_uncertainty_df.set_index('Date', inplace=True)
        self.syn_profiles_df = pd.DataFrame(self.syn_profiles.T, columns=self.syn_factor_columns)
        self.syn_contributions_df = pd.DataFrame(self.syn_contributions, columns=self.syn_factor_columns)
        if self.verbose:
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
        if self.verbose:
            logger.info("Synthetic source apportionment instance created.")

    def get_data(self):
        """
        Get the synthetic data and uncertainty dataframes to use with the DataHandler.

        Returns
        -------
            pd.DataFrame, pd.DataFrame
        """
        self._generate_data()
        self._generate_dfs()
        self._create_sa()
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
            if self.verbose:
                logger.info("Searching all models in the batch to find which has the highest average correlation mapping.")
            self.factor_compare = FactorCompare(input_df=self.syn_data_df,
                                                uncertainty_df=self.syn_uncertainty_df,
                                                base_profile_df=self.syn_profiles_df,
                                                base_contribution_df=self.syn_contributions_df,
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

    def plot_synthetic_contributions(self):
        """
        Plot the factor contribution matrix.
        """

        curve_figure = go.Figure()
        for i in range(self.factors_n):
            contribution_i = self.syn_contributions[:, i]
            curve_figure.add_trace(go.Scatter(x=list(self.time_steps), y=contribution_i,
                                              name=self.syn_factor_columns[i], mode='lines+markers'))
        curve_figure.update_layout(title_text="Factor Contributions (W)", width=1200, height=800)
        curve_figure.update_yaxes(title_text="Conc")
        curve_figure.show()

    def plot_comparison(self, model_i: int = None):
        """
        Plot the results of the output comparison for the model with the highest correlated mapping, if model_i is not
        specified. Otherwise, plots the output comparison of model_i to the synthetic profiles.

        Parameters
        ----------
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

        # color_map = px.colors.sample_colorscale("plasma", [n / (self.factors_n - 1) for n in range(self.factors_n)])
        # r_color_map = px.colors.sample_colorscale("jet", [n / (100 - 1) for n in range(100)])

        syn_H = self.syn_profiles
        norm_syn_H = 100 * (syn_H / syn_H.sum(axis=0))
        _H = self.batch_sa.results[model_i].H
        norm_H = 100 * (_H / _H.sum(axis=0))

        syn_W = self.syn_contributions
        norm_syn_W = 100 * (syn_W / syn_W.sum(axis=0))
        _W = self.batch_sa.results[model_i].W
        norm_W = 100 * (_W / _W.sum(axis=0))

        factors_n = min(len(self.factor_compare.sa_factors), len(self.factor_compare.base_factors))

        if not self.factor_compare.base_k:
            subplot_titles = [f"Synthetic Factor {i} : Modelled {factor_compare['factor_map'][i - 1]}" for i in
                              range(1, factors_n + 1)]
        else:
            subplot_titles = [f"Modelled Factor {i} : Synthetic {factor_compare['factor_map'][i - 1]}" for i in
                              range(1, factors_n + 1)]
        for i in range(1, factors_n + 1):
            map_i = int(factor_compare['factor_map'][i - 1].split(" ")[1])
            if not self.factor_compare.base_k:
                syn_i = i - 1
                mod_i = map_i - 1

            else:
                syn_i = map_i - 1
                mod_i = i - 1
            i_r2 = factor_compare['best_factor_r'][i-1]
            i_r2_con = factor_compare['best_contribution_r'][i-1]
            label = (subplot_titles[i - 1] + " - R2: " + str(round(i_r2, 4)),
                     subplot_titles[i - 1] + " - R2: " + str(round(i_r2_con, 4)), "", "",)
            h_fig = make_subplots(rows=2, cols=2, subplot_titles=label, vertical_spacing=0.01, row_heights=[0.6, 0.4])
            h_fig.add_trace(go.Bar(name=f"Synthetic Profile f{syn_i + 1}", x=self.syn_columns, y=norm_syn_H[syn_i],
                                   marker_color="black"), row=1, col=1)
            h_fig.add_trace(go.Bar(name=f"Modelled Profile f{mod_i + 1}", x=self.syn_columns, y=norm_H[mod_i],
                                   marker_color="green"), row=1, col=1)
            h_fig.add_trace(
                go.Bar(name="", x=self.syn_columns, y=norm_syn_H[syn_i] - norm_H[mod_i], marker_color="blue",
                       showlegend=False), row=2, col=1)
            h_fig.add_trace(go.Scatter(name=f"Synthetic Contribution f{syn_i + 1}", x=self.syn_data_df.index,
                                       y=norm_syn_W[:, syn_i], line_color="black", mode='lines+markers'), row=1, col=2)
            h_fig.add_trace(go.Scatter(name=f"Model Contribution f{mod_i + 1}", x=self.syn_data_df.index,
                                       y=norm_W[:, mod_i], line_color="green", mode='lines+markers'), row=1, col=2)
            h_fig.add_trace(
                go.Scatter(name="", x=self.syn_data_df.index, y=norm_syn_W[:, syn_i] - norm_W[:, mod_i],
                           marker_color="blue", showlegend=False), row=2, col=2)
            h_fig.update_yaxes(title_text="Synthetic Profile", row=1, col=1, title_standoff=3)
            h_fig.update_yaxes(title_text="Difference", row=2, col=1)
            h_fig.update_yaxes(title_text="Scaled Concentrations", row=1, col=2)
            h_fig.update_xaxes(row=1, showticklabels=False)
            h_fig.update_yaxes(row=2, col=2, title_text="Residuals")
            h_fig.update_yaxes(row=2, col=1, range=[-50, 50])
            h_fig.update_layout(title_text=f"Mapped Factor Comparison - Model: {model_i + 1}",
                                width=1600, height=800, hovermode='x', showlegend=True)
            h_fig.show()

    def plot_profile_comparison(self, model_i: int = None):
        """
        Plot the results of the output comparison for the model with the highest correlated mapping, if model_i is not
        specified. Otherwise, plots the output comparison of model_i to the synthetic profiles.

        Parameters
        ----------
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

        factors_n = min(len(self.factor_compare.sa_factors), len(self.factor_compare.base_factors))

        if not self.factor_compare.base_k:
            subplot_titles = [f"Synthetic Factor {i} : Modelled {factor_compare['factor_map'][i - 1]}" for i in
                              range(1, factors_n + 1)]
        else:
            subplot_titles = [f"Modelled Factor {i} : Synthetic {factor_compare['factor_map'][i - 1]}" for i in
                              range(1, factors_n + 1)]
        for i in range(1, factors_n + 1):
            map_i = int(factor_compare['factor_map'][i - 1].split(" ")[1])
            if not self.factor_compare.base_k:
                syn_i = i - 1
                mod_i = map_i - 1
            else:
                syn_i = map_i - 1
                mod_i = i - 1
            abs_res = np.round(np.abs(norm_syn_H[syn_i] - norm_H[mod_i]).sum(), 2)
            label = (f"{subplot_titles[i - 1]} - Residual: {abs_res}", "", "")
            h_fig = make_subplots(rows=3, cols=1, vertical_spacing=0.05, subplot_titles=label,
                                  row_heights=[0.5, 0.08, 0.3])
            h_fig.add_trace(
                go.Bar(name=f"Syn Factor {syn_i + 1}", x=self.syn_columns, y=norm_syn_H[syn_i], marker_color="black"),
                row=1, col=1)
            h_fig.add_trace(
                go.Bar(name=f"Modelled Factor {mod_i+1}", x=self.syn_columns, y=norm_H[mod_i],
                       marker_color="green"), row=1, col=1)
            i_r2 = factor_compare['best_factor_r'][i - 1]
            h_fig.add_trace(go.Bar(name="R2", x=(i_r2,), orientation='h', marker_color=r_color_map[int(100 * i_r2)],
                                   text=np.round(i_r2, 4), textposition="inside", hoverinfo='text',
                                   hovertemplate="R2: %{x:4f}<extra></extra>", showlegend=False), row=2, col=1)
            h_fig.add_trace(
                go.Bar(name="", x=self.syn_columns, y=norm_syn_H[syn_i] - norm_H[mod_i], marker_color="blue",
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

    def save(self, sim_name: str = "synthetic", output_directory: str = "."):
        """
        Save the generated synthetic data and uncertainty datasets, and the simulator instance binary.

        Parameters
        ----------
        sim_name : str
            The name for the data and uncertainty dataset files.
        output_directory : str
            The path to the directory where the files will be saved.

        Returns
        -------
        bool
            True if save is successful, otherwise False.
        """
        output_directory = Path(output_directory)
        if not output_directory.is_absolute():
            logger.error("Provided output directory is not an absolute path. Must provide an absolute path.")
            return False
        if os.path.exists(output_directory):
            file_path = os.path.join(output_directory, "esat_simulator.pkl")
            with open(file_path, "wb") as save_file:
                pickle.dump(self, save_file)
                logger.info(f"ESAT Simulator instance saved to pickle file: {file_path}")

            data_file_path = os.path.join(output_directory, f"{sim_name}_data.csv")
            self.syn_data_df.to_csv(data_file_path)
            logger.info(f"ESAT synthetic data saved to file: {data_file_path}")

            uncertainty_file_path = os.path.join(output_directory, f"{sim_name}_uncertainty.csv")
            self.syn_uncertainty_df.to_csv(uncertainty_file_path)
            logger.info(f"ESAT synthetic uncertainty saved to file: {uncertainty_file_path}")

            profiles_file_path = os.path.join(output_directory, f"{sim_name}_profiles.csv")
            self.syn_profiles_df.to_csv(profiles_file_path)
            logger.info(f"ESAT synthetic profiles saved to file: {profiles_file_path}")

            contribution_file_path = os.path.join(output_directory, f"{sim_name}_contributions.csv")
            self.syn_contributions_df.to_csv(contribution_file_path)
            logger.info(f"ESAT synthetic contributions saved to file: {contribution_file_path}")
            return True
        return False

    @staticmethod
    def load(file_path: str):
        """
        Load a previously saved ESAT Simulator pickle file.

        Parameters
        ----------
        file_path : str
           File path to a previously saved ESAT Simulator pickle file

        Returns
        -------
        Simulator
           On successful load, will return a previously saved Simulator object. Will return None on load fail.
        """
        file_path = Path(file_path)
        if not file_path.is_absolute():
            if not file_path.is_absolute():
                logger.error("Provided directory is not an absolute path. Must provide an absolute path.")
                return None
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as pfile:
                    sim = pickle.load(pfile)
                    return sim
            except pickle.PickleError as p_error:
                logger.error(f"Failed to load Simulator pickle file {file_path}. \nError: {p_error}")
                return None
        else:
            logger.error(f"Simulator file load failed, specified pickle file does not exist. File Path: {file_path}")
            return None
