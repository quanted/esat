import copy
import logging
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from esat.data.datahandler import DataHandler
from esat.model.sa import SA
from esat.model.batch_sa import BatchSA
from esat.utils import min_timestep

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelAnalysis:
    """
    Class for running model analysis and generating plots.
    A collection of model statistic methods and plot generation functions.

    Parameters
    ----------
    datahandler : DataHandler
        The datahandler instance used for processing the input and uncertainty datasets used by the SA model.
    model : SAModel
        A completed SA model with output used for calculating model statistics and generating plots.
    selected_model : int
        If SA model is part of a batch, the model id/index that will be used for plot labels.
    """
    def __init__(self,
                 datahandler: DataHandler,
                 model: SA,
                 selected_model: int = None
                 ):
        """
        Constructor method
        """
        self.dh = datahandler
        self.model = model
        self.selected_model = selected_model
        self.statistics = None
        self.residual_metrics = None
        self.V_prime_k = None
        self.V_prime_plot = None

    def aggregate_factors_for_plotting(self):
        """
        Aggregate each factor's V_prime_k for plotting, reducing to max_samples using DataHandler's binning.

        Returns
        -------
        dict
            Dictionary mapping factor index to aggregated V_prime_k DataFrame.
        """
        W = self.model.W
        H = self.model.H
        factors = self.model.factors
        factor_agg = {}
        for k in range(factors):
            V_prime_k = np.outer(W[:, k], H[k])
            # Use DataHandler's aggregation for consistent plotting
            agg_df = self.dh.aggregate_output(V_prime_k)
            factor_agg[k] = agg_df
        # sum of aggregated factors, into v_prime (same dimensions as V)
        factor_vprime = sum([agg_df.values for agg_df in factor_agg.values()])
        factor_vprime = pd.DataFrame(factor_vprime, columns=self.dh.input_data_df.columns,
                                     index=self.dh.input_data_df.index)
        self.V_prime_plot = factor_vprime
        self.V_prime_k = factor_agg
        return factor_agg, factor_vprime

    def features_metrics(self, est_V: np.ndarray = None):
        """
        Create a dataframe of the feature metrics and error for model analysis.

        Parameters
        ----------
        est_V : np.ndarray
            Overrides the use of the ESAT model's WH matrix in the residual calculation. Default = None.

        Returns
        -------
            pd.DataFrame
                The features of the input dataset compared to the results of the model, as a pd.DataFrame
        """
        V = self.model.V
        if est_V is None:
            est_V = self.model.WH
        features = self.dh.features

        feature_mean = []
        feature_var = []
        est_means = []
        est_vars = []
        feature_rmse = []
        for column_i in range(V.shape[1]):
            c_mean = round(float(V[:, column_i].mean()), 4)
            c_var = round(float(V[:, column_i].var()), 4)
            est_mean = round(float(est_V[:, column_i].mean()), 4)
            est_var = round(float(est_V[:, column_i].var()), 4)
            c_rmse = round(np.sqrt(np.sum((V[:, column_i] - est_V[:, column_i]) ** 2)) / 14, 4)
            feature_mean.append(c_mean)
            feature_var.append(c_var)
            est_means.append(est_mean)
            est_vars.append(est_var)
            feature_rmse.append(c_rmse)
        df = pd.DataFrame(data={
            "Feature": features,
            "Input Mean": feature_mean,
            "Input Var": feature_var,
            "Est Mean": est_means,
            "Est Var": est_vars,
            "RMSE": feature_rmse,
        })
        self.residual_metrics = df
        return df

    def calculate_statistics(self, results: np.ndarray = None):
        """
        Calculate general statistics from the results of the NMF model run.

        Will generate a pd.DataFrame with a set of metrics for each feature. The resulting dataframe will be accessible
        as .statistics. These metrics focus on residual analysis, including Norm tests of the residuals with three
        different metrics for testing the norm.

        Parameters
        ----------
        results : np.ndarray
            The default behavior is for this function to use the ESAT model WH matrix for calculating metrics, this can
            be overriden by providing np.ndarray in the 'results' parameter. Default = None.

        """

        statistics = {"Features": [], "Category": [], "r2": [], "Intercept": [], "Intercept SE": [], "Slope": [],
                      "Slope SE": [], "SE": [], "SE Regression": [],
                      "Anderson Normal Residual": [], "Anderson Statistic": [],
                      "Shapiro Normal Residuals": [], "Shapiro PValue": [],
                      "KS Normal Residuals": [], "KS PValue": [], "KS Statistic": []}
        cats = copy.copy(self.dh.metrics['Category'])
        results = self.model.WH if results is None else results
        residuals = self.dh.input_data_df - results
        scaled_residuals = residuals / self.dh.uncertainty_data
        for feature_idx, x_label in enumerate(self.dh.features):
            observed_data = self.dh.input_data_df[x_label]
            predicted_data = results[:, feature_idx]

            i_residuals = residuals[x_label]
            i_sresiduals = scaled_residuals[x_label]
            se = np.std(i_residuals) / np.sqrt(len(i_residuals))

            stats_results = stats.linregress(observed_data, predicted_data)
            shap_stat, shap_p = stats.shapiro(i_sresiduals)
            anderson = stats.anderson(i_sresiduals, dist='norm')
            loc, scale = np.mean(i_sresiduals), np.std(i_sresiduals, ddof=1)
            cdf = stats.norm(loc, scale).cdf
            ktest = stats.ks_1samp(i_sresiduals, cdf)
            category = cats.loc[x_label]
            se_regression = np.sqrt(1 - stats_results.rvalue**2) * np.std(predicted_data)

            normal_residuals = "No"
            for i, cv in enumerate(anderson.critical_values):
                if anderson.statistic < cv:
                    normal_residuals = str(anderson.significance_level[i])
                    break

            statistics["Features"].append(x_label)
            statistics["Category"].append(category)
            statistics["r2"].append(stats_results.rvalue**2)
            statistics["Slope"].append(stats_results.slope)
            statistics["Intercept"].append(stats_results.intercept)
            statistics["SE"].append(se)
            statistics["SE Regression"].append(se_regression)
            statistics["Slope SE"].append(stats_results.stderr)
            statistics["Intercept SE"].append(stats_results.intercept_stderr)
            statistics["KS PValue"].append(ktest.pvalue)
            statistics["KS Statistic"].append(ktest.statistic)
            statistics["Shapiro PValue"].append(shap_p)
            statistics["Anderson Statistic"].append(anderson.statistic)
            statistics["Anderson Normal Residual"].append(normal_residuals)
            statistics["Shapiro Normal Residuals"].append("Yes" if shap_p >= 0.05 else "No")
            statistics["KS Normal Residuals"].append("Yes" if ktest.pvalue >= 0.05 else "No")

        self.statistics = pd.DataFrame(data=statistics)

    def plot_residual_histogram(self,
                                feature_idx: int = None,
                                feature_name: str = None,
                                abs_threshold: float = 3.0,
                                est_V: np.ndarray = None,
                                show: bool = True
                                ):
        """
        Create a plot of a histogram of the residuals for a specific feature.

        Parameters
        ----------
        feature_idx : int, optional
            The index of the feature to plot.
        feature_name : str, optional
            The name of the feature to plot.
        abs_threshold : float
            The function generates a list of residuals that exceed this limit, the absolute value of the limit.
        est_V : np.ndarray
            Overrides the use of the ESAT model's WH matrix in the residual calculation. Default = None.
        show : bool
            If True, the plot will be displayed. Default is True.
        """
        V = self.model.V
        if est_V is None:
            est_V = self.model.WH
        U = self.model.U

        # Determine feature index
        if feature_idx is not None:
            if feature_idx < 0 or feature_idx >= len(self.dh.features):
                logger.info(f"Invalid feature_idx provided, must be between 0 and {len(self.dh.features) - 1}")
                return
            feature = self.dh.features[feature_idx]
        elif feature_name is not None:
            if feature_name not in self.dh.features:
                logger.info(f"Invalid feature_name provided, must be one of {self.dh.features}")
                return
            feature = feature_name
            feature_idx = self.dh.features.index(feature_name)
        else:
            logger.info("Either feature_idx or feature_name must be provided.")
            return

        residuals_data = (V[:, feature_idx] - est_V[:, feature_idx]) / U[:, feature_idx]
        dist_fig = ff.create_distplot([residuals_data], [feature], curve_type='normal')
        normal_x = dist_fig.data[1]['x']
        normal_y = dist_fig.data[1]['y']
        dist_fig2 = ff.create_distplot([residuals_data], [feature], curve_type='kde')
        kde_x = dist_fig2.data[1]['x']
        kde_y = dist_fig2.data[1]['y']

        residual_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.2, 0.8], vertical_spacing=0.01,
                                     subplot_titles=["Scaled Residuals", None])

        residual_fig.add_trace(
            go.Histogram(x=residuals_data, histnorm='probability', name='Histogram'),
            row=2, col=1
        )
        residual_fig.add_trace(
            go.Box(x=residuals_data, boxmean=True, name="Dist"),
            row=1, col=1
        )
        residual_fig.add_trace(go.Scatter(x=normal_x, y=normal_y, mode='lines', name='Normal'), row=2, col=1)
        residual_fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode='lines', name='KDE'), row=2, col=1)

        residual_fig.update_layout(
            title=f"Residual Histogram for {feature}",
            yaxis2_title="Probability",
            width=1200,
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        if show:
            residual_fig.show()
            return None
        else:
            residuals_df = pd.DataFrame(residuals_data, index=self.dh.input_data_df.index, columns=[feature])
            if abs_threshold != -1:
                residuals_df = residuals_df[residuals_df[feature].abs() > abs_threshold]
            return residual_fig, residuals_df

    def plot_estimated_observed(self,
                                feature_idx: int = None,
                                feature_name: str = None,
                                show: bool = True):
        """
        Create a plot that shows the estimated concentrations of a feature vs the observed concentrations.

        Parameters
        ----------
        feature_idx : int, optional
            The index of the feature to plot.
        feature_name : str, optional
            The name of the feature to plot.
        show : bool
            If True, the plot will be displayed. Default is True.
        """
        # Determine feature index
        if feature_idx is not None:
            if feature_idx < 0 or feature_idx >= len(self.dh.features):
                logger.info(f"Invalid feature_idx provided, must be between 0 and {len(self.dh.features) - 1}")
                return None
            feature = self.dh.features[feature_idx]
        elif feature_name is not None:
            if feature_name not in self.dh.features:
                logger.info(f"Invalid feature_name provided, must be one of {self.dh.features}")
                return None
            feature = feature_name
            feature_idx = self.dh.features.index(feature_name)
        else:
            logger.info("Either feature_idx or feature_name must be provided.")
            return None
        if self.V_prime_plot is None:
            self.aggregate_factors_for_plotting()

        observed_data = self.dh.input_data_plot[feature]
        predicted_data = self.V_prime_plot.iloc[:, feature_idx]

        # Perform regression
        A = np.vstack([observed_data.values, np.ones(len(observed_data))]).T
        m, c = np.linalg.lstsq(A, predicted_data, rcond=None)[0]
        m1, c1 = np.linalg.lstsq(A, observed_data, rcond=None)[0]

        # Create the plot
        xy_plot = go.Figure()
        xy_plot.add_trace(go.Scatter(x=observed_data, y=predicted_data, mode='markers', name="Data"))
        xy_plot.add_trace(go.Scatter(x=observed_data, y=(m * observed_data + c),
                                     line=dict(color='red', dash='dash', width=1), name='Regression'))
        xy_plot.add_trace(go.Scatter(x=observed_data, y=(m1 * observed_data + c1),
                                     line=dict(color='blue', width=1), name='One-to-One'))

        xy_plot.update_layout(
            title=f"Observed/Predicted Scatter Plot - {feature}",
            width=800,
            height=600,
            xaxis_title="Observed Concentrations",
            yaxis_title="Predicted Concentrations",
            hovermode='x'
        )

        if show:
            xy_plot.show()
            return None
        else:
            return xy_plot

    def plot_estimated_timeseries(self,
                                  feature_idx: int = None,
                                  feature_name: str = None,
                                  show: bool = True):
        """
        Create a plot that shows the estimated values of a timeseries for a specific feature.

        Parameters
        ----------
        feature_idx : int, optional
            The index of the feature to plot.
        feature_name : str, optional
            The name of the feature to plot.
        show : bool
            If True, the plot will be displayed. Default is True.
        """
        # Determine feature index
        if feature_idx is not None:
            if feature_idx < 0 or feature_idx >= len(self.dh.features):
                logger.info(f"Invalid feature_idx provided, must be between 0 and {len(self.dh.features) - 1}")
                return None
            feature = self.dh.features[feature_idx]
        elif feature_name is not None:
            if feature_name not in self.dh.features:
                logger.info(f"Invalid feature_name provided, must be one of {self.dh.features}")
                return None
            feature = feature_name
            feature_idx = self.dh.features.index(feature_name)
        else:
            logger.info("Either feature_idx or feature_name must be provided.")
            return None

        if self.V_prime_plot is None:
            self.aggregate_factors_for_plotting()

        observed_data = self.dh.input_data_plot[feature].values
        predicted_data = self.V_prime_plot.iloc[:, feature_idx]

        data_df = pd.DataFrame(data={"observed": observed_data, "predicted": predicted_data},
                               index=self.dh.input_data_plot.index)
        data_df.index = pd.to_datetime(data_df.index)
        data_df = data_df.sort_index()
        data_df = data_df.resample(min_timestep(data_df)).mean()

        ts_subplot = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.8, 0.2])

        ts_subplot.add_trace(go.Scatter(x=data_df.index, y=data_df["observed"], line=dict(width=1),
                                        mode='lines+markers', name="Observed Concentrations"), row=1, col=1)
        ts_subplot.add_trace(go.Scatter(x=data_df.index, y=data_df["predicted"], line=dict(width=1),
                                        mode='lines+markers', name="Predicted Concentrations"), row=1, col=1)
        ts_subplot.add_trace(go.Scatter(x=data_df.index, y=data_df["observed"] - data_df["predicted"],
                                        line=dict(width=1), mode='lines', name="Residuals"), row=2, col=1)

        ts_subplot.update_layout(
            title=f"Estimated Time-series for {feature} - Model {self.selected_model+1}",
            width=1200,
            height=800,
            yaxis_title="Concentrations",
            hovermode='x unified'
        )

        if show:
            ts_subplot.show()
            return None
        else:
            return ts_subplot

    def plot_factor_profile(self,
                            factor_idx: int,
                            H: np.ndarray = None,
                            W: np.ndarray = None,
                            show: bool = True
                            ):
        """
        Create a bar plot of a factor profile.

        Parameters
        ----------
        factor_idx : int
            The index of the factor to plot (1 -> k).
        H : np.ndarray
            Overrides the factor profile matrix in the ESAT model used for the plot.
        W : np.ndarray
            Overrides the factor contribution matrix in the ESAT model used for the plot.
        show : bool
            If True, the plot will be displayed. Default is True.
        """
        #TODO: Manage case where the number of samples is very high and plotting will take a long time.
        if factor_idx > self.model.factors or factor_idx < 1:
            logger.info(f"Invalid factor provided, must be between 1 and {self.model.factors}")
            return None
        factor_label = f"Factor {factor_idx}"
        factor_idx_l = factor_idx
        factor_idx = factor_idx - 1
        if H is None:
            factors_data = self.model.H[factor_idx]
            factors_sum = self.model.H.sum(axis=0)
        else:
            factors_data = H[factor_idx]
            factors_sum = H.sum(axis=0)
        if W is None:
            factor_contribution = self.model.W[:, factor_idx]
        else:
            factor_contribution = W[:, factor_idx]

        factor_matrix = np.matmul(factor_contribution.reshape(len(factor_contribution), 1), [factors_data])

        factor_conc_sums = factor_matrix.sum(axis=0)
        factor_conc_sums[factor_conc_sums == 0.0] = 1e-12

        norm_profile = 100 * (factors_data / factors_sum)

        norm_contr = factor_contribution / factor_contribution.mean()
        #
        data_df = pd.DataFrame(data={factor_label: norm_contr}, index=self.dh.input_data_df.index)
        data_df[factor_label] = norm_contr
        data_df.index = pd.to_datetime(data_df.index)
        data_df = data_df.sort_index()
        data_df = data_df.resample(min_timestep(data_df)).mean()

        profile_plot = make_subplots(specs=[[{"secondary_y": True}]], rows=1, cols=1)
        profile_plot.add_trace(go.Scatter(x=self.dh.features, y=norm_profile, mode='markers', marker=dict(color='red'),
                                          name="% of Features"), secondary_y=True, row=1, col=1)
        profile_plot.add_trace(go.Bar(x=self.dh.features, y=factor_conc_sums, marker_color='rgb(158,202,225)',
                                      marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6,
                                      name='Conc. of Features'), secondary_y=False, row=1, col=1)
        profile_plot.update_yaxes(title_text="Conc. of Features", secondary_y=False, row=1, col=1,
                                  type="log",
                                  range=[0, np.log10(factor_conc_sums).max()]
                                  )
        profile_plot.update_yaxes(title_text="% of Features", secondary_y=True, row=1, col=1, range=[0, 100])
        profile_plot.update_layout(title=f"Factor Profile - Model {self.selected_model+1} - Factor {factor_idx_l}",
                                   width=1200, height=600, hovermode='x unified')
        if show:
            profile_plot.show()

        contr_plot = go.Figure()
        contr_plot.add_trace(go.Scatter(x=data_df.index, y=data_df[factor_label], mode='lines+markers',
                                        name="Normalized Contributions", line=dict(color='blue')))
        contr_plot.update_layout(title=f"Factor Contributions - Model {self.selected_model+1} - Factor {factor_idx_l}",
                                 width=1200, height=600, showlegend=True,
                                 legend=dict(orientation="h", xanchor="right", yanchor="bottom", x=1, y=1.02))
        contr_plot.update_yaxes(title_text="Normalized Contributions")
        if show:
            contr_plot.show()
            return None
        else:
            return profile_plot, contr_plot

    def plot_all_factors(self, factor_list: list = None, H: np.ndarray = None, W: np.ndarray = None, show: bool = True):
        """
        Create a vertical set of subplots for all factor profiles, similar to plot_factor_profile.

        Parameters
        ----------
        factor_list : list
            A list of factor indices to plot, if None will plot all factors.
        H : np.ndarray
            Overrides the factor profile matrix in the ESAT model used for the plot.
        W : np.ndarray
            Overrides the factor contribution matrix in the ESAT model used for the plot.
        """
        if H is None:
            H = self.model.H
        if W is None:
            W = self.model.W

        num_factors = self.model.factors
        fig = make_subplots(rows=num_factors,
                            cols=1,
                            specs=[[{"secondary_y": True}] for _ in range(num_factors)],
                            shared_xaxes=True,
                            vertical_spacing=0.01,
                            subplot_titles=[f"Factor {i + 1}" for i in range(num_factors)])

        for factor_idx in range(num_factors):
            if factor_list is not None:
                if factor_idx not in factor_list:
                    continue
            factors_data = H[factor_idx]
            factors_sum = H.sum(axis=0)
            factor_contribution = W[:, factor_idx]

            factor_matrix = np.matmul(factor_contribution.reshape(len(factor_contribution), 1), [factors_data])
            factor_conc_sums = factor_matrix.sum(axis=0)
            factor_conc_sums[factor_conc_sums == 0.0] = 1e-12
            norm_profile = 100 * (factors_data / factors_sum)

            fig.add_trace(go.Bar(x=self.dh.features, y=factor_conc_sums, name="Conc. of Features",
                                 marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                                 marker_line_width=1.5, opacity=0.6, legendgroup="group1", showlegend=(factor_idx==0)), secondary_y=False, row=factor_idx + 1, col=1)

            fig.add_trace(go.Scatter(x=self.dh.features, y=norm_profile, mode='markers', marker=dict(color='red'),
                                     name="% of Features", legendgroup="group2", showlegend=(factor_idx==0)), secondary_y=True, row=factor_idx + 1, col=1)

            fig.update_yaxes(title_text="Conc. of Features", type="log",
                             range=[0, np.log10(factor_conc_sums).max()], secondary_y=False, row=factor_idx + 1, col=1)
            fig.update_yaxes(title_text="% of Features", range=[0, 100], secondary_y=True, row=factor_idx + 1, col=1)

        fig.update_layout(title=f"Factor Profiles - Model {self.selected_model+1}", width=1200, height=600 * num_factors,
                          hovermode='x unified')
        if show:
            fig.show()
            return None
        else:
            return fig

    def plot_all_factors_3d(self, H=None, W=None, show: bool = True, plot_type: str = "profile"):
        """
        Create a 3D bar plot of the factor profiles and their contributions.
        Parameters
        ----------
        H : np.ndarray, optional
            The factor profile matrix, if None will use the model's H matrix.
        W : np.ndarray, optional
            The factor contribution matrix, if None will use the model's W matrix.
        show : bool
            If True, the plot will be displayed. Default is True.
        plot_type : str
            Should be either "profile", "conc", or "both".

        Returns
        -------

        """
        if H is None:
            H = self.model.H
        if W is None:
            W = self.model.W

        num_factors = self.model.factors
        num_features = len(self.dh.features)
        conc_z = np.zeros((num_factors, num_features))
        percent_z = np.zeros((num_factors, num_features))

        for factor_idx in range(num_factors):
            factors_data = H[factor_idx]
            factor_contribution = W[:, factor_idx]
            factor_matrix = np.matmul(factor_contribution.reshape(len(factor_contribution), 1), [factors_data])
            factor_conc_sums = factor_matrix.sum(axis=0)
            factor_conc_sums[factor_conc_sums == 0.0] = 1e-12
            conc_z[factor_idx, :] = np.log10(factor_conc_sums + 1)

            factors_sum = H.sum(axis=0)
            percent_z[factor_idx, :] = 100 * (factors_data / factors_sum)

        # Bar parameters
        bar_width = 0.85
        bar_depth = 0.35

        def _create_mesh3d(z_data, hover_label):
            X, Y, Z, I, J, K, color, hovertext = [], [], [], [], [], [], [], []
            idx = 0
            for i in range(num_factors):
                for j in range(num_features):
                    x0 = j - bar_width / 2
                    x1 = j + bar_width / 2
                    y0 = i - bar_depth / 2
                    y1 = i + bar_depth / 2
                    z0 = 0
                    z1 = z_data[i, j]

                    # 8 vertices of the cuboid
                    vertices = [
                        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),  # bottom
                        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)  # top
                    ]
                    X.extend([v[0] for v in vertices])
                    Y.extend([v[1] for v in vertices])
                    Z.extend([v[2] for v in vertices])
                    color.extend([z1] * 8)

                    # Add hover text
                    hovertext.extend([f"Factor: {i + 1}<br>Feature: {self.dh.features[j]}<br>Value: {z1:.2f}"] * 8)

                    # 12 triangles (2 per face)
                    v = idx * 8
                    faces = [
                        (0, 1, 2), (0, 2, 3),  # bottom
                        (4, 5, 6), (4, 6, 7),  # top
                        (0, 1, 5), (0, 5, 4),  # front
                        (1, 2, 6), (1, 6, 5),  # right
                        (2, 3, 7), (2, 7, 6),  # back
                        (3, 0, 4), (3, 4, 7)  # left
                    ]
                    for f in faces:
                        I.append(v + f[0])
                        J.append(v + f[1])
                        K.append(v + f[2])
                    idx += 1
            return go.Mesh3d(
                x=X, y=Y, z=Z,
                i=I, j=J, k=K,
                intensity=color,
                colorscale='Plasma',
                opacity=1.0,
                showscale=True,
                hoverinfo="text",
                text=hovertext
            )

        fig = go.Figure()

        add_button = False
        # Add 3D bar graphs for "Conc. of Features"
        if plot_type.lower() in ["conc", "both"]:
            fig.add_trace(_create_mesh3d(conc_z, "Conc. of Features"))
        else:
            # Add 3D bar graphs for "% of Features"
            fig.add_trace(_create_mesh3d(percent_z, "% of Features"))
        if plot_type.lower() == "both":
            fig.data[1].visible = False  # Initially hide "% of Features"
            add_button = True
        # Update layout and add buttons for toggling plot types
        fig.update_layout(
            title="3D Factor Profiles",
            width=1200,
            height=800,
            scene=dict(
                xaxis=dict(title="Features"),
                yaxis=dict(title="Factors"),
                zaxis=dict(title="Value")
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.9,
                    y=1.1,
                    buttons=[
                        dict(
                            label="Conc. of Features",
                            method="update",
                            args=[{
                                "visible": [True, False],  # trace visibility
                                'scene.zaxis.range': [1, conc_z.max()]
                            }]
                        ),
                        dict(
                            label="% of Features",
                            method="update",
                            args=[{
                                "visible": [False, True],  # trace visibility
                                'scene.zaxis.range': [0, 100]
                            }]
                        )
                    ] if add_button else None
                )
            ]
        )
        if show:
            fig.show()
            return None
        else:
            return fig

    def plot_factor_fingerprints(self, grouped: bool = False, show: bool = True):
        """
        Create a stacked bar plot of the factor profile, fingerprints.

        """
        factors_data = self.model.H
        normalized_factors_data = 100 * (factors_data / factors_data.sum(axis=0))

        fig = go.Figure()
        for idx in range(self.model.factors-1, -1, -1):
            fig.add_trace(go.Bar(name=f"Factor {idx+1}", x=self.dh.features, y=normalized_factors_data[idx]))
            fig.update_layout(title=f"Factor Fingerprints - Model {self.selected_model+1}",
                              width=1200, height=800, barmode='group' if grouped else 'stack',
                              hovermode='x unified')
        fig.update_yaxes(title_text="% Feature Concentration", range=[0, 100])
        if show:
            fig.show()
        else:
            return fig


    def plot_g_space(self,
                     factor_1: int,
                     factor_2: int,
                     show: bool = True
                     ):
        """
        Create a scatter plot showing a factor contributions vs another factor contributions.

        Parameters
        ----------
        factor_1 : int
            The index of the factor to plot along the x-axis.
        factor_2 : int
            The index of the factor to plot along the y-axis.
        show : bool
            If True, the plot will be displayed. Default is True.

        """
        if factor_1 > self.model.factors or factor_1 < 1:
            logger.info(f"Invalid factor_1 provided, must be between 1 and {self.model.factors}")
            return
        if factor_2 > self.model.factors or factor_2 < 1:
            logger.info(f"Invalid factor_2 provided, must be between 0 and {self.model.factors}")
            return

        factors_contr = self.model.W
        normalized_factors_contr = factors_contr / factors_contr.sum(axis=0)
        f1_idx = factor_1 - 1
        f2_idx = factor_2 - 1
        fig = go.Figure(data=go.Scatter(
            x=normalized_factors_contr[:, f1_idx],
            y=normalized_factors_contr[:, f2_idx], mode='markers')
        )
        fig.update_layout(title=f"G-Space Plot - Model {self.selected_model+1}", width=800, height=800)
        fig.update_yaxes(title_text=f"Factor {factor_2} Contributions (avg=1)")
        fig.update_xaxes(title_text=f"Factor {factor_1} Contributions (avg=1)")
        if show:
            fig.show()
        else:
            return fig

    def plot_factor_contributions(self,
                                  feature_idx: int,
                                  contribution_threshold: float = 0.05,
                                  show: bool = True
                                  ):
        """
        Create a plot of the factor contributions and the normalized contribution.

        Parameters
        ----------
        feature_idx : int
            The index of the feature to plot.
        contribution_threshold : float
            The contribution percentage of a factor above which to include on the plot.
        show : bool
            If True, the plot will be displayed. Default is True.

        """
        if feature_idx > self.dh.input_data_df.shape[1] - 1 or feature_idx < 0:
            logger.info(f"Invalid feature index provided, must not be negative and be less than {self.dh.input_data_df.shape[1]-1}")
            return
        if 50.0 > contribution_threshold < 0:
            logger.info(f"Invalid contribution threshold provided, must be between 0.0 and 50.0")
            return
        x_label = self.dh.input_data_df.columns[feature_idx]

        factors_data = self.model.H
        normalized_factors_data = 100 * (factors_data / factors_data.sum(axis=0))

        feature_contr = normalized_factors_data[:, feature_idx]
        feature_contr_inc = []
        feature_contr_labels = []
        feature_legend = {}
        for idx in range(feature_contr.shape[0]-1, -1, -1):
            idx_l = idx+1
            if feature_contr[idx] > contribution_threshold:
                feature_contr_inc.append(feature_contr[idx])
                feature_contr_labels.append(f"Factor {idx_l}")
                feature_legend[f"Factor {idx_l}"] = f"Factor {idx_l} = {factors_data[idx:, feature_idx]}"
        feature_fig = go.Figure(data=[go.Pie(labels=feature_contr_labels, values=feature_contr_inc,
                                             hoverinfo="label+value", textinfo="percent")])
        feature_fig.update_layout(title=f"Factor Contributions to Feature: {x_label} - Model {self.selected_model+1}", width=1200, height=600,
                                  legend_title_text=f"Factor Contribution > {contribution_threshold}%")
        if show:
            feature_fig.show()

        factors_contr = self.model.W
        normalized_factors_contr = 100 * (factors_contr / factors_contr.sum(axis=0))
        factor_labels = [f"Factor {i}" for i in range(1, normalized_factors_contr.shape[1]+1)]
        contr_df = pd.DataFrame(normalized_factors_contr, columns=factor_labels)
        contr_df.index = pd.to_datetime(self.dh.input_data_df.index)
        contr_df = contr_df.sort_index()
        contr_df = contr_df.resample(min_timestep(contr_df)).mean()

        contr_fig = go.Figure()
        for factor in factor_labels:
            contr_fig.add_trace(go.Scatter(x=contr_df.index, y=contr_df[factor], mode='lines+markers', name=factor))
        converged = "Converged Model" if self.model.converged else "Unconverged Model"
        contr_fig.update_layout(title=f"Factor Contributions (avg=1) From Base Model #{self.selected_model+1} ({converged})",
                                width=1200, height=600, hovermode='x unified',
                                legend=dict(orientation="h", xanchor="right", yanchor="bottom", x=1, y=1.02))
        contr_fig.update_yaxes(title_text="Normalized Contribution")
        if show:
            contr_fig.show()
        else:
            return feature_fig, contr_fig

    def plot_factor_composition(self):
        """
        Creates a radar plot of the composition of all the factors to all features.

        """
        categories = self.dh.features
        profile_p = self.model.H / self.model.H.sum(axis=0)

        profile_radar = go.Figure()
        for f in range(self.model.factors):
            fH = profile_p[f]
            profile_radar.add_trace(go.Scatterpolar(
                r=fH,
                theta=categories,
                fill='toself',
                name=f"Factor {f+1}",
                hoverinfo="all",
                mode="lines+markers+text"
            ))
        profile_radar.update_layout(title="Factor Profile Composition", showlegend=True, width=1400, height=1200)
        profile_radar.show()

    def plot_factor_surface(self, factor_idx: int = 1, feature_idx: int = None,
                            percentage: bool = True, zero_threshold: float = 1e-4):
        """
        Creates a 3d surface plot of the specified factor_idx's concentration percentage or mass.

        Parameters
        ----------
        factor_idx : int
           The factor index to plot showing all features for that factor, if factor_idx is none will show the
           feature_idx for all factors.
        feature_idx : int
           The feature to include in the plot if factor_idx is none, otherwise will show all features for a specified
           factor_idx.
        percentage : bool
           Plot the concentration as a scaled value, percentage of the sum of all factors, or as the calculated mass.
           Default = True.
        zero_threshold : float
           Values below this threshold are considered zero on the plot.

        """
        if factor_idx is None and feature_idx is None:
            logger.warning("A factor or feature index must be provided.")
            return
        if factor_idx is not None:
            if factor_idx > self.model.factors or factor_idx < 1:
                logger.warning(f"Invalid factor_idx provided, must be between 1 and {self.model.factors}")
                return
            factor_idx = factor_idx - 1
        if feature_idx is not None:
            if feature_idx < 1 or feature_idx > self.model.n:
                logger.warning(f"Invalid feature_idx provided, must be between 1 and {self.model.n}")
                return
            feature_idx = feature_idx - 1

        factor_matrices = []
        percent_matrices = []
        for f in range(self.model.factors):
            fW = self.model.W[:, f]
            fW = fW.reshape(len(fW), 1)
            fH = self.model.H[f]
            f_matrix = np.multiply(fW, fH)
            factor_matrices.append(f_matrix)
            percent_matrices.append(f_matrix / self.model.V)

        _y = self.dh.input_data_df.index
        z_title = "Percentage (%)" if percentage else "Mass"
        x_labels = []
        x_label_values = []

        if factor_idx is None:
            trace_name = self.dh.features[feature_idx]
            plot_title = f"{trace_name} Concentrations for All Factors"
            _z = []
            for i in range(len(factor_matrices)):
                i_z = percent_matrices[i][:, feature_idx] if percentage else factor_matrices[i][:, feature_idx]
                _z.append(i_z)
            _x = [f"Factor {i}" for i in range(1, self.model.factors + 1)]
            _z = np.array(_z).T
            x_labels = _x
            x_label_values = _x
        else:
            plot_title = f"Feature Concentrations for Factor {factor_idx + 1}"
            trace_name = f"Factor {factor_idx + 1}"
            _z = percent_matrices[factor_idx] if percentage else factor_matrices[factor_idx]
            _z[_z < zero_threshold] = np.nan
            _x = self.dh.features

            for f in range(self.model.n):
                if not all(np.isnan(_z[:, f])):
                    x_labels.append(_x[f])
                    x_label_values.append(f)

        matrix_plot = go.Figure()
        matrix_plot.add_trace(
            go.Surface(x=_x, y=_y, z=_z, opacity=1.0, name=trace_name, showscale=True, showlegend=False,
                       colorscale='spectral'))
        matrix_plot.update_layout(scene=dict(
            xaxis=dict(title="", nticks=len(x_labels), ticktext=x_labels, tickvals=x_label_values),
            yaxis=dict(title=""),
            zaxis=dict(title=f'Concentration {z_title}')),
                                  title=plot_title, width=1200, height=1200,
                                  margin=dict(l=65, r=50, b=65, t=60))
        matrix_plot.show()


class BatchAnalysis:
    """
    Class for running batch solution analysis.

    Parameters
    ----------
    batch_sa : BatchSA
        A completed ESAT batch source apportionment to run solution analysis on.
    """
    def __init__(self, batch_sa: BatchSA, data_handler: DataHandler = None):
        self.batch_sa = batch_sa
        self.data_handler = data_handler
        self.aggregated_output = {}

    def plot_loss(self, show: bool = True):
        """
        Plot the loss value for each model in the batch solution as it changes over time.

        A model will stop updating if the convergence criteria is met, which can be identified by the models that stop
        before reaching max iterations. The ideal loss curve should represent a y=1/x hyperbola, but because of the
        data uncertainty the curve may not be entirely smooth.
        """
        q_fig = go.Figure()
        for i, result in enumerate(self.batch_sa.results):
            if result is not None:
                q_fig.add_trace(
                    go.Scatter(x=list(range(len(result.q_list))), y=result.q_list, name=f"Model {i + 1}", mode='lines'))
        q_fig.update(layout_title_text=f"Batch Q(True) vs Iterations. Max Iterations: {self.batch_sa.max_iter}")
        q_fig.update_layout(width=1200, height=600, hovermode='x')
        q_fig.update_xaxes(title_text="Iterations")
        q_fig.update_yaxes(title_text="Q(True)")
        if show:
            q_fig.show()
            return None
        return q_fig

    def plot_loss_distribution(self, show: bool = True):
        """
        Plot the distribution of batch model Q(True) and Q(Robust).

        A very broad distribution is often a result of a 'loose' convergence criteria, increasing converge_n and
        decreasing converge_delta will narrow the criteria. If the Q(True) and Q(Robust) distributions are very similar
        the solution may be overfit, where enough sources/factors are available to capture the majority of outline
        behavior. In this case, reducing the number of factors can resolve overfitting the model.
        """
        qt_list = []
        qr_list = []
        model = []
        for i, result in enumerate(self.batch_sa.results):
            if result is not None:
                model.append(i)
                qt_list.append(result.Qtrue)
                qr_list.append(result.Qrobust)

        b_q_df = pd.DataFrame(data={"Q(True)": qt_list, "Q(Robust)": qr_list, 'Model': model})
        b_q_fig = go.Figure(data=[
            go.Box(y=b_q_df['Q(True)'], boxpoints="all", notched=True, name="Q(True)", marker_size=3, text=model),
            go.Box(y=b_q_df["Q(Robust)"], boxpoints="all", notched=True, name="Q(Robust)", marker_size=3, text=model)
        ])
        b_q_fig.update(layout_title_text="Batch Models Loss Distribution")
        b_q_fig.update_yaxes(title_text="Loss (Q)")
        b_q_fig.update_xaxes(title_text="")
        b_q_fig.update_traces(hovertemplate='Model: %{text}<br>%{x}: %{y:.2f}<extra></extra>')
        b_q_fig.update_layout(width=800, height=800)
        if show:
            b_q_fig.show()
            return None
        return b_q_fig

    def plot_temporal_residuals(self, feature_idx: int, show: bool = True):
        """
        Plot the temporal residuals for a specified feature.
        Only the best model's residuals are visible initially; others are legendonly.
        """
        if feature_idx is None:
            raise ValueError("feature_idx must be provided.")

        if self.data_handler is None:
            V = self.batch_sa.V
            V_primes = [r.WH for r in self.batch_sa.results if r is not None]
            features = [f"Feature {i + 1}" for i in range(V.shape[1])]
            x = list(range(1, V.shape[0] + 1))
            feature_label = features[feature_idx]
            col_idx = feature_idx
        else:
            V = self.data_handler.input_data_plot.values
            if len(self.aggregated_output) == 0:
                for i, r in enumerate(self.batch_sa.results):
                    if r is not None:
                        self.aggregated_output[i] = self.data_handler.aggregate_output(r.WH)
            # Ensure V_primes is a list, not a dict
            V_primes = [self.aggregated_output.get(i, None) for i in range(len(self.batch_sa.results))]
            feature_label = self.data_handler.features[feature_idx]
            col_idx = feature_idx
            x = self.data_handler.input_data_plot.index

        temporal_fig = go.Figure()
        # Add input trace
        y_input = V[:, col_idx]
        temporal_fig.add_trace(
            go.Scatter(x=x, y=y_input, name="Input", line=dict(dash='dash', width=2), visible=True)
        )

        # Add residual traces for each model
        for i, V_prime in enumerate(V_primes):
            if V_prime is None:
                continue
            if self.data_handler is not None:
                if not isinstance(V_prime, pd.DataFrame):
                    V_prime = pd.DataFrame(V_prime, columns=self.data_handler.features, index=x)
                y_res = V[:, col_idx] - V_prime[feature_label].values
            else:
                y_res = V[:, col_idx] - V_prime[:, col_idx]
            visible = True if i == self.batch_sa.best_model else "legendonly"
            temporal_fig.add_trace(
                go.Scatter(x=x, y=y_res, mode='lines', name=f"Model {i + 1}", visible=visible)
            )

        temporal_fig.update_layout(
            width=1200,
            height=600,
            hovermode='x',
            title=f"Model Temporal Residuals - Feature: {feature_label}"
        )
        temporal_fig.update_yaxes(title_text="Conc.")
        if show:
            temporal_fig.show()
            return None
        return temporal_fig
