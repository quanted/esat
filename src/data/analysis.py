from plotly.subplots import make_subplots
import copy
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from src.data.datahandler import DataHandler
from src.model.batch_nmf import BatchNMF


class CompareAnalyzer:

    def __init__(self,
                 input_df,
                 pmf_profile_df,
                 pmf_contributions_df,
                 ls_profile_df,
                 ws_profile_df,
                 ls_mapping,
                 ws_mapping,
                 ls_contributions_df,
                 ws_contributions_df,
                 features,
                 datetimestamps
                 ):
        self.input_df = input_df
        self.pmf_profile_df = pmf_profile_df
        self.ls_profile_df = ls_profile_df
        self.ws_profile_df = ws_profile_df

        self.ls_mapping = ls_mapping
        self.ws_mapping = ws_mapping

        self.pmf_contributions_df = pmf_contributions_df
        self.ls_contributions_df = ls_contributions_df
        self.ws_contributions_df = ws_contributions_df

        self.features = features
        self.factor_n = pmf_profile_df.shape[1] - 1
        self.factor_columns = [f"Factor {i}" for i in range(1, self.factor_n + 1)]
        self.datetimestamps = datetimestamps

    def _standardize(self, data, include_features: bool = False):
        std_data = data[self.factor_columns].div(data[self.factor_columns].sum(axis=1), axis=0)
        if include_features:
            std_data["features"] = self.features
        return std_data

    def _matmul(self, contributions, profile):
        data = np.matmul(contributions.values, profile.values.T)
        data_df = pd.DataFrame(data, columns=self.features)
        data_df["Date"] = pd.to_datetime(self.datetimestamps, format="%m/%d/%y %H:%M")
        data_df.set_index("Date", inplace=True)
        return data_df

    def plot_factor_contribution(self, feature: str = None, feature_i: int = 0):
        if feature not in self.features:
            feature = self.features[feature_i % len(self.features)]

        pmf_norm = self._standardize(data=self.pmf_profile_df, include_features=True)
        ls_norm = self._standardize(data=self.ls_profile_df, include_features=True)
        ws_norm = self._standardize(data=self.ws_profile_df, include_features=True)

        pmf_feature_profile = pmf_norm[self.features == feature]
        ls_feature_profile = ls_norm[self.features == feature]
        ws_feature_profile = ws_norm[self.features == feature]

        pmf_pie = go.Pie(labels=self.factor_columns, values=pmf_feature_profile[self.factor_columns].values[0], name="PMF")
        ls_pie = go.Pie(labels=self.factor_columns, values=ls_feature_profile[self.factor_columns].values[0], name="LS-NMF")
        ws_pie = go.Pie(labels=self.factor_columns, values=ws_feature_profile[self.factor_columns].values[0], name="WS-NMF")

        profile_subplot = make_subplots(rows=1, cols=3,
                                        specs=[[{"type": "domain"}, {"type": "domain"}, {"type": "domain"}]],
                                        subplot_titles=["PMF", "LS-NMF", "WS-NMF"])
        profile_subplot.add_trace(pmf_pie, row=1, col=1)
        profile_subplot.add_trace(ls_pie, row=1, col=2)
        profile_subplot.add_trace(ws_pie, row=1, col=3)
        profile_subplot.layout.title = f"Factor Contributions : {feature}"
        profile_subplot.layout.height = 600
        profile_subplot.show()

    def plot_fingerprints(self, ls_nmf_r2: -1, ws_nmf_r2: -1):
        pmf_fp = []
        ws_fp = []
        ls_fp = []

        pmf_norm = self._standardize(data=self.pmf_profile_df, include_features=False)
        ls_norm = self._standardize(data=self.ls_profile_df, include_features=False)
        ws_norm = self._standardize(data=self.ws_profile_df, include_features=False)

        for factor_n in range(self.factor_n - 1, -1, -1):
            pmf_n = go.Bar(x=self.pmf_profile_df["species"],
                           y=(100 * pmf_norm[self.factor_columns[factor_n]]), name=f"Factor {factor_n + 1}",
                           legendgroup="PMF")
            ws_n = go.Bar(x=self.ws_profile_df["species"], y=(100 * ws_norm[self.ws_mapping[factor_n]]),
                          name=f"Factor {int(self.ws_mapping[factor_n].split(' ')[1])}", legendgroup="WS-NMF")
            ls_n = go.Bar(x=self.ls_profile_df["species"], y=(100 * ls_norm[self.ls_mapping[factor_n]]),
                          name=f"Factor {int(self.ls_mapping[factor_n].split(' ')[1])}", legendgroup="LS-NMF")
            pmf_fp.append(pmf_n)
            ws_fp.append(ws_n)
            ls_fp.append(ls_n)

        pmf_fig = go.Figure(data=pmf_fp)
        pmf_fig.update_layout(barmode='stack')
        pmf_fig.update_yaxes(title_text="Species Concentration %")
        pmf_fig.layout.height = 600
        pmf_fig.layout.title = "PMF Factor Fingerprints"
        pmf_fig.show()

        ls_fig = go.Figure(data=ls_fp)
        ls_fig.update_layout(barmode='stack')
        ls_fig.update_yaxes(title_text="Species Concentration %")
        ls_fig.layout.height = 600
        if ls_nmf_r2 > 0:
            ls_fig.layout.title = f"LS-NMF Factor Fingerprints - R2: {round(ls_nmf_r2, 3)}"
        else:
            ls_fig.layout.title = f"LS-NMF Factor Fingerprints"
        ls_fig.show()

        ws_fig = go.Figure(data=ws_fp)
        ws_fig.update_layout(barmode='stack')
        ws_fig.update_yaxes(title_text="Species Concentration %")
        ws_fig.layout.height = 600
        if ws_nmf_r2 > 0:
            ws_fig.layout.title = f"WS-NMF Factor Fingerprints - R2: {round(ws_nmf_r2, 3)}"
        else:
            ws_fig.layout.title = "WS-NMF Factor Fingerprints"
        ws_fig.show()

    def plot_factors(self):

        profile_subplot = make_subplots(rows=6, cols=1, specs=[[{"type": "bar"}], [{"type": "bar"}], [{"type": "bar"}],
                                                               [{"type": "bar"}], [{"type": "bar"}], [{"type": "bar"}]])

        pmf_norm = self._standardize(data=self.pmf_profile_df, include_features=False)
        ls_norm = self._standardize(data=self.ls_profile_df, include_features=False)
        ws_norm = self._standardize(data=self.ws_profile_df, include_features=False)

        for factor_n in range(0, self.factor_n):
            pmf_profile_trace = go.Bar(x=self.pmf_profile_df["species"],
                                       y=pmf_norm[self.factor_columns[factor_n]], name=f"PMF Factor {factor_n + 1}",
                                       legendgroup=f"{factor_n + 1}")
            ws_profile_trace = go.Bar(x=self.ws_profile_df["species"], y=ws_norm[self.ws_mapping[factor_n]],
                                      name=f"WS-NMF {self.ws_mapping[factor_n]}", legendgroup=f"{factor_n + 1}")
            ls_profile_trace = go.Bar(x=self.ls_profile_df["species"], y=ls_norm[self.ls_mapping[factor_n]],
                                      name=f"LS-NMF {self.ls_mapping[factor_n]}", legendgroup=f"{factor_n + 1}")

            profile_subplot.add_trace(pmf_profile_trace, row=factor_n + 1, col=1)
            profile_subplot.add_trace(ws_profile_trace, row=factor_n + 1, col=1)
            profile_subplot.add_trace(ls_profile_trace, row=factor_n + 1, col=1)

        profile_subplot.layout.title = "PMF - NMF-PY : Normalized Factor Profiles"
        profile_subplot.layout.height = 1800
        profile_subplot.show()

    def plot_feature_timeseries(self, factor_n: int, feature_n, show_input: bool = True):

        pmf_f_p = self.pmf_profile_df[self.factor_columns[factor_n]].values
        pmf_f_p = pmf_f_p.reshape(1, len(pmf_f_p))
        pmf_f_c = self.pmf_contributions_df[self.factor_columns[factor_n]].values
        pmf_f_c = pmf_f_c.reshape(len(pmf_f_c), 1)
        pmf_f_prod = np.matmul(pmf_f_c, pmf_f_p)
        pmf_f_df = pd.DataFrame(pmf_f_prod, columns=self.pmf_profile_df["species"])
        pmf_f_df["Datetime"] = self.datetimestamps

        ls_f_p = self.ls_profile_df[self.ls_mapping[factor_n]].values
        ls_f_p = ls_f_p.reshape(1, len(ls_f_p))
        ls_f_c = self.ls_contributions_df[self.ls_mapping[factor_n]].values
        ls_f_c = ls_f_c.reshape(len(ls_f_c), 1)
        ls_f_prod = np.matmul(ls_f_c, ls_f_p)
        ls_f_df = pd.DataFrame(ls_f_prod, columns=self.pmf_profile_df["species"])
        ls_f_df["Datetime"] = self.datetimestamps

        ws_f_p = self.ws_profile_df[self.ws_mapping[factor_n]].values
        ws_f_p = ws_f_p.reshape(1, len(ws_f_p))
        ws_f_c = self.ws_contributions_df[self.ws_mapping[factor_n]].values
        ws_f_c = ws_f_c.reshape(len(ws_f_c), 1)
        ws_f_prod = np.matmul(ws_f_c, ws_f_p)
        ws_f_df = pd.DataFrame(ws_f_prod, columns=self.pmf_profile_df["species"])
        ws_f_df["Datetime"] = self.datetimestamps

        if type(feature_n) == int:
            feature_n = [feature_n]
        elif type(feature_n) == str:
            if feature_n == "all":
                feature_n = range(0, self.pmf_profile_df.shape[0])

        for feature_i in feature_n:

            ts_fig = go.Figure()
            ts_fig.add_trace(
                go.Scatter(x=pmf_f_df["Datetime"], y=pmf_f_df[self.features[feature_i]], name=f"PMF - {self.features[feature_i]}"))
            ts_fig.add_trace(
                go.Scatter(x=ls_f_df["Datetime"], y=ls_f_df[self.features[feature_i]], name=f"LS - {self.features[feature_i]}"))
            ts_fig.add_trace(
                go.Scatter(x=ws_f_df["Datetime"], y=ws_f_df[self.features[feature_i]], name=f"WS - {self.features[feature_i]}"))
            if show_input:
                ts_fig.add_trace(go.Scatter(x=pmf_f_df["Datetime"], y=self.input_df[self.features[feature_i]],
                                            name=f"Input - {self.features[feature_i]}", line=dict(dash='dot')))

            ts_fig.layout.title = f"PMF - NMF-PY : Factor {factor_n + 1} Timeseries : {self.features[feature_i]}"
            ts_fig.layout.height = 600
            ts_fig.show()

    def timeseries_plot(self, feature: str = None, feature_i: int = 0):

        pmf_data_df = self._matmul(contributions=self.pmf_contributions_df[self.factor_columns],
                                   profile=self.pmf_profile_df[self.factor_columns])
        ls_data_df = self._matmul(contributions=self.ls_contributions_df[self.factor_columns],
                                  profile=self.ls_profile_df[self.factor_columns])
        ws_data_df = self._matmul(contributions=self.ws_contributions_df[self.factor_columns],
                                  profile=self.ws_profile_df[self.factor_columns])

        if feature not in self.features:
            feature = self.features[feature_i]
        pmf_ts = pmf_data_df[feature].resample("1D").mean()
        ls_ts = ls_data_df[feature].resample("1D").mean()
        ws_ts = ws_data_df[feature].resample("1D").mean()
        data_ts = self.input_df[feature].resample("1D").mean()

        ts_fig = go.Figure()
        ts_fig.add_trace(go.Scatter(x=pmf_ts.index, y=pmf_ts, name=f"PMF - {feature}"))
        ts_fig.add_trace(go.Scatter(x=ls_ts.index, y=ls_ts, name=f"LS - {feature}"))
        ts_fig.add_trace(go.Scatter(x=ws_ts.index, y=ws_ts, name=f"WS - {feature}"))
        ts_fig.add_trace(go.Scatter(x=data_ts.index, y=data_ts, name=f"Input - {feature}", line=dict(dash='dot')))

        ts_fig.layout.title = f"PMF - NMF-PY Timeseries : Feature {feature}"
        ts_fig.layout.height = 600
        ts_fig.show()

    def feature_histogram(self, feature: str = None, feature_i: int = 0, normalized: bool = False, threshold: float = 3.0):

        pmf_data_df = self._matmul(contributions=self.pmf_contributions_df[self.factor_columns],
                                   profile=self.pmf_profile_df[self.factor_columns])
        ls_data_df = self._matmul(contributions=self.ls_contributions_df[self.factor_columns],
                                  profile=self.ls_profile_df[self.factor_columns])
        ws_data_df = self._matmul(contributions=self.ws_contributions_df[self.factor_columns],
                                  profile=self.ws_profile_df[self.factor_columns])

        pmf_residuals = self.input_df.subtract(pmf_data_df)
        ls_residuals = self.input_df.subtract(ls_data_df)
        ws_residuals = self.input_df.subtract(ws_data_df)

        if feature not in self.features:
            feature = self.features[feature_i]
        pmf_his = pmf_residuals[feature]
        ls_his = ls_residuals[feature]
        ws_his = ws_residuals[feature]
        if normalized:
            pmf_trace = go.Histogram(x=pmf_his, histnorm='probability', name="PMF")
            ls_trace = go.Histogram(x=ls_his, histnorm='probability', name="LS-NMF")
            ws_trace = go.Histogram(x=ws_his, histnorm='probability', name="WS-NMF")
        else:
            pmf_trace = go.Histogram(x=pmf_his, name="PMF")
            ls_trace = go.Histogram(x=ls_his, name="LS-NMF")
            ws_trace = go.Histogram(x=ws_his, name="WS-NMF")
        table_data = [
            ["min", "mean", "max", f">|{threshold}|"],
            [pmf_his.min(), pmf_his.mean(), pmf_his.max(), pmf_his[pmf_his.abs() >= threshold].count()],
            [ls_his.min(), ls_his.mean(), ls_his.max(), ls_his[ls_his.abs() >= threshold].count()],
            [ws_his.min(), ws_his.mean(), ws_his.max(), ws_his[ws_his.abs() >= threshold].count()]
        ]
        table_trace = go.Table(header=dict(values=["metric", "PMF", "LS-NMF", "WS-NMF"]), cells=dict(values=table_data))

        hist_subplot = make_subplots(rows=4, cols=1, specs=[[{"type": "bar"}], [{"type": "bar"}], [{"type": "bar"}],
                                                            [{"type": "table"}]])
        hist_subplot.add_trace(pmf_trace, row=1, col=1)
        hist_subplot.add_trace(ls_trace, row=2, col=1)
        hist_subplot.add_trace(ws_trace, row=3, col=1)
        hist_subplot.add_trace(table_trace, row=4, col=1)
        hist_subplot.layout.title = f"Residual Analysis - {feature}"
        hist_subplot.layout.height = 800
        hist_subplot.show()


class ModelAnalysis:
    def __init__(self,
                 datahandler: DataHandler,
                 model: BatchNMF,
                 selected_model: int = None
                 ):
        self.dh = datahandler
        self.model = model
        self.selected_model = selected_model
        self.statistics = None

    def calculate_statistics(self, results=None):

        statistics = {"Features": [], "Category": [], "r2": [], "Intercept": [], "Intercept SE": [], "Slope": [],
                      "Slope SE": [], "SE": [], "SE Regression": [],
                      "Anderson Normal Residual": [], "Anderson Statistic": [],
                      "Shapiro Normal Residuals": [], "Shapiro PValue": [],
                      "KS Normal Residuals": [], "KS PValue": [], "KS Statistic": []}
        cats = copy.copy(self.dh.metrics['Category'])
        results = self.model.results[self.selected_model]['wh'] if results is None else results
        residuals = self.dh.input_data - results
        scaled_residuals = residuals / self.dh.uncertainty_data
        for feature_idx, x_label in enumerate(self.dh.features):
            observed_data = self.dh.input_data[x_label]
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

    def plot_residual_histogram(self, feature_idx: int, abs_threshold: float = 3.0, est_V = None):
        if feature_idx > self.dh.input_data.shape[1] - 1 or feature_idx < 0:
            print(f"Invalid feature index provided, must be between 0 and {self.dh.input_data.shape[1]}")
            return
        V = self.model.V[:, feature_idx]
        if est_V is None:
            est_V = self.model.results[self.selected_model]['wh'][:, feature_idx]
        else:
            est_V = est_V[:, feature_idx]
        U = self.model.U[:, feature_idx]
        feature = self.dh.features[feature_idx]

        residuals = pd.DataFrame(data={f'{feature}': (V - est_V)/U, 'datetime': self.dh.input_data.index})
        residuals_data = [residuals[feature].values]
        dist_fig = ff.create_distplot(residuals_data, ['distplot'], curve_type='normal')
        normal_x = dist_fig.data[1]['x']
        normal_y = dist_fig.data[1]['y']
        dist_fig2 = ff.create_distplot(residuals_data, ['distplot'], curve_type='kde')
        normal_x2 = dist_fig2.data[1]['x']
        normal_y2 = dist_fig2.data[1]['y']
        residual_fig = px.histogram(residuals, x=feature, histnorm='probability', marginal='box')
        residual_fig.add_trace(go.Scatter(x=normal_x, y=normal_y, mode='lines', name='Normal'))
        residual_fig.add_trace(go.Scatter(x=normal_x2, y=normal_y2, mode='lines', name='KDE'))
        residual_fig.update_layout(title=f"{feature}", xaxis_title="Scaled Residuals", yaxis_title="Percent",
                                   width=1200, height=600, showlegend=True)
        residual_fig.update_traces(marker_line_width=1, marker_line_color="white")
        residual_fig.show()

        threshold_residuals = residuals[residuals[feature].abs() >= abs_threshold]
        return threshold_residuals

    def plot_estimated_observed(self, feature_idx: int):
        if feature_idx > self.dh.input_data.shape[1] - 1 or feature_idx < 0:
            print(f"Invalid feature index provided, must between 0 and {self.dh.input_data.shape[1]}")
            return
        x_label = self.dh.input_data.columns[feature_idx]

        observed_data = self.dh.input_data[x_label]
        predicted_data = self.model.results[self.selected_model]['wh'][:, feature_idx]

        A = np.vstack([observed_data.values, np.ones(len(observed_data))]).T
        m, c = np.linalg.lstsq(A, predicted_data, rcond=None)[0]

        m1, c1 = np.linalg.lstsq(A, observed_data, rcond=None)[0]

        xy_plot = go.Figure()
        xy_plot.add_trace(go.Scatter(x=observed_data, y=predicted_data, mode='markers', name="Data"))
        xy_plot.add_trace(go.Scatter(x=observed_data, y=(m*observed_data + c),
                                     line=dict(color='red', dash='dash', width=1), name='Regression'))
        xy_plot.add_trace(go.Scatter(x=observed_data, y=(m1*observed_data + c1),
                                     line=dict(color='blue', width=1), name='One-to-One'))
        xy_plot.update_layout(title=f"Observed/Predicted Scatter Plot - {x_label}", width=800, height=600,
                              xaxis_title="Observed Concentrations", yaxis_title="Predicted Concentrations")
        xy_plot.update_xaxes(range=[0, observed_data.max() + 0.5])
        xy_plot.update_yaxes(range=[0, predicted_data.max() + 0.5])
        xy_plot.show()

    def plot_estimated_timeseries(self, feature_idx: int):
        if feature_idx > self.dh.input_data.shape[1] - 1 or feature_idx < 0:
            print(f"Invalid feature index provided, must be between 0 and {self.dh.input_data.shape[1]}")
            return
        x_label = self.dh.input_data.columns[feature_idx]

        observed_data = self.dh.input_data[x_label].values
        predicted_data = self.model.results[self.selected_model]['wh'][:, feature_idx]

        data_df = pd.DataFrame(data={"observed": observed_data, "predicted": predicted_data}, index=self.dh.input_data.index)
        data_df.index = pd.to_datetime(data_df.index)
        data_df = data_df.sort_index()
        data_df = data_df.resample('D').mean()

        ts_subplot = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.8, 0.2])

        ts_subplot.add_trace(go.Scatter(x=data_df.index, y=data_df["observed"], line=dict(width=1), mode='lines+markers', name="Observed Concentrations"), row=1, col=1)
        ts_subplot.add_trace(go.Scatter(x=data_df.index, y=data_df["predicted"], line=dict(width=1), mode='lines+markers', name="Predicted Concentrations"), row=1, col=1)
        ts_subplot.add_trace(go.Scatter(x=data_df.index, y=data_df["observed"] - data_df["predicted"], line=dict(width=1), mode='lines', name="Residuals"), row=2, col=1)

        ts_subplot.update_layout(title_text=f"{x_label} - Model {self.selected_model}", width=1200, height=800, yaxis_title="Concentrations")
        ts_subplot.show()

    def plot_factor_profile(self, factor_idx: int, H=None, W=None):
        if factor_idx > self.model.factors - 1 or factor_idx < 0:
            print(f"Invalid factor provided, must be between 0 and {self.model.factors - 1}")
            return

        factor_label = f"Factor {factor_idx}"
        if H is None:
            factors_data = self.model.results[self.selected_model]['H'][factor_idx]
            factors_sum = self.model.results[self.selected_model]['H'].sum(axis=0)
        else:
            factors_data = H[factor_idx]
            factors_sum = H.sum(axis=0)
        if W is None:
            factor_contribution = self.model.results[self.selected_model]['W'][:, factor_idx]
        else:
            factor_contribution = W[:, factor_idx]

        factor_matrix = np.matmul(factor_contribution.reshape(len(factor_contribution), 1), [factors_data])

        factor_conc_sums = factor_matrix.sum(axis=0)
        factor_conc_sums[factor_conc_sums == 0.0] = 1e-12

        norm_profile = 100 * (factors_data / factors_sum)

        norm_contr = factor_contribution / factor_contribution.mean()
        #
        data_df = pd.DataFrame(data={factor_label: norm_contr}, index=self.dh.input_data.index)
        data_df[factor_label] = norm_contr
        data_df.index = pd.to_datetime(data_df.index)
        data_df = data_df.sort_index()
        data_df = data_df.resample('D').mean()

        profile_plot = make_subplots(specs=[[{"secondary_y": True}]], rows=1, cols=1)
        profile_plot.add_trace(go.Scatter(x=self.dh.features, y=norm_profile, mode='markers', marker=dict(color='red'), name="% of Features"), secondary_y=True, row=1, col=1)
        profile_plot.add_trace(go.Bar(x=self.dh.features, y=factor_conc_sums, marker_color='rgb(158,202,225)',
                                        marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6, name='Conc. of Features'), secondary_y=False, row=1, col=1)
        profile_plot.update_yaxes(title_text="Conc. of Features", secondary_y=False, row=1, col=1,
                                  type="log",
                                  range=[0, np.log10(factor_conc_sums).max()]
                                  )
        profile_plot.update_yaxes(title_text="% of Features", secondary_y=True, row=1, col=1, range=[0, 100])
        profile_plot.update_layout(title=f"Factor Profile - Model {self.selected_model} - Factor {factor_idx}", width=1200, height=600)
        profile_plot.show()

        contr_plot = go.Figure()
        contr_plot.add_trace(go.Scatter(x=data_df.index, y=data_df[factor_label], mode='lines+markers', name="Normalized Contributions", line=dict(color='blue')))
        contr_plot.update_layout(title=f"Factor Contributions - Model {self.selected_model} - Factor {factor_idx}",
                                 width=1200, height=600, showlegend=True,
                                 legend=dict(orientation="h", xanchor="right", yanchor="bottom", x=1, y=1.02))
        contr_plot.update_yaxes(title_text="Normalized Contributions")
        contr_plot.show()

    def plot_factor_fingerprints(self):
        factors_data = self.model.results[self.selected_model]['H']
        normalized_factors_data = 100 * (factors_data / factors_data.sum(axis=0))

        fig = go.Figure()
        for idx in range(self.model.factors-1, -1, -1):
            fig.add_trace(go.Bar(name=f"Factor {idx}", x=self.dh.features, y=normalized_factors_data[idx]))
        fig.update_layout(title=f"Factor Fingerprints - Model {self.selected_model}",
                          width=1200, height=800, barmode='stack')
        fig.update_yaxes(title_text="% Feature Concentration", range=[0, 100])
        fig.show()

    def plot_g_space(self, factor_1: int, factor_2: int):
        if factor_1 > self.model.factors - 1 or factor_1 < 0:
            print(f"Invalid factor_1 provided, must be between 0 and {self.model.factors - 1}")
            return
        if factor_2 > self.model.factors - 1 or factor_2 < 0:
            print(f"Invalid factor_2 provided, must be between 0 and {self.model.factors - 1}")
            return

        factors_contr = self.model.results[self.selected_model]['W']
        normalized_factors_contr = factors_contr / factors_contr.sum(axis=0)

        fig = go.Figure(data=go.Scatter(
            x=normalized_factors_contr[:, factor_1],
            y=normalized_factors_contr[:, factor_2], mode='markers')
        )
        fig.update_layout(title=f"G-Space Plot - Model {self.selected_model}", width=800, height=800)
        fig.update_yaxes(title_text=f"Factor {factor_2} Contributions (avg=1)")
        fig.update_xaxes(title_text=f"Factor {factor_1} Contributions (avg=1)")
        fig.show()

    def plot_factor_contributions(self, feature_idx: int, contribution_threshold: float = 0.05):
        if feature_idx > self.dh.input_data.shape[1] - 1 or feature_idx < 0:
            print(f"Invalid feature index provided, must not be negative and be less than {self.dh.input_data.shape[1]-1}")
            return
        if 50.0 > contribution_threshold < 0:
            print(f"Invalid contribution threshold provided, must be between 0.0 and 50.0")
            return
        x_label = self.dh.input_data.columns[feature_idx]

        factors_data = self.model.results[self.selected_model]['H']
        normalized_factors_data = 100 * (factors_data / factors_data.sum(axis=0))

        feature_contr = normalized_factors_data[:, feature_idx]
        feature_contr_inc = []
        feature_contr_labels = []
        feature_legend = {}
        for idx in range(feature_contr.shape[0]-1, -1, -1):
            if feature_contr[idx] > contribution_threshold:
                feature_contr_inc.append(feature_contr[idx])
                feature_contr_labels.append(f"Factor {idx}")
                feature_legend[f"Factor {idx}"] = f"Factor {idx} = {factors_data[idx:, feature_idx]}"
        feature_fig = go.Figure(data=[go.Pie(labels=feature_contr_labels, values=feature_contr_inc)])
        feature_fig.update_layout(title=f"{x_label} - Model {self.selected_model}", width=1200, height=600,
                                  legend_title_text=f"Factor Contribution > {0.05}%")
        feature_fig.show()

        factors_contr = self.model.results[self.selected_model]['W']
        normalized_factors_contr = 100 * (factors_contr / factors_contr.sum(axis=0))
        factor_labels = [f"Factor {i}" for i in range(normalized_factors_contr.shape[1])]
        contr_df = pd.DataFrame(normalized_factors_contr, columns=factor_labels)
        contr_df.index = pd.to_datetime(self.dh.input_data.index)
        contr_df = contr_df.sort_index()
        contr_df = contr_df.resample('D').mean()

        contr_fig = go.Figure()
        for factor in factor_labels:
            contr_fig.add_trace(go.Scatter(x=contr_df.index, y=contr_df[factor], mode='lines+markers', name=factor))
        converged = "Converged Model" if self.model.results[self.selected_model]['converged'] else "Unconverged Model"
        contr_fig.update_layout(title=f"Factor Contributions (avg=1) From Base Model #{self.selected_model} ({converged})",
                                width=1200, height=600,
                                legend=dict(orientation="h", xanchor="right", yanchor="bottom", x=1, y=1.02))
        contr_fig.update_yaxes(title_text="Normalized Contribution")
        contr_fig.show()
