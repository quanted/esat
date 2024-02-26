from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import plotly.graph_objects as go


class CompareAnalyzer:
    """
    Compare ESAT output with the PMF5 output.
    """

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
                                        subplot_titles=["PMF", "LS-NMF ", "WS-NMF"])
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
