import logging
import numpy as np
import plotly.graph_objects as go
from src.error.bootstrap import Bootstrap
from src.error.displacement import Displacement

logger = logging.getLogger("NMF")
logger.setLevel(logging.INFO)


class Error:

    def __init__(self, bs: Bootstrap = None, disp: Displacement = None):
        self.bs = bs
        self.disp = disp
        if bs is not None:
            self.feature_labels = self.bs.feature_labels
            self.model_selected = self.bs.model_selected
        if disp is not None:
            self.feature_labels = self.disp.feature_labels
            self.model_selected = self.disp.selected_model

    def plot_summary(self, factor_i: int):
        disp_dQ = 4
        if self.bs is None and self.disp is None:
            logging.error("Must complete and provide an instance of either displacement or bootstrap or both")
            return
        error_plot = go.Figure()

        if self.bs is not None:
            base_Wi = self.bs.base_W[:, factor_i]
            base_Wi = base_Wi.reshape(len(base_Wi), 1)
            base_Hi = [self.bs.base_H[factor_i]]
            base_sums = np.matmul(base_Wi, base_Hi).sum(axis=0)
            base_sums[base_sums < 1e-4] = 1e-4
            bs_data = np.array(self.bs.bs_factor_contributions[factor_i])
            bs_data[bs_data < 1e-4] = 1e-4
            error_plot.add_trace(
                go.Bar(x=self.feature_labels, y=bs_data.max(axis=0) - bs_data.min(axis=0), base=bs_data.min(axis=0),
                       name="BS", marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)'))
        if self.disp is not None:
            base_Wi = self.disp.W[:, factor_i]
            base_Wi = base_Wi.reshape(len(base_Wi), 1)
            base_Hi = [self.disp.H[factor_i]]
            base_sums = np.matmul(base_Wi, base_Hi).sum(axis=0)
            base_sums[base_sums < 1e-4] = 1e-4

            selected_data = self.disp.compiled_results.loc[self.disp.compiled_results["factor"] == factor_i].loc[
                self.disp.compiled_results["dQ"] == disp_dQ]
            conc = selected_data["conc"]
            conc[conc < 1e-4] = 1e-4
            error_plot.add_trace(
                go.Bar(x=self.feature_labels, y=selected_data.conc_max - selected_data.conc_min,
                       base=selected_data.conc_min, name="Disp",
                       marker_color='rgb(171,245,106)', marker_line_color='rgb(128,216,52)'))
        error_plot.add_trace(go.Scatter(x=self.feature_labels, y=base_sums, mode='markers', name="Base",
                                    marker=dict(size=12, color="red", symbol="line-ew", line_width=1,
                                                line_color="red")))
        error_plot.update_layout(
            title=f"Error Estimation Concentration Summary - Model {self.model_selected} - Factor {factor_i}", width=1200,
            height=600, showlegend=True, barmode='group')
        error_plot.update_traces(selector=dict(type="bar"), hovertemplate='Max: %{value}<br>Min: %{base}')
        error_plot.update_yaxes(title_text="Concentration (log)", type="log")
        error_plot.show()


