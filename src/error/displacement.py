import logging
import copy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
from src.utils import q_loss, compare_all_factors
from src.model.nmf import NMF


logger = logging.getLogger("NMF")
logger.setLevel(logging.INFO)


class Displacement:

    dQmax = [32, 16, 8, 4]

    def __init__(self, batch_nmf, feature_labels, selected_model=None, max_search: int = 50, threshold_dQ: float = 0.1):
        self.batch_nmf = batch_nmf

        self.selected_model = selected_model if selected_model is not None else self.batch_nmf.best_model
        self.V = batch_nmf.V
        self.U = batch_nmf.U
        self.H = batch_nmf.results[self.selected_model]["H"]
        self.W = batch_nmf.results[self.selected_model]["W"]
        self.base_Q = batch_nmf.results[self.selected_model]["Q(true)"]
        self.feature_labels = feature_labels

        self.max_search = max_search
        self.threshold_dQ = threshold_dQ

        self.increase_results = {}
        self.decrease_results = {}

        self.swap_table = np.zeros(shape=(len(self.dQmax), self.H.shape[0]))
        self.count_table = np.zeros(shape=(len(self.dQmax), self.H.shape[0]))
        self.compiled_results = None

    def run(self):
        self._increase_disp()
        self._decrease_disp()
        self._compile_results()

    def summary(self):
        largest_dQ_inc = self.compiled_results["dQ_drop"].max()
        largest_dQ_dec = self.compiled_results["dQ_drop"].min()
        largest_dQ_change = largest_dQ_inc if np.abs(largest_dQ_inc) > np.abs(largest_dQ_dec) else largest_dQ_dec
        table_labels = ["dQ Max"]
        for i in range(self.batch_nmf.factors):
            table_labels.append(f"Factor {i}")
        table_data = np.round(100 * (self.swap_table/self.count_table), 2)
        dq_list = list(reversed(self.dQmax))
        dq_list = np.reshape(dq_list, newshape=(len(dq_list), 1))
        table_data = np.hstack((dq_list, table_data))
        table_plot = go.Figure(data=[go.Table(
            header=dict(values=table_labels),
            cells=dict(values=table_data.T)
        )])
        table_plot.update_layout(title=f"Swap % - Largest dQ Change: {round(largest_dQ_change, 2)}",
                                 width=600, height=200, margin={'t': 50, 'l': 25, 'b': 10, 'r': 25})
        table_plot.show()

    def plot_results(self, factor: int, dQ: int = 4):
        selected_data = self.compiled_results.loc[self.compiled_results["factor"] == factor].loc[
            self.compiled_results["dQ"] == dQ]

        disp_profile = go.Figure()
        disp_profile.add_trace(
            go.Scatter(x=selected_data.feature, y=100 * selected_data.profile, mode='markers', marker=dict(color='blue'),
                       name="Base Run"))
        disp_profile.add_trace(go.Bar(x=selected_data.feature,
                                      y=100 * (selected_data.profile_max - selected_data.profile_min),
                                      base=100 * selected_data.profile_min, name="DISP Range"))
        disp_profile.update_traces(selector=dict(type="bar"), marker_color='rgb(158,202,225)',
                                   marker_line_color='rgb(8,48,107)',
                                   marker_line_width=1.5, opacity=0.6)
        disp_profile.update_layout(
            title=f"Variability in Percentage of Features - Model {self.selected_model} - Factor {factor} - dQ {dQ}",
            width=1200, height=600, showlegend=True)
        disp_profile.update_yaxes(title_text="Percentage", range=[0, 100])
        disp_profile.update_traces(selector=dict(type="bar"), hovertemplate='Max: %{value}<br>Min: %{base}')
        disp_profile.show()

        conc = selected_data["conc"]
        conc[conc < 1e-4] = 1e-4
        disp_conc = go.Figure()
        disp_conc.add_trace(
            go.Scatter(x=selected_data.feature, y=conc, mode='markers', marker=dict(color='blue'),
                       name="Base Run"))
        disp_conc.add_trace(
            go.Bar(x=selected_data.feature, y=selected_data.conc_max - selected_data.conc_min,
                   base=selected_data.conc_min,
                   name="DISP Range"))
        disp_conc.update_traces(selector=dict(type="bar"), marker_color='rgb(158,202,225)',
                                marker_line_color='rgb(8,48,107)',
                                marker_line_width=1.5, opacity=0.6, hovertemplate='Max: %{value}<br>Min: %{base}')
        disp_conc.update_layout(
            title=f"Variability in Concentration of Features - Model {self.selected_model} - Factor {factor} - dQ {dQ}",
            width=1200, height=600, showlegend=True)
        disp_conc.update_yaxes(title_text="Concentration (log)", type="log")
        disp_conc.show()

    def _increase_disp(self):
        print("DISP - Testing increasing value changes to H")
        for factor_i in tqdm(range(self.H.shape[0]), desc=" Factors", position=0):
            factor_results = {}
            for feature_j in tqdm(range(self.H.shape[1]), desc=f"Factor {factor_i} - Features", position=0, leave=True):
                new_H = copy.copy(self.H)
                high_modifier = 2.0
                high_found = False
                i_results = {}
                high_search_i = 0
                max_dQ = 0
                max_high_search = 5000

                while not high_found:
                    new_value = self.H[factor_i, feature_j] * high_modifier
                    new_H[factor_i, feature_j] = new_value
                    disp_i_Q = q_loss(V=self.V, U=self.U, W=self.W, H=new_H)
                    dQ = np.abs(self.base_Q - disp_i_Q)
                    if dQ < self.dQmax[0]:
                        high_modifier *= 2
                    else:
                        high_found = True
                    high_search_i += 1
                    if dQ > max_dQ:
                        max_dQ = dQ
                    if high_search_i >= max_high_search:
                        logging.warn(f"Failed to find upper bound modifier within search limit. "
                                     f"Factor: {factor_i}, Feature: {feature_j}, Max iterations: {max_high_search}, max dQ: {max_dQ}")
                        break
                for i in range(len(self.dQmax)):
                    low_modifier = 1.0
                    modifier = (high_modifier + low_modifier) / 2.0
                    value_found = False
                    search_i = 0
                    while not value_found:
                        new_H = copy.copy(self.H)
                        new_value = self.H[factor_i, feature_j] * modifier
                        new_H[factor_i, feature_j] = new_value
                        disp_i_Q = q_loss(V=self.V, U=self.U, W=self.W, H=new_H)
                        dQ = np.abs(self.base_Q - disp_i_Q)
                        if dQ > self.dQmax[i]:
                            high_modifier = modifier
                            modifier = (modifier + low_modifier) / 2.0
                        elif dQ < self.dQmax[i] - self.threshold_dQ:
                            low_modifier = modifier
                            modifier = (high_modifier + modifier) / 2.0
                        else:
                            value_found = True
                        search_i += 1
                        if dQ > max_dQ:
                            max_dQ = dQ
                    disp_i_nmf = NMF(V=self.V, U=self.U,
                                     factors=self.batch_nmf.factors, method=self.batch_nmf.method,
                                     seed=self.batch_nmf.seed, optimized=self.batch_nmf.optimized, verbose=False)
                    disp_i_nmf.initialize(H=new_H)
                    disp_i_nmf.train(max_iter=self.batch_nmf.max_iter, converge_delta=self.batch_nmf.converge_delta,
                                     converge_n=self.batch_nmf.converge_n, robust_mode=False)
                    factor_swap = compare_all_factors(disp_i_nmf.H, self.H)
                    scaled_profiles = new_H / new_H.sum(axis=0)
                    percent = scaled_profiles[factor_i, feature_j]
                    factor_W = self.W[:, factor_i]
                    factor_matrix = np.matmul(factor_W.reshape(len(factor_W), 1), [new_H[factor_i]])
                    factor_conc_sum = factor_matrix.sum(axis=0)
                    factor_conc_i = factor_conc_sum[feature_j]
                    i_results[self.dQmax[i]] = {"dQ": dQ, "value": new_value, "percent": percent,
                                                "search steps": search_i, "swap": factor_swap, "conc": factor_conc_i,
                                                "Q_drop": self.base_Q - disp_i_nmf.Qtrue}
                factor_results[f"feature-{feature_j}"] = i_results
            self.increase_results[f"factor-{factor_i}"] = factor_results

    def _decrease_disp(self):
        print("DISP - Testing decreasing value changes to H")
        for factor_i in tqdm(range(self.H.shape[0]), desc=" Factors", position=0):
            factor_results = {}
            for feature_j in tqdm(range(self.H.shape[1]), desc=f" Factor {factor_i} - Features", position=0, leave=True):
                i_results = {}
                for i in range(len(self.dQmax)):
                    high_mod = 1.0
                    low_mod = 0.0
                    modifier = (high_mod + low_mod) / 2
                    value_found = False
                    new_H = None
                    max_dQ = 0
                    search_i = 0
                    p_mod = 0.0
                    while not value_found:
                        new_H = copy.copy(self.H)
                        new_value = self.H[factor_i, feature_j] * modifier
                        new_H[factor_i, feature_j] = new_value
                        disp_i_Q = q_loss(V=self.V, U=self.U, W=self.W, H=new_H)
                        dQ = np.abs(self.base_Q - disp_i_Q)
                        if dQ > self.dQmax[i]:
                            low_mod = modifier
                            modifier = (modifier + high_mod) / 2
                        elif dQ < self.dQmax[i] - self.threshold_dQ:
                            high_mod = modifier
                            modifier = (low_mod + modifier) / 2
                        else:
                            value_found = True
                        if np.abs(
                                p_mod - modifier) <= 1e-8:  # small value, considered zero. Or no change in the modifier.
                            value_found = True
                        search_i += 1
                        if dQ > max_dQ:
                            max_dQ = dQ
                        p_mod = modifier
                    disp_i_nmf = NMF(V=self.V, U=self.U,
                                     factors=self.batch_nmf.factors, method=self.batch_nmf.method,
                                     seed=self.batch_nmf.seed, optimized=self.batch_nmf.optimized, verbose=False)
                    disp_i_nmf.initialize(H=new_H)
                    disp_i_nmf.train(max_iter=self.batch_nmf.max_iter, converge_delta=self.batch_nmf.converge_delta,
                                     converge_n=self.batch_nmf.converge_n, robust_mode=False)
                    factor_swap = compare_all_factors(disp_i_nmf.H, self.H)
                    scaled_profiles = new_H / new_H.sum(axis=0)
                    percent = scaled_profiles[factor_i, feature_j]
                    factor_W = self.W[:, factor_i]
                    factor_matrix = np.matmul(factor_W.reshape(len(factor_W), 1), [new_H[factor_i]])
                    factor_conc_sum = factor_matrix.sum(axis=0)
                    factor_conc_i = factor_conc_sum[feature_j]
                    i_results[self.dQmax[i]] = {"dQ": dQ, "value": new_value, "percent": percent,
                                                "search steps": search_i, "swap": factor_swap, "conc": factor_conc_i,
                                                "Q_drop": self.base_Q - disp_i_nmf.Qtrue}
                factor_results[f"feature-{feature_j}"] = i_results
            self.decrease_results[f"factor-{factor_i}"] = factor_results

    def _compile_results(self):
        scaled_profiles = self.H / self.H.sum(axis=0)
        compiled_data = {"dQ":[], "factor":[], "feature":[], "profile":[], "profile_max":[], "profile_min":[], "conc":[],
                       "conc_max":[], "conc_min":[], "dQ_drop": []}
        for dQ in self.dQmax:
            for factor_i in range(self.H.shape[0]):
                factor_W = self.W[:, factor_i]
                factor_matrix = np.matmul(factor_W.reshape(len(factor_W), 1), [self.H[factor_i]])
                factor_conc = factor_matrix.sum(axis=0)
                for feature_j in range(self.H.shape[1]):
                    feature_inc_results = self.increase_results[f"factor-{factor_i}"][f"feature-{feature_j}"][dQ]
                    feature_dec_results = self.decrease_results[f"factor-{factor_i}"][f"feature-{feature_j}"][dQ]
                    compiled_data["profile_max"].append(feature_inc_results["percent"])
                    compiled_data["profile_min"].append(feature_dec_results["percent"])
                    compiled_data["profile"].append(scaled_profiles[factor_i, feature_j])
                    conc_max = feature_inc_results["conc"]
                    compiled_data["conc_max"].append(conc_max if conc_max > 1e-4 else 1e-4)
                    conc_min = feature_dec_results["conc"]
                    compiled_data["conc_min"].append(conc_min if conc_min > 1e-4 else 1e-4)
                    compiled_data["conc"].append(factor_conc[feature_j])
                    inc_dQ = feature_inc_results["Q_drop"]
                    dec_dQ = feature_dec_results["Q_drop"]
                    compiled_data["dQ_drop"].append(inc_dQ if np.abs(inc_dQ) > np.abs(dec_dQ) else dec_dQ)
                    compiled_data["dQ"].append(dQ)
                    compiled_data["factor"].append(factor_i)
                    compiled_data["feature"].append(self.feature_labels[feature_j])
        self.compiled_results = pd.DataFrame(data=compiled_data)

        factor_i = 0
        for factor, factor_results in self.increase_results.items():
            for feature, feature_results in factor_results.items():
                dQ_i = 0
                for dQ, dQ_results in feature_results.items():
                    self.count_table[dQ_i, factor_i] += 1
                    if dQ_results["swap"]:
                        self.swap_table[dQ_i, factor_i] += 1
                    dQ_i += 1
            factor_i += 1
        factor_i = 0
        for factor, factor_results in self.decrease_results.items():
            for feature, feature_results in factor_results.items():
                dQ_i = 0
                for dQ, dQ_results in feature_results.items():
                    self.count_table[dQ_i, factor_i] += 1
                    if dQ_results["swap"]:
                        self.swap_table[dQ_i, factor_i] += 1
                    dQ_i += 1
            factor_i += 1
