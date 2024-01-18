import sys
import os
import types

module_path = os.path.abspath(os.path.join('..', "nmf_py"))
sys.path.append(module_path)

from src.model.nmf import NMF
from src.data.datahandler import DataHandler
from src.error.bootstrap import Bootstrap
from src.utils import q_loss, qr_loss, np_encoder
from tqdm import trange
from datetime import datetime
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import logging
import copy
import pickle
import json
import time


logger = logging.getLogger("NMF")
logger.setLevel(logging.INFO)


class Fpeak:
    """

    """
    def __init__(self, base_model: NMF, data_handler: DataHandler, fpeaks: list = None, s: float = 0.1, S: list = None):

        self.base = base_model
        self.dh = data_handler
        self.fpeaks = fpeaks
        if self.fpeaks is None:
            self.fpeaks = (0.5, -0.5, 1.0, -1.0, 1.5)
        else:
            if 0.0 in self.fpeaks:
                logger.error(f"Invalid value in fpeak list, 0.0 is not a valid value for fpeak. "
                             f"Use values != 0.0, either less than or greater than zero.")
                self.fpeaks.remove(0.0)
        self.s = s
        self.S = S
        if self.S is None:
            self.S = np.ones(shape=self.base.W.shape[0]) * self.s
        else:
            if len(self.S) != self.base.W.shape[0]:
                logger.error(f"Invalid S array. Array must be of size: {self.base.W.shape[0]}, "
                             f"provided size: {len(self.S)}")
                logger.info(f"Using default values for S array.")
                self.S = np.ones(shape=self.base.W.shape[0]) * self.s

        self.V = self.base.V
        self.U = self.base.U
        self.We = self.base.We
        self.WeV = np.multiply(self.We, self.V)
        self.factors = self.base.factors

        self.base_W = self.base.W
        self.base_H = self.base.H

        self.results_df = None
        self.results = {}

        self.bs_results = {}
        self.bs_seed = None
        self.bs_n = None
        self.bs_block_size = None
        self.bs_threshold = None

    def qaux_loss(self, Wp, D):
        r = np.square(self.base_W + D - Wp)
        qaux = np.divide(r.sum(axis=1), np.square(self.S))
        return np.sum(qaux)

    def ls_nmf_w(self, W):
        WH = np.matmul(W, self.base_H)
        W_num = np.matmul(self.WeV, self.base_H.T)
        W_den = np.matmul(np.multiply(self.We, WH), self.base_H.T)
        W = np.multiply(W, np.divide(W_num, W_den))
        return W

    def ls_nmf_h(self, W, H):
        WH = np.matmul(W, H)
        H_num = np.matmul(W.T, self.WeV)
        H_den = np.matmul(W.T, np.multiply(self.We, WH))
        H = np.multiply(H, np.divide(H_num, H_den))
        return H

    def run(self, max_iter: int = 5000, converge_delta: float = 1e-4, converge_n: int = 20):
        for fp in self.fpeaks:
            phi = np.full(shape=(self.factors, self.factors), fill_value=fp)
            for i in range(self.factors):
                phi[i, i] = 0.0
            t_iter = trange(max_iter, desc=f"W Update - Q(robust): NA, Q(main): NA, Q(aux): NA",
                            position=0, leave=True)
            W_i = self.base_W
            H_i = self.base_H
            Qaux_i, Qm = None, None
            converged = False
            qa_list = []
            qm_list = []
            qd_list = []
            for i in t_iter:
                WtW = np.matmul(W_i.transpose(), W_i)
                D = np.matmul(np.matmul(W_i, np.linalg.inv(WtW)), phi)
                W_d = D + W_i
                W_d[W_d < 0.0] = 0.0
                W_i = self.ls_nmf_w(W=W_d)
                Qm_i = q_loss(V=self.V, U=self.U, W=W_i, H=H_i)
                Qmr_i, _ = qr_loss(V=self.V, U=self.U, W=W_i, H=H_i)
                Qaux_i = self.qaux_loss(Wp=W_i, D=D)
                Qm = Qm_i + Qaux_i
                t_iter.set_description(f"W Update - Q(robust): {round(Qmr_i, 4)}, Q(main): {round(Qm, 4)}, "
                                       f"Q(aux): {round(Qaux_i, 4)}")
                qa_list.append(Qaux_i)
                qm_list.append(Qm)
                if len(qa_list) > converge_n:
                    qa_list.pop(0)
                    qm_list.pop(0)
                    qd_list.append(qm_list[-1] - qm_list[-2])
                    if len(qd_list) > converge_n:
                        qd_list.pop(0)
                    if np.abs(qa_list[0] - qa_list[-1]) <= converge_delta:
                        converged = True
                        break
            t_iter = trange(max_iter, desc=f"H Update - Q(robust): NA, Q(main): NA, Q(aux): NA",
                            position=0, leave=True)
            qh_list = []
            for i in t_iter:
                H_i = self.ls_nmf_h(W=W_i, H=H_i)
                Qm_i = q_loss(V=self.V, U=self.U, W=W_i, H=H_i)
                Qr_i, _ = qr_loss(V=self.V, U=self.U, W=W_i, H=H_i)
                qh_list.append(Qm_i)
                t_iter.set_description(f"H Update - Q(robust): {round(Qr_i, 2)}, Q(main): {round(Qm_i, 2)}")
                if len(qh_list) > converge_n:
                    qh_list.pop(0)
                    if np.abs(qh_list[0] - qh_list[-1]) <= 1e-2:
                        break
            nmf_i = NMF(V=self.V, U=self.U, factors=self.factors, method=self.base.method, seed=self.base.seed,
                        optimized=self.base.optimized, verbose=self.base.verbose)
            nmf_i.initialize(H=H_i, W=W_i)
            nmf_i.converged = converged
            nmf_i.Qtrue = q_loss(V=self.V, U=self.U, W=nmf_i.W, H=nmf_i.H)
            nmf_i.Qrobust, _ = qr_loss(V=self.V, U=self.U, W=nmf_i.W, H=nmf_i.H)
            nmf_i.metadata["completion_date"] = datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S %Z")
            nmf_i.metadata["max_iterations"] = int(max_iter)
            nmf_i.metadata["converge_delta"] = float(converge_delta)
            nmf_i.metadata["converge_n"] = int(converge_n)
            nmf_i.metadata["robust_mode"] = False
            Qm_2 = nmf_i.Qrobust + Qaux_i
            self.results[str(fp)] = {
                'model': nmf_i,
                'Strength': str(fp),
                'Q(Aux)': Qaux_i,
                'Q(M)': Qm_2,
            }
            del nmf_i
        self._compile_results()

    def run_bs(self, bootstrap_n: int = 20, block_size: int = None, threshold: float = 0.6, seed: int = None):
        self.bs_n = bootstrap_n
        self.bs_block_size = block_size
        self.bs_threshold = threshold
        if len(self.results.keys()) == 0:
            logger.error("Fpeak models must be created before running bootstrap on the results.")
            return
        if seed is None:
            seed = self.base.seed
        self.bs_seed = seed
        for fp, fp_result in self.results.items():
            fp_model = fp_result["model"]
            fp_bs = Bootstrap(nmf=fp_model, feature_labels=self.dh.features, model_selected=fp, bootstrap_n=bootstrap_n, block_size=block_size, threshold=threshold, seed=seed)
            fp_bs.run()
            self.bs_results[fp] = fp_bs

    def display_bs_results(self, fpeak, factor_idx):
        if factor_idx is not None:
            if factor_idx > self.base.factors or factor_idx < 1:
                logger.warn(f"Invalid factor_idx provided, must be between 1 and {self.base.factors}")
                return
        if fpeak not in self.fpeaks:
            fpeak = self.fpeaks[0]
        fpeak = str(fpeak)
        self.bs_results[fpeak].summary()
        self.bs_results[fpeak].plot_results(factor=factor_idx)

    def _compile_results(self):
        df_data = {'Strength': [], 'dQ(Robust)': [], 'Q(Robust)': [], '% dQ(Robust)': [], 'Q(Aux)': [], 'Q(True)': [],
                   'Converged': []}
        for fp, data in self.results.items():
            df_data['Strength'].append(fp)
            df_data['dQ(Robust)'].append(round(data['model'].Qrobust - self.base.Qrobust, 2))
            df_data['Q(Robust)'].append(round(data['model'].Qrobust, 2))
            df_data['% dQ(Robust)'].append(round(100 * ((data['model'].Qrobust / self.base.Qrobust)-1), 2))
            df_data['Q(Aux)'].append(round(data['Q(Aux)'], 2))
            df_data['Q(True)'].append(round(data['model'].Qtrue, 2))
            df_data['Converged'].append(data['model'].converged)
        self.results_df = pd.DataFrame(data=df_data)

    def plot_profile_contributions(self, factor_idx, fpeak):
        if fpeak not in self.fpeaks:
            logger.warn(f"fpeak is not a value calculated. Provided fpeak: {fpeak}. "
                        f"Valid fpeak values are: {self.fpeaks}")
            logger.info(f"Defaulting to first fpeak value. Fpeak={self.fpeaks[0]}")
            fpeak = self.fpeaks[0]
        if factor_idx is not None:
            if factor_idx > self.base.factors or factor_idx < 1:
                logger.warn(f"Invalid factor_idx provided, must be between 1 and {self.base.factors}")
                return
        fpeak = str(fpeak)
        self.plot_profile(factor_idx=factor_idx, fpeak=fpeak)
        self.plot_contributions(factor_idx=factor_idx, fpeak=fpeak)

    def plot_profile(self, factor_idx, fpeak):
        factor_label = factor_idx
        factor_idx = factor_idx - 1

        selected_fpeak = self.results[fpeak]
        b_W = self.base_W[:, factor_idx]
        b_H = self.base_H[factor_idx]
        b_H_sum = self.base_H.sum(axis=0)
        b_factor_matrix = np.matmul(b_W.reshape(len(b_W), 1), [b_H])

        b_factor_conc_sum = b_factor_matrix.sum(axis=0)
        b_factor_conc_sum[b_factor_conc_sum == 0.0] = 1e-12

        i_W = selected_fpeak['model'].W[:, factor_idx]
        i_H = selected_fpeak['model'].H[factor_idx]
        i_H_sum = selected_fpeak['model'].H.sum(axis=0)
        i_factor_matrix = np.matmul(i_W.reshape(len(i_W), 1), [i_H])

        i_factor_conc_sum = i_factor_matrix.sum(axis=0)
        i_factor_conc_sum[i_factor_conc_sum == 0] = 1e-12

        b_norm_H = np.round(100 * (b_H / b_H_sum), 2)
        i_norm_H = np.round(100 * (i_H / i_H_sum), 2)

        fig = make_subplots(specs=[[{"secondary_y": True}]], rows=1, cols=1)
        fig.add_trace(go.Scatter(x=self.dh.features, y=b_norm_H, mode="markers", marker=dict(color='gray'),
                                 name="Base % of Features", opacity=0.8), secondary_y=True, row=1, col=1)
        fig.add_trace(go.Scatter(x=self.dh.features, y=i_norm_H, mode="markers", marker=dict(color='red'),
                                 name="FPeak % of Features", opacity=0.6), secondary_y=True, row=1, col=1)
        fig.add_trace(go.Bar(x=self.dh.features, y=b_factor_conc_sum, marker_color='rgb(203,203,203)',
                             marker_line_color='rgb(186,186,186)', marker_line_width=1.5, opacity=0.6,
                             name='Base Conc. of Features'), secondary_y=False, row=1, col=1)
        fig.add_trace(go.Bar(x=self.dh.features, y=i_factor_conc_sum, marker_color='rgb(134,236,168)',
                             marker_line_color='rgb(125,220,157)', marker_line_width=1.5, opacity=0.6,
                             name='FPeak Conc. of Features'), secondary_y=False, row=1, col=1)
        fig.update_layout(width=1200, height=600, title=f"Fpeak Factor Profile - FP={fpeak} - Factor {factor_label}",
                          barmode='group', scattermode='group', hovermode="x unified")
        fig.update_yaxes(type="log", secondary_y=False, range=[0, np.log10(b_factor_conc_sum).max()], row=1, col=1)
        fig.update_yaxes(secondary_y=True, range=[0, 100])
        fig.show()

    def plot_contributions(self, factor_idx, fpeak):
        factor_label = f"Factor {factor_idx}"
        factor_idx = factor_idx - 1

        selected_fpeak = self.results[fpeak]

        b_W = self.base.W[:, factor_idx]
        i_W = selected_fpeak['model'].W[:, factor_idx]

        b_norm_contr = b_W / b_W.mean()
        b_data_df = copy.copy(self.dh.input_data)
        b_data_df[factor_label] = b_norm_contr
        b_data_df.index = pd.to_datetime(b_data_df.index)
        b_data_df = b_data_df.sort_index()
        b_data_df = b_data_df.resample('D').mean()

        i_norm_contr = i_W / i_W.mean()
        i_data_df = copy.copy(self.dh.input_data)
        i_data_df[factor_label] = i_norm_contr
        i_data_df.index = pd.to_datetime(i_data_df.index)
        i_data_df = i_data_df.sort_index()
        i_data_df = i_data_df.resample('D').mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=b_data_df.index, y=b_data_df[factor_label], mode='lines+markers',
                                 marker_color='rgb(186,186,186)', name="Base Factor Contributions"))
        fig.add_trace(go.Scatter(x=i_data_df.index, y=i_data_df[factor_label], mode='lines+markers',
                                 marker_color='rgb(125,220,157)', name="FPeak Factor Contributions"))
        fig.update_layout(width=1200, height=800,
                          title=f"Fpeak Factor Contributions - Fpeak={fpeak} - Factor {factor_label}",
                          hovermode="x unified")
        fig.update_yaxes(title_text="Factor Contributions")
        fig.show()

    def plot_factor_fingerprints(self, fpeak):
        if fpeak not in self.fpeaks:
            logger.warn(f"fpeak is not a value calculated. Provided fpeak: {fpeak}. "
                        f"Valid fpeak values are: {self.fpeaks}")
            logger.info(f"Defaulting to first fpeak value. Fpeak={self.fpeaks[0]}")
            fpeak = self.fpeaks[0]
        fpeak = str(fpeak)
        selected_fpeak = self.results[fpeak]
        b_H = self.base_H
        fp_H = selected_fpeak['model'].H

        b_normalized = 100 * (b_H / b_H.sum(axis=0))
        fp_normalized = 100 * (fp_H / fp_H.sum(axis=0))

        fp_factors_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       subplot_titles=("Base Profile", "Fpeak Profile"), vertical_spacing=0.075)
        colors = px.colors.sequential.Viridis_r
        for idx in range(self.base.factors - 1, -1, -1):
            fp_factors_fig.add_trace(go.Bar(name=f"Base Factor {idx + 1}", x=self.dh.features, y=b_normalized[idx],
                                            marker_color=colors[idx]), row=1, col=1)
            fp_factors_fig.add_trace(
                go.Bar(name=f"Fpeak Factor {idx + 1}", x=self.dh.features, y=fp_normalized[idx],
                       marker_color=colors[idx]), row=2, col=1)
        fp_factors_fig.update_layout(title=f"Fpeak Factor Fingerprints - Fpeak={fpeak}", width=1200, height=800,
                                     barmode='stack', hovermode='x unified')
        fp_factors_fig.update_yaxes(title_text="% Feature Concentration", range=[0, 100])
        fp_factors_fig.show()

    def plot_g_space(self, fpeak, factor_idx1, factor_idx2, show_base: bool = False, show_delta: bool = False):
        if fpeak not in self.fpeaks:
            logger.warn(f"fpeak is not a value calculated. Provided fpeak: {fpeak}. "
                        f"Valid fpeak values are: {self.fpeaks}")
            logger.info(f"Defaulting to first fpeak value. Fpeak={self.fpeaks[0]}")
            fpeak = self.fpeaks[0]
        if factor_idx1 is not None:
            if factor_idx1 > self.base.factors or factor_idx1 < 1:
                logger.warn(f"Invalid factor_idx1 provided, must be between 1 and {self.base.factors}")
                return
        if factor_idx2 is not None:
            if factor_idx2 > self.base.factors or factor_idx2 < 1:
                logger.warn(f"Invalid factor_idx2 provided, must be between 1 and {self.base.factors}")
                return
        fpeak = str(fpeak)
        f1_idx = factor_idx1 - 1
        f2_idx = factor_idx2 - 1

        selected_fpeak = self.results[fpeak]
        b_W = self.base_W
        fp_W = selected_fpeak['model'].W

        b_normalized_factors_contr = b_W / b_W.sum(axis=0)
        fp_normalized_factors_contr = fp_W / fp_W.sum(axis=0)

        if show_delta:
            arrows = ((fp_normalized_factors_contr[:, f1_idx] - b_normalized_factors_contr[:, f1_idx]),
                      (fp_normalized_factors_contr[:, f2_idx] - b_normalized_factors_contr[:, f2_idx]))
            fp_g_fig = ff.create_quiver(x=b_normalized_factors_contr[:, f1_idx],
                                        y=b_normalized_factors_contr[:, f2_idx], u=arrows[0], v=arrows[1],
                                        name="Fpeak Delta", line_width=1, arrow_scale=0.01, scale=0.99)
            fp_g_fig.add_trace(
                go.Scatter(x=fp_normalized_factors_contr[:, f1_idx], y=fp_normalized_factors_contr[:, f2_idx],
                           mode='markers', name="Fpeak"))
            fp_g_fig.add_trace(
                go.Scatter(x=b_normalized_factors_contr[:, f1_idx], y=b_normalized_factors_contr[:, f2_idx],
                           mode='markers', name="Base"))
        else:
            fp_g_fig = go.Figure()
            fp_g_fig.add_trace(
                go.Scatter(x=fp_normalized_factors_contr[:, f1_idx], y=fp_normalized_factors_contr[:, f2_idx],
                           mode='markers', name="Fpeak"))
            if show_base:
                fp_g_fig.add_trace(
                    go.Scatter(x=b_normalized_factors_contr[:, f1_idx], y=b_normalized_factors_contr[:, f2_idx],
                               mode='markers', name="Base"))
        fp_g_fig.update_layout(title=f"Fpeak G-Space Plot - Fpeak={fpeak}", width=800, height=800)
        fp_g_fig.update_yaxes(title_text=f"Factor {factor_idx1} Contributions (avg=1)")
        fp_g_fig.update_xaxes(title_text=f"Factor {factor_idx2} Contributions (avg=1)")
        fp_g_fig.show()

    def plot_factor_contributions(self, fpeak, feature_idx, threshold: float = 0.06):
        if fpeak not in self.fpeaks:
            logger.warn(f"fpeak is not a value calculated. Provided fpeak: {fpeak}. "
                        f"Valid fpeak values are: {self.fpeaks}")
            logger.info(f"Defaulting to first fpeak value. Fpeak={self.fpeaks[0]}")
            fpeak = self.fpeaks[0]
        if feature_idx is not None:
            if feature_idx < 1 or feature_idx > self.base.n:
                logger.warn(f"Invalid feature_idx provided, must be between 1 and {self.base.n}")
                return
            feature_idx = feature_idx - 1
        else:
            feature_idx = 0
        fpeak = str(fpeak)
        x_label = self.dh.input_data.columns[feature_idx]
        factors_data = self.results[fpeak]['model'].H
        normalized_factors_data = 100 * (factors_data / factors_data.sum(axis=0))

        feature_contr = normalized_factors_data[:, feature_idx]
        feature_contr_inc = []
        feature_contr_labels = []
        feature_legend = {}
        for idx in range(feature_contr.shape[0] - 1, -1, -1):
            idx_l = idx + 1
            if feature_contr[idx] > threshold:
                feature_contr_inc.append(feature_contr[idx])
                feature_contr_labels.append(f"Factor {idx_l}")
                feature_legend[f"Factor {idx_l}"] = f"Factor {idx_l} = {factors_data[idx:, feature_idx]}"
        feature_fig = go.Figure(data=[
            go.Pie(labels=feature_contr_labels, values=feature_contr_inc, hoverinfo="label+value", textinfo="percent")])
        feature_fig.update_layout(title=f"Factor Contributions to Feature: {x_label} - Fpeak={fpeak}", width=1200,
                                  height=600,
                                  legend_title_text=f"Factor Contribution > {threshold}%")
        feature_fig.show()

        factors_contr = self.results[fpeak]['model'].W
        normalized_factors_contr = 100 * (factors_contr / factors_contr.sum(axis=0))
        factor_labels = [f"Factor {i}" for i in range(1, normalized_factors_contr.shape[1] + 1)]
        contr_df = pd.DataFrame(normalized_factors_contr, columns=factor_labels)
        contr_df.index = pd.to_datetime(self.dh.input_data.index)
        contr_df = contr_df.sort_index()
        contr_df = contr_df.resample('D').mean()

        contr_fig = go.Figure()
        for factor in factor_labels:
            contr_fig.add_trace(go.Scatter(x=contr_df.index, y=contr_df[factor], mode='lines+markers', name=factor))
        contr_fig.update_layout(title=f"Factor Contributions (avg=1) - Fpeak={fpeak}",
                                width=1200, height=600,
                                legend=dict(orientation="h", xanchor="right", yanchor="bottom", x=1, y=1.02))
        contr_fig.update_yaxes(title_text="Normalized Contribution")
        contr_fig.show()
