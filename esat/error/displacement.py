import logging
import copy
import pickle
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from tqdm import tqdm
from esat.utils import compare_all_factors, np_encoder
from esat.metrics import q_loss, EPSILON
from esat.model.sa import SA

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class Displacement:
    """
    The displacement method (DISP) for error estimation explores the rotational ambiguity in the solution by assessing
    the largest range of source profile values without an appreciable increase in the loss value (Q).

    The DISP method finds the required change in a factor profile feature value to cause a specific increase in the
    loss function value (dQ). The change is found for dQ=(4, 8, 16, 32) and for both increasing and decreasing
    changes to the factor profile feature values. The search for these changes is limited to max_search steps, where
    a step is the binary search for the value based upon the bounds of the initial value. Factor profile values must
    be greater than 0, so once the modified value is below 1e-8 or the modified value is no longer changing between
    steps the search is stopped and the final value in the search used.

    The process is repeated for all factors and features, if there are factors=k, features=N, dQ_N=4 then this
    process is completed 2*k*N*4 times.

    Parameters
    ----------
    sa : SA
       The base model to run the DISP method on.
    feature_labels : list
       The list of feature, column, labels from the original input dataset. Provided in the data handler.
    model_selected : int
       The index of the model selected in the case of a batch NMF run, used for labeling.
    max_search : int
       The maximum number of search steps to complete when trying to find a factor feature value. Default = 50
    threshold_dQ : float
       The threshold range of the dQ value for the factor feature value to be considered found. I.E, dQ=4 and
       threshold_dQ=0.1, than any value between 3.9 and 4.0 will be considered valid.
    features : list
       A list of the feature indices to run DISP on, default is None which will run DISP on all features.
    """

    dQmax = [32, 16, 8, 4]

    def __init__(self,
                 sa: SA,
                 feature_labels: list,
                 model_selected: int = -1,
                 max_search: int = 50,
                 threshold_dQ: float = 0.1,
                 features: list = None
                 ):
        """
        Constructor method.
        """
        self.sa = sa
        self.selected_model = model_selected
        self.V = self.sa.V
        self.U = self.sa.U
        self.H = self.sa.H + EPSILON
        self.W = self.sa.W
        self.base_Q = self.sa.Qtrue
        self.feature_labels = feature_labels
        self.features = features if features is not None else [i for i in range(len(self.feature_labels))]
        self.excluded_features = set(range(len(self.feature_labels))).difference(set(self.features))
        self.factors = self.H.shape[0]

        self.max_search = max_search
        self.threshold_dQ = threshold_dQ

        self.increase_results = {}
        self.decrease_results = {}

        self.swap_table = np.zeros(shape=(len(self.dQmax), self.factors))
        self.count_table = np.zeros(shape=(len(self.dQmax), self.factors))
        self.compiled_results = None
        self.metadata = {
            "selected_model": self.selected_model,
            "features": self.features,
            "excluded_features": self.excluded_features,
            "max_search": self.max_search,
            "threshold_dQ": self.threshold_dQ
        }

    def run(self, batch: int = -1):
        """
        Run the DISP method on the provided SA model.

        Parameters
        ----------
        batch : int
           Batch number identifier, used for labeling DISP during parallel runs with BS-DISP.
        """
        logger.info(f"Starting DISP for batch: {batch}")
        self._increase_disp(batch=batch)
        self._decrease_disp(batch=batch)
        self._compile_results()
        logger.info(f"Completed DISP for batch: {batch}")

    def summary(self):
        """
        Print the summary table showing the largest change in dQ and the % of factor flips that occurred.
        """
        largest_dQ_inc = self.compiled_results["dQ_drop"].max()
        largest_dQ_dec = self.compiled_results["dQ_drop"].min()
        largest_dQ_change = largest_dQ_inc if np.abs(largest_dQ_inc) > np.abs(largest_dQ_dec) else largest_dQ_dec
        table_labels = ["dQ Max"]
        for i in range(self.factors):
            table_labels.append(f"Factor {i+1}")
        table_data = np.round(100 * (self.swap_table/self.count_table), 2)
        dq_list = list(reversed(self.dQmax))
        dq_list = np.reshape(dq_list, newshape=(len(dq_list), 1))
        table_data = np.hstack((dq_list, table_data))

        logger.info(f"Largest dQ Decrease: {round(largest_dQ_change, 2)}")
        table_plot = go.Figure(data=[go.Table(
            header=dict(values=table_labels),
            cells=dict(values=table_data.T)
        )])
        table_plot.update_layout(title=f"Swap %", width=600, height=200, margin={'t': 50, 'l': 25, 'b': 10, 'r': 25})
        table_plot.show()

    def plot_results(self,
                     factor: int,
                     dQ: int = 4
                     ):
        """
        Plot the DISP results for a specified factor and dQ value. The output results are grouped by dQ, with dQ=4 being
        the default value displayed for results.

        Parameters
        ----------
        factor : int
           The index of the DISP factor results to display.
        dQ : int
           The dQ value to show in the results, valid values are (4, 8, 16, 32). Default = 4, will use default if
           invalid value provided.

        """
        dQ = dQ if dQ in self.dQmax else 4
        self.plot_profile(factor=factor, dQ=dQ)
        self.plot_contribution(factor=factor, dQ=dQ)

    def plot_profile(self,
                     factor: int,
                     dQ: int = 4
                     ):
        """
        Plot the DISP factor profile results.

        Parameters
        ----------
        factor : int
           The index of the DISP factor results to display.
        dQ : int
           The dQ value to show in the results, valid values are (4, 8, 16, 32). Default = 4, will use default if
           invalid value provided.

        """
        if factor > self.factors or factor < 1:
            logger.info(f"Invalid factor provided, must be between 1 and {self.factors}")
            return
        factor_label = factor
        factor = factor - 1
        dQ = dQ if dQ in self.dQmax else 4
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
            title=f"Variability in Percentage of Features - Model {self.selected_model} - Factor {factor_label} - dQ {dQ}",
            width=1200, height=600, showlegend=True, hovermode='x unified')
        disp_profile.update_yaxes(title_text="Percentage", range=[0, 100])
        disp_profile.update_traces(selector=dict(type="bar"), hovertemplate='Max: %{value}<br>Min: %{base}')
        disp_profile.show()

    def plot_contribution(self,
                          factor: int,
                          dQ: int = 4
                          ):
        """
        Plot the DISP factor contribution results.

        Parameters
        ----------
        factor : int
           The index of the DISP factor results to display.
        dQ : int
           The dQ value to show in the results, valid values are (4, 8, 16, 32). Default = 4, will use default if
           invalid value provided.

        """
        if factor > self.factors or factor < 1:
            logger.info(f"Invalid factor provided, must be between 1 and {self.factors}")
            return
        factor_label = factor
        factor = factor - 1
        dQ = dQ if dQ in self.dQmax else 4
        selected_data = self.compiled_results.loc[self.compiled_results["factor"] == factor].loc[
            self.compiled_results["dQ"] == dQ]

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
            title=f"Variability in Concentration of Features - Model {self.selected_model} - Factor {factor_label} - dQ {dQ}",
            width=1200, height=600, showlegend=True, hovermode='x unified')
        disp_conc.update_yaxes(title_text="Concentration (log)", type="log")
        disp_conc.show()

    def _increase_disp(self, batch: int = -1):
        """
        Run the increasing change DISP method on all factors and features.

        Parameters
        ----------
        batch : int
           Batch number identifier, used for labeling DISP during parallel runs with BS-DISP.

        """
        for factor_i in tqdm(range(self.H.shape[0]), desc="Increasing value for factors", position=0, leave=True):
            factor_results = {}
            for feature_j in tqdm(self.features, desc=f"+ : Batch {batch}, Factor {factor_i+1} - Features", position=0, leave=True):
                new_H =  copy.copy(self.H)
                high_modifier = 2.0
                high_found = False
                i_results = {}
                high_search_i = 0
                max_dQ = 0
                max_high_search = 100

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
                                     f"Batch :{batch}, Factor: {factor_i+1}, Feature: {feature_j}, "
                                     f"Max iterations: {max_high_search}, max dQ: {max_dQ}")
                        break
                new_value = 0.0
                for i in range(len(self.dQmax)):
                    low_modifier = 1.0
                    modifier = (high_modifier + low_modifier) / 2.0
                    value_found = False
                    search_i = 0
                    while not value_found:
                        new_H =  copy.copy(self.H)
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
                        if search_i >= max_high_search:
                            value_found = True
                    disp_i_sa = SA(V=self.V, U=self.U,
                                     factors=self.sa.factors, method=self.sa.method,
                                     seed=self.sa.seed, optimized=self.sa.optimized, verbose=False)
                    disp_i_sa.initialize(H=new_H)
                    disp_i_sa.train(max_iter=self.sa.metadata["max_iterations"],
                                     converge_delta=self.sa.metadata["converge_delta"],
                                     converge_n=self.sa.metadata["converge_n"], robust_mode=False)
                    factor_swap = compare_all_factors(disp_i_sa.H, self.H)
                    scaled_profiles = new_H / new_H.sum(axis=0)
                    percent = scaled_profiles[factor_i, feature_j]
                    factor_W = self.W[:, factor_i]
                    factor_matrix = np.matmul(factor_W.reshape(len(factor_W), 1), [new_H[factor_i]])
                    factor_conc_sum = factor_matrix.sum(axis=0)
                    factor_conc_i = factor_conc_sum[feature_j]
                    i_results[self.dQmax[i]] = {"dQ": dQ, "value": new_value, "percent": percent,
                                                "search steps": search_i, "swap": factor_swap, "conc": factor_conc_i,
                                                "Q_drop": self.base_Q - disp_i_sa.Qtrue}
                factor_results[f"feature-{feature_j}"] = i_results
            self.increase_results[f"factor-{factor_i+1}"] = factor_results

    def _decrease_disp(self, batch: int = -1):
        """
        Run the decreasing change DISP method on all factors and features.

        Parameters
        ----------
        batch : int
           Batch number identifier, used for labeling DISP during parallel runs with BS-DISP.
        """
        for factor_i in tqdm(range(self.H.shape[0]), desc="Decreasing value for factors", position=0, leave=True):
            factor_results = {}
            for feature_j in tqdm(self.features, desc=f"- : Batch {batch}, Factor {factor_i+1} - Features", position=0, leave=True):
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
                    max_search_i = 50
                    while not value_found:
                        new_H =  copy.copy(self.H)
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
                        if np.abs(p_mod - modifier) <= 1e-8:  # small value, considered zero. Or no change of modifier.
                            value_found = True
                        search_i += 1
                        if dQ > max_dQ:
                            max_dQ = dQ
                        p_mod = modifier
                        if search_i >= max_search_i:
                            value_found = True
                    disp_i_sa = SA(V=self.V, U=self.U,
                                     factors=self.sa.factors, method=self.sa.method,
                                     seed=self.sa.seed, optimized=self.sa.optimized, verbose=False)
                    disp_i_sa.initialize(H=new_H)
                    disp_i_sa.train(max_iter=self.sa.metadata["max_iterations"],
                                     converge_delta=self.sa.metadata["converge_delta"],
                                     converge_n=self.sa.metadata["converge_n"], robust_mode=False)
                    factor_swap = compare_all_factors(disp_i_sa.H, self.H)
                    scaled_profiles = new_H / new_H.sum(axis=0)
                    percent = scaled_profiles[factor_i, feature_j]
                    factor_W = self.W[:, factor_i]
                    factor_matrix = np.matmul(factor_W.reshape(len(factor_W), 1), [new_H[factor_i]])
                    factor_conc_sum = factor_matrix.sum(axis=0)
                    factor_conc_i = factor_conc_sum[feature_j]
                    i_results[self.dQmax[i]] = {"dQ": dQ, "value": new_value, "percent": percent,
                                                "search steps": search_i, "swap": factor_swap, "conc": factor_conc_i,
                                                "Q_drop": self.base_Q - disp_i_sa.Qtrue}
                factor_results[f"feature-{feature_j}"] = i_results
            self.decrease_results[f"factor-{factor_i+1}"] = factor_results

    def _compile_results(self):
        """
        Compile and calculate the DISP summary statistics from all runs and each dQ value.

        Results are found in the self.compiled_results dataframe.

        """
        scaled_profiles = self.H / self.H.sum(axis=0)
        compiled_data = {"dQ": [], "factor": [], "feature": [], "profile": [], "profile_max": [], "profile_min": [],
                         "conc": [], "conc_max": [], "conc_min": [], "dQ_drop": []}
        for dQ in self.dQmax:
            for factor_i in range(self.H.shape[0]):
                factor_label = factor_i + 1
                factor_W = self.W[:, factor_i]
                factor_matrix = np.matmul(factor_W.reshape(len(factor_W), 1), [self.H[factor_i]])
                factor_conc = factor_matrix.sum(axis=0)
                for feature_j in range(self.H.shape[1]):
                    if feature_j in self.excluded_features:
                        profile_max = scaled_profiles[factor_i, feature_j]
                        profile_min = scaled_profiles[factor_i, feature_j]
                        conc_max = factor_conc[feature_j]
                        conc_min = factor_conc[feature_j]
                        dQ_drop = 0.0
                    else:
                        feature_inc_results = self.increase_results[f"factor-{factor_label}"][f"feature-{feature_j}"][dQ]
                        feature_dec_results = self.decrease_results[f"factor-{factor_label}"][f"feature-{feature_j}"][dQ]
                        profile_max = feature_inc_results["percent"]
                        profile_min = feature_dec_results["percent"]
                        conc_max = feature_inc_results["conc"]
                        conc_min = feature_dec_results["conc"]
                        inc_dQ = feature_inc_results["Q_drop"]
                        dec_dQ = feature_dec_results["Q_drop"]
                        dQ_drop = inc_dQ if inc_dQ < dec_dQ else dec_dQ
                    compiled_data["profile_max"].append(profile_max)
                    compiled_data["profile_min"].append(profile_min)
                    compiled_data["profile"].append(scaled_profiles[factor_i, feature_j])
                    compiled_data["conc_max"].append(conc_max if conc_max > 1e-4 else 1e-4)
                    compiled_data["conc_min"].append(conc_min if conc_min > 1e-4 else 1e-4)
                    compiled_data["conc"].append(factor_conc[feature_j])
                    compiled_data["dQ_drop"].append(dQ_drop if dQ_drop < 0.0 else 0.0)
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

    def save(self, disp_name: str,
             output_directory: str,
             pickle_result: bool = True
             ):
        """
        Save the DISP results.
        Parameters
        ----------
        disp_name : str
            The name to use for the DISP pickle file.
        output_directory : str
            The output directory to save the DISP pickle file to.
        pickle_result : bool
            Pickle the disp model. Default = True.

        Returns
        -------
        str
           The path to the saved file.

        """
        output_directory = Path(output_directory)
        if not output_directory.is_absolute():
            logger.error("Provided output directory is not an absolute path. Must provide an absolute path.")
            return None
        if os.path.exists(output_directory):
            file_path = os.path.join(output_directory, f"{disp_name}.pkl")
            if pickle_result:
                with open(file_path, "wb") as save_file:
                    pickle.dump(self, save_file)
                    logger.info(f"DISP SA output saved to pickle file: {file_path}")
            else:
                file_path = output_directory
                meta_file = os.path.join(output_directory, f"{disp_name}-metadata.json")
                with open(meta_file, "w") as mfile:
                    json.dump(self.metadata, mfile, default=np_encoder)
                    logger.info(f"DISP SA model metadata saved to file: {meta_file}")
                increase_file = os.path.join(output_directory, f"{disp_name}-increase-disp.json")
                with open(increase_file, "w") as incfile:
                    json.dump(self.increase_results, incfile, default=np_encoder)
                    logger.info(f"DISP SA model increasing results saved to file: {increase_file}")
                decrease_file = os.path.join(output_directory, f"{disp_name}-decrease-disp.json")
                with open(decrease_file, "w") as decfile:
                    json.dump(self.decrease_results, decfile, default=np_encoder)
                    logger.info(f"DISP SA model decreasing results saved to file: {decrease_file}")
                swap_file = os.path.join(output_directory, f"{disp_name}-swaptable.csv")
                with open(swap_file, 'w') as stfile:
                    table_labels = ["dQ Max"]
                    for i in range(self.factors):
                        table_labels.append(f"Factor {i + 1}")
                    table_data = np.round(100 * (self.swap_table / self.count_table), 2)
                    dq_list = list(reversed(self.dQmax))
                    dq_list = np.reshape(dq_list, newshape=(len(dq_list), 1))
                    table_data = np.hstack((dq_list, table_data))
                    np.savetxt(stfile, table_data, delimiter=',', header=", ".join(table_labels))
                    logger.info(f"DISP SA swap table saved to file: {swap_file}")
                compiled_file = os.path.join(output_directory, f"{disp_name}-results.csv")
                with open(compiled_file, 'w') as cfile:
                    self.compiled_results.to_csv(cfile, index=False, lineterminator='\n')
                    logger.info(f"DISP SA compiled results saved to file: {compiled_file}")
            return file_path
        else:
            logger.error(f"Output directory does not exist. Specified directory: {output_directory}")
            return None

    @staticmethod
    def load(file_path: str):
        """
        Load a previously saved DISP SA pickle file.

        Parameters
        ----------
        file_path : str
           File path to a previously saved DISP SA pickle file

        Returns
        -------
        Displacement
           On successful load, will return a previously saved DISP NMF object. Will return None on load fail.
        """
        file_path = Path(file_path)
        if not file_path.is_absolute():
            logger.error("Provided path is not an absolute path. Must provide an absolute path.")
            return None
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as pfile:
                    disp = pickle.load(pfile)
                    return disp
            except pickle.PickleError as p_error:
                logger.error(f"Failed to load DISP pickle file {file_path}. \nError: {p_error}")
                return None
        else:
            logger.error(f"DISP load file failed, specified pickle file does not exist. File Path: {file_path}")
            return None
