import datetime
import logging
import pickle
import os
import copy
import time
import json
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import plotly.graph_objects as go
from pathlib import Path
from esat.model.sa import SA
from esat.error.bootstrap import Bootstrap
from esat.error.displacement import Displacement
from esat.utils import np_encoder

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class BSDISP:
    """
    The Bootstrap-Displacement (BS-DISP) method combines both the Bootstrap and Displacement methods to estimate the
    errors with both random and rotational ambiguity. For each BS run/dataset, the DISP method is run on that dataset.

    The BS-DISP method uses a base model, and an optional BS instance. For each bootstrap run, BS dataset, DISP will
    be run on each BS model for the specified features. If no features are specified then all features are run on
    DISP.

    Parameters
    ----------
    sa : SA
       A completed SA base model that used the same data and uncertainty datasets.
    feature_labels : list
       The labels for the features, columns of the dataset, specified from the data handler.
    model_selected : int
       The index of the model selected from a batch NMF run, used for labeling.
    bootstrap: Bootstrap
       A previously complete BS model.
    bootstrap_n : int
       The number of bootstrap runs to make.
    block_size : int
       The block size for the BS resampling.
    threshold : float
       The correlation threshold that must be met for a BS factor to be mapped to a base model factor, factor
       correlations must be greater than the threshold or are labeled unmapped.
    max_search : int
       The maximum number of search steps to complete when trying to find a factor feature value. Default = 50
    threshold_dQ : float
       The threshold range of the dQ value for the factor feature value to be considered found. I.E, dQ=4 and
       threshold_dQ=0.1, than any value between 3.9 and 4.0 will be considered valid.
    features: list
       A list of the feature indices to run DISP on, default is None which will run DISP on all features.
    seed : int
       The random seed for random resampling of the BS datasets. The base model random seed is used for all BS runs,
       which result in the same initial W matrix.
    """

    dQmax = [4, 2, 1, 0.5]

    def __init__(self,
                 sa: SA,
                 feature_labels: list,
                 model_selected: int = -1,
                 bootstrap: Bootstrap = None,
                 bootstrap_n: int = 20,
                 block_size: int = 10,
                 threshold: float = 0.6,
                 max_search: int = 50,
                 threshold_dQ: float = 0.1,
                 features: list = None,
                 seed: int = None,
                 ):
        """
        Constructor method.
        """
        self.sa = sa
        self.feature_labels = feature_labels
        self.model_selected = model_selected
        self.factors = self.sa.factors
        self.bootstrap = bootstrap
        self.bootstrap_n = bootstrap_n if bootstrap is None else bootstrap.bootstrap_n
        self.block_size = block_size if bootstrap is None else bootstrap.block_size
        self.threshold = threshold if bootstrap is None else bootstrap.threshold
        self.max_search = max_search
        self.threshold_dQ = threshold_dQ
        self.features = features
        self.seed = seed if bootstrap is None else bootstrap.bs_seed

        self.disp_results = {}
        self.compiled_results = None
        self.swap_table = np.zeros(shape=(len(self.dQmax), self.factors))
        self.count_table = np.zeros(shape=(len(self.dQmax), self.factors))
        self.n_drops = 0
        self.disp_swap = 0
        self.fit_swaps = -1
        self.metadata = {
            "model_selected": self.model_selected,
            "bs-block_size": self.block_size,
            "bs-threshold": self.threshold,
            "disp-max_search": self.max_search,
            "disp-threshold_dQ": self.threshold_dQ,
            "features": self.features,
            "seed": self.seed
        }

    def run(self,
            parallel: bool = True,
            keep_H: bool = True,
            reuse_seed: bool = True,
            block: bool = True,
            overlapping: bool = False):
        """
        Run the BS-DISP error estimation method. If no prior BS run had been completed, this will execute a BS run and
        then a DISP for each of the BS runs.

        Parameters
        ----------
        keep_H : bool
           When retraining the SA models using the resampled input and uncertainty datasets, keep the base model H
           matrix instead of reinitializing. The W matrix is always reinitialized when SA is run on the BS datasets.
           Default = True
        reuse_seed : bool
           Reuse the base model seed for initializing the W matrix, and the H matrix if keep_H = False. Default = True
        block : bool
           Use block resampling instead of full resampling. Default = True
        overlapping : bool
           Allow resampled blocks to overlap. Default = False

        """
        self.metadata["parallel"] = parallel
        self.metadata["keep_H"] = keep_H
        self.metadata["reuse_seed"] = reuse_seed
        self.metadata["block"] = block
        self.metadata["overlapping"] = overlapping

        if self.bootstrap is None:
            logger.info(f"Running new Bootstrap instance with {self.bootstrap_n} runs and block size {self.block_size}")
            # Run BS
            self.bootstrap = Bootstrap(sa=self.sa, feature_labels=self.feature_labels,
                                       model_selected=self.model_selected, block_size=self.block_size,
                                       bootstrap_n=self.bootstrap_n, threshold=self.threshold, seed=self.seed)
            self.bootstrap.run(keep_H=keep_H, reuse_seed=reuse_seed, block=block, overlapping=overlapping)
            logger.info("Bootstrap model created for BS-DISP instance\n")

        bs_keys = list(self.bootstrap.bs_results.keys())
        t0 = time.time()
        if parallel:
            cpus = mp.cpu_count()
            cpus = cpus - 1 if cpus > 1 else 1
            pool = mp.Pool(processes=cpus)
            p_args = []
            for i, bs_key in enumerate(bs_keys):
                i_model = self.bootstrap.bs_results[bs_key]["model"]
                i_args = (bs_key, i_model, self.feature_labels, self.model_selected, self.threshold_dQ,
                          self.max_search, self.features, self.dQmax)
                p_args.append(i_args)
            results = pool.starmap(BSDISP._parallel_disp, p_args)
            pool.close()
            pool.join()
            # for result in pool.starmap(BSDISP._parallel_disp, p_args, chunksize=10):
            for result in results:
                i, i_disp = result
                self.disp_results[i] = i_disp
        else:
            for bs_key in tqdm(bs_keys, desc="BS-DISP - Displacement Stage", position=0, leave=True):
                bs_result = self.bootstrap.bs_results[bs_key]
                bs_model = bs_result["model"]
                i_disp = Displacement(sa=bs_model,
                                      feature_labels=self.feature_labels,
                                      model_selected=self.model_selected,
                                      threshold_dQ=self.threshold_dQ,
                                      max_search=self.max_search,
                                      features=self.features
                                      )
                i_disp.dQmax = self.dQmax
                i_disp.run(batch=bs_key)
                self.disp_results[bs_key] = i_disp
        t1 = time.time()
        logger.info(f"Completed all BS-DISP calculations, BS runs: {self.bootstrap_n}, "
                    f"Features: {len(self.feature_labels)}, Factors: {self.factors}, "
                    f"Runtime: {str(datetime.timedelta(seconds=t1-t0))} hr:min:sec")
        self.__compile_results()

    @staticmethod
    def _parallel_disp(bs_key, bs_model, feature_labels, model_selected, threshold_dQ, max_search, features, dQmax):
        t0 = time.time()
        logger.info(f"Starting Displacement Stage for BS run {bs_key}.")
        i_disp = Displacement(sa=bs_model, feature_labels=feature_labels, model_selected=model_selected,
                              threshold_dQ=threshold_dQ, max_search=max_search, features=features)
        i_disp.dQmax = dQmax
        i_disp.run(batch=bs_key)
        t1 = time.time()
        logger.info(f"Completed Displacement Stage for BS run {bs_key}. Runtime: {round(t1-t0, 2)} secs")
        return bs_key, i_disp

    def __compile_results(self):
        """
        Calculate the merging statistics and metrics for the bs-disp results.
        """
        key0 = list(self.disp_results.keys())[0]
        disp_result = self.disp_results[key0].compiled_results
        profiles = disp_result["profile"]
        profiles_max = disp_result["profile_max"]
        profiles_min = disp_result["profile_min"]
        conc = disp_result["conc"]
        conc_max = disp_result["conc_max"]
        conc_min = disp_result["conc_min"]
        dQ_drop = disp_result["dQ_drop"]
        disp_profiles = [profiles]
        disp_conc = [conc]
        for result_i in range(1, len(self.disp_results.keys())):
            disp_result_i = self.disp_results[result_i].compiled_results
            profile_i = disp_result_i["profile"]
            profile_max_i = disp_result_i["profile_max"]
            profile_min_i = disp_result_i["profile"]
            conc_i = disp_result_i["conc"]
            conc_max_i = disp_result_i["conc_max"]
            conc_min_i = disp_result_i["conc_min"]
            dQ_drop_i = disp_result_i["dQ_drop"]
            disp_profiles.append(profile_i)
            profiles_max = np.max([profiles_max, profile_max_i], axis=0)
            profiles_min = np.min([profiles_min, profile_min_i], axis=0)
            disp_conc.append(conc_i)
            conc_max = np.max([conc_max, conc_max_i], axis=0)
            conc_min = np.min([conc_min, conc_min_i], axis=0)
            dQ_drop = np.min([dQ_drop, dQ_drop_i.values], axis=0)
            if any(dQ_drop_i < 0.0):
                self.n_drops += 1
        self.compiled_results =  copy.copy(self.disp_results[key0].compiled_results)
        self.compiled_results["profiles"] = np.mean(disp_profiles, axis=0)
        self.compiled_results["profile_max"] = profiles_max
        self.compiled_results["profile_min"] = profiles_min
        self.compiled_results["conc"] = np.mean(disp_conc, axis=0)
        self.compiled_results["conc_max"] = conc_max
        self.compiled_results["conc_min"] = conc_min
        self.compiled_results["dQ_drop"] = dQ_drop

        for result_i in self.disp_results.keys():
            self.swap_table = self.swap_table + self.disp_results[result_i].swap_table
            self.count_table = self.count_table + self.disp_results[result_i].count_table
            if np.count_nonzero(self.disp_results[result_i].swap_table) > 0:
                self.disp_swap += 1
        fit_swap = 0
        _table = self.bootstrap.mapping_df.iloc[:, 1:self.factors+1].values
        for i in range(0, self.factors):
            fit_swap += 1 if _table[i, i] < self.bootstrap_n else 0
        self.fit_swaps = fit_swap

    def summary(self):
        """
        Prints a summary of the BS-DISP results table.

        Summary shows the largest change in Q across all DISP runs, the % of cases with a drop of Q, swap in best
        fit and swap in DISP phase. Followed by the swap % table as shown in the regular DISP summary. The dQmax values
        in BS-DISP differ from DISP to account for increased variability, BS-DISP dQmax values are (0.5, 1, 2, 4) while
        DISP dQmax values are (4, 8, 16, 32)
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
        logger.info(f"Largest dQ decrease: {round(largest_dQ_change, 2)}, # of Q drops: {self.n_drops}, "
                    f"# of best fit swaps: {self.fit_swaps}, # of DISP swaps: {self.disp_swap}")
        table_plot = go.Figure(data=[go.Table(
            header=dict(values=table_labels),
            cells=dict(values=table_data.T)
        )])
        table_plot.update_layout(title=f"Swap %", width=600, height=200, margin={'t': 50, 'l': 25, 'b': 10, 'r': 25})
        table_plot.show()

    def plot_results(self,
                     factor: int,
                     dQ: float = 0.5
                     ):
        """
        Plot the BS-DISP results for a specified factor and dQ value. The output results are grouped by dQ, with dQ=0.5
        being the default value displayed for results.

        Parameters
        ----------
        factor : int
           The index of the BS-DISP factor results to display.
        dQ : float
           The dQ value to show in the results, valid values are (0.5, 1, 2, 4). Default = 0.5, will use default if
           invalid value provided.

        """
        dQ = dQ if dQ in self.dQmax else 0.5
        self.plot_profile(factor=factor, dQ=dQ)
        self.plot_contribution(factor=factor, dQ=dQ)

    def plot_profile(self,
                     factor: int,
                     dQ: float = 0.5
                     ):
        """
        Plot the BS-DISP factor profile results.

        Parameters
        ----------
        factor : int
           The index of the BS-DISP factor results to display.
        dQ : float
           The dQ value to show in the results, valid values are (0.5, 1, 2, 4). Default = 0.4, will use default if
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
                                      base=100 * selected_data.profile_min, name="BS-DISP Range"))
        disp_profile.update_traces(selector=dict(type="bar"), marker_color='rgb(158,202,225)',
                                   marker_line_color='rgb(8,48,107)',
                                   marker_line_width=1.5, opacity=0.6)
        disp_profile.update_layout(
            title=f"Variability in Percentage of Features - Model {self.model_selected} - Factor {factor_label} - dQ {dQ}",
            width=1200, height=600, showlegend=True, hovermode='x unified')
        disp_profile.update_yaxes(title_text="Percentage", range=[0, 100])
        disp_profile.update_traces(selector=dict(type="bar"), hovertemplate='Max: %{value}<br>Min: %{base}')
        disp_profile.show()

    def plot_contribution(self,
                          factor: int,
                          dQ: float = 0.5
                          ):
        """
        Plot the BS-DISP factor contribution results.

        Parameters
        ----------
        factor : int
           The index of the BS-DISP factor results to display.
        dQ : float
           The dQ value to show in the results, valid values are (0.5, 1, 2, 4). Default = 0.5, will use default if
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
                   name="BS-DISP Range"))
        disp_conc.update_traces(selector=dict(type="bar"), marker_color='rgb(158,202,225)',
                                marker_line_color='rgb(8,48,107)',
                                marker_line_width=1.5, opacity=0.6, hovertemplate='Max: %{value}<br>Min: %{base}')
        disp_conc.update_layout(
            title=f"Variability in Concentration of Features - Model {self.model_selected} - Factor {factor_label} - dQ {dQ}",
            width=1200, height=600, showlegend=True, hovermode='x unified')
        disp_conc.update_yaxes(title_text="Concentration (log)", type="log")
        disp_conc.show()

    def save(self, bsdisp_name: str,
             output_directory: str,
             pickle_result: bool = True
             ):
        """
        Save the BS-DISP results.
        Parameters
        ----------
        bsdisp_name : str
            The name to use for the BS-DISP pickle file.
        output_directory : str
            The output directory to save the BS-DISP pickle file to.
        pickle_result : bool
            Pickle the BS-DISP model. Default = True.

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
            file_path = os.path.join(output_directory, f"{bsdisp_name}.pkl")
            if pickle_result:
                with open(file_path, "wb") as save_file:
                    pickle.dump(self, save_file)
                    logger.info(f"BS-DISP SA output saved to pickle file: {file_path}")
            else:
                meta_file = os.path.join(output_directory, f"{bsdisp_name}-metadata.json")
                with open(meta_file, "w") as mfile:
                    json.dump(self.metadata, mfile, default=np_encoder)
                    logger.info(f"BSDISP SA model metadata saved to file: {meta_file}")
                self.bootstrap.save(bs_name=bsdisp_name, output_directory=str(output_directory), pickle_result=False)
                for k, disp in self.disp_results.items():
                    disp.save(disp_name=bsdisp_name + "-" + str(k), output_directory=str(output_directory),
                              pickle_result=False)
                compiled_file = os.path.join(output_directory, f"{bsdisp_name}-results.csv")
                with open(compiled_file, 'w') as cfile:
                    self.compiled_results.to_csv(cfile, index=False, lineterminator='\n')
            return file_path
        else:
            logger.error(f"Output directory does not exist. Specified directory: {output_directory}")
            return None

    @staticmethod
    def load(file_path: str):
        """
        Load a previously saved BS-DISP SA pickle file.

        Parameters
        ----------
        file_path : str
           File path to a previously saved BS-DISP SA pickle file

        Returns
        -------
        BSDISP
           On successful load, will return a previously saved BS-DISP SA object. Will return None on load fail.
        """
        file_path = Path(file_path)
        if not file_path.is_absolute():
            logger.error("Provided path is not an absolute path. Must provide an absolute path.")
            return None
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as pfile:
                    bsdisp = pickle.load(pfile)
                    return bsdisp
            except pickle.PickleError as p_error:
                logger.error(f"Failed to load BS-DISP pickle file {file_path}. \nError: {p_error}")
                return None
        else:
            logger.error(f"BS-DISP load file failed, specified pickle file does not exist. File Path: {file_path}")
            return None
