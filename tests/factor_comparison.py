import math
import os
import json
import numpy as np
import pandas as pd
from itertools import permutations, combinations
import multiprocessing as mp
from tqdm import tqdm
from src.model.batch_nmf import BatchNMF


class FactorComp:

    def __init__(self, pmf_profile_file, pmf_contribution_file, factors, features, batch_nmf=None, nmf_output_file=None, residuals_path=None, method="all"):
        self.nmf_output_file = nmf_output_file
        self.pmf_profile_file = pmf_profile_file
        self.pmf_contribution_file = pmf_contribution_file
        self.factors = factors
        self.features = features

        self.pmf_residuals_path = residuals_path
        self.pmf_residuals = None

        self.factor_columns = None

        self.pmf_profiles_df = None
        self.pmf_profile_p_df = None
        self.pmf_profile_t_df = None
        self.pmf_contribution_df = None
        self.pmf_WH = {}
        self._parse_pmf_output()
        self._calculate_pmf_wh()

        self.batch_nmf = batch_nmf
        self.nmf_model_dfs = {}
        self.nmf_Q = {}
        if nmf_output_file is not None or batch_nmf is not None:
            self._parse_nmf_output()
        self.factor_map = None
        self.best_model = None
        self.best_factor_r = None
        self.best_avg_r = None
        self.best_factor_r_avg = None
        self.best_contribution_r = None
        self.best_contribution_r_avg = None
        self.best_wh_r = None
        self.best_wh_r_avg = None

        self.method = method if method in ('all', 'H', 'W', 'WH') else 'all'
        # 'all': equal weight between correlation of W, H and WH, 'H': only, 'W': only, 'WH': only

    def _parse_pmf_output(self):
        if not os.path.exists(self.pmf_profile_file):
            print(f"No pmf output found at: {self.pmf_profile_file}")
            return
        profiles = self.factors + 2

        pmf_profiles = []
        pmf_profile_p = []
        pmf_profile_t = []

        column_labels = None

        with open(self.pmf_profile_file, 'r') as open_file:
            profile_strings = open_file.read()
            t = profile_strings.split('\n')
            j = 0
            for line in t:
                i = line.split('\t')
                if len(i) == profiles:
                    if i[0] == '' and i[1] == '':
                        i[0] = "run"
                        i[1] = "species"
                        column_labels = i
                        continue
                    if j < len(self.features):
                        pmf_profiles.append(i)
                    elif j < 2 * len(self.features):
                        pmf_profile_p.append(i)
                    elif j < 3 * len(self.features):
                        pmf_profile_t.append(i)
                    j += 1
            pmf_profiles_df = pd.DataFrame(pmf_profiles, columns=column_labels)
            pmf_profile_p_df = pd.DataFrame(pmf_profile_p, columns=column_labels)
            pmf_profile_t_df = pd.DataFrame(pmf_profile_t, columns=column_labels)
            pmf_profiles_df.drop('run', axis=1, inplace=True)
            pmf_profile_p_df.drop('run', axis=1, inplace=True)
            pmf_profile_t_df.drop('run', axis=1, inplace=True)

        df_columns = list(pmf_profiles_df.columns)

        self.factor_columns = df_columns[1:]
        factor_types = {}
        for f in self.factor_columns:
            factor_types[f] = 'float'
        self.pmf_profiles_df = pmf_profiles_df.astype(factor_types)
        self.pmf_profile_p_df = pmf_profile_p_df.astype(factor_types)
        self.pmf_profile_t_df = pmf_profile_t_df.astype(factor_types)

        if self.pmf_residuals_path:
            if os.path.exists(self.pmf_residuals_path):
                with open(self.pmf_residuals_path) as pmf_file:
                    file_lines = pmf_file.read().split("\n")
                    header = file_lines[3].split("\t")
                    data = []
                    for i in range(4, len(file_lines)):
                        if file_lines[i] == "":
                            break
                        data.append(file_lines[i].split("\t"))
                    self.pmf_residuals = pd.DataFrame(data, columns=header)
                    self.pmf_residuals.drop('Base_Run', axis=1, inplace=True)
                    self.pmf_residuals = self.pmf_residuals.set_index("Date_Time")
                    column_types = {}
                    for k in list(self.pmf_residuals.columns):
                        column_types[k] = np.float32
                    self.pmf_residuals = self.pmf_residuals.astype(column_types)
        if self.pmf_contribution_file:
            if os.path.exists(self.pmf_contribution_file):
                column_row = 4
                data_start_row = 5
                dates = []
                pmf_contribution_data = []
                pmf_contribution_columns = None

                with open(self.pmf_contribution_file, 'r') as open_file:
                    contribution_strings = open_file.read()
                    rows = contribution_strings.split('\n')
                    for i, row in enumerate(rows):
                        if i == column_row - 1:
                            pmf_contribution_columns = row.split('\t')[2:]
                        elif i >= data_start_row - 1:
                            row_cells = row.split('\t')
                            if len(row_cells) > 1:
                                dates.append(row_cells[1])
                                pmf_contribution_data.append(row_cells[2:])
                self.pmf_contribution_df = pd.DataFrame(pmf_contribution_data, columns=pmf_contribution_columns)
                self.pmf_contribution_df["Datetime"] = dates

                factor_types = {}
                for f in pmf_contribution_columns:
                    factor_types[f] = 'float'
                self.pmf_contribution_df = self.pmf_contribution_df.astype(factor_types)

    def _calculate_pmf_wh(self):
        if self.pmf_profiles_df is not None and self.pmf_contribution_df is not None:
            for factor in self.factor_columns:
                pmf_W_f = self.pmf_contribution_df[factor].to_numpy()
                pmf_H_f = self.pmf_profiles_df[factor].to_numpy()
                pmf_W_f = pmf_W_f.reshape(len(pmf_W_f), 1)
                pmf_WH_f = np.multiply(pmf_W_f, pmf_H_f)
                self.pmf_WH[factor] = pmf_WH_f

    def _parse_nmf_output(self):
        if self.batch_nmf is None:
            if not os.path.exists(self.nmf_output_file):
                print(f"No nmf output found at: {self.nmf_output_file}")
                return
            else:
                self.batch_nmf = BatchNMF.load(self.nmf_output_file)
        species_columns = self.features
        for i, i_nmf in enumerate(self.batch_nmf.results):
            nmf_h_data = i_nmf.H
            nmf_w_data = i_nmf.W
            nmf_wh_data = i_nmf.WH
            nmf_wh_data = nmf_wh_data.reshape(nmf_wh_data.shape[1], nmf_wh_data.shape[0])

            nmf_h_df = pd.DataFrame(nmf_h_data, columns=species_columns, index=self.factor_columns)
            nmf_w_df = pd.DataFrame(nmf_w_data, columns=self.factor_columns)
            nmf_wh_df = pd.DataFrame(nmf_wh_data.T, columns=species_columns)

            nmf_wh_e = {}
            for factor in self.factor_columns:
                nmf_H_f = nmf_h_df.loc[factor].to_numpy()
                nmf_W_f = nmf_w_df[factor].to_numpy()
                nmf_W_f = nmf_W_f.reshape(len(nmf_W_f), 1)
                nmf_WH_f = np.multiply(nmf_W_f, nmf_H_f)
                nmf_wh_e[factor] = nmf_WH_f

            self.nmf_model_dfs[i] = {"WH": nmf_wh_df, "W": nmf_w_df, "H": nmf_h_df, 'WH-element': nmf_wh_e}
            self.nmf_Q[i] = i_nmf.Qtrue

    def compare(self, PMF_Q=None, verbose: bool = True):
        correlation_results = {}
        contribution_results = {}
        wh_results = {}
        for m in tqdm(range(len(self.nmf_model_dfs)), desc="Calculating correlation between factors from each epoch"):
            correlation_results[m] = {}
            contribution_results[m] = {}
            wh_results[m] = {}
            nmf_m = self.nmf_model_dfs[m]["H"]
            nmf_contribution_m = self.nmf_model_dfs[m]["W"]
            nmf_wh = self.nmf_model_dfs[m]["WH-element"]
            for i in self.factor_columns:
                pmf_i = self.pmf_profiles_df[i].astype(float)
                pmf_contribution_i = self.pmf_contribution_df[i].astype(float)
                pmf_wh = self.pmf_WH[i].flatten()
                for j in self.factor_columns:
                    nmf_j = nmf_m.loc[j].astype(float)
                    r2 = self.calculate_correlation(pmf_factor=pmf_i, nmf_factor=nmf_j)
                    correlation_results[m][f"pmf-{i}_nmf-{j}"] = r2
                    nmf_contribution_j = nmf_contribution_m[j].astype(float)
                    r2_2 = self.calculate_correlation(pmf_factor=pmf_contribution_i, nmf_factor=nmf_contribution_j)
                    contribution_results[m][f"pmf-{i}_nmf-{j}"] = r2_2
                    nmf_wh_f = nmf_wh[j].astype(float).flatten()
                    r2_3 = self.calculate_correlation(pmf_wh, nmf_wh_f)
                    wh_results[m][f"pmf-{i}_nmf-{j}"] = r2_3

        # factor_permutations = list(permutations(self.factor_columns, len(self.factor_columns)))
        print(f"Number of permutations for {self.factors} factors: {math.factorial(self.factors)}")
        best_r = 0.0
        best_perm = None
        best_model = None
        best_factor_r = None
        best_contribution_r = None
        best_contribution_r_avg = None
        best_factor_r_avg = None
        best_wh_r = None
        best_wh_r_avg = None

        permutations_n = math.factorial(self.factors)
        factors_max = 100000

        pool = mp.Pool()

        for m in tqdm(range(len(self.nmf_model_dfs)), desc="Calculating average correlation for all permutations for each epoch"):
            # Each Model
            permutation_results = {}
            model_contribution_results = {}
            factor_contribution_results = {}

            factor_permutations = []
            for factor_i, factor in enumerate(list(permutations(self.factor_columns, len(self.factor_columns)))):
                factor_permutations.append(factor)
                if len(factor_permutations) >= factors_max or factor_i == permutations_n - 1:
                    pool_inputs = [(factor, correlation_results[m], contribution_results[m], wh_results[m]) for factor in factor_permutations]
                    for pool_results in pool.starmap(self.combine_factors, pool_inputs):
                        factor, r_avg, r_values, c_r_avg, c_r_values, wh_r_avg, wh_r_values = pool_results
                        permutation_results[factor] = (r_avg, r_values)
                        model_contribution_results[factor] = (c_r_avg, c_r_values)
                        factor_contribution_results[factor] = (wh_r_avg, wh_r_values)

                        if self.method == "all":
                            model_avg_r = (r_avg + c_r_avg + wh_r_avg) / 3.0
                        elif self.method == "W":
                            model_avg_r = c_r_avg
                        elif self.method == "H":
                            model_avg_r = r_avg
                        elif self.method == "WH":
                            model_avg_r = wh_r_avg

                        if model_avg_r > best_r:
                            best_r = model_avg_r
                            best_perm = factor
                            best_model = m
                            best_factor_r = r_values
                            best_factor_r_avg = r_avg
                            best_contribution_r = c_r_values
                            best_contribution_r_avg = c_r_avg
                            best_wh_r = wh_r_values
                            best_wh_r_avg = wh_r_avg
                    factor_permutations = []
        self.best_model = best_model
        self.best_factor_r = best_factor_r
        self.best_avg_r = best_r
        self.best_factor_r_avg = best_factor_r_avg
        self.best_contribution_r = best_contribution_r
        self.best_contribution_r_avg = best_contribution_r_avg
        self.best_wh_r = best_wh_r
        self.best_wh_r_avg = best_wh_r_avg
        self.factor_map = list(best_perm)
        if verbose:
            print(f"R2 - Model: {best_model}, Best permutations: {list(best_perm)}, Average R2: {self.best_avg_r}, \n"
                  f"Profile R2 Avg: {self.best_factor_r_avg}, Contribution R2 Avg: {self.best_contribution_r_avg}, "
                  f"WH R2 Avg: {self.best_wh_r_avg}\n"
                  f"Profile R2: {self.best_factor_r}, \n"
                  f"Contribution R2: {self.best_contribution_r}, \n"
                  f"WH R2: {self.best_wh_r}\n"
                  )
            print(f"PMF5 Q(true): {PMF_Q}, NMF-PY Model {best_model} Q(true): {self.nmf_Q[best_model]}")

    @staticmethod
    def calculate_correlation(pmf_factor, nmf_factor):
        pmf_f = pmf_factor.astype(float)
        nmf_f = nmf_factor.astype(float)
        corr_matrix = np.corrcoef(nmf_f, pmf_f)
        corr = corr_matrix[0, 1]
        r_sq = corr ** 2
        return r_sq

    def combine_factors(self, factors, model_correlation, model_contributions, factor_contributions):
        r_values = []
        r_values_2 = []
        r_values_3 = []
        for i, f in enumerate(factors):
            r2 = model_correlation[f"pmf-{self.factor_columns[i]}_nmf-{f}"]
            r2_2 = model_contributions[f"pmf-{self.factor_columns[i]}_nmf-{f}"]
            r2_3 = factor_contributions[f"pmf-{self.factor_columns[i]}_nmf-{f}"]
            r_values.append(r2)
            r_values_2.append(r2_2)
            r_values_3.append(r2_3)
        r_avg = np.mean(r_values)
        r_avg_2 = np.mean(r_values_2)
        r_avg_3 = np.mean(r_values_3)
        return factors, r_avg, r_values, r_avg_2, r_values_2, r_avg_3, r_values_3

