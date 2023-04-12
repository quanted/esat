import os
import json
import numpy as np
import pandas as pd
from itertools import permutations
import multiprocessing as mp
from tqdm import tqdm


class FactorComp:

    def __init__(self, nmf_output, pmf_output, factors, species, residuals_path=None):
        self.nmf_output = nmf_output
        self.pmf_output = pmf_output
        self.factors = factors
        self.species = species

        self.pmf_residuals_path = residuals_path
        self.pmf_residuals = None

        self.factor_columns = None

        self.pmf_profiles_df = None
        self.pmf_profile_p_df = None
        self.pmf_profile_t_df = None
        self._parse_pmf_output()

        self.nmf_epochs_dfs = {}
        self.nmf_Q = {}
        self._parse_nmf_output()
        self.factor_map = None
        self.best_model = None
        self.best_factor_r = None
        self.best_avg_r = None

    def _parse_pmf_output(self):
        if not os.path.exists(self.pmf_output):
            print(f"No pmf output found at: {self.pmf_output}")
            return
        profiles = self.factors + 2

        pmf_profiles = []
        pmf_profile_p = []
        pmf_profile_t = []

        column_labels = None

        with open(self.pmf_output, 'r') as open_file:
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
                    if j < self.species:
                        pmf_profiles.append(i)
                    elif j < 2 * self.species:
                        pmf_profile_p.append(i)
                    elif j < 3 * self.species:
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

    def _parse_nmf_output(self):
        if not os.path.exists(self.nmf_output):
            print(f"No nmf output found at: {self.nmf_output}")
            return
        if self.pmf_profiles_df is None:
            print(f"PMF output must be loaded prior to loading NMF outputs")
            return
        with open(self.nmf_output, 'br') as json_file:
            json_data = json.load(json_file)
            species_columns = np.array(self.pmf_profiles_df["species"])
            for i in range(len(json_data)):
                nmf_h_data = np.array(json_data[i]["H"])
                nmf_w_data = np.array(json_data[i]["W"])
                nmf_wh_data = np.array(json_data[i]["wh"])
                nmf_wh_data = nmf_wh_data.reshape(nmf_wh_data.shape[1], nmf_wh_data.shape[0])

                nmf_h_df = pd.DataFrame(nmf_h_data, columns=species_columns, index=self.factor_columns)
                nmf_w_df = pd.DataFrame(nmf_w_data, columns=self.factor_columns)
                nmf_wh_df = pd.DataFrame(nmf_wh_data.T, columns=species_columns)

                self.nmf_epochs_dfs[i] = {"WH": nmf_wh_df, "W": nmf_w_df, "H": nmf_h_df}
                self.nmf_Q[i] = json_data[i]["Q"]

    def compare(self, PMF_Q=None, verbose: bool = True, parallel: bool = True):
        correlation_results = {}
        for m in tqdm(range(len(self.nmf_epochs_dfs)), desc="Calculating correlation between factors from each epoch"):
            correlation_results[m] = {}
            nmf_m = self.nmf_epochs_dfs[m]["H"]
            for i in self.factor_columns:
                pmf_i = self.pmf_profiles_df[i].astype(float)
                for j in self.factor_columns:
                    nmf_j = nmf_m.loc[j].astype(float)
                    r2 = self.calculate_correlation(pmf_factor=pmf_i, nmf_factor=nmf_j)
                    correlation_results[m][f"pmf-{i}_nmf-{j}"] = r2

        factor_permutations = list(permutations(self.factor_columns, len(self.factor_columns)))
        # TODO: factor_permutations is too large at 12 factors: 479,001,600 permutations
        #  (need alternative way of managing the permutation list)
        print(f"Number of permutations for {self.factors} factors: {len(factor_permutations)}")
        best_r = 0.0
        best_perm = None
        best_model = None
        best_factor_r = None

        pool = mp.Pool()

        for m in tqdm(range(len(self.nmf_epochs_dfs)), desc="Calculating average correlation for all permutations for each epoch"):
            # Each Model
            permutation_results = {}
            if parallel:
                pool_inputs = [(factor, correlation_results[m]) for factor in factor_permutations]
                for pool_results in pool.starmap(self.combine_factors, pool_inputs):
                    factor, r_avg, r_values = pool_results
                    permutation_results[factor] = (r_avg, r_values)
                    if r_avg > best_r:
                        best_r = r_avg
                        best_perm = factor
                        best_model = m
                        best_factor_r = r_values
            else:
                for n in range(len(factor_permutations)):
                    r_values = []
                    for i, f in enumerate(factor_permutations[n]):
                        r2 = correlation_results[m][f"pmf-{self.factor_columns[i]}_nmf-{f}"]
                        r_values.append(r2)
                    r_avg = np.mean(r_values)
                    permutation_results[n] = (r_avg, r_values)
                    if r_avg > best_r:
                        best_r = r_avg
                        best_perm = factor_permutations[n]
                        best_model = m
                        best_factor_r = r_values
        self.best_model = best_model
        self.best_factor_r = best_factor_r
        self.best_avg_r = best_r
        self.factor_map = list(best_perm)
        if verbose:
            print(f"R2 - Model: {best_model}, Best permutations: {list(best_perm)}, Average: {best_r}, "
                  f"Factors: {best_factor_r}")
            print(f"PMF5 Q(true): {PMF_Q}, NMF-PY Model {best_model} Q(true): {self.nmf_Q[best_model]}")

    def calculate_correlation(self, pmf_factor, nmf_factor):
        pmf_f = pmf_factor.astype(float)
        nmf_f = nmf_factor.astype(float)
        corr_matrix = np.corrcoef(nmf_f, pmf_f)
        corr = corr_matrix[0, 1]
        r_sq = corr ** 2
        return r_sq

    def combine_factors(self, factors, model_correlation):
        r_values = []
        for i, f in enumerate(factors):
            r2 = model_correlation[f"pmf-{self.factor_columns[i]}_nmf-{f}"]
            r_values.append(r2)
        r_avg = np.mean(r_values)
        return (factors, r_avg, r_values)
