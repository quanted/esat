import os
import json
import numpy as np
import pandas as pd
from itertools import permutations


class FactorComp:

    def __init__(self, nmf_output, pmf_output, factors, species):
        self.nmf_output = nmf_output
        self.pmf_output = pmf_output
        self.factors = factors
        self.species = species

        self.factor_columns = None

        self.pmf_profiles_df = None
        self.pmf_profile_p_df = None
        self.pmf_profile_t_df = None
        self._parse_pmf_output()

        self.nmf_epochs_dfs = {}
        self._parse_nmf_output()

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
                # i = str(i)
                nmf_h_data = np.array(json_data[i]["H"])
                nmf_w_data = np.array(json_data[i]["W"])
                nmf_wh_data = np.array(json_data[i]["wh"])
                nmf_wh_data = nmf_wh_data.reshape(nmf_wh_data.shape[1], nmf_wh_data.shape[0])

                nmf_h_df = pd.DataFrame(nmf_h_data, columns=species_columns, index=self.factor_columns)
                nmf_w_df = pd.DataFrame(nmf_w_data, columns=self.factor_columns)
                nmf_wh_df = pd.DataFrame(nmf_wh_data.T, columns=species_columns)

                self.nmf_epochs_dfs[i] = {"WH": nmf_wh_df, "W": nmf_w_df, "H": nmf_h_df}

    def compare(self):
        factor_permutations = list(permutations(self.factor_columns, len(self.factor_columns)))

        all_r2 = {}

        best_permutation_r = None
        best_r = float("-inf")
        best_model = None
        best_factor_r = None

        for e in range(len(self.nmf_epochs_dfs)):
            # e = str(e)
            nmf_h = self.nmf_epochs_dfs[e]["H"]
            all_r = []
            for perm in factor_permutations:
                values = []
                for i in range(len(self.factor_columns)):
                    pmf_i = self.pmf_profiles_df[self.factor_columns[i]].astype(float)
                    nmf_i = nmf_h.loc[perm[i]].astype(float)
                    corr_matrix = np.corrcoef(nmf_i, pmf_i)
                    corr = corr_matrix[0, 1]
                    r_sq = corr ** 2
                    values.append(r_sq)
                r_avg = np.mean(values)
                r_max = np.max(values)
                r_min = np.min(values)
                if r_avg > best_r:
                    best_r = r_avg
                    best_permutation_r = perm
                    best_model = e
                    best_factor_r = values
                all_r.append((perm, r_avg))
            all_r2[e] = all_r
        print(f"R2 - Model: {best_model}, Best permutations: {list(best_permutation_r)}, Average: {best_r}, "
              f"Factors: {best_factor_r}")
