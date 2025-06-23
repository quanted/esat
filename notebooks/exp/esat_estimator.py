import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score

from esat.model.batch_sa import BatchSA
from esat.model.sa import SA
from esat.data.datahandler import DataHandler
from esat_eval.simulator import Simulator

from tqdm import tqdm

import time
import copy
import pickle
import warnings
import logging

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)


def prepare_data(V, U, i_selection):
    _V = pd.DataFrame(V.copy()[i_selection, :])
    _U = pd.DataFrame(U.copy()[i_selection, :])

    for f in _V.columns:
        _V[f] = pd.to_numeric(_V[f])
        _U[f] = pd.to_numeric(_U[f])
    return _V.to_numpy(), _U.to_numpy()


class Factor:
    def __init__(self,
                 factor_id,
                 profile,
                 model_id
                 ):
        self.factor_id = factor_id
        self.profile = profile
        self.model_id = model_id
        self.cluster_id = None
        self.cor = None

    def assign(self, cluster_id, cor):
        self.cluster_id = cluster_id
        self.cor = cor

    def deallocate(self):
        self.cluster_id = None
        self.cor = None

    def distance(self, cluster):
        f1 = np.array(self.profile).astype(float)
        f2 = np.array(cluster).astype(float)
        corr_matrix = np.corrcoef(f2, f1)
        corr = corr_matrix[0, 1]
        r_sq = corr ** 2
        return r_sq


class Model:
    def __init__(self,
                 model_id):
        self.model_id = model_id
        self.factors = []

        self.score = None

    def add_factor(self, factor):
        self.factors.append(factor)


class Cluster:
    def __init__(self,
                 cluster_id,
                 centroid: np.ndarray
                 ):
        self.cluster_id = cluster_id
        self.centroid = centroid
        self.factors = []
        self.count = 0

        self.mean_r2 = 0
        self.std = 0
        self.wcss = -1
        self.volume = 0
        self.min_values = np.full(len(centroid), np.nan)
        self.max_values = np.full(len(centroid), np.nan)

    def __len__(self):
        return self.count

    def add(self, factor: Factor, cor: float):
        factor.assign(cluster_id=self.cluster_id, cor=cor)
        self.factors.append(factor)
        self.count += 1
        self.min_values = np.fmin(self.min_values, factor.profile)
        self.max_values = np.fmax(self.max_values, factor.profile)
        self.mean_r2 = np.mean([factor.cor for factor in self.factors])
        self.std = np.std([factor.profile for factor in self.factors], axis=0)
        self.wcss = np.sum([np.square(self.centroid - factor.profile) for factor in self.factors])
        self.volume = np.prod(self.max_values - self.min_values)

    def purge(self):
        for factor in self.factors: factor.deallocate()
        self.factors = []
        self.count = 0
        self.mean_r2 = 0
        self.std = 0
        self.wcss = -1
        self.volume = 0
        self.min_values = self.centroid
        self.max_values = self.centroid

    def recalculate(self):
        if len(self.factors) > 0:
            factor_matrix = np.array([factor.profile for factor in self.factors])
            new_centroid = np.mean(factor_matrix, axis=0)
            self.centroid = new_centroid


class BatchFactorCatalog:
    def __init__(self,
                 n_factors: int,
                 n_features: int,
                 threshold: float = 0.8,
                 seed: int = 42
                 ):
        self.n_factors = n_factors
        self.n_features = n_features
        self.threshold = threshold

        self.rng = np.random.default_rng(seed)

        self.models = {}
        self.model_count = 0
        self.factors = {}
        self.factor_count = 0

        # Min and max values for all factor vectors, used for random initialization of the centroids in clustering
        self.factor_min = None
        self.factor_max = None

        self.clusters = {}
        self.dropped_clusters = []
        self.max_clusters_found = 0

        self.bcss = float("inf")
        self.sil = float("-inf")
        self.membership_p = 0.05
        self.primary_factors = []
        self.primary_clusters = []

        self.state = {}

    def results(self):
        results = {}
        for c, cluster in self.clusters.items():
            results[cluster.cluster_id] = {
                "count": len(cluster),
                "mean_r2": cluster.mean_r2,
                "std": cluster.std
            }
        return results

    def metrics(self, membership_p: float = None):
        if membership_p is None:
            membership_p = self.membership_p
        else:
            self.membership_p = membership_p
        all_factors = np.array([v.profile for k, v in self.factors.items()])
        factor_assignments = np.array([v.cluster_id for k, v in self.factors.items()])
        cluster_centroids = [(c, cluster.centroid) for c, cluster in self.clusters.items()]

        i_cluster, i_centroids = zip(*cluster_centroids)
        i_centroids = np.array(i_centroids)

        df_pca0 = pd.DataFrame(all_factors)
        factor_columns = df_pca0.columns
        df_pca0["Cluster"] = factor_assignments

        df_centroids0 = pd.DataFrame(i_centroids, index=list(i_cluster))
        cluster_columns = df_centroids0.columns
        assigned_centroids, cluster_size = np.unique(factor_assignments, return_counts=True)
        df_centroids0["Cluster"] = list(i_cluster)
        df_centroids0 = df_centroids0.loc[assigned_centroids]
        df_centroids0["count"] = cluster_size

        point_cluster_n = []
        for i in range(len(all_factors)):
            i_cluster_count = df_centroids0[df_centroids0["Cluster"] == df_pca0["Cluster"].iloc[i]]["count"].values
            point_cluster_n.append(i_cluster_count)
        cluster_n_threshold = int(len(all_factors) * membership_p)
        df_pca0["cluster_n"] = point_cluster_n
        df_pca0["cluster_n"] = df_pca0["cluster_n"].astype(int)
        df_pca0 = df_pca0[df_pca0["cluster_n"] > cluster_n_threshold]

        df_centroids0 = df_centroids0[df_centroids0["count"] > cluster_n_threshold]

        all_factors = df_pca0[factor_columns].values
        factor_assignments = df_pca0["Cluster"].values
        # Calculate Silhouette Score
        self.primary_factors = list(all_factors)
        self.primary_clusters = list(set(factor_assignments))
        if len(all_factors) > len(self.primary_clusters) > 1:
            self.sil = silhouette_score(all_factors, factor_assignments)

        # Calculate between-cluster sum of squares
        cluster_centroids = [(len(cluster), cluster.centroid) for c_id, cluster in self.clusters.items() if
                             c_id in df_centroids0["Cluster"]]
        overall_mean = np.mean(all_factors, axis=0)
        bcss = 0.0
        for c_count, c_centroid in cluster_centroids:
            bcss += c_count * np.sum((c_centroid - overall_mean) ** 2)
        self.bcss = bcss

    def add_model(self, model: SA, norm: bool = True):
        model_id = self.model_count
        model_factor_ids = []
        norm_H = model.H / np.sum(model.H, axis=0)
        i_model = Model(model_id=model_id)
        for i in range(model.H.shape[0]):
            factor_id = self.factor_count
            self.factor_count += 1
            model_factor_ids.append(factor_id)
            i_H = norm_H if norm else model.H
            factor = Factor(factor_id=factor_id, profile=i_H[i], model_id=model_id)

            i_model.add_factor(factor)
            self.factors[factor_id] = factor
            self.update_ranges(i_H[i])

        self.models[str(model_id)] = i_model
        self.model_count += 1

    def compare(self, matrix):
        compare_results = {}
        for i in range(matrix.shape[0]):
            i_H = matrix[i]
            i_cor = 0.0
            best_cluster = None
            for c, cluster in self.clusters.items():
                cluster_cor = self.distance(i_H, cluster.centroid)
                if cluster_cor > i_cor:
                    i_cor = cluster_cor
                    best_cluster = cluster.cluster_id
            compare_results[i] = {"cluster_id": best_cluster, "r2": i_cor}
        return compare_results

    def score(self):
        # iterate over all models, get the membership count the cluster that each factor is mapped to.
        for model_id, model in self.models.items():
            model_score = 0.0
            for factor in model.factors:
                if factor.cluster_id not in self.clusters.keys():
                    logger.info(f"Factor {factor.factor_id} assigned to non-existent cluster {factor.cluster_id}")
                    factor_score = 0
                    # factor.cluster_id = -1
                else:
                    factor_score = len(self.clusters[factor.cluster_id])
                model_score += factor_score
            model.score = model_score
        # self.metrics()

    def update_ranges(self, factor):
        if self.factor_min is None and self.factor_max is None:
            self.factor_min = copy.copy(factor)
            self.factor_max = copy.copy(factor)
        else:
            self.factor_min = np.minimum(self.factor_min, factor)
            self.factor_max = np.maximum(self.factor_max, factor)

    def initialize_clusters(self):
        for k in range(self.n_factors):
            new_centroid = np.zeros(self.n_features)
            for i in range(self.n_features):
                i_v = self.rng.uniform(low=self.factor_min[i], high=self.factor_max[i])
                new_centroid[i] = i_v
            cluster = Cluster(cluster_id=k, centroid=new_centroid)
            self.clusters[k] = cluster

    def purge_clusters(self):
        for c, cluster in self.clusters.items():
            cluster.purge()

    def distance(self, factor1, factor2):
        f1 = np.array(factor1).astype(float)
        f2 = np.array(factor2).astype(float)
        corr_matrix = np.corrcoef(f2, f1)
        corr = corr_matrix[0, 1]
        r_sq = corr ** 2
        return r_sq

    def calculate_centroids(self):
        new_centroid_matrix = []
        for c, cluster in self.clusters.items():
            cluster.recalculate()
            new_centroid_matrix.append(cluster.centroid)
        return np.array(new_centroid_matrix)

    def cluster_cleanup(self):
        drop_clusters = set()
        cluster_keys = list(self.clusters.keys())
        for i, i_key in enumerate(cluster_keys[:len(cluster_keys) - 1]):
            cluster_i = self.clusters[i_key]
            for j, j_key in enumerate(cluster_keys[i + 1:]):
                if j_key == i_key:
                    continue
                cluster_j = self.clusters[j_key]
                ij_cor = self.distance(cluster_i.centroid, cluster_j.centroid)
                if ij_cor > self.threshold:
                    smaller_cluster = i_key if len(cluster_i) < len(cluster_j) else j_key
                    if smaller_cluster not in drop_clusters:
                        drop_clusters.add(smaller_cluster)
        for i_key, cluster in self.clusters.items():
            if len(cluster) == 0:
                drop_clusters.add(i_key)
        new_centroid_matrix = []
        for i, cluster in self.clusters.items():
            new_centroid_matrix.append(cluster.centroid)
        for cluster in drop_clusters:
            self.clusters[cluster].purge()
        return np.array(new_centroid_matrix)

    def save_state(self, iteration):
        factor_assignment = np.array([v.cluster_id for k, v in self.factors.items()])
        cluster_centroids = [(c, cluster.centroid) for c, cluster in self.clusters.items() if
                             cluster is not None]
        self.state[iteration] = {"assignment": factor_assignment, "cluster_centroids": cluster_centroids}
        self.max_clusters_found = max(self.max_clusters_found, len(cluster_centroids))

    def matrix_difference(self, i_centroids, j_centroids):
        if i_centroids.shape == j_centroids.shape:
            distance = np.linalg.norm(i_centroids - j_centroids, axis=1)
            centroid_shifts = np.mean(distance)
        else:
            min_shape = (min(i_centroids.shape[0], j_centroids.shape[0]), min(i_centroids.shape[1], j_centroids.shape[1]))
            centroid_shifts = np.mean(np.linalg.norm(i_centroids[:min_shape[0], :min_shape[1]] - j_centroids[:min_shape[0], :min_shape[1]],axis=1))
            if i_centroids.shape[0] > j_centroids.shape[0]:
                centroid_shifts += np.mean((len(i_centroids[min_shape[0]:]) / i_centroids.shape[0]) * i_centroids[min_shape[0]:])
            else:
                centroid_shifts += np.mean((len(j_centroids[min_shape[0]:]) / j_centroids.shape[0]) * j_centroids[min_shape[0]:])
        return centroid_shifts

    def cluster(self, max_iterations: int = 20, threshold: float = None, early_stopping: bool = True):
        self.initialize_clusters()
        centroids = self.calculate_centroids()
        converged = False
        current_iter = 0
        if threshold is None:
            threshold = self.threshold
        else:
            self.threshold = threshold
        with tqdm(total=max_iterations, desc="Running clustering. N Clusters: NA, Added: NA") as pbar:
            while not converged:
                if current_iter >= max_iterations:
                    logger.info(
                        f"{self.n_factors} Factor Clustering did not converge after {max_iterations} iterations.")
                    break
                self.purge_clusters()

                model_list = self.rng.permutation(list(self.models.keys()))
                for model_i in model_list:
                    model_factors = [factor.factor_id for factor in self.models[model_i].factors]
                    factor_dist = {}
                    factor_hi = {}
                    # Calculate distances for all factors in the model to all centroids and then order the distances.
                    for factor_i in model_factors:
                        distances = [(j, self.distance(self.factors[factor_i].profile, cluster.centroid)) for j, cluster
                                     in self.clusters.items()]
                        distances.sort(key=lambda x: x[1], reverse=True)
                        factor_dist[str(factor_i)] = distances
                        factor_hi[str(factor_i)] = distances[0]
                    already_assigned = []
                    factor_hi = dict(sorted(factor_hi.items(), key=lambda x: x[1], reverse=True))
                    # Assign factors to clusters, if model hasn't contributed to the cluster already and if the correlation is above the threshold
                    for factor_id in factor_hi.keys():
                        # iterate through list of clusters in order of highest correlation.
                        cluster_idx = -1
                        for cluster_i, correlation_i in factor_dist[factor_id]:
                            if cluster_i not in already_assigned and correlation_i >= threshold:
                                cluster_idx = cluster_i
                                break
                        if cluster_idx != -1:
                            self.clusters[cluster_idx].add(factor=self.factors[int(factor_id)],
                                                           cor=factor_hi[factor_id][1])
                            already_assigned.append(cluster_idx)
                        else:
                            new_cluster_id = self.dropped_clusters.pop(0) if len(self.dropped_clusters) > 0 else len(
                                self.clusters)

                            new_cluster = Cluster(cluster_id=new_cluster_id,
                                                  centroid=self.factors[int(factor_id)].profile)
                            new_cluster.add(factor=self.factors[int(factor_id)], cor=1.0)
                            self.clusters[new_cluster_id] = new_cluster
                            already_assigned.append(new_cluster_id)

                # Recalculate centroids of clusters
                self.save_state(iteration=current_iter)
                new_centroids = self.calculate_centroids()

                if (self.matrix_difference(i_centroids=new_centroids,
                                           j_centroids=centroids) < 0.0001 and current_iter > 3 and early_stopping) or (
                        current_iter >= max_iterations):
                    converged = True

                pbar.update(1)
                pbar.set_description(
                    f"Running {self.n_factors} Factor Clustering. N Clusters: {len(new_centroids)}, Added: {len(new_centroids) - len(centroids)}")
                centroids = new_centroids
                current_iter += 1
        self.score()


class FactorCountOptimizer:
    def __init__(self, contamination='auto', n_estimators=500, random_state=42):
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.rng = np.random.default_rng(random_state)
        self.feature_cols = ['membership_threshold', 's/n', 'bcss',
                             'sil', 'mean_wcss', 'mean_r2', 'qtrue', 'qrobust']
        self.is_trained = False
        self.training_stats = {}

        self.isolation_forest = IsolationForest(random_state=random_state, contamination="auto",n_estimators=n_estimators)

        self.grouped_data = None
        self.global_mean = None
        self.global_var = None
        self.global_count = None

    def update_globals(self, new_batch):
        new_batch = new_batch[self.feature_cols].values
        n = len(new_batch)
        if self.global_mean is None:  # Initialization
            self.global_mean = np.mean(new_batch, axis=0)
            self.global_var = np.var(new_batch, axis=0)
            self.global_count = n
        else:
            new_mean = np.mean(new_batch, axis=0)
            new_var = np.var(new_batch, axis=0)
            new_count = n

            delta = new_mean - self.global_mean
            m_a = self.global_var * self.global_count
            m_b = new_var * new_count

            self.global_count += new_count
            self.global_mean += delta * new_count / self.global_count
            self.global_var = (m_a + m_b + delta ** 2 * self.global_count * new_count / (self.global_count + new_count)) / self.global_count

    def generate_scenario(self, scenario_id=None, true_k=None):
        """Generate a single synthetic scenario with varied parameters"""
        if scenario_id is not None:
            np.random.seed(scenario_id)

        # Vary the experimental conditions
        n_samples = np.random.randint(500, 2000)
        n_features = np.random.randint(8, 40)
        true_factors = np.random.randint(3, 9) if true_k is None else true_k
        noise_min = np.random.uniform(0.1, 0.2)
        noise_max = np.random.uniform(0.2, 0.4)
        noise_scale = np.random.uniform(0.1, 0.3)
        outliers = np.random.uniform(0.1, 0.3)
        outlier_mag = np.random.uniform(1.5, 5.0)

        # Generate clustered data
        simulator = Simulator(seed=self.rng.integers(low=0, high=1e10),
                              factors_n=true_factors,
                              features_n=n_features,
                              samples_n=n_samples,
                              outliers=True,
                              outlier_p=outliers,
                              outlier_mag=outlier_mag,
                              contribution_max=2,
                              noise_mean_min=noise_min,
                              noise_mean_max=noise_max,
                              noise_scale=noise_scale,
                              uncertainty_mean_min=0.04,
                              uncertainty_mean_max=0.07,
                              uncertainty_scale=0.01,
                              verbose=False
                              )
        syn_input_df, syn_uncertainty_df = simulator.get_data()
        data_handler = DataHandler.load_dataframe(input_df=syn_input_df, uncertainty_df=syn_uncertainty_df)
        V, U = data_handler.get_data()

        return V, U, true_factors, noise_max - noise_min

    def run_scenario(self, V=None, U=None, true_factors: int = None, k_min: int = 3, k_max: int = 12, scenario_id=None):
        """Run a single scenario and return the metrics"""
        if V is None or U is None:
            V, U, true_factors, noise_level = self.generate_scenario(scenario_id, true_k=true_factors)

        method = "ls-nmf"
        converge_delta = 0.1
        converge_n = 25

        subset_size = 100
        max_batches = 2
        n_models = 5
        max_iter = 10000

        i_selection = self.rng.choice(V.shape[0], size=subset_size, replace=False, shuffle=True)
        i_V, i_U = prepare_data(V=V, U=U, i_selection=i_selection)

        change_p = 1.0

        k_batches = {}
        # Generate subset batched models
        with tqdm(range(max_batches * (k_max - k_min + 1)), desc="Generating subset profiles", leave=False) as pbar:
            for k in range(k_min, k_max + 1):
                i_batches = []
                for i in range(max_batches):
                    j_selection = self.rng.choice(i_V.shape[0], size=int(subset_size * change_p), replace=False, shuffle=True)
                    idx_change = self.rng.choice(subset_size, size=int(subset_size * change_p), replace=False, shuffle=True)
                    i_selection[idx_change] = j_selection
                    i_V, i_U = prepare_data(V=V, U=U, i_selection=i_selection)

                    batch_sa = BatchSA(V=i_V, U=i_U, factors=k, models=n_models, method=method,
                                       seed=self.rng.integers(low=0, high=1e8), max_iter=max_iter,
                                       converge_delta=converge_delta, converge_n=converge_n, verbose=False)
                    _ = batch_sa.train()
                    i_batches.append(batch_sa)
                    pbar.update(1)
                    pbar.set_description(f"Generating subset profiles. K: {k}")
                k_batches[k] = i_batches

        k_catalog = {}
        k_loss = {}
        for k in range(k_min, k_max + 1):
            factor_catalog = BatchFactorCatalog(n_factors=k, n_features=V.shape[1], threshold=0.8, seed=42)
            batch_qtrue = []
            batch_qrobust = []
            for i_batch in k_batches[k]:
                for sa in i_batch.results:
                    factor_catalog.add_model(model=sa, norm=True)
                    batch_qtrue.append(sa.Qtrue)
                    batch_qrobust.append(sa.Qrobust)
            k_loss[k] = {"Q(True)": round(float(np.mean(batch_qtrue)), 2),
                         "Q(Robust)": round(float(np.mean(batch_qrobust)), 2),
                         "ratio": round(float(np.mean(batch_qtrue) / np.mean(batch_qrobust)), 2)}
            factor_catalog.cluster(max_iterations=15, threshold=.8)
            k_catalog[k] = factor_catalog

        k_membership = {}
        for k, fc in k_catalog.items():
            membership_searching = True
            membership_p = 0.1
            min_p = 0.0
            max_p = 0.2
            max_i = 10
            p_clusters = -1
            while membership_searching and max_i > 0:
                fc.metrics(membership_p=membership_p)
                p_clusters = len(fc.primary_clusters)
                if p_clusters > k:
                    min_p = membership_p
                    membership_p = (max_p + membership_p) / 2.0
                elif p_clusters < k:
                    max_p = membership_p
                    membership_p = (membership_p + min_p) / 2.0
                else:
                    membership_searching = False
                max_i -= 1
            k_membership[k] = membership_p

        range_catalog = {}
        for factors, fc in k_catalog.items():
            cluster_wcss = []
            cluster_r2 = []
            membership_p = k_membership[factors]
            count_threshold = int(fc.factor_count * membership_p)
            for c_id, cluster in fc.clusters.items():
                if len(cluster) >= count_threshold:
                    cluster_wcss.append(cluster.wcss)
                    cluster_r2.append(cluster.mean_r2)
            range_catalog[factors] = {
                "true_factors": true_factors,
                "count": len(fc.clusters),
                "primary_factors": len(fc.primary_factors),
                "primary_clusters": len(fc.primary_clusters),
                "membership_threshold": membership_p,
                "s/n": len(fc.primary_factors) / len(fc.factors),
                "bcss": np.round(fc.bcss, 4),
                "sil": np.round(fc.sil, 4),
                "mean_wcss": np.round(np.mean(cluster_wcss), 4),
                "mean_r2": np.round(np.mean(cluster_r2), 4),
                "qtrue": k_loss[factors]["Q(True)"],
                "qrobust": k_loss[factors]["Q(Robust)"]
            }
        metrics_df = pd.DataFrame(range_catalog)

        return metrics_df, true_factors

    def train_model(self, metrics_df):
        metrics_data = metrics_df[self.feature_cols].values
        metrics_data = (metrics_data - self.global_mean) / np.sqrt(self.global_var)
        self.isolation_forest.fit(metrics_data)

    def predict(self, metrics_df):
        metrics_data = metrics_df[self.feature_cols].values
        metric_data = (metrics_data - self.global_mean) / np.sqrt(self.global_var)
        predictions = self.isolation_forest.predict(metric_data)
        return predictions

    def train(self, n_scenarios=1000, batches=10, k_min=3, k_max=12):
        """Train the isolation forest model"""
        batch_metrics = []
        batch_factors = []
        grouped_data = {}
        training_accuracy = []
        all_anomaly_scores = []
        optimal_anomaly_scores = []
        all_training_metrics = []

        with tqdm(range(n_scenarios), desc="Training scenarios", leave=True) as pbar:
            for i in range(n_scenarios):
                i_metrics, i_factors = self.run_scenario(k_min=k_min, k_max=k_max)
                batch_metrics.append(i_metrics)
                batch_factors.append(i_factors)
                if len(batch_metrics) == batches:
                    # Concatenate batch metrics and factors
                    batch_metrics_df = pd.concat(batch_metrics, axis=0)
                    self.update_globals(batch_metrics_df)
                    self.train_model(batch_metrics_df)

                    predictions = self.predict(batch_metrics_df)
                    accuracy = [x == y for x, y in zip(predictions, batch_factors)]
                    difference = np.mean(predictions - batch_factors)
                    error = np.sum(np.abs(predictions - batch_factors)) / len(predictions)
                    accuracy_val = np.sum(accuracy) / len(accuracy)
                    pbar.set_description(f"Training scenarios. Accuracy: {accuracy_val:.2f}, Error: {error:.2f}, Difference: {difference:.2f}")

                    # Collect anomaly scores for all and optimal factor counts
                    anomaly_scores = -self.isolation_forest.decision_function(
                        (batch_metrics_df[self.feature_cols].values - self.global_mean) / np.sqrt(self.global_var)
                    )
                    batch_metrics_df['anomaly_score'] = anomaly_scores
                    batch_metrics_df['is_optimal'] = batch_metrics_df['factor_count'] == batch_factors
                    optimal_anomaly_scores.extend(batch_metrics_df[batch_metrics_df['is_optimal']]['anomaly_score'].tolist())
                    all_training_metrics.append(batch_metrics_df.assign(scenario_id=i))
                    training_accuracy.append(accuracy_val)

                    batch_factors = []
                    batch_metrics = []
                grouped_data[i] = i_metrics
                pbar.update(1)

        self.training_stats = {
            'total_scenarios': n_scenarios,
            'total_factor_combinations': len(grouped_data),
            'mean_accuracy': np.mean(training_accuracy),
            'std_accuracy': np.std(training_accuracy),
            'mean_anomaly_score_all': np.mean(all_anomaly_scores),
            'std_anomaly_score_all': np.std(all_anomaly_scores),
            'mean_anomaly_score_optimal': np.mean(optimal_anomaly_scores),
            'std_anomaly_score_optimal': np.std(optimal_anomaly_scores)
        }
        self.grouped_data = grouped_data
        self.is_trained = True
        logger.info("Model training completed!")

    def predict_optimal_factor_count(self, metrics_df, confidence_method='ensemble'):
        """Predict optimal factor count with confidence metrics"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        metrics_data = metrics_df[self.feature_cols].values

        # Scale features
        metrics_scaled = (metrics_data - self.global_mean) / np.sqrt(self.global_var)

        # Get anomaly scores
        anomaly_scores = -self.isolation_forest.decision_function(metrics_scaled)
        metrics_df['anomaly_score'] = anomaly_scores

        # Find optimal factor count
        optimal_idx = np.argmax(anomaly_scores)
        optimal_factor_count = metrics_df.iloc[optimal_idx]['factor_count']
        max_anomaly_score = anomaly_scores[optimal_idx]

        # Calculate confidence metrics
        confidence_metrics = self._calculate_confidence_metrics(
            anomaly_scores, metrics_df, confidence_method
        )

        return {
            'optimal_factor_count': optimal_factor_count,
            'anomaly_score': max_anomaly_score,
            'confidence_metrics': confidence_metrics,
            'all_scores': metrics_df[['factor_count', 'anomaly_score']].to_dict('records')
        }

    def _calculate_confidence_metrics(self, anomaly_scores, grouped_metrics, method='ensemble'):
        """Calculate various confidence metrics"""
        max_score = np.max(anomaly_scores)
        second_max_score = np.partition(anomaly_scores, -2)[-2]

        confidence_metrics = {
            'score_separation': max_score - second_max_score,
            'score_percentile': np.mean(anomaly_scores >= max_score) * 100,
            'z_score_from_training': (max_score - self.training_stats['mean_anomaly_score_all']) /
                                     self.training_stats['std_anomaly_score_all'],
            'optimal_likelihood': self._calculate_optimal_likelihood(max_score),
            'stability_confidence': self._calculate_stability_confidence(anomaly_scores)
        }

        # Ensemble confidence (combination of multiple metrics)
        if method == 'ensemble':
            weights = {
                'score_separation': 0.3,
                'optimal_likelihood': 0.4,
                'stability_confidence': 0.3
            }

            # Normalize metrics to 0-1 scale
            norm_separation = min(1.0, confidence_metrics['score_separation'] / 0.1)
            norm_likelihood = confidence_metrics['optimal_likelihood']
            norm_stability = confidence_metrics['stability_confidence']

            confidence_metrics['ensemble_confidence'] = (
                    weights['score_separation'] * norm_separation +
                    weights['optimal_likelihood'] * norm_likelihood +
                    weights['stability_confidence'] * norm_stability
            )

        return confidence_metrics

    def _calculate_optimal_likelihood(self, score):
        """Calculate likelihood that this score represents optimal solution"""
        # Based on training statistics
        optimal_mean = self.training_stats['mean_anomaly_score_optimal']
        optimal_std = self.training_stats['std_anomaly_score_optimal']

        # Probability density function approximation
        z_score = (score - optimal_mean) / optimal_std
        likelihood = max(0, 1 - abs(z_score) / 3)  # Simple approximation
        return likelihood

    def _calculate_stability_confidence(self, anomaly_scores):
        """Calculate confidence based on score distribution stability"""
        # Higher confidence when there's a clear winner
        sorted_scores = np.sort(anomaly_scores)[::-1]
        if len(sorted_scores) < 2:
            return 1.0

        # Ratio of top score to second score
        ratio = sorted_scores[0] / (sorted_scores[1] + 1e-10)
        confidence = min(1.0, (ratio - 1) * 2)  # Scale to 0-1
        return max(0, confidence)

    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            'isolation_forest': self.isolation_forest,
            'feature_cols': self.feature_cols,
            'training_stats': self.training_stats,
            'global_mean': self.global_mean,
            'global_var': self.global_var,
            'global_count': self.global_count,
            'grouped_data': self.grouped_data
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.isolation_forest = model_data['isolation_forest']
        self.scaler = model_data['scaler']
        self.feature_cols = model_data['feature_cols']
        self.training_stats = model_data['training_stats']
        self.global_mean = model_data['global_mean']
        self.global_var = model_data['global_var']
        self.global_count = model_data['global_count']
        self.grouped_data = model_data['grouped_data']
        self.is_trained = True
        print(f"Model loaded from {filepath}")


def demonstrate_generalized_model(test_k:int = 5):
    """Demonstrate the generalized isolation forest approach"""

    # Create and train the model
    optimizer = FactorCountOptimizer(n_estimators=500)

    # Generate training data (large synthetic dataset)
    print("=== Training Phase ===")
    optimizer.train(n_scenarios=1, k_min=3, k_max=8)  # Reduced for demo, use 1000+ in practice

    model_save_path = "factor_optimizer_model_test.pkl"
    # Save the model
    optimizer.save_model(model_save_path)

    # Test on new synthetic data
    print("\n=== Testing Phase ===")
    V, U, true_factors, noise_level = optimizer.generate_scenario(scenario_id=9999, true_k=test_k)
    i_metrics, i_factors = optimizer.run_scenario(V=V, U=U, true_factors=true_factors, k_min=3, k_max=12)

    # Make prediction with confidence
    result = optimizer.predict_optimal_factor_count(i_metrics)

    print(f"True factor count: {test_k}")
    print(f"Predicted optimal factor count: {result['optimal_factor_count']}")
    print(f"Anomaly score: {result['anomaly_score']:.4f}")
    print("\nConfidence Metrics:")
    for metric, value in result['confidence_metrics'].items():
        print(f"  {metric}: {value:.4f}")

    # Visualize results
    plt.figure(figsize=(15, 10))

    # Plot 1: Anomaly scores for test case
    plt.subplot(2, 3, 1)
    scores_df = pd.DataFrame(result['all_scores'])
    plt.bar(scores_df['factor_count'], scores_df['anomaly_score'])
    plt.axvline(x=test_k, color='r', linestyle='--', label=f'True ({test_k})')
    plt.axvline(x=result['optimal_factor_count'], color='g', linestyle='--',
                label=f'Predicted ({result["optimal_factor_count"]})')
    plt.xlabel('Factor Count')
    plt.ylabel('Anomaly Score')
    plt.title('Test Case: Anomaly Scores')
    plt.legend()

    # Plot 2: Training statistics
    plt.subplot(2, 3, 2)
    training_grouped = optimizer.grouped_data.groupby(['factor_count', 'is_optimal'])['anomaly_score'].mean().unstack()
    if True in training_grouped.columns and False in training_grouped.columns:
        plt.plot(training_grouped.index, training_grouped[True], 'go-', label='Optimal')
        plt.plot(training_grouped.index, training_grouped[False], 'ro-', label='Non-optimal')
    plt.xlabel('Factor Count')
    plt.ylabel('Mean Anomaly Score')
    plt.title('Training Data: Score Distribution')
    plt.legend()

    # Plot 3: Confidence metrics
    plt.subplot(2, 3, 3)
    conf_metrics = result['confidence_metrics']
    metrics_to_plot = ['score_separation', 'optimal_likelihood', 'stability_confidence', 'ensemble_confidence']
    values = [conf_metrics.get(metric, 0) for metric in metrics_to_plot]
    plt.bar(range(len(metrics_to_plot)), values)
    plt.xticks(range(len(metrics_to_plot)), [m.replace('_', '\n') for m in metrics_to_plot], rotation=45)
    plt.ylabel('Confidence Value')
    plt.title('Confidence Metrics')
    plt.ylim(0, 1)

    # Plot 4: Test metrics trends
    plt.subplot(2, 3, 4)
    test_grouped = i_metrics.groupby('factor_count')[['sil', 'mean_r2', 'primary_total_ratio']].mean()
    for col in test_grouped.columns:
        plt.plot(test_grouped.index, test_grouped[col], marker='o', label=col)
    plt.axvline(x=test_k, color='r', linestyle='--')
    plt.xlabel('Factor Count')
    plt.ylabel('Metric Value')
    plt.title('Test Case: Metric Trends')
    plt.legend()

    # Plot 5: Q-values
    plt.subplot(2, 3, 5)
    plt.plot(test_grouped.index,
             test_grouped.index.map(lambda x: i_metrics[i_metrics['factor_count'] == x]['qtrue'].mean()),
             'o-', label='Qtrue')
    plt.plot(test_grouped.index,
             test_grouped.index.map(lambda x: i_metrics[i_metrics['factor_count'] == x]['qrobust'].mean()),
             's-', label='Qrobust')
    plt.axvline(x=test_k, color='r', linestyle='--')
    plt.xlabel('Factor Count')
    plt.ylabel('Q Value')
    plt.title('Test Case: Q Values')
    plt.legend()

    # Plot 6: Accuracy across training scenarios
    plt.subplot(2, 3, 6)
    accuracy_data = []
    for scenario_id in optimizer.grouped_data['scenario_id'].unique()[:20]:  # Sample for visualization
        scenario_data = optimizer.grouped_data[optimizer.grouped_data['scenario_id'] == scenario_id]
        true_factor = scenario_data['true_factors'].iloc[0]

        scenario_grouped = scenario_data.groupby('factor_count')['anomaly_score'].mean()
        predicted_factor = scenario_grouped.idxmax()

        accuracy_data.append({
            'scenario': scenario_id,
            'true_factor': true_factor,
            'predicted_factor': predicted_factor,
            'correct': true_factor == predicted_factor
        })

    accuracy_df = pd.DataFrame(accuracy_data)
    accuracy_rate = accuracy_df['correct'].mean()

    plt.bar(['Correct', 'Incorrect'],
            [accuracy_rate, 1 - accuracy_rate],
            color=['green', 'red'], alpha=0.7)
    plt.ylabel('Proportion')
    plt.title(f'Training Accuracy: {accuracy_rate:.2%}')
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

    return optimizer, result


# Run the demonstration
if __name__ == "__main__":
    optimizer, test_result = demonstrate_generalized_model()
