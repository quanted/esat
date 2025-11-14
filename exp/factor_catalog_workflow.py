import os
import time

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

import multiprocessing as mp

import plotly.colors
import plotly.subplots as sp
import plotly.graph_objs as go

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from factor_catalog import BatchFactorCatalog
from esat.model.batch_sa import BatchSA
from esat_eval.simulator import Simulator
from esat.data.datahandler import DataHandler


def calculate_metrics(catalog, membership_threshold=0.01):
    all_factors = np.array([v.profile for k, v in catalog.factors.items()])
    factor_assignments = np.array([v.cluster_id for k, v in catalog.factors.items()])
    cluster_centroids = [(c, cluster.centroid) for c, cluster in catalog.clusters.items()]

    i_cluster, i_centroids = zip(*cluster_centroids)
    i_centroids = np.array(i_centroids)

    df_pca0 = pd.DataFrame(all_factors)
    df_pca0["Cluster"] = factor_assignments

    df_centroids0 = pd.DataFrame(i_centroids, index=list(i_cluster))
    assigned_centroids, cluster_size = np.unique(factor_assignments, return_counts=True)
    df_centroids0["Cluster"] = list(i_cluster)
    df_centroids0 = df_centroids0.loc[assigned_centroids]
    df_centroids0["count"] = cluster_size

    point_cluster_n = []
    for i in range(len(all_factors)):
        i_cluster_count = df_centroids0[df_centroids0["Cluster"] == df_pca0["Cluster"].iloc[i]]["count"].values
        point_cluster_n.append(i_cluster_count)
    cluster_n_threshold = int(len(all_factors) * membership_threshold)
    df_pca0["cluster_n"] = point_cluster_n
    df_pca0["cluster_n"] = df_pca0["cluster_n"].astype(int)
    df_pca0 = df_pca0[df_pca0["cluster_n"] > cluster_n_threshold]

    # Filtered data and labels
    filtered_data = df_pca0.drop(columns=["Cluster", "cluster_n"]).values
    filtered_labels = df_pca0["Cluster"].values

    # Number of clusters : The number of clusters identified after filtering. Helps track how many groups are found at each parameter setting.
    n_clusters = len(np.unique(filtered_labels))

    # Silhouette score : Measures how similar each point is to its own cluster compared to other clusters. Ranges from -1 to 1; higher values indicate better-defined, well-separated clusters.
    silhouette = silhouette_score(filtered_data, filtered_labels) if n_clusters > 1 else np.nan

    # Calinski-Harabasz Index : The ratio of between-cluster dispersion to within-cluster dispersion. Higher values indicate more distinct, well-separated clusters.
    calinski_harabasz = calinski_harabasz_score(filtered_data, filtered_labels) if n_clusters > 1 else np.nan

    # Davies-Bouldin Index : The average similarity between each cluster and its most similar one. Lower values indicate better clustering (clusters are compact and well-separated).
    davies_bouldin = davies_bouldin_score(filtered_data, filtered_labels) if n_clusters > 1 else np.nan

    # Dunn Index
    # The ratio of the smallest distance between clusters to the largest intra-cluster distance. Higher values mean clusters are well-separated and compact.
    def dunn_index(X, labels):
        from scipy.spatial.distance import cdist
        unique_labels = np.unique(labels)
        clusters = [X[labels == l] for l in unique_labels]
        # Min inter-cluster distance
        min_inter = np.inf
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = cdist(clusters[i], clusters[j])
                if dist.size > 0:
                    min_inter = min(min_inter, np.min(dist))
        # Max intra-cluster distance
        max_intra = 0
        for cluster in clusters:
            if len(cluster) > 1:
                dist = cdist(cluster, cluster)
                max_intra = max(max_intra, np.max(dist))
        return min_inter / max_intra if max_intra > 0 else np.nan

    dunn = dunn_index(filtered_data, filtered_labels) if n_clusters > 1 else np.nan

    if n_clusters > 0:
        # Fit GMM for BIC/AIC
        # BIC: A model selection criterion for Gaussian Mixture Models. Lower values suggest a better fit with an appropriate penalty for model complexity (number of clusters).
        # AIC: Similar to BIC but with a different penalty for model complexity. Lower values indicate a better fit.
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=0)
        gmm.fit(filtered_data)
        bic = gmm.bic(filtered_data)
        aic = gmm.aic(filtered_data)
    else:
        bic = np.nan
        aic = np.nan

    return {
        "n_clusters": n_clusters,
        "silhouette": silhouette,
        "bic": bic,
        "aic": aic,
        "calinski_harabasz": calinski_harabasz,
        "davies_bouldin": davies_bouldin,
        "dunn": dunn
    }

def plot_metrics_single_k(clustering_results, threshold_criteria, k):
    metrics_names = ["n_clusters", "silhouette", "bic", "aic"]
    fig = sp.make_subplots(
        rows=1, cols=4,
        subplot_titles=metrics_names
    )

    for j, metric in enumerate(metrics_names, start=1):
        y = [clustering_results[t][metric] for t in threshold_criteria]
        fig.add_trace(
            go.Scatter(x=threshold_criteria, y=y, mode="lines+markers", name=metric),
            row=1, col=j
        )
        fig.update_xaxes(title_text="Threshold", row=1, col=j)
        fig.update_yaxes(title_text=metric, row=1, col=j)

    fig.update_layout(
        height=350, width=1200,
        title_text=f"Clustering Metrics by Threshold (k={k})",
        showlegend=False
    )
    fig.show()

def plot_metrics_all_factors(all_results, threshold_criteria, true_factor_count=None, qtrue_means=None, mse_means=None):
    metrics_names = [
        "n_clusters", "silhouette", "bic", "aic",
        "calinski_harabasz", "davies_bouldin", "dunn"
    ]
    factor_counts = sorted(all_results.keys())

    # Assign a color to each threshold
    palette = plotly.colors.qualitative.Plotly
    color_map = {thresh: palette[i % len(palette)] for i, thresh in enumerate(threshold_criteria)}

    # Add one more subplot for QTrue mean and one for MSE mean
    fig = sp.make_subplots(
        rows=3, cols=3,
        subplot_titles=metrics_names + ["Mean MSE", "QTrue Mean"]
    )

    for idx, metric in enumerate(metrics_names):
        row = idx // 3 + 1
        col = idx % 3 + 1
        for threshold in threshold_criteria:
            y = [all_results[k][threshold][metric] for k in factor_counts]
            fig.add_trace(
                go.Scatter(
                    x=factor_counts,
                    y=y,
                    mode="lines+markers",
                    name=f"thresh={threshold}",
                    legendgroup=f"thresh={threshold}",
                    showlegend=(idx == 0),
                    line=dict(color=color_map[threshold]),
                    marker=dict(color=color_map[threshold])
                ),
                row=row, col=col
            )
        fig.update_xaxes(title_text="Factor Count", row=row, col=col)
        fig.update_yaxes(title_text=metric, row=row, col=col)
        if true_factor_count is not None:
            fig.add_vline(
                x=true_factor_count,
                line_dash="dash",
                line_color="red",
                row=row, col=col
            )

    # Add Mean MSE plot at (row=3, col=2)
    if mse_means is not None:
        mse_x = sorted(mse_means.keys())
        mse_y = [mse_means[k] for k in mse_x]
        fig.add_trace(
            go.Scatter(
                x=mse_x,
                y=mse_y,
                mode="lines+markers",
                name="Mean MSE",
                line=dict(color="blue"),
                marker=dict(color="blue")
            ),
            row=3, col=2
        )
        fig.update_xaxes(title_text="Factor Count", row=3, col=2)
        fig.update_yaxes(title_text="Mean MSE", row=3, col=2)
        if true_factor_count is not None:
            fig.add_vline(
                x=true_factor_count,
                line_dash="dash",
                line_color="red",
                row=3, col=2
            )

    # Add QTrue mean plot at (row=3, col=3)
    if qtrue_means is not None:
        qtrue_x = sorted(qtrue_means.keys())
        qtrue_y = [qtrue_means[k] for k in qtrue_x]
        fig.add_trace(
            go.Scatter(
                x=qtrue_x,
                y=qtrue_y,
                mode="lines+markers",
                name="QTrue Mean",
                line=dict(color="black"),
                marker=dict(color="black")
            ),
            row=3, col=3
        )
        fig.update_xaxes(title_text="Factor Count", row=3, col=3)
        fig.update_yaxes(title_text="QTrue Mean", row=3, col=3)
        if true_factor_count is not None:
            fig.add_vline(
                x=true_factor_count,
                line_dash="dash",
                line_color="red",
                row=3, col=3
            )

    fig.update_layout(
        height=1050, width=1400,
        title_text="Clustering Metrics Across All Factor Counts",
    )
    fig.show()

def plot_metrics_all_thresholds(all_results, threshold_criteria, true_factor_count=None):
    metrics_names = [
        "n_clusters", "silhouette", "bic", "aic",
        "calinski_harabasz", "davies_bouldin", "dunn"
    ]
    factor_counts = sorted(all_results.keys())

    fig = sp.make_subplots(
        rows=3, cols=3,
        subplot_titles=metrics_names
    )

    for idx, metric in enumerate(metrics_names):
        row = idx // 3 + 1
        col = idx % 3 + 1
        for k in factor_counts:
            y = [all_results[k][threshold][metric] for threshold in threshold_criteria]
            fig.add_trace(
                go.Scatter(
                    x=threshold_criteria,
                    y=y,
                    mode="lines+markers",
                    name=f"k={k}",
                    legendgroup=f"k={k}",
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )
            # Highlight the true factor count with a marker
            if true_factor_count is not None and k == true_factor_count:
                fig.add_trace(
                    go.Scatter(
                        x=threshold_criteria,
                        y=y,
                        mode="markers",
                        marker=dict(color="red", size=10, symbol="star"),
                        name="True Factor Count",
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )
        fig.update_xaxes(title_text="Threshold", row=row, col=col)
        fig.update_yaxes(title_text=metric, row=row, col=col)

    fig.update_layout(
        height=1050, width=1400,
        title_text="Clustering Metrics Across All Thresholds",
    )
    fig.show()

def compute_metrics_for_threshold(args):
    k, threshold, batch_sa_results, n_features, seed = args
    factor_catalog = BatchFactorCatalog(n_factors=k, n_features=n_features, threshold=threshold, seed=seed)
    for sa in batch_sa_results:
        factor_catalog.add_model(model=sa, norm=True)
    factor_catalog.cluster(max_iterations=15, early_stopping=False)
    metrics = calculate_metrics(factor_catalog, membership_threshold=0.01)
    return threshold, metrics

if __name__ == "__main__":

    method = "ls-nmf"  # "ls-nmf", "ws-nmf"
    models = 20  # the number of models to train
    init_method = "col_means"  # default is column means "col_means", "kmeans", "cmeans"
    seed = 42  # random seed for initialization
    converge_delta = 0.1  # convergence criteria for the change in loss, Q
    converge_n = 25  # convergence criteria for the number of steps where the loss changes by less than converge_delta

    n_factors = int(np.random.random_integers(low=3, high=6))
    # n_factors = 9
    n_features = 40
    n_samples = 1000
    max_iter = 20000
    nm = n_features * n_samples
    rng = np.random.default_rng(seed)

    simulator = Simulator(seed=int(rng.integers(low=0, high=1e10)),
                          factors_n=n_factors,
                          features_n=n_features,
                          samples_n=n_samples,
                          )
    syn_input_df, syn_uncertainty_df = simulator.get_data()
    data_handler = DataHandler.load_dataframe(input_df=syn_input_df, uncertainty_df=syn_uncertainty_df)
    V, U = data_handler.get_data()

    # threshold_criteria = [0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    threshold_criteria = [0.8, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.99]

    all_results = {}
    qtrue_means = {}  # Store QTrue means for each k
    mse_means = {}  # Store mean MSE for each k

    max_factors = 8  # maximum number of factors to evaluate
    t0 = time.time()
    for k in range(2, max_factors+1):
        batch_sa = BatchSA(V=V, U=U, factors=k, models=models, method=method,
                           seed=int(rng.integers(low=0, high=1e8)), max_iter=max_iter,
                           converge_delta=converge_delta, converge_n=converge_n, verbose=False)
        _ = batch_sa.train()
        qtrue_values = [sa.Qtrue for sa in batch_sa.results]
        mean_qtrue = np.mean(qtrue_values)
        qtrue_means[k] = mean_qtrue

        mse_values = [sa.Qtrue/nm for sa in batch_sa.results]
        mean_mse = np.mean(mse_values)
        mse_means[k] = mean_mse

        clustering_results = {}
        args_list = [
            (k, threshold, batch_sa.results, n_features, seed)
            for threshold in threshold_criteria
        ]
        with mp.Pool() as pool:
            results = pool.map(compute_metrics_for_threshold, args_list)
            for threshold, metrics in results:
                clustering_results[threshold] = metrics
        all_results[k] = clustering_results

    plot_metrics_all_factors(
        all_results=all_results,
        threshold_criteria=threshold_criteria,
        true_factor_count=n_factors,
        qtrue_means=qtrue_means,
        mse_means=mse_means
    )
    plot_metrics_all_thresholds(all_results=all_results, threshold_criteria=threshold_criteria, true_factor_count=n_factors)
    t1 = time.time()
    print(f"Total time taken: {(t1 - t0)/60:.2f} minutes")