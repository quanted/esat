import os
import copy
import logging

import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go

import matplotlib.pyplot as plt


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn

from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE, MDS

from tqdm import tqdm

from esat.model.sa import SA
from esat.model.batch_sa import BatchSA
from esat.data.datahandler import DataHandler
from esat_eval.simulator import Simulator


logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)  # Latent space
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, encoding_dim, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, encoding_dim)
        self.decoder = Decoder(encoding_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def build_autoencoder(input_dim, encoding_dim):
    autoencoder = Autoencoder(input_dim, encoding_dim)
    encoder = autoencoder.encoder
    return autoencoder, encoder


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

    def plot(self):
        n_features = len(self.centroid)
        factor_matrix = np.array([factor.profile for factor in self.factors])

        box_plot = go.Figure()
        for i in range(n_features):
            box_plot.add_trace(go.Box(
                y=factor_matrix[:, i],
                boxpoints="all",
                jitter=0.5,
                whiskerwidth=0.2,
                marker_size=2,
                line_width=1,
                name=f"Feature {i + 1}")
            )
        box_plot.add_trace(go.Scatter(
            x=np.arange(n_features),
            y=self.centroid,
            name="Centroid",
            mode='markers',
            marker=dict(color='red', size=10)
        ))
        box_plot.update_layout(title="Clustered Factor Profile", width=1200, height=800)
        box_plot.show()


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
        model_assignment = np.array([v.model_id for k, v in self.factors.items()])
        factor_assignments = np.array([v.cluster_id for k, v in self.factors.items()])
        cluster_centroids = [(c, cluster.centroid) for c, cluster in self.clusters.items()]

        i_cluster, i_centroids = zip(*cluster_centroids)
        i_length = len(i_cluster)
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
        i_centroids = df_centroids0[cluster_columns].values
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

    def plot(self, count_threshold: int = 3, method="mds", membership_p: float = 0.0):
        all_factors = np.array([v.profile for k, v in self.factors.items()])
        model_assignment = np.array([v.model_id for k, v in self.factors.items()])
        factor_assignments = np.array([v.cluster_id for k, v in self.factors.items()])
        cluster_centroids = [(c, cluster.centroid) for c, cluster in self.clusters.items()]

        i_cluster, i_centroids = zip(*cluster_centroids)
        i_length = len(i_cluster)
        i_centroids = np.array(i_centroids)

        df_pca0 = pd.DataFrame(all_factors)
        factor_columns = df_pca0.columns
        df_pca0["Cluster"] = factor_assignments
        df_pca0["text"] = "Profile " + df_pca0["Cluster"].astype(str) + " Model: " + model_assignment.astype(str)

        df_centroids0 = pd.DataFrame(i_centroids, index=list(i_cluster))
        cluster_columns = df_centroids0.columns
        assigned_centroids, cluster_size = np.unique(factor_assignments, return_counts=True)
        df_centroids0["Cluster"] = list(i_cluster)
        df_centroids0 = df_centroids0.loc[assigned_centroids]
        df_centroids0["count"] = cluster_size

        if membership_p > 0.0:
            point_cluster_n = []
            for i in range(len(all_factors)):
                i_cluster_count = df_centroids0[df_centroids0["Cluster"] == df_pca0["Cluster"].iloc[i]]["count"].values
                point_cluster_n.append(i_cluster_count)
            cluster_n_threshold = int(len(all_factors) * membership_p)
            df_pca0["cluster_n"] = point_cluster_n
            df_pca0["cluster_n"] = df_pca0["cluster_n"].astype(int)
            df_pca0 = df_pca0[df_pca0["cluster_n"] > cluster_n_threshold]

            df_centroids0 = df_centroids0[df_centroids0["count"] > cluster_n_threshold]
        elif count_threshold > 0:
            df_centroids0 = df_centroids0[df_centroids0["count"] > count_threshold]

        all_factors = df_pca0[factor_columns].values
        i_centroids = df_centroids0[cluster_columns].values

        if method.lower() == "tsne":
            samples = np.vstack((all_factors, i_centroids))
            factor_model = TSNE(n_components=3, random_state=0, perplexity=min(50, len(samples)))
            reduction_results = factor_model.fit_transform(samples)
            factor_reduction = reduction_results[:len(all_factors), :]
            pca_centroids = reduction_results[len(all_factors):, :]
        elif method.lower() == "mds":
            factor_model = MDS(n_components=3, random_state=0, metric=True, max_iter=300)
            reduction_results = factor_model.fit_transform(np.vstack((all_factors, i_centroids)))
            factor_reduction = reduction_results[:len(all_factors), :]
            pca_centroids = reduction_results[len(all_factors):, :]
        else:
            autoencoder, encoder = build_autoencoder(input_dim=i_centroids.shape[1], encoding_dim=3)
            autoencoder.compile(optimizer='adam', loss='mse')
            autoencoder.fit(all_factors, all_factors, epochs=50, batch_size=16, shuffle=True, validation_split=0.2,
                            verbose=0)
            factor_reduction = encoder.predict(all_factors, verbose=0)
            pca_centroids = encoder.predict(i_centroids, verbose=0)

        df_pca = pd.DataFrame(factor_reduction, columns=['PCA1', 'PCA2', 'PCA3'])
        df_pca["Cluster"] = df_pca0["Cluster"].values
        df_pca["text"] = df_pca0["text"].values

        df_centroids = pd.DataFrame(pca_centroids, columns=['PCA1', 'PCA2', 'PCA3'], index=list(df_centroids0.index))
        df_centroids["text"] = "Centroid " + df_centroids.index.astype(str)
        df_centroids["Cluster"] = df_centroids0["Cluster"].values
        df_centroids["count"] = df_centroids0["count"].values

        color_map = self.generate_continuous_colormap(len(df_centroids["Cluster"]), colormap_name='rainbow')
        full_colormap = dict(zip(list(df_centroids["Cluster"]), [color[1] for color in color_map]))
        df_pca["color"] = df_pca["Cluster"].map(full_colormap)
        df_centroids["color"] = df_centroids["Cluster"].map(full_colormap)

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=df_pca["PCA1"],
            y=df_pca["PCA2"],
            z=df_pca["PCA3"],
            mode='markers',
            marker=dict(
                size=3,
                color=df_pca["color"],
                opacity=0.5
            ),
            text=df_pca["text"],
            hoverinfo='text'
        ))
        fig.add_trace(go.Scatter3d(
            x=df_centroids['PCA1'],
            y=df_centroids['PCA2'],
            z=df_centroids['PCA3'],
            mode='markers',
            marker=dict(
                size=3,
                color="black",
                symbol='x',
                opacity=0.5
            ),
            text=df_centroids["text"],
            hoverinfo='text'
        ))
        # Update layout for better visualization
        fig.update_layout(
            title=f"{self.n_factors} Factor Clustering - Method: {method}",
            scene=dict(
                xaxis_title="PCA1",
                yaxis_title="PCA2",
                zaxis_title="PCA3"
            ),
            showlegend=False,
            height=800,
            width=1000,
            margin=dict(l=5, r=5, b=5, t=50)
        )
        # Show the plot
        fig.show()

    def generate_continuous_colormap(self, n, colormap_name='viridis'):
        # Get the colormap from matplotlib
        cmap = plt.get_cmap(colormap_name)

        # Generate colors from the colormap
        colors = [cmap(i / (n - 1)) for i in range(n)]

        # Convert RGBA to RGB and then to hex
        colors_hex = ['#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)) for r, g, b, a in colors]

        # Create a Plotly color scale
        plotly_color_scale = [(i / (n - 1), color) for i, color in enumerate(colors_hex)]

        return plotly_color_scale

    def animate(self, to_file: bool = False, base_matrix=None, profiled_matrix=None, method="mds", membership_p: float = 0.0):
        all_factors = np.array([v.profile for k, v in self.factors.items()])
        model_assignment = np.array([v.model_id for k, v in self.factors.items()])
        required_members = int(len(all_factors) * membership_p)

        color_map = self.generate_continuous_colormap(len(self.clusters), colormap_name='rainbow')
        full_colormap = dict(zip(list(self.clusters.keys()), [color[1] for color in color_map]))

        sample_lengths = []
        samples = []
        for i in range(len(self.state)):
            cluster_centroids = self.state[i]["cluster_centroids"]
            i_cluster, i_centroids = zip(*cluster_centroids)
            i_centroids = np.array(i_centroids)
            samples.append(i_centroids)
            sample_lengths.append(len(i_centroids))
        i_centroids = np.vstack(samples)

        sample_list = [all_factors, i_centroids]
        samples_n = np.sum(sample_lengths) + len(all_factors)
        plot_base = False
        if base_matrix is not None:
            sample_list.append(base_matrix)
            plot_base = True
        plot_profiled = False
        if profiled_matrix is not None:
            sample_list.append(profiled_matrix)
            plot_profiled = True

        if method.lower() == "tsne":
            samples = np.vstack(sample_list)
            factor_model = TSNE(n_components=3, random_state=0, perplexity=min(50, len(samples)))
            reduction_results = factor_model.fit_transform(samples)
            factor_reduction = reduction_results[:len(all_factors), :]
            cluster_centroids = reduction_results[len(all_factors):samples_n, :]
            i_max = samples_n
            if plot_base:
                base_points = reduction_results[i_max:i_max + base_matrix.shape[0], :]
                i_max += base_matrix.shape[0]
            if plot_profiled:
                profiled_points = reduction_results[i_max:i_max + profiled_matrix.shape[0], :]
        elif method.lower() == "mds":
            samples = np.vstack(sample_list)
            factor_model = MDS(n_components=3, random_state=0, metric=True, max_iter=300)
            reduction_results = factor_model.fit_transform(samples)
            factor_reduction = reduction_results[:len(all_factors), :]
            cluster_centroids = reduction_results[len(all_factors):samples_n, :]
            i_max = samples_n
            if plot_base:
                base_points = reduction_results[i_max:i_max + base_matrix.shape[0], :]
                i_max += base_matrix.shape[0]
            if plot_profiled:
                profiled_points = reduction_results[i_max:i_max + profiled_matrix.shape[0], :]
        else:
            autoencoder, encoder = build_autoencoder(input_dim=i_centroids.shape[1], encoding_dim=3)
            autoencoder.compile(optimizer='adam', loss='mse')
            autoencoder.fit(all_factors, all_factors, epochs=50, batch_size=16, shuffle=True, validation_split=0.2,
                            verbose=0)
            factor_reduction = encoder.predict(all_factors, verbose=0)
            cluster_centroids = encoder.predict(i_centroids, verbose=0)
            if plot_base:
                base_points = encoder.predict(base_matrix, verbose=0)
            if plot_profiled:
                profiled_points = encoder.predict(profiled_matrix, verbose=0)

        cumulative_lengths = np.cumsum([0] + sample_lengths)
        state_centroids = [cluster_centroids[cumulative_lengths[i]: cumulative_lengths[i + 1]] for i in
                           range(len(sample_lengths))]

        if base_matrix is not None:
            df_base = pd.DataFrame(base_points, columns=['x', 'y', 'z'])

        if profiled_matrix is not None:
            df_profiled = pd.DataFrame(profiled_points, columns=['x', 'y', 'z'])

        df_pca0 = pd.DataFrame(factor_reduction, columns=['x', 'y', 'z'])
        df_pca0["text"] = "P: " + df_pca0.index.astype(str) + ", M: " + model_assignment.astype(str)

        frames = []
        for i in range(len(self.state)):
            cluster_centroids = self.state[i]["cluster_centroids"]
            i_cluster, i_centroids = zip(*cluster_centroids)
            factor_assignments = self.state[i]["assignment"]
            i_centroids = state_centroids[i]

            i_length = len(i_cluster)
            i_centroids = np.array(i_centroids)

            df_pca = pd.DataFrame(factor_reduction, columns=['x', 'y', 'z'])
            factor_columns = df_pca.columns
            df_pca["Cluster"] = factor_assignments
            df_pca["text"] = "P: " + df_pca0.index.astype(str) + ", M: " + model_assignment.astype(str) + ", C:" + \
                             df_pca["Cluster"].astype(str)

            df_centroids = pd.DataFrame(i_centroids, columns=['x', 'y', 'z'], index=list(i_cluster))
            cluster_columns = df_centroids.columns
            assigned_centroids, cluster_size = np.unique(factor_assignments, return_counts=True)
            df_centroids["text"] = "Centroid " + df_centroids.index.astype(str)
            df_centroids["Cluster"] = list(i_cluster)
            df_centroids = df_centroids.loc[assigned_centroids]
            df_centroids["count"] = cluster_size
            df_centroids = df_centroids[df_centroids["count"] > required_members]

            df_pca["color"] = df_pca["Cluster"].map(full_colormap)
            df_centroids["color"] = df_centroids["Cluster"].map(full_colormap)

            data = [
                go.Scatter3d(
                    x=df_pca["x"],
                    y=df_pca["y"],
                    z=df_pca["z"],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=df_pca["color"],
                        opacity=0.4
                    ),
                    text=df_pca["text"],
                    hoverinfo='text'
                ),
                go.Scatter3d(
                    x=df_centroids['x'],
                    y=df_centroids['y'],
                    z=df_centroids['z'],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color="black",
                        symbol='x',
                        opacity=0.75
                    ),
                    text=df_centroids["text"],
                    hoverinfo='text'
                )
            ]
            if plot_base:
                data.append(go.Scatter3d(
                    x=df_base['x'],
                    y=df_base['y'],
                    z=df_base['z'],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color="black",
                        symbol='cross'
                    ),
                    name="Base Factor"
                ))
            if plot_profiled:
                data.append(go.Scatter3d(
                    x=df_profiled['x'],
                    y=df_profiled['y'],
                    z=df_profiled['z'],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color="green",
                        symbol='cross'
                    ),
                    name="P Factor"
                ))
            frames.append(
                go.Frame(data=data,
                         layout=go.Layout(
                             annotations=[
                                 dict(
                                     x=1,
                                     y=1,
                                     showarrow=False,
                                     text=f"Iteration: {i + 1}/{len(self.state)}",
                                     xref="paper",
                                     yref="paper",
                                     font=dict(size=14)
                                 )
                             ]
                         ),
                         name=str(i)
                         )
            )
            if i == 0:
                state0 = [
                    go.Scatter3d(
                        x=df_pca0["x"],
                        y=df_pca0["y"],
                        z=df_pca0["z"],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color="gray",
                            opacity=0.4
                        ),
                        text=df_pca0["text"],
                        hoverinfo='text'
                    ),
                    go.Scatter3d(
                        x=df_centroids['x'],
                        y=df_centroids['y'],
                        z=df_centroids['z'],
                        mode='markers',
                        marker=dict(
                            size=2,
                            color="black",
                            symbol='x',
                            opacity=0.75
                        ),
                        text=df_centroids["text"],
                        hoverinfo='text'
                    )
                ]
                if plot_base:
                    state0.append(go.Scatter3d(
                        x=df_base['x'],
                        y=df_base['y'],
                        z=df_base['z'],
                        mode='markers',
                        marker=dict(
                            size=4,
                            color="black",
                            symbol='cross'
                        ),
                        name="Base Factor"
                    ))
                if plot_profiled:
                    state0.append(go.Scatter3d(
                        x=df_profiled['x'],
                        y=df_profiled['y'],
                        z=df_profiled['z'],
                        mode='markers',
                        marker=dict(
                            size=4,
                            color="green",
                            symbol='cross'
                        ),
                        name="P Factor"
                    ))
        df_pca0 = pd.DataFrame(factor_reduction, columns=['x', 'y', 'z'])

        fig = go.Figure(
            data=state0,
            layout=go.Layout(
                title="Factor Profile Clustering",
                height=1000,
                width=1000,
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None,
                                   dict(frame=dict(duration=500, redraw=True), fromcurrent=True, mode="immediate")]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                        dict(
                            args=[[0], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                            label="Reset",
                            method="animate"),
                    ]
                )],
                annotations=[
                    dict(
                        x=1,
                        y=1,
                        showarrow=False,
                        text="Iteration: NA",
                        xref="paper",
                        yref="paper",
                        font=dict(size=14)
                    )
                ],
                showlegend=False
            ),
            frames=frames
        )
        if to_file:
            pio.write_html(fig, file="factor_clustering.html", auto_open=False)
        else:
            fig.show()

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
                else:
                    factor_score = len(self.clusters[factor.cluster_id])
                model_score += factor_score
            model.score = model_score

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
            min_shape = (
            min(i_centroids.shape[0], j_centroids.shape[0]), min(i_centroids.shape[1], j_centroids.shape[1]))
            centroid_shifts = np.mean(
                np.linalg.norm(i_centroids[:min_shape[0], :min_shape[1]] - j_centroids[:min_shape[0], :min_shape[1]],
                               axis=1))
            if i_centroids.shape[0] > j_centroids.shape[0]:
                centroid_shifts += np.mean(
                    (len(i_centroids[min_shape[0]:]) / i_centroids.shape[0]) * i_centroids[min_shape[0]:])
            else:
                centroid_shifts += np.mean(
                    (len(j_centroids[min_shape[0]:]) / j_centroids.shape[0]) * j_centroids[min_shape[0]:])
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



if __name__ == "__main__":

    method = "ls-nmf"  # "ls-nmf", "ws-nmf"
    models = 20  # the number of models to train
    init_method = "col_means"  # default is column means "col_means", "kmeans", "cmeans"
    seed = 42  # random seed for initialization
    converge_delta = 0.1  # convergence criteria for the change in loss, Q
    converge_n = 25  # convergence criteria for the number of steps where the loss changes by less than converge_delta

    n_factors = 6
    n_features = 20
    n_samples = 1000
    batch_size = 250
    max_batches = 10
    i_batches = 0
    n_models = 5
    max_iter = 20000

    rng = np.random.default_rng(seed)

    simulator = Simulator(seed=rng.integers(low=0, high=1e10),
                          factors_n=n_factors,
                          features_n=n_features,
                          samples_n=n_samples,
                          )
    syn_input_df, syn_uncertainty_df = simulator.get_data()
    data_handler = DataHandler.load_dataframe(input_df=syn_input_df, uncertainty_df=syn_uncertainty_df)
    V, U = data_handler.get_data()

    batch_sa = BatchSA(V=V, U=U, factors=n_factors, models=n_models, method=method,
                       seed=int(rng.integers(low=0, high=1e8)), max_iter=max_iter,
                       converge_delta=converge_delta, converge_n=converge_n, verbose=False)
    _ = batch_sa.train()

    factor_catalog = BatchFactorCatalog(n_factors=n_factors, n_features=n_features, threshold=0.8, seed=seed)
    for sa in batch_sa.results:
        factor_catalog.add_model(model=sa, norm=True)
    factor_catalog.cluster(max_iterations=50)
