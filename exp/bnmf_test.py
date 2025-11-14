import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from bayes_nmf import BayesianNMF

from utils import generate_dataset, q_loss
from esat.model.sa import SA
from esat.model.batch_sa import BatchSA
from factor_catalog import BatchFactorCatalog


def compare_nmf_results(V, U, trueH, trueW, species_names=None, sample_names=None, init_H=None, init_W=None, factor_search: bool=False):
    """
    Test BayesianNMF and compare results to known ground truth

    Parameters:
    -----------
    V : array-like, shape (n_samples, n_species)
        Observed concentration matrix
    U : array-like, shape (n_samples, n_species)
        Uncertainty matrix
    trueH : array-like, shape (n_samples, n_factors)
        True factor contributions
    trueW : array-like, shape (n_species, n_factors)
        True factor profiles
    species_names : list, optional
        Names of chemical species
    sample_names : list, optional
        Sample identifiers
    """

    # Initialize BayesianNMF with known number of factors
    n_true_factors = trueH.shape[0]

    print(f"Testing BayesianNMF with {n_true_factors} factors")
    print(f"Data shape: {V.shape}")
    print(f"True W shape: {trueW.shape}")
    print(f"True H shape: {trueH.shape}")

    # Create and fit model
    model = BayesianNMF(
        n_factors=n_true_factors if not factor_search else None,
        max_factors=8,
        auto_factor_selection=factor_search,  # Use known number
        uncertainty_estimation=True
    )
    # Fit model
    print("\nFitting Bayesian NMF model...")
    model.fit(V, U, species_names=species_names, sample_names=sample_names,
              init_H=init_H, init_W=init_W,
              n_samples=2000, n_tune=1200, n_chains=12)

    # Get estimated results
    estimated_H = model.get_factor_profiles(normalize=False)  # (n_species, n_factors)
    estimated_W = model.get_factor_contributions()  # (n_samples, n_factors)

    print("\nModel fitted successfully!")
    print(f"Estimated W shape: {estimated_W.shape}")
    print(f"Estimated H shape: {estimated_H.shape}")

    # Solve factor matching problem (factors may be in different order)
    W_matched, H_matched, factor_mapping = match_factors(
        estimated_W, estimated_H, trueW, trueH
    )

    # Calculate comparison metrics
    results = calculate_comparison_metrics(
        V, trueW, trueH, W_matched, H_matched, model
    )

    # Print results
    print_comparison_results(results, factor_mapping)

    # Create visualizations
    create_comparison_plots_plotly(
        trueW, trueH, W_matched, H_matched,
        species_names, factor_mapping, results
    )

    return {
        'model': model,
        'estimated_W': W_matched,
        'estimated_H': H_matched,
        'factor_mapping': factor_mapping,
        'metrics': results
    }


def match_factors(estimated_W, estimated_H, trueW, trueH):
    """
    Match estimated factors to true factors using Hungarian algorithm
    Factors can be in different order and need to be aligned
    """
    print("\nMatching factors using cosine similarity...")

    # Normalize profiles for comparison

    true_H_norm = trueH / trueH.sum(axis=0)
    est_H_norm = estimated_H / estimated_H.sum(axis=0)

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(true_H_norm, est_H_norm)

    # Use Hungarian algorithm to find optimal matching
    # Convert to cost matrix (1 - similarity)
    cost_matrix = 1 - similarity_matrix
    true_indices, est_indices = linear_sum_assignment(cost_matrix)

    # Reorder estimated factors to match true factors
    W_matched = estimated_W[:, est_indices]
    H_matched = estimated_H[est_indices]

    # Create mapping dictionary
    factor_mapping = {}
    for i, (true_idx, est_idx) in enumerate(zip(true_indices, est_indices)):
        similarity = similarity_matrix[true_idx, est_idx]
        factor_mapping[f'True_Factor_{true_idx + 1}'] = {
            'estimated_factor': f'Est_Factor_{est_idx + 1}',
            'cosine_similarity': similarity,
            'matched_as': f'Factor_{i + 1}'
        }

    return W_matched, H_matched, factor_mapping


def calculate_comparison_metrics(V, trueW, trueH, estimated_W, estimated_H, model):
    """Calculate comprehensive comparison metrics"""

    # Reconstruction accuracy
    true_recon = np.dot(trueW, trueH)
    est_recon = np.dot(estimated_W, estimated_H)

    # Overall metrics
    metrics = {
        'reconstruction': {
            'true_vs_observed_r2': calculate_r2(V, true_recon),
            'estimated_vs_observed_r2': calculate_r2(V, est_recon),
            'true_vs_estimated_r2': calculate_r2(true_recon, est_recon),
            'rmse_true_vs_est': np.sqrt(np.mean((true_recon - est_recon) ** 2))
        }
    }

    # Factor contribution comparison (W matrices)
    W_cosine_sims = []
    W_correlations = []
    for i in range(trueW.shape[1]):
        # Cosine similarity
        cos_sim = cosine_similarity([trueW[:, i]], [estimated_W[:, i]])[0, 0]
        W_cosine_sims.append(cos_sim)

        # Pearson correlation
        corr = np.corrcoef(trueW[:, i], estimated_W[:, i])[0, 1]
        W_correlations.append(corr)

    metrics['contributions'] = {
        'cosine_similarities': W_cosine_sims,
        'correlations': W_correlations,
        'mean_cosine_similarity': np.mean(W_cosine_sims),
        'mean_correlation': np.mean(W_correlations)
    }

    scaled_trueH = trueH / trueH.sum(axis=0)
    scaled_estimated_H = estimated_H / estimated_H.sum(axis=0)

    # Factor profiles comparison (H matrices)
    H_cosine_sims = []
    H_correlations = []
    for i in range(trueH.shape[0]):
        # Cosine similarity
        cos_sim = cosine_similarity([scaled_trueH[i]], [scaled_estimated_H[i]])[0, 0]
        H_cosine_sims.append(cos_sim)

        # Pearson correlation
        corr = np.corrcoef(scaled_trueH[i], scaled_estimated_H[i])[0, 1]
        H_correlations.append(corr)

    metrics['profiles'] = {
        'cosine_similarities': H_cosine_sims,
        'correlations': H_correlations,
        'mean_cosine_similarity': np.mean(H_cosine_sims),
        'mean_correlation': np.mean(H_correlations)
    }

    # Model diagnostics
    if hasattr(model, 'get_model_diagnostics'):
        metrics['model_diagnostics'] = model.get_model_diagnostics()

    return metrics


def calculate_r2(y_true, y_pred):
    """Calculate R-squared"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def print_comparison_results(results, factor_mapping):
    """Print detailed comparison results"""

    print("\n" + "=" * 60)
    print("BAYESIAN NMF COMPARISON RESULTS")
    print("=" * 60)

    # Factor matching results
    print("\nFactor Matching Results:")
    for true_factor, mapping in factor_mapping.items():
        print(f"{true_factor} → {mapping['estimated_factor']} "
              f"(similarity: {mapping['cosine_similarity']:.3f})")

    # Reconstruction accuracy
    print(f"\nReconstruction Accuracy:")
    recon = results['reconstruction']
    print(f"True vs Observed R²: {recon['true_vs_observed_r2']:.4f}")
    print(f"Estimated vs Observed R²: {recon['estimated_vs_observed_r2']:.4f}")
    print(f"True vs Estimated R²: {recon['true_vs_estimated_r2']:.4f}")
    print(f"RMSE (True vs Estimated): {recon['rmse_true_vs_est']:.4f}")

    # Profile comparison
    print(f"\nFactor Profiles (H) Comparison:")
    profiles = results['profiles']
    print(f"Mean Cosine Similarity: {profiles['mean_cosine_similarity']:.4f}")
    print(f"Mean Correlation: {profiles['mean_correlation']:.4f}")

    for i, (cos_sim, corr) in enumerate(zip(profiles['cosine_similarities'],
                                            profiles['correlations'])):
        print(f"  Factor {i + 1}: Cosine={cos_sim:.3f}, Corr={corr:.3f}")

    # Contribution comparison
    print(f"\nFactor Contributions (W) Comparison:")
    contribs = results['contributions']
    print(f"Mean Cosine Similarity: {contribs['mean_cosine_similarity']:.4f}")
    print(f"Mean Correlation: {contribs['mean_correlation']:.4f}")

    for i, (cos_sim, corr) in enumerate(zip(contribs['cosine_similarities'],
                                            contribs['correlations'])):
        print(f"  Factor {i + 1}: Cosine={cos_sim:.3f}, Corr={corr:.3f}")

    # Model diagnostics
    if 'model_diagnostics' in results:
        print(f"\nModel Diagnostics:")
        diag = results['model_diagnostics']
        for key, value in diag.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")


def create_comparison_plots_plotly(trueW, trueH, estimated_W, estimated_H,
                                   species_names, factor_mapping, results):
    n_factors = trueW.shape[1]
    n_species = trueH.shape[1]

    scaled_trueH = trueH / trueH.sum(axis=0)
    scaled_estimated_H = estimated_H / estimated_H.sum(axis=0)

    # 1. Factor Profiles Comparison
    profile_corrs = results['profiles']['correlations']
    profile_titles = []
    for i in range(n_factors):
        mapping = factor_mapping.get(f'True_Factor_{i + 1}', {})
        est_label = mapping.get('estimated_factor', f'Est_Factor_{i + 1}').split("_")[-1]
        profile_titles.append(
            f"True F{i + 1} → Est F{est_label}<br>Pearson r: {profile_corrs[i]:.3f}"
        )
    fig1 = make_subplots(
        rows=1, cols=n_factors,
        shared_yaxes=True,
        subplot_titles=profile_titles
    )
    for i in range(n_factors):
        fig1.add_trace(go.Bar(
            x=species_names if species_names else [f"Species {j + 1}" for j in range(n_species)],
            y=scaled_trueH[i],
            name=f"True F{i + 1}",
            marker_color=f"rgba(0,0,255,{0.3 + 0.7 * i / n_factors})",
            opacity=0.75,
            legendgroup=f"True F{i + 1}",
            showlegend=True
        ), row=1, col=i+1)
        fig1.add_trace(go.Bar(
            x=species_names if species_names else [f"Species {j + 1}" for j in range(n_species)],
            y=scaled_estimated_H[i],
            name=f"Est F{i + 1}",
            marker_color=f"rgba(255,0,0,{0.3 + 0.7 * i / n_factors})",
            opacity=0.75,
            legendgroup=f"Est F{i + 1}",
            showlegend=True
        ), row=1, col=i+1)
    fig1.update_yaxes(title_text="Normalized %", row=1, col=1)
    fig1.update_xaxes(title_text="Species")
    fig1.update_layout(
        title="Factor Profiles Comparison",
        barmode="group",
        height=400,
        width=1600,
        hovermode="x unified"
    )
    fig1.show()

    # 2. Factor Contributions Time Series
    contrib_corrs = results['contributions']['correlations']
    contrib_titles = [
        f"Factor {i+1}<br>Pearson r: {contrib_corrs[i]:.3f}"
        for i in range(n_factors)
    ]
    fig2 = make_subplots(
        rows=n_factors, cols=1,
        vertical_spacing=0.05,
        shared_xaxes=True,
        subplot_titles=contrib_titles
    )
    for i in range(n_factors):
        fig2.add_trace(go.Scatter(
            y=trueW[:, i], mode="lines", name=f"True F{i + 1}", line=dict(color="blue", dash="solid"),
            legendgroup=f"True F{i + 1}", showlegend=True
        ), row=i+1, col=1)
        fig2.add_trace(go.Scatter(
            y=estimated_W[:, i], mode="lines", name=f"Est F{i + 1}", line=dict(color="red", dash="dot"),
            legendgroup=f"Est F{i + 1}", showlegend=True
        ), row=i+1, col=1)
    fig2.update_yaxes(title_text="Contribution")
    fig2.update_xaxes(title_text="Sample", row=n_factors, col=1)
    fig2.update_layout(
        title="Factor Contributions Comparison",
        height=1600,
        width=1200,
        hovermode="x unified"
    )
    fig2.show()

    # 3. Scatter Plots for Each Factor
    fig3 = make_subplots(
        rows=2, cols=n_factors,
        subplot_titles=[f"Factor {i+1}" for i in range(n_factors)],
        vertical_spacing=0.075,
    )
    for i in range(n_factors):
        # Profile scatter
        fig3.add_trace(go.Scatter(
            x=scaled_trueH[i], y=scaled_estimated_H[i],
            mode="markers", name=f"Profile F{i + 1}",
            marker=dict(color="purple", opacity=0.6),
            showlegend=False
        ), row=1, col=1+i)
        # 1:1 line
        fig3.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", name="1:1 Line", line=dict(dash="dash", color="gray"),
            showlegend=False
        ), row=1, col=1+i)

        # Contribution scatter
        fig3.add_trace(go.Scatter(
            x=trueW[:, i], y=estimated_W[:, i],
            mode="markers", name=f"Contrib F{i + 1}",
            marker=dict(color="green", opacity=0.6),
            showlegend=False
        ), row=2, col=1+i)
        # 1:1 line
        max_val = max(np.max(trueW[:, i]), np.max(estimated_W[:, i]))
        fig3.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines", name="1:1 Line", line=dict(dash="dash", color="gray"),
            showlegend=False
        ), row=2, col=1+i)
        fig3.update_xaxes(title_text="True")

    for i in range(n_factors):
        # Profile scatter (row 1): y-axis [0, 1]
        fig3.update_xaxes(range=[0, 1], row=1, col=i + 1)
        fig3.update_yaxes(range=[0, 1], row=1, col=i + 1)
        # Contribution scatter (row 2): y-axis [0, max_val]
        max_val = max(np.max(trueW[:, i]), np.max(estimated_W[:, i]), 1e-8)
        fig3.update_yaxes(range=[0, max_val], row=2, col=i + 1)
        fig3.update_xaxes(range=[0, max_val], row=2, col=i + 1)

    fig3.update_yaxes(title_text="Estimated", row=1, col=1)
    fig3.update_yaxes(title_text="Estimated", row=2, col=1)
    fig3.add_annotation(
        text="Factor Profiles",
        xref="paper", yref="paper",
        x=0.5, y=1.08,  # Centered above row 1
        xanchor="center", yanchor="bottom",
        showarrow=False,
        font=dict(size=20)
    )
    fig3.add_annotation(
        text="Factor Contributions",
        xref="paper", yref="paper",
        x=0.5, y=0.48,  # Centered above row 2 (adjust y as needed)
        xanchor="center", yanchor="bottom",
        showarrow=False,
        font=dict(size=20)
    )
    fig3.update_layout(
        title="Factor Scatter Comparison",
        height=800,
        width=1600,
        hovermode="x unified"
    )
    fig3.show()

def plot_profile_uncertainty_plotly(model, species_names=None):
    H_samples = model.trace.posterior['H'].stack(sample=("chain", "draw")).values
    H_samples = np.transpose(H_samples, (2, 0, 1))  # (n_samples, n_factors, n_species)
    n_factors = H_samples.shape[1]
    n_species = H_samples.shape[2]

    fig = make_subplots(
        rows=1,
        cols=n_factors,
        subplot_titles=[f"Factor {i+1}" for i in range(n_factors)],
        shared_yaxes=True
    )
    for i in range(n_factors):
        H_factor_samples = H_samples[:, i, :]
        H_factor_samples_norm = H_factor_samples / H_factor_samples.sum(axis=1, keepdims=True)
        for j in range(n_species):
            fig.add_trace(
                go.Violin(
                    y=H_factor_samples_norm[:, j],
                    name=species_names[j] if species_names else f"Species {j+1}",
                    box_visible=True,
                    meanline_visible=True,
                    legendgroup=species_names[j] if species_names else f"Species {j+1}",
                    showlegend=(i == n_factors-1),
                    opacity=0.75,
                ),
                row=1, col=i+1
            )
    fig.update_layout(
        title="Estimated Factor Profiles (Posterior Violin)",
        yaxis_title="Normalized Profile",
        # violingap=0,
        # violinmode='group',
        hovermode="x unified",
        width=1600,
        height=400
    )
    fig.update_yaxes(range=[0, 1])
    fig.show()

def plot_contributions_ribbon_plotly(model, sample_names=None, downsample=2):
    W_samples = model.trace.posterior['W'].stack(sample=("chain", "draw")).values
    W_samples = np.transpose(W_samples, (2, 0, 1))  # (n_samples, n_factors, n_draws)
    n_samples = W_samples.shape[1]
    n_factors = W_samples.shape[2]
    idx = np.arange(0, n_samples, downsample)
    x = idx if sample_names is None else [sample_names[j] for j in idx]

    fig = make_subplots(rows=n_factors, cols=1, subplot_titles=[f"Factor {i+1}" for i in range(n_factors)])
    for i in range(n_factors):
        W_factor_samples = W_samples[:, i, :]
        median = np.median(W_factor_samples, axis=1)[idx]
        lower = np.percentile(W_factor_samples, 2.5, axis=1)[idx]
        upper = np.percentile(W_factor_samples, 97.5, axis=1)[idx]

        fig.add_trace(go.Scatter(
            x=x, y=median, mode='lines', name='Median',
            line=dict(color='blue'),
            showlegend=(i == 0),
            hovertemplate=(
                'Sample: %{x}<br>'
                'Median: %{y:.3f}<br>'
                'Lower (2.5%): %{customdata[0]:.3f}<br>'
                'Upper (97.5%): %{customdata[1]:.3f}<extra></extra>'
            ),
            customdata=np.stack([lower, upper], axis=-1)
        ), row=i+1, col=1)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=(i == 0),
            name='95% CI',
            opacity=0.75
        ), row=i+1, col=1)
        fig.update_yaxes(title_text="Contribution", row=i+1, col=1)
    fig.update_layout(
        title="Factor Contributions (95% CI)",
        hovermode="x unified",
        height=1600,
        width=1600
    )
    fig.update_xaxes(title_text="Sample")
    fig.show()

def plot_contributions_confidence_intervals_plotly(model, sample_names=None):
    W_samples = model.trace.posterior['W'].stack(sample=("chain", "draw")).values
    W_samples = np.transpose(W_samples, (2, 0, 1))  # (n_samples, n_factors, n_draws)
    n_samples = W_samples.shape[1]
    n_factors = W_samples.shape[2]
    x = np.arange(n_samples) if sample_names is None else sample_names

    fig = make_subplots(rows=n_factors, cols=1, subplot_titles=[f"Factor {i+1}" for i in range(n_factors)])
    for i in range(n_factors):
        W_factor_samples = W_samples[:, i, :]
        median = np.median(W_factor_samples, axis=1)
        lower = np.percentile(W_factor_samples, 2.5, axis=1)
        upper = np.percentile(W_factor_samples, 97.5, axis=1)
        error_y = dict(
            type='data',
            symmetric=False,
            array=upper - median,
            arrayminus=median - lower,
            thickness=1.5,
            width=3
        )
        fig.add_trace(go.Scatter(
            x=x, y=median,
            mode='markers',
            error_y=error_y,
            name='95% CI',
            marker=dict(color='blue'),
            showlegend=(i == 0)
        ), row=i+1, col=1)
    fig.update_layout(
        title="Factor Contributions (95% CI)",
        yaxis_title="Contribution",
        hovermode="x unified",
        height=1600,
        width=1600
    )
    fig.show()

def plot_factor_all_features_timeseries_with_ci(model, factor_idx, sample_names=None, ci=95, feature_names=None):
    """
    Plot time series with CI ribbon for all features of a specified factor as subplots.

    Parameters
    ----------
    model : fitted BayesianNMF model with .trace
    factor_idx : int
        Index of the factor (column in W, row in H)
    sample_names : list or None
        Optional x-axis labels for samples
    ci : float
        Confidence interval percentage (default 95)
    feature_names : list or None
        Optional feature/species names for subplot titles
    """
    # Extract posterior samples
    W_samples = model.trace.posterior['W'].stack(sample=("chain", "draw")).values
    H_samples = model.trace.posterior['H'].stack(sample=("chain", "draw")).values

    n_samples = W_samples.shape[0]
    n_features = H_samples.shape[1]
    x = np.arange(n_samples) if sample_names is None else sample_names

    fig = make_subplots(
        rows=n_features, cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{feature_names[j] if feature_names else f'Feature {j + 1}'}" for j in range(n_features)],
        vertical_spacing=0.03
    )

    for j in range(n_features):
        contrib = W_samples[:, factor_idx, :] * H_samples[factor_idx, j, :]
        median = np.median(contrib, axis=1)
        lower = np.percentile(contrib, (100 - ci) / 2, axis=1)
        upper = np.percentile(contrib, 100 - (100 - ci) / 2, axis=1)

        customdata = np.stack([lower, upper], axis=-1)

        fig.add_trace(go.Scatter(
            x=x, y=median,
            mode='lines',
            name=feature_names[j] if feature_names else f"Feature {j + 1}",
            line=dict(color='blue'),
            customdata=customdata,
            hovertemplate=(
                'Sample: %{x}<br>'
                'Median: %{y:.3f}<br>'
                'Lower (' + str((100 - ci) / 2) + '%): %{customdata[0]:.3f}<br>'
                'Upper (' + str(100 - (100 - ci) / 2) + '%): %{customdata[1]:.3f}<extra></extra>'
            )
        ), row=j + 1, col=1)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=(j == 0),
            name=f'{ci}% CI',
            opacity=0.65
        ), row=j + 1, col=1)
        fig.update_yaxes(title_text="Contribution", row=j + 1, col=1)

    fig.update_layout(
        title=f"Time Series for All Features (Factor {factor_idx + 1})",
        xaxis_title="Sample" if sample_names is None else "Sample Name",
        hovermode="x unified",
        height=300 * n_features,
        width=1000
    )
    fig.show()


# Example usage:
if __name__ == "__main__":
    # This assumes V, U, trueH, trueW are already defined
    # If not, here's how to create test data:
    random_seed = 11
    print("Creating synthetic test data...")
    np.random.seed(np.random.random_integers(0, 1e5))

    # Define dimensions
    n_samples = 100
    n_species = 10
    n_factors = 4

    V, U, true_k, trueH, trueW, sim = generate_dataset(true_k=n_factors, n_samples=n_samples, n_features=n_species)
    # Species names
    # species_names = ['EC', 'OC', 'SO4', 'NO3', 'NH4', 'Si', 'Cl', 'Na']
    species_names = ['EC', 'OC', 'SO4', 'NO3', 'NH4', 'Si', 'Cl', 'Na', 'K', 'Ca']
    batch_test = BatchSA(
        V=V,
        U=U,
        factors=true_k,
        models=20,
        method='ls-nmf',
        seed=random_seed
    )
    _ = batch_test.train()
    # sim.compare(batch_test)
    # sim.plot_comparison()
    #
    # factor_catalog = BatchFactorCatalog(n_factors=n_factors, n_features=n_species, threshold=0.8, seed=random_seed)
    # for sa in batch_test.results:
    #     factor_catalog.add_model(model=sa, norm=True)
    # factor_catalog.cluster(max_iterations=50)

    init_H = batch_test.results[batch_test.best_model].H
    init_W = batch_test.results[batch_test.best_model].W
    # init_H = init_H / init_H.sum(axis=0)

    print("Running BayesianNMF comparison...")

    # Run the comparison
    comparison_results = compare_nmf_results(
        V, U, trueH=trueH, trueW=trueW,
        species_names=species_names,
        init_H=None, init_W=None,
        factor_search=True
    )

    baye_qloss = q_loss(V, U, W=comparison_results['estimated_W'], H=comparison_results['estimated_H'])
    print(f"\nBayesian NMF Q-loss: {float(baye_qloss):.4f}")
    comparison_results["model"].plot_trace(var_names=['H', 'W'])
    comparison_results["model"].plot_posterior(var_names=['H', 'W'])
    comparison_results["model"].plot_pair(var_names=['H', 'W'])
    comparison_results["model"].plot_energy()

    # Create a new SA instance using BayesianNMF outputs
    # sa_bayes = SA(V=V, U=U, factors=sim.factors_n, seed=sim.seed)
    # sa_bayes.W = comparison_results['model'].get_factor_contributions()   # shape: (samples, factors)
    # sa_bayes.H = comparison_results['model'].get_factor_profiles(normalize=False)  # shape: (factors, features)
    # sa_bayes.WH = sa_bayes.W @ sa_bayes.H

    # fc_results = factor_catalog.compare(matrix=sa_bayes.H)
    # for k, v in fc_results.items():
    #     cluster_id = v['cluster_id']
    #     print(f"BNMF Factor {k}: {v['r2']:.3f} closest to cluster {cluster_id} containing {(100*len(factor_catalog.clusters[cluster_id])/factor_catalog.factor_count):.2f}% membership")
    # print("")

    # Build a new BatchSA with just this model
    # batch_sa_bayes = BatchSA(V=V, U=U, factors=true_k, models=1)
    # batch_sa_bayes.results = [sa_bayes]  # Manually set results if needed
    # batch_sa_bayes.best_model = 0
    #
    # # Compare the BayesianNMF-based model to the true data
    # sim.compare(batch_sa_bayes)
    # sim.plot_comparison()
    # Plot uncertainty in factor profiles

    # plot_profile_uncertainty_plotly(comparison_results['model'], species_names=species_names)
    # plot_factor_all_features_timeseries_with_ci(comparison_results['model'], factor_idx=0, sample_names=None, ci=95, feature_names=species_names)

    print("\nComparison complete!")
    print("Check the generated plots and printed metrics above.")

    # model_to_graphviz(comparison_results['model'].model).render("bnmf-test")