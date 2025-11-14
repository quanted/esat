"""
ARD Factor Search for Source Apportionment
=========================================

This module provides a standalone ARD (Automatic Relevance Determination) factor search class
for optimal factor count determination in Bayesian NMF, along with a test procedure.
"""

import numpy as np
import warnings
from typing import Optional, List, Dict

from hbayes_nmf import HierarchicalBayesianNMF
from utils import generate_dataset, q_loss


class ARDFactorSearch:
    """
    Standalone ARD factor search for Bayesian NMF
    """
    def __init__(self, max_factors: int = 12, min_factors: int = 2, seed: int = 42):
        self.max_factors = max_factors
        self.min_factors = min_factors
        self.seed = seed
        self.model = None
        self.results = None
        self.diagnostics = None

    def run(self, X: np.ndarray, X_uncertainty: Optional[np.ndarray] = None,
            species_names: Optional[List[str]] = None,
            n_samples: int = 2000, n_tune: int = 1000, n_chains: int = 4,
            target_accept: float = 0.95, true_k: Optional[int] = None) -> Dict:
        """
        Run ARD factor search on the provided data
        """
        self.model = HierarchicalBayesianNMF(
            max_factors=self.max_factors,
            min_factors=self.min_factors,
            approach="ard",
            uncertainty_estimation=True,
            seed=self.seed
        )
        self.model.fit(
            X=X,
            X_uncertainty=X_uncertainty,
            species_names=species_names,
            n_samples=n_samples,
            n_tune=n_tune,
            n_chains=n_chains,
            target_accept=target_accept,
            true_k=true_k
        )
        self.results = self.model.selection_result
        self.diagnostics = getattr(self.model, 'factor_diagnostics', None)
        return {
            'optimal_factors': self.model.optimal_factors,
            'factor_profiles': self.model.factor_profiles,
            'factor_contributions': self.model.factor_contributions,
            'diagnostics': self.diagnostics,
            'model': self.model
        }

    def summary(self):
        if self.results is None:
            print("No results available. Run ARD factor search first.")
            return
        print(f"ARD Factor Search Summary:")
        print(f"  - Optimal factors: {self.model.optimal_factors}")
        print(f"  - Approach: {self.model.approach}")
        # Print confidence values for each factor
        factor_conf = getattr(self.results, 'factor_confidence', None)
        if factor_conf is not None:
            print("  - Factor confidence values:")
            for i, conf in enumerate(factor_conf):
                print(f"    Factor {i+1}: confidence = {conf:.3f}")
        if self.diagnostics:
            print(f"  - Profile diagnostics available (see .diagnostics)")
            diag = self.diagnostics.get('profile_diagnostics',None)
            if diag is not None:
                normalized_profiles = diag['normalized_profiles']
                top_species_indices = diag['top_species_indices']
                profile_similarity = diag['profile_similarity']
                print("\nProfile Summary:")
                for i, profile in enumerate(normalized_profiles):
                    print(f"  Factor {i+1}:")
                    top_idx = top_species_indices[i]
                    top_vals = profile[top_idx]
                    print(f"    Top species indices: {top_idx.tolist()}")
                    print(f"    Top species values: {[f'{v:.2f}' for v in top_vals]}")
                    print(f"    Profile (truncated): {[f'{v:.2f}' for v in profile[:6]]} ...")
                min_sim = np.min(profile_similarity[np.triu_indices_from(profile_similarity, k=1)])
                max_sim = np.max(profile_similarity[np.triu_indices_from(profile_similarity, k=1)])
                print(f"\nProfile distinctness (cosine similarity): min={min_sim:.2f}, max={max_sim:.2f}")
                if max_sim > 0.95:
                    print("  [!] Warning: Some factors are highly similar (max cosine similarity > 0.95)")
                # Print pruned factors and reasons
                pruned = diag.get('pruned_factors', None)
                if pruned and pruned['indices']:
                    print("\nPruned factors (redundant/diffuse):")
                    for idx, reason in zip(pruned['indices'], pruned['reasons']):
                        print(f"  - Factor {idx+1}: {reason}")
                imp_pruned = diag.get('importance_pruned_factors', None)
                if imp_pruned and imp_pruned['indices']:
                    print("\nPruned factors (low importance):")
                    for idx, reason in zip(imp_pruned['indices'], imp_pruned['reasons']):
                        print(f"  - Factor {idx+1}: {reason}")
                # Print fallback status
                if diag.get('fallback_triggered', False):
                    print("\n[!] Fallback logic triggered: thresholds were relaxed to avoid under/over-selection.")
            else:
                print(f"  - No profile diagnostics available.")
        else:
            print(f"  - No diagnostics available.")

    def plot(self):
        if self.model:
            self.model.plot_factor_selection_results()
        else:
            print("No model available to plot.")

# Test procedure for ARD factor search
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    print("=== ARD Factor Search Test ===")
    # Generate synthetic dataset
    # seed = 42
    seed = np.random.randint(2**10)
    np.random.seed(seed)
    true_k = np.random.randint(low=3, high=9)
    n_samples = np.random.randint(low=150, high=500)
    n_features = np.random.randint(low=8, high=30)

    V, U, true_k, true_H, true_W, sim = generate_dataset(
        true_k=true_k, n_samples=n_samples, n_features=n_features, seed=seed
    )
    species_names = [f'Species_{i+1}' for i in range(V.shape[1])]
    print(f"Generated dataset: {V.shape[0]} samples × {V.shape[1]} species")
    print(f"True number of factors: {true_k}")
    print(f"Random Seed: {seed}")

    # Run ARD factor search
    ard_search = ARDFactorSearch(max_factors=12, min_factors=2, seed=42)
    results = ard_search.run(V, U, species_names=species_names,
                            n_samples=1500, n_tune=750, n_chains=4, true_k=true_k)
    ard_search.summary()

    # Print factor selection accuracy
    print(f"ARD detected: {results['optimal_factors']} factors (true: {true_k})")
    accuracy = "✓" if results['optimal_factors'] == true_k else "✗"
    print(f"Factor selection accuracy: {accuracy}")

    # Calculate Q-loss
    q_loss_value = q_loss(V, U, results['factor_contributions'], results['factor_profiles'])
    print(f"Q-loss: {q_loss_value:.2f}")

    # Plot results
    ard_search.plot()

    # Show diagnostics (if available)
    if results['diagnostics']:
        print("\nDiagnostics:")
        diag = results['diagnostics']
        for i in range(len(diag['lambda_W_mean'])):
            print(f"Factor {i+1}: Prec_W={diag['lambda_W_mean'][i]:.2f}, Prec_H={diag['lambda_H_mean'][i]:.2f}, Strength={diag['combined_strength'][i]:.1f}")
    print("=== End of ARD Factor Search Test ===")
