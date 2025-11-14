"""
Infinite Beta Process Bayesian NMF Factor Search
================================================

This module provides a standalone infinite factor model (Beta Process) factor search class
for optimal factor count determination in Bayesian NMF, along with a test procedure.
"""

import numpy as np
import warnings
from typing import Optional, List, Dict

from hbayes_nmf import HierarchicalBayesianNMF
from utils import generate_dataset, q_loss


class InfiniteBetaBNMF:
    """
    Standalone infinite factor model (Beta Process) factor search for Bayesian NMF
    """
    def __init__(self, max_factors: int = 12, min_factors: int = 2, seed: int = 42):
        self.max_factors = max_factors
        self.min_factors = min_factors
        self.seed = seed
        self.model = None
        self.results = None

    def run(self, X: np.ndarray, X_uncertainty: Optional[np.ndarray] = None,
            species_names: Optional[List[str]] = None,
            n_samples: int = 3000, n_tune: int = 1500, n_chains: int = 4,
            target_accept: float = 0.95, fast_mode: bool = False) -> Dict:
        """
        Run infinite factor model (Beta Process) factor search on the provided data
        If fast_mode is True, use reduced sampling and factor count for speed.
        """
        if fast_mode:
            n_samples = 1000
            n_tune = 500
            n_chains = 4
            self.max_factors = min(self.max_factors, 12)
            target_accept = 0.90
            print("[Fast Mode] Using reduced sampling and factor count for speed.")
        self.model = HierarchicalBayesianNMF(
            max_factors=self.max_factors,
            min_factors=self.min_factors,
            approach="infinite",
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
            target_accept=target_accept
        )
        self.results = self.model.selection_result
        return {
            'optimal_factors': self.model.optimal_factors,
            'factor_profiles': self.model.factor_profiles,
            'factor_contributions': self.model.factor_contributions,
            'factor_probabilities': getattr(self.results, 'factor_probabilities', None),
            'model': self.model
        }

    def _adaptive_factor_selection(self, cumulative_threshold: float = 0.8, min_weight: float = 0.10):
        """
        Select factors adaptively based on cumulative stick-breaking weights and minimum weight cutoff.
        Returns selected factor indices and weights.
        """
        factor_probs = getattr(self.results, 'factor_probabilities', None)
        if factor_probs is None:
            return [], []
        weights = np.array(factor_probs)
        sorted_idx = np.argsort(weights)[::-1]
        cum_weights = np.cumsum(weights[sorted_idx])
        n_selected = np.searchsorted(cum_weights, cumulative_threshold) + 1
        selected_factors = sorted_idx[:n_selected]
        # Apply more stringent minimum weight cutoff
        selected_factors = [i for i in selected_factors if weights[i] >= min_weight]
        selected_weights = weights[selected_factors]
        # Diagnostic: print cumulative weights and elbow
        print("    [Diagnostics] Cumulative stick-breaking weights:")
        for i, idx in enumerate(sorted_idx):
            print(f"      Factor {idx+1}: weight = {weights[idx]:.3f}, cumulative = {cum_weights[i]:.3f}")
        return selected_factors, selected_weights.tolist()

    def _elbow_factor_selection(self, min_weight: float = 0.10):
        """
        Option 3: Select number of factors using the elbow method on sorted stick-breaking weights.
        Uses the maximum distance to chord (kneedle) method for robust elbow detection.
        Returns selected factor indices and weights.
        """
        factor_probs = getattr(self.results, 'factor_probabilities', None)
        if factor_probs is None:
            return [], []
        weights = np.array(factor_probs)
        sorted_idx = np.argsort(weights)[::-1]
        sorted_weights = weights[sorted_idx]
        n = len(sorted_weights)
        # Normalize x and y
        x = np.arange(n)
        y = (sorted_weights - sorted_weights.min()) / (sorted_weights.max() - sorted_weights.min() + 1e-8)
        # Chord endpoints
        p1 = np.array([0, y[0]])
        p2 = np.array([n-1, y[-1]])
        # Compute distances to chord
        def point_line_dist(pt, a, b):
            return np.linalg.norm(np.cross(b-a, a-pt)) / (np.linalg.norm(b-a) + 1e-8)
        dists = np.array([point_line_dist(np.array([i, y[i]]), p1, p2) for i in range(n)])
        elbow_idx = np.argmax(dists)
        # Select all factors up to elbow, but apply min_weight threshold
        selected_factors = [i for i in sorted_idx[:elbow_idx+1] if weights[i] >= min_weight]
        selected_weights = weights[selected_factors]
        return selected_factors, selected_weights.tolist()

    def _min_weight_factor_selection(self, min_weight: float = 0.05):
        """
        Select all factors with stick-breaking weight above min_weight.
        Returns selected factor indices and weights.
        """
        factor_probs = getattr(self.results, 'factor_probabilities', None)
        if factor_probs is None:
            return [], []
        weights = np.array(factor_probs)
        selected_factors = [i for i, w in enumerate(weights) if w >= min_weight]
        selected_weights = weights[selected_factors]
        return selected_factors, selected_weights.tolist()

    def summary(self):
        if self.results is None:
            print("No results available. Run infinite factor search first.")
            return
        print(f"Infinite Beta Process Factor Search Summary:")
        print(f"  - [Option 1] Model optimal factor count: {self.model.optimal_factors}")
        print(f"  - Approach: {self.model.approach}")
        factor_probs = getattr(self.results, 'factor_probabilities', None)
        if factor_probs is not None:
            print("  - Factor probabilities (stick-breaking weights):")
            for i, prob in enumerate(factor_probs):
                print(f"    Factor {i+1}: weight = {prob:.3f}")
            # Option 2: Adaptive threshold selection (80% cumulative weight, min weight 0.10)
            selected_factors2, selected_weights2 = self._adaptive_factor_selection(cumulative_threshold=0.8, min_weight=0.10)
            print(f"  - [Option 2] Adaptive estimated optimal factor count: {len(selected_factors2)}")
            print(f"    Selected factors: {selected_factors2}")
            print(f"    Selected weights: {[f'{w:.3f}' for w in selected_weights2]}")
            # Option 3: Elbow method
            selected_factors3, selected_weights3 = self._elbow_factor_selection(min_weight=0.10)
            print(f"  - [Option 3] Elbow estimated optimal factor count: {len(selected_factors3)}")
            print(f"    Selected factors: {selected_factors3}")
            print(f"    Selected weights: {[f'{w:.3f}' for w in selected_weights3]}")
            # Diagnostics: Try multiple thresholds and min_weights
            for cum_thresh in [0.8, 0.85, 0.9]:
                sel_factors_adapt, sel_weights_adapt = self._adaptive_factor_selection(cumulative_threshold=cum_thresh, min_weight=0.02)
                print(f"  - [Diagnostics] Adaptive (cum_thresh={cum_thresh}, min_weight=0.02): count={len(sel_factors_adapt)}, factors={sel_factors_adapt}, weights={[f'{w:.3f}' for w in sel_weights_adapt]}")
            for mw in [0.02, 0.05, 0.10]:
                sel_factors_elbow, sel_weights_elbow = self._elbow_factor_selection(min_weight=mw)
                print(f"  - [Diagnostics] Elbow (min_weight={mw}): count={len(sel_factors_elbow)}, factors={sel_factors_elbow}, weights={[f'{w:.3f}' for w in sel_weights_elbow]}")
            for mw in [0.02, 0.05, 0.10]:
                sel_factors_minw, sel_weights_minw = self._min_weight_factor_selection(min_weight=mw)
                print(f"  - [Diagnostics] MinWeight (min_weight={mw}): count={len(sel_factors_minw)}, factors={sel_factors_minw}, weights={[f'{w:.3f}' for w in sel_weights_minw]}")
            # Confidence note
            predictions = [self.model.optimal_factors, len(selected_factors2), len(selected_factors3)]
            most_common = max(set(predictions), key=predictions.count)
            n_agree = predictions.count(most_common)
            if n_agree >= 2:
                print(f"  - [Confidence] {n_agree} methods agree on {most_common} factors. High confidence.")
            else:
                print(f"  - [Confidence] Methods disagree. Low confidence.")
        # Print Q-loss, MSE, RMSE together, in order
        try:
            # Use the original input matrix and predicted matrix from selected factors
            X_true = getattr(self.model, 'X_input', None)
            if X_true is None:
                X_true = self.model.contributions_df.values @ self.model.profiles_df.values.T
            X_pred = self.model.factor_contributions @ self.model.factor_profiles
            unc = getattr(self.model, 'X_uncertainty', None)
            # Q-loss
            try:
                if unc is not None:
                    ql = q_loss(X_true, unc, self.model.factor_contributions, self.model.factor_profiles)
                    print(f"  - Q-loss: {ql:.2f}")
                else:
                    print("  - Q-loss: N/A (uncertainty not available)")
            except Exception as e:
                print(f"  - Q-loss: N/A ({e})")
            # MSE, RMSE
            try:
                from sklearn.metrics import mean_squared_error
                mse = mean_squared_error(X_true.flatten(), X_pred.flatten())
                rmse = np.sqrt(mse)
                print(f"  - MSE: {mse:.4f}")
                print(f"  - RMSE: {rmse:.4f}")
            except Exception as e:
                print(f"  - MSE: N/A ({e})")
                print(f"  - RMSE: N/A ({e})")
        except Exception as e:
            print(f"  - Q-loss: N/A ({e})")
            print(f"  - MSE: N/A ({e})")
            print(f"  - RMSE: N/A ({e})")

    def plot(self):
        if self.model:
            self.model.plot_factor_selection_results()
        else:
            print("No model available to plot.")

# Test procedure for infinite factor model
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    print("=== Infinite Beta Process Factor Search Test ===")
    # Generate synthetic dataset
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

    # Run infinite factor search (fast mode)
    inf_search = InfiniteBetaBNMF(max_factors=12, min_factors=2, seed=42)
    results = inf_search.run(V, U, species_names=species_names, n_samples=3000,
                             n_tune=1500, n_chains=6, target_accept=0.99, fast_mode=True)
    inf_search.summary()
    # Print factor selection accuracy
    print(f"Infinite Beta Process detected: {results['optimal_factors']} factors (true: {true_k})")
    accuracy = "✓" if results['optimal_factors'] == true_k else "✗"
    print(f"Factor selection accuracy: {accuracy}")
    # Calculate Q-loss
    q_loss_value = q_loss(V, U, results['factor_contributions'], results['factor_profiles'])
    print(f"Q-loss: {q_loss_value:.2f}")
    # Plot results
    inf_search.plot()

    print("=== End of Infinite Beta Process Factor Search Test ===")
