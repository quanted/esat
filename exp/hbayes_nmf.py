"""
Hierarchical Bayesian NMF for Optimal Factor Count Determination
================================================================

This module implements several approaches for determining the optimal number of factors
in Bayesian Non-negative Matrix Factorization for source apportionment:

1. Hierarchical Bayesian NMF with Automatic Relevance Determination (ARD)
2. Discrete Factor Search using Bayesian Model Comparison
3. Nested Model Comparison with Cross-Validation
4. Infinite Factor Model with Beta Process
"""

import os
import warnings
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import arviz as az
import pymc as pm
import pytensor.tensor as pt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

try:
    from utils import generate_dataset, q_loss
except ImportError:
    print("Warning: utils module not found. Some functions may not work.")

pio.renderers.default = "browser"
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class FactorSelectionResult:
    """Container for factor selection results"""
    optimal_factors: int
    model_evidence: Dict[int, float]
    information_criteria: Dict[int, Dict[str, float]]
    stability_metrics: Dict[int, float]
    cross_validation_scores: Dict[int, float]
    posterior_samples: Optional[Dict] = None
    factor_probabilities: Optional[np.ndarray] = None
    factor_confidence: Optional[np.ndarray] = None


class HierarchicalBayesianNMF:
    """
    Hierarchical Bayesian NMF with multiple approaches for optimal factor determination
    """

    def __init__(self,
                 max_factors: int = 15,
                 min_factors: int = 2,
                 approach: str = "ard",  # "ard", "discrete_search", "nested_comparison", "infinite"
                 uncertainty_estimation: bool = True,
                 seed: int = 42):
        """
        Initialize Hierarchical Bayesian NMF

        Parameters
        ----------
        max_factors : int
            Maximum number of factors to consider
        min_factors : int
            Minimum number of factors to consider
        approach : str
            Factor selection approach:
            - "ard": Automatic Relevance Determination
            - "discrete_search": Discrete factor search with model comparison
            - "nested_comparison": Nested model comparison with CV
            - "infinite": Infinite factor model with Beta Process
        uncertainty_estimation : bool
            Whether to estimate uncertainty
        seed : int
            Random seed
        """
        self.max_factors = max_factors
        self.min_factors = min_factors
        self.approach = approach
        self.uncertainty_estimation = uncertainty_estimation
        self.seed = seed

        # Results storage
        self.optimal_factors = None
        self.factor_profiles = None
        self.factor_contributions = None
        self.uncertainty_bounds = None
        self.model_diagnostics = {}
        self.trace = None
        self.selection_result = None

    def fit(self,
            X: np.ndarray,
            X_uncertainty: Optional[np.ndarray] = None,
            species_names: Optional[List[str]] = None,
            sample_names: Optional[List[str]] = None,
            n_samples: int = 2000,
            n_tune: int = 1000,
            n_chains: int = 4,
            target_accept: float = 0.95,
            true_k: Optional[int] = None) -> 'HierarchicalBayesianNMF':
        """
        Fit the hierarchical Bayesian NMF model with optimal factor determination

        Parameters
        ----------
        X : np.ndarray
            Concentration matrix (n_samples, n_species)
        X_uncertainty : np.ndarray, optional
            Uncertainty matrix
        species_names : List[str], optional
            Species names
        sample_names : List[str], optional
            Sample names
        n_samples : int
            Number of MCMC samples
        n_tune : int
            Number of tuning steps
        n_chains : int
            Number of MCMC chains
        target_accept : float
            Target acceptance rate

        Returns
        -------
        self : HierarchicalBayesianNMF
            Fitted model
        """
        X = np.ascontiguousarray(X, dtype=np.float64)

        if X_uncertainty is None:
            X_uncertainty = np.maximum(0.05 * X, 0.01 * np.mean(X, axis=0))
        X_uncertainty = np.ascontiguousarray(X_uncertainty, dtype=np.float64)

        print(f"Fitting Hierarchical Bayesian NMF using '{self.approach}' approach...")

        if self.approach == "ard":
            self.selection_result, self.factor_diagnostics = self._fit_ard_model(
                X, X_uncertainty, n_samples, n_tune, n_chains, target_accept, true_k=true_k
            )
        elif self.approach == "discrete_search":
            self.selection_result = self._fit_discrete_search(
                X, X_uncertainty, n_samples, n_tune, n_chains, target_accept
            )
        elif self.approach == "nested_comparison":
            self.selection_result = self._fit_nested_comparison(
                X, X_uncertainty, n_samples, n_tune, n_chains, target_accept
            )
        elif self.approach == "infinite":
            self.selection_result = self._fit_infinite_model(
                X, X_uncertainty, n_samples, n_tune, n_chains, target_accept
            )
        else:
            raise ValueError(f"Unknown approach: {self.approach}")

        # Extract results
        self._extract_results(X, species_names, sample_names)

        return self

    def _fit_ard_model(self, X, X_uncertainty, n_samples, n_tune, n_chains, target_accept, true_k=None):
        """
        Automatic Relevance Determination approach using hierarchical priors
        """
        print("Using Automatic Relevance Determination (ARD) approach...")

        N, D = X.shape
        K = self.max_factors

        with pm.Model() as model:
            # Hierarchical prior on factor relevance (ARD)
            # Use more informative priors for better factor selection
            alpha_lambda = pm.Gamma('alpha_lambda', alpha=2, beta=2)  # More informative prior

            # Factor-specific precision parameters with shared hyperprior
            lambda_W = pm.Gamma('lambda_W', alpha=alpha_lambda, beta=1, shape=K)
            lambda_H = pm.Gamma('lambda_H', alpha=alpha_lambda, beta=1, shape=K)

            # Factor matrices with ARD priors
            W = pm.Exponential('W', lam=lambda_W[None, :], shape=(N, K))
            H = pm.Exponential('H', lam=lambda_H[:, None], shape=(K, D))

            # Reconstruction
            X_recon = pm.math.dot(W, H)

            # Likelihood with given uncertainty
            pm.Normal('X_obs', mu=X_recon, sigma=X_uncertainty, observed=X)

            # Sample
            self.trace = pm.sample(
                draws=n_samples, tune=n_tune, chains=n_chains,
                target_accept=target_accept, random_seed=self.seed,
                cores=min(n_chains, os.cpu_count() - 1),
                progressbar=True, return_inferencedata=True
            )

        # Enhanced factor relevance determination with balanced criteria
        lambda_W_samples = self.trace.posterior['lambda_W'].values
        lambda_H_samples = self.trace.posterior['lambda_H'].values
        W_samples = self.trace.posterior['W'].values
        H_samples = self.trace.posterior['H'].values

        # Calculate factor strengths (total contribution across samples/species)
        W_mean = W_samples.mean(axis=(0, 1))  # (N, K)
        H_mean = H_samples.mean(axis=(0, 1))  # (K, D)

        # Multiple strength metrics
        factor_strength_W = np.sum(W_mean, axis=0)  # Sum across samples for each factor
        factor_strength_H = np.sum(H_mean, axis=1)  # Sum across species for each factor
        combined_strength = factor_strength_W * factor_strength_H

        # Factor mass contribution (more realistic measure)
        factor_mass = np.sum(W_mean * H_mean.sum(axis=1), axis=0)

        # Alternative method: use precision parameters
        lambda_W_mean = lambda_W_samples.mean(axis=(0, 1))
        lambda_H_mean = lambda_H_samples.mean(axis=(0, 1))

        # Combined precision score (geometric mean)
        combined_precision = np.sqrt(lambda_W_mean * lambda_H_mean)

        # Method 1: Precision-based pruning (weight H much more than W, and stricter threshold)
        precision_threshold_W = np.percentile(lambda_W_mean, 70)  # Stricter: top 30% only
        precision_threshold_H = np.percentile(lambda_H_mean, 60)  # Stricter: top 40% only
        low_precision_W = lambda_W_mean < precision_threshold_W
        low_precision_H = lambda_H_mean < precision_threshold_H
        precision_score = 0.85 * low_precision_H.astype(float) + 0.15 * low_precision_W.astype(float)
        precision_active = precision_score >= 0.7  # Stricter threshold for activation

        # Method 2: Strength-based selection (more lenient)
        strength_threshold = np.percentile(combined_strength, 60)  # Top 40%
        mass_threshold = np.percentile(factor_mass, 60)

        strength_active = (combined_strength > strength_threshold) | (factor_mass > mass_threshold)

        # Method 3: Relative contribution method (more lenient)
        total_strength = np.sum(combined_strength)
        relative_contribution = combined_strength / (total_strength + 1e-8)
        contribution_threshold = 0.03  # Must contribute at least 3% (reduced from 5%)
        contribution_active = relative_contribution > contribution_threshold

        # Method 4: Signal-to-noise ratio (more lenient)
        factor_snr = np.array([
            np.mean(W_mean[:, k]) / (np.std(W_mean[:, k]) + 1e-8) for k in range(K)
        ])
        snr_threshold = np.percentile(factor_snr, 50)  # More lenient
        snr_active = factor_snr > snr_threshold

        # Method 5: Profile coherence (more lenient)
        profile_coherence = np.array([
            np.max(H_mean[k, :]) / (np.mean(H_mean[k, :]) + 1e-8) for k in range(K)
        ])
        coherence_threshold = np.percentile(profile_coherence, 50)  # More lenient
        coherence_active = profile_coherence > coherence_threshold

        # Method 6: Factor importance (new) - based on reconstruction contribution
        X_recon_full = W_mean @ H_mean
        factor_importance = np.array([
            np.var(W_mean[:, k:k+1] @ H_mean[k:k+1, :]) / (np.var(X_recon_full) + 1e-8)
            for k in range(K)
        ])
        importance_threshold = 0.01  # Must explain at least 1% of variance
        importance_active = factor_importance > importance_threshold

        # Improved voting system with weighted criteria
        # Give more weight to the most reliable indicators
        vote_weights = {
            'precision': 1.0,
            'strength': 2.0,
            'contribution': 2.5,
            'importance': 2.0,
            'snr': 1.0,
            'coherence': 1.0
        }

        # Calculate weighted votes (use precision_score instead of precision_active)
        weighted_votes = (
            precision_score * vote_weights['precision'] +
            strength_active.astype(float) * vote_weights['strength'] +
            contribution_active.astype(float) * vote_weights['contribution'] +
            importance_active.astype(float) * vote_weights['importance'] +
            snr_active.astype(float) * vote_weights['snr'] +
            coherence_active.astype(float) * vote_weights['coherence']
        )

        # Total possible votes
        total_possible_votes = sum(vote_weights.values())
        # RELAXED: Lower threshold for selection
        vote_threshold = total_possible_votes * 0.55  # Was 0.7, now 0.55

        # Primary selection: weighted voting
        active_factors = weighted_votes >= vote_threshold

        # RELAXED: Lower contribution and importance thresholds
        contribution_threshold = 0.025  # Was 0.05, now 0.025
        importance_threshold = 0.008    # Was 0.02, now 0.008
        contribution_active = relative_contribution > contribution_threshold
        importance_active = factor_importance > importance_threshold

        # Secondary selection: must pass BOTH contribution threshold AND importance threshold
        active_factors = active_factors & (contribution_active & importance_active)

        # Hard cap: allow a buffer for extra factors
        hard_cap = min(K, max(2, K//2 + 4))  # Was +2, now +4
        n_active = np.sum(active_factors)

        fallback_triggered = False
        if n_active > hard_cap:
            # Create comprehensive factor scores
            factor_scores = (
                relative_contribution * 0.35 +  # 35% contribution weight
                factor_importance * 0.25 +       # 25% importance weight
                (1 - combined_precision / np.max(combined_precision)) * 0.20 +  # 20% precision weight
                factor_snr / np.max(factor_snr) * 0.10 +  # 10% SNR weight
                profile_coherence / np.max(profile_coherence) * 0.10  # 10% coherence weight
            )
            # Keep top factors by score
            top_indices = np.argsort(factor_scores)[-hard_cap:]
            active_factors = np.zeros(K, dtype=bool)
            active_factors[top_indices] = True
            n_active = hard_cap

        # More robust minimum required factors logic
        min_required = max(2, min(self.min_factors, N//15, D//4))
        if n_active < min_required or n_active < self.min_factors:
            # Score all factors and keep the best ones
            factor_scores = (
                relative_contribution * 0.35 +
                factor_importance * 0.25 +
                (1 - combined_precision / np.max(combined_precision)) * 0.20 +
                factor_snr / np.max(factor_snr) * 0.10 +
                profile_coherence / np.max(profile_coherence) * 0.10
            )
            top_indices = np.argsort(factor_scores)[-self.min_factors:]
            active_factors = np.zeros(K, dtype=bool)
            active_factors[top_indices] = True
            n_active = self.min_factors
            fallback_triggered = True

        self.optimal_factors = n_active

        # Calculate model evidence for ARD model
        model_evidence = self._calculate_model_evidence_ard(X, X_uncertainty)

        # Post-selection fit check: if fit is poor, relax thresholds and re-select
        W_tmp = W_mean[:, active_factors]
        H_tmp = H_mean[active_factors, :]
        X_pred_tmp = W_tmp @ H_tmp
        r2_tmp = r2_score(X.flatten(), X_pred_tmp.flatten())
        q_loss_tmp = np.sum(((X - X_pred_tmp) / X_uncertainty) ** 2)
        if r2_tmp < 0.05 or q_loss_tmp > 1.5 * np.sum(((X - W_mean @ H_mean) / X_uncertainty) ** 2):
            # RELAXED: Lower thresholds further
            vote_threshold = total_possible_votes * 0.45
            contribution_threshold = 0.015
            importance_threshold = 0.005
            contribution_active = relative_contribution > contribution_threshold
            importance_active = factor_importance > importance_threshold
            active_factors = weighted_votes >= vote_threshold
            active_factors = active_factors & (contribution_active & importance_active)
            n_active = np.sum(active_factors)
            fallback_triggered = True
            # Re-apply hard cap and min_required
            if n_active > hard_cap:
                factor_scores = (
                    relative_contribution * 0.35 +
                    factor_importance * 0.25 +
                    (1 - combined_precision / np.max(combined_precision)) * 0.20 +
                    factor_snr / np.max(factor_snr) * 0.10 +
                    profile_coherence / np.max(profile_coherence) * 0.10
                )
                top_indices = np.argsort(factor_scores)[-hard_cap:]
                active_factors = np.zeros(K, dtype=bool)
                active_factors[top_indices] = True
                n_active = hard_cap
            if n_active < min_required or n_active < self.min_factors:
                factor_scores = (
                    relative_contribution * 0.35 +
                    factor_importance * 0.25 +
                    (1 - combined_precision / np.max(combined_precision)) * 0.20 +
                    factor_snr / np.max(factor_snr) * 0.10 +
                    profile_coherence / np.max(profile_coherence) * 0.10
                )
                top_indices = np.argsort(factor_scores)[-self.min_factors:]
                active_factors = np.zeros(K, dtype=bool)
                active_factors[top_indices] = True
                n_active = self.min_factors
                fallback_triggered = True
            self.optimal_factors = n_active

        # --- Compute factor confidence values ---
        diagnostics_matrix = np.stack([
            factor_importance,
            relative_contribution,
            (1 - combined_precision / np.max(combined_precision)),
            factor_snr / np.max(factor_snr),
            profile_coherence / np.max(profile_coherence),
            weighted_votes / np.max(weighted_votes)
        ], axis=1)  # shape: (K, 6)
        weights = np.array([
            vote_weights['importance'],
            vote_weights['contribution'],
            vote_weights['precision'],
            vote_weights['snr'],
            vote_weights['coherence'],
            2.0  # voting score weight
        ])
        raw_confidence = diagnostics_matrix @ weights
        confidence_norm = (raw_confidence - raw_confidence.min()) / (raw_confidence.max() - raw_confidence.min() + 1e-8)
        factor_confidence = np.zeros(K)
        factor_confidence[active_factors] = confidence_norm[active_factors]

        # --- Post-selection pruning: Remove redundant, diffuse, and low-importance factors ---
        # Only for factors currently marked as active
        H_sel = H_mean[active_factors, :]
        n_active = np.sum(active_factors)
        if n_active > 0:
            norm_profiles = np.array([h / (np.sum(h) + 1e-8) for h in H_sel])
            sim_matrix = cosine_similarity(norm_profiles)
            pruned = np.zeros(n_active, dtype=bool)
            pruned_reasons = [None] * n_active
            imp_pruned = np.zeros(n_active, dtype=bool)
            imp_pruned_reasons = [None] * n_active
            for i in range(n_active):
                # Redundant: high similarity to another factor
                if np.any(sim_matrix[i, :] > 0.95) and np.sum(sim_matrix[i, :] > 0.95) > 1:
                    pruned[i] = True
                    pruned_reasons[i] = 'Redundant profile (cosine similarity > 0.95)'
                # Diffuse: max value too low
                if np.max(norm_profiles[i]) < 0.08:
                    pruned[i] = True
                    pruned_reasons[i] = 'Diffuse profile (max < 0.08)'
                # Low importance
                imp_val = factor_importance[active_factors][i]
                if imp_val < 0.008:
                    imp_pruned[i] = True
                    imp_pruned_reasons[i] = 'Low importance (< 0.008)'
            # Remove pruned factors
            keep_mask = ~(pruned | imp_pruned)
            # If all would be pruned, keep top by importance
            if np.sum(keep_mask) == 0:
                top_idx = np.argsort(factor_importance[active_factors])[-1:]
                keep_mask[top_idx] = True
            # Map back to full active_factors
            active_indices = np.where(active_factors)[0]
            final_active = np.zeros_like(active_factors)
            final_active[active_indices[keep_mask]] = True
            active_factors = final_active
            n_active = np.sum(active_factors)
            fallback_triggered = True
        # If still too many, keep top by importance/contribution
        hard_cap = min(K, max(2, K//2 + 2))
        if n_active > hard_cap:
            scores = factor_importance * 0.6 + relative_contribution * 0.4
            top_idx = np.argsort(scores[active_factors])[-hard_cap:]
            active_indices = np.where(active_factors)[0]
            final_active = np.zeros_like(active_factors)
            final_active[active_indices[top_idx]] = True
            active_factors = final_active
            n_active = np.sum(active_factors)
            fallback_triggered = True
        self.optimal_factors = n_active

        # Calculate model evidence for ARD model
        model_evidence = self._calculate_model_evidence_ard(X, X_uncertainty)

        # Post-selection fit check: if fit is poor, relax thresholds and re-select
        W_tmp = W_mean[:, active_factors]
        H_tmp = H_mean[active_factors, :]
        X_pred_tmp = W_tmp @ H_tmp
        r2_tmp = r2_score(X.flatten(), X_pred_tmp.flatten())
        q_loss_tmp = np.sum(((X - X_pred_tmp) / X_uncertainty) ** 2)
        if r2_tmp < 0.05 or q_loss_tmp > 1.5 * np.sum(((X - W_mean @ H_mean) / X_uncertainty) ** 2):
            # RELAXED: Lower thresholds further
            vote_threshold = total_possible_votes * 0.45
            contribution_threshold = 0.015
            importance_threshold = 0.005
            contribution_active = relative_contribution > contribution_threshold
            importance_active = factor_importance > importance_threshold
            active_factors = weighted_votes >= vote_threshold
            active_factors = active_factors & (contribution_active & importance_active)
            n_active = np.sum(active_factors)
            fallback_triggered = True
            # Re-apply hard cap and min_required
            if n_active > hard_cap:
                factor_scores = (
                    relative_contribution * 0.35 +
                    factor_importance * 0.25 +
                    (1 - combined_precision / np.max(combined_precision)) * 0.20 +
                    factor_snr / np.max(factor_snr) * 0.10 +
                    profile_coherence / np.max(profile_coherence) * 0.10
                )
                top_indices = np.argsort(factor_scores)[-hard_cap:]
                active_factors = np.zeros(K, dtype=bool)
                active_factors[top_indices] = True
                n_active = hard_cap
            if n_active < min_required or n_active < self.min_factors:
                factor_scores = (
                    relative_contribution * 0.35 +
                    factor_importance * 0.25 +
                    (1 - combined_precision / np.max(combined_precision)) * 0.20 +
                    factor_snr / np.max(factor_snr) * 0.10 +
                    profile_coherence / np.max(profile_coherence) * 0.10
                )
                top_indices = np.argsort(factor_scores)[-self.min_factors:]
                active_factors = np.zeros(K, dtype=bool)
                active_factors[top_indices] = True
                n_active = self.min_factors
                fallback_triggered = True
            self.optimal_factors = n_active

        # --- Compute factor confidence values ---
        diagnostics_matrix = np.stack([
            factor_importance,
            relative_contribution,
            (1 - combined_precision / np.max(combined_precision)),
            factor_snr / np.max(factor_snr),
            profile_coherence / np.max(profile_coherence),
            weighted_votes / np.max(weighted_votes)
        ], axis=1)  # shape: (K, 6)
        weights = np.array([
            vote_weights['importance'],
            vote_weights['contribution'],
            vote_weights['precision'],
            vote_weights['snr'],
            vote_weights['coherence'],
            2.0  # voting score weight
        ])
        raw_confidence = diagnostics_matrix @ weights
        confidence_norm = (raw_confidence - raw_confidence.min()) / (raw_confidence.max() - raw_confidence.min() + 1e-8)
        factor_confidence = np.zeros(K)
        factor_confidence[active_factors] = confidence_norm[active_factors]

        # --- Profile diagnostics ---
        normalized_profiles = []
        top_species_indices = []
        top_species_values = []
        profile_similarity = np.zeros((n_active, n_active))
        pruned_indices = []
        pruned_reasons = []
        imp_pruned_indices = []
        imp_pruned_reasons = []
        # For selected factors only
        H_sel = H_mean[active_factors, :]
        for i, profile in enumerate(H_sel):
            norm_profile = profile / (np.sum(profile) + 1e-8)
            normalized_profiles.append(norm_profile)
            top_idx = np.argsort(norm_profile)[-5:][::-1]
            top_species_indices.append(top_idx)
            top_species_values.append(norm_profile[top_idx])
        # Cosine similarity between profiles
        for i in range(n_active):
            for j in range(n_active):
                profile_similarity[i, j] = cosine_similarity([normalized_profiles[i]], [normalized_profiles[j]])[0, 0]
        # Pruned factors: redundant (high similarity) or diffuse (low max value)
        # Now use the pruned info from above
        for i in range(n_active):
            # These indices are now after pruning, so just mark as empty
            pass
        profile_diagnostics = {
            'normalized_profiles': normalized_profiles,
            'top_species_indices': top_species_indices,
            'top_species_values': top_species_values,
            'profile_similarity': profile_similarity,
            'pruned_factors': {'indices': [], 'reasons': []},
            'importance_pruned_factors': {'indices': [], 'reasons': []},
            'fallback_triggered': fallback_triggered
        }

        factor_diagnostics = {
            'factor_importance': factor_importance,
            'relative_contribution': relative_contribution,
            'combined_precision': combined_precision,
            'factor_snr': factor_snr,
            'profile_coherence': profile_coherence,
            'weighted_votes': weighted_votes,
            'vote_weights': vote_weights,
            'factor_confidence': factor_confidence,
            'lambda_W_mean': lambda_W_mean,
            'lambda_H_mean': lambda_H_mean,
            'combined_strength': combined_strength,
            'profile_diagnostics': profile_diagnostics
        }

        return FactorSelectionResult(
            optimal_factors=self.optimal_factors,
            model_evidence={self.optimal_factors: model_evidence},
            information_criteria={},
            stability_metrics={},
            cross_validation_scores={},
            posterior_samples={'W': W_samples, 'H': H_samples},
            factor_probabilities=active_factors.astype(float),
            factor_confidence=factor_confidence
        ), factor_diagnostics

    def _fit_discrete_search(self, X, X_uncertainty, n_samples, n_tune, n_chains, target_accept):
        """
        Discrete factor search using Bayesian model comparison
        """
        print("Using Discrete Factor Search with Bayesian Model Comparison...")

        results = {}
        model_evidences = {}
        information_criteria = {}
        stability_metrics = {}

        # Fit models for each factor count
        for k in range(self.min_factors, self.max_factors + 1):
            print(f"Fitting model with {k} factors...")

            result = self._fit_single_factor_model(
                X, X_uncertainty, k, n_samples//2, n_tune//2, n_chains, target_accept
            )

            results[k] = result
            model_evidences[k] = result['log_marginal_likelihood']
            information_criteria[k] = {
                'waic': result['waic'],
                'loo': result['loo'],
                'bic': result['bic']
            }
            stability_metrics[k] = result['stability']

        # Select optimal factors using model evidence
        optimal_k = max(model_evidences.keys(), key=lambda k: model_evidences[k])

        # Refit optimal model with full sampling
        print(f"Refitting optimal model with {optimal_k} factors...")
        final_result = self._fit_single_factor_model(
            X, X_uncertainty, optimal_k, n_samples, n_tune, n_chains, target_accept
        )

        self.optimal_factors = optimal_k
        self.trace = final_result['trace']

        return FactorSelectionResult(
            optimal_factors=optimal_k,
            model_evidence=model_evidences,
            information_criteria=information_criteria,
            stability_metrics=stability_metrics,
            cross_validation_scores={}
        )

    def _fit_nested_comparison(self, X, X_uncertainty, n_samples, n_tune, n_chains, target_accept):
        """
        Nested model comparison with cross-validation
        """
        print("Using Nested Model Comparison with Cross-Validation...")

        cv_scores = {}
        model_evidences = {}
        information_criteria = {}

        # Cross-validation setup
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.seed)

        for k in range(self.min_factors, self.max_factors + 1):
            print(f"Cross-validating model with {k} factors...")

            cv_scores_k = []
            for train_idx, val_idx in kfold.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                U_train, U_val = X_uncertainty[train_idx], X_uncertainty[val_idx]

                # Fit on training data
                result = self._fit_single_factor_model(
                    X_train, U_train, k, n_samples//4, n_tune//4, 2, target_accept
                )

                # Predict on validation data
                W_mean = result['W_mean']
                H_mean = result['H_mean']
                X_pred = W_mean @ H_mean

                # Calculate log-likelihood on validation set
                log_lik = self._calculate_log_likelihood(X_val, X_pred, U_val)
                cv_scores_k.append(log_lik)

            cv_scores[k] = np.mean(cv_scores_k)

            # Also fit full model for comparison
            full_result = self._fit_single_factor_model(
                X, X_uncertainty, k, n_samples//2, n_tune//2, n_chains, target_accept
            )
            model_evidences[k] = full_result['log_marginal_likelihood']
            information_criteria[k] = {
                'waic': full_result['waic'],
                'loo': full_result['loo'],
                'bic': full_result['bic']
            }

        # Select optimal factors using CV score
        optimal_k = max(cv_scores.keys(), key=lambda k: cv_scores[k])

        # Refit optimal model
        print(f"Refitting optimal model with {optimal_k} factors...")
        final_result = self._fit_single_factor_model(
            X, X_uncertainty, optimal_k, n_samples, n_tune, n_chains, target_accept
        )

        self.optimal_factors = optimal_k
        self.trace = final_result['trace']

        return FactorSelectionResult(
            optimal_factors=optimal_k,
            model_evidence=model_evidences,
            information_criteria=information_criteria,
            stability_metrics={},
            cross_validation_scores=cv_scores
        )

    def _fit_infinite_model(self, X, X_uncertainty, n_samples, n_tune, n_chains, target_accept):
        """
        Infinite factor model using Beta Process (adaptive selection)
        """
        print("Using Infinite Factor Model with Beta Process...")
        N, D = X.shape
        K = self.max_factors

        with pm.Model() as model:
            # Beta process / stick-breaking construction
            alpha = pm.Gamma('alpha', alpha=.5, beta=1.0)  # Concentration parameter
            v = pm.Beta('v', alpha=1, beta=alpha, shape=K)

            # Stick-breaking weights
            stick_segments = pm.Deterministic('stick_segments', v * pt.concatenate([[1], pt.cumprod(1 - v)[:-1]]))
            pi = pm.Deterministic('pi', stick_segments / pt.sum(stick_segments))

            # Factor matrices
            H = pm.Exponential('H', lam=1.25, shape=(K, D))

            lam_W = pm.Gamma('lam_W', alpha=.5, beta=1.0, shape=K)
            W = pm.Exponential('W', lam=lam_W[None, :], shape=(N, K))

            # Weighted reconstruction
            X_recon = pm.math.dot(W * pi, H)

            # Likelihood
            pm.Normal('X_obs', mu=X_recon, sigma=X_uncertainty, observed=X)

            # Sample
            self.trace = pm.sample(
                draws=n_samples, tune=n_tune, chains=n_chains,
                target_accept=target_accept, random_seed=self.seed,
                cores=min(n_chains, os.cpu_count() - 2),
                progressbar=True, return_inferencedata=True,
                max_treedepth=12
            )

        pi_samples = self.trace.posterior['pi'].values
        pi_mean = pi_samples.mean(axis=(0, 1))

        # --- Stricter selection: only factors with pi_mean > min_weight ---
        min_weight = 0.12  # Increased threshold for more robust pruning
        active_factors = pi_mean > min_weight
        n_active = np.sum(active_factors)
        print(f"Selected factors (pi_mean > {min_weight}): {np.where(active_factors)[0].tolist()}")
        print(f"Selected weights: {[f'{float(pi_mean[i]):.3f}' for i in np.where(active_factors)[0]]}")

        # If too few factors, add next highest-weight factors up to min_factors
        if n_active < self.min_factors:
            top_idx = np.argsort(-pi_mean)[:self.min_factors]
            final_active = np.zeros_like(active_factors)
            final_active[top_idx] = True
            active_factors = final_active
            n_active = np.sum(active_factors)
            print(f"[Enforced min_factors] Selected factors: {np.where(active_factors)[0].tolist()}")
            print(f"[Enforced min_factors] Selected weights: {[f'{float(pi_mean[i]):.3f}' for i in np.where(active_factors)[0]]}")
        self.optimal_factors = n_active

        model_evidence = self._calculate_model_evidence_infinite()

        # Extract mean factor matrices for selected factors
        W_samples = self.trace.posterior['W'].values
        H_samples = self.trace.posterior['H'].values
        W_flat = W_samples.reshape(-1, W_samples.shape[-2], W_samples.shape[-1])
        H_flat = H_samples.reshape(-1, H_samples.shape[-2], H_samples.shape[-1])
        factor_contributions = np.mean(W_flat, axis=0)[:, active_factors]
        factor_profiles = np.mean(H_flat, axis=0)[active_factors, :]
        ql = q_loss(V=X, W=factor_contributions, H=factor_profiles, U=X_uncertainty)
        print(f"Reconstruction Q loss with {n_active} factors: {ql:.4f}")
        return FactorSelectionResult(
            optimal_factors=self.optimal_factors,
            model_evidence={self.optimal_factors: model_evidence},
            information_criteria={},
            stability_metrics={},
            cross_validation_scores={},
            factor_probabilities=pi_mean
        )

    def _fit_single_factor_model(self, X, X_uncertainty, k, n_samples, n_tune, n_chains, target_accept):
        """
        Fit a single Bayesian NMF model with k factors
        """
        N, D = X.shape

        with pm.Model() as model:
            # Simple exponential priors
            W = pm.Exponential('W', lam=1.0, shape=(N, k))
            H = pm.Exponential('H', lam=1.0, shape=(k, D))

            # Reconstruction
            X_recon = pm.math.dot(W, H)

            # Likelihood - store as named variable for WAIC calculation
            obs = pm.Normal('X_obs', mu=X_recon, sigma=X_uncertainty, observed=X)

            # Sample
            trace = pm.sample(
                draws=n_samples, tune=n_tune, chains=n_chains,
                target_accept=target_accept, random_seed=self.seed,
                cores=min(n_chains, os.cpu_count() - 1),
                progressbar=False, return_inferencedata=True
            )

        # Calculate metrics
        W_samples = trace.posterior['W'].values
        H_samples = trace.posterior['H'].values
        W_mean = W_samples.mean(axis=(0, 1))
        H_mean = H_samples.mean(axis=(0, 1))

        # Model comparison metrics - handle potential WAIC/LOO failures
        try:
            waic_result = az.waic(trace)
            waic = waic_result.waic
        except (TypeError, KeyError, ValueError) as e:
            print(f"Warning: WAIC calculation failed for k={k}: {e}")
            # Fallback: calculate manually using log likelihood
            X_pred = W_mean @ H_mean
            log_lik = self._calculate_log_likelihood(X, X_pred, X_uncertainty)
            # Simple WAIC approximation
            waic = -2 * log_lik + 2 * k * (N + D)

        try:
            loo_result = az.loo(trace)
            loo = loo_result.loo
        except (TypeError, KeyError, ValueError) as e:
            print(f"Warning: LOO calculation failed for k={k}: {e}")
            # Use WAIC as fallback
            loo = waic

        # BIC approximation
        X_pred = W_mean @ H_mean
        log_lik = self._calculate_log_likelihood(X_pred, X, X_uncertainty)
        n_params = k * (N + D)
        bic = -2 * log_lik + np.log(X.size) * n_params

        # Stability metric (consistency across chains)
        stability = self._calculate_stability(H_samples)

        # Log marginal likelihood approximation
        log_marginal_likelihood = -0.5 * waic  # Simple approximation

        return {
            'trace': trace,
            'W_mean': W_mean,
            'H_mean': H_mean,
            'waic': waic,
            'loo': loo,
            'bic': bic,
            'stability': stability,
            'log_marginal_likelihood': log_marginal_likelihood
        }

    def _calculate_model_evidence_ard(self, X, X_uncertainty):
        """
        Calculate model evidence for ARD model using Laplace approximation
        """
        # Simplified implementation - in practice, use more sophisticated methods
        # like thermodynamic integration or bridge sampling
        W_samples = self.trace.posterior['W'].values
        H_samples = self.trace.posterior['H'].values

        # Use mean as MAP estimate
        W_map = W_samples.mean(axis=(0, 1))
        H_map = H_samples.mean(axis=(0, 1))

        # Log likelihood at MAP
        X_pred = W_map @ H_map
        log_lik_map = self._calculate_log_likelihood(X, X_pred, X_uncertainty)

        # Simple approximation (should use proper Hessian calculation)
        n_params = W_map.size + H_map.size
        log_evidence = log_lik_map - 0.5 * n_params * np.log(2 * np.pi)

        return log_evidence

    def _calculate_model_evidence_infinite(self):
        """
        Calculate model evidence for infinite factor model
        """
        # Placeholder - implement proper evidence calculation
        return 0.0

    def _calculate_log_likelihood(self, X_obs, X_pred, X_uncertainty):
        """
        Calculate log-likelihood of observations given predictions
        """
        residuals = (X_obs - X_pred) / X_uncertainty
        log_lik = -0.5 * np.sum(residuals**2) - 0.5 * np.sum(np.log(2 * np.pi * X_uncertainty**2))
        return log_lik

    def _calculate_stability(self, H_samples):
        """
        Calculate stability metric based on profile consistency across chains
        """
        n_chains = H_samples.shape[0]
        n_draws = H_samples.shape[1]

        if n_chains < 2:
            return 1.0

        # Normalize profiles
        H_norm = H_samples / (H_samples.sum(axis=-1, keepdims=True) + 1e-8)

        # Calculate pairwise similarities between chains
        similarities = []
        for i in range(n_chains):
            for j in range(i + 1, n_chains):
                # Use last half of samples (after burn-in)
                H_i = H_norm[i, n_draws//2:].mean(axis=0)
                H_j = H_norm[j, n_draws//2:].mean(axis=0)

                sim = np.mean([cosine_similarity([H_i[k]], [H_j[k]])[0, 0]
                              for k in range(H_i.shape[0])])
                similarities.append(sim)

        return np.mean(similarities) if similarities else 1.0

    def _extract_results(self, X, species_names=None, sample_names=None):
        """
        Extract factor profiles and contributions from trace
        """
        if self.trace is None:
            return

        W_samples = self.trace.posterior['W'].values
        H_samples = self.trace.posterior['H'].values

        if self.approach == "ard":
            active_factors = self.selection_result.factor_probabilities.astype(bool)
            W_samples = W_samples[:, :, :, active_factors]
            H_samples = H_samples[:, :, active_factors, :]
        elif self.approach == "infinite":
            pi_mean = self.selection_result.factor_probabilities
            min_weight = 0.07
            active_factors = pi_mean > min_weight
            n_active = np.sum(active_factors)
            print(f"[Extract] Selected factors (pi_mean > {min_weight}): {np.where(active_factors)[0].tolist()}")
            print(f"[Extract] Selected weights: {[f'{float(pi_mean[i]):.3f}' for i in np.where(active_factors)[0]]}")
            hard_cap = min(self.max_factors, self.min_factors + 3, self.max_factors // 2 + 3)
            if n_active > hard_cap:
                top_idx = np.argsort(-pi_mean[active_factors])[:hard_cap]
                active_indices = np.where(active_factors)[0]
                final_active = np.zeros_like(active_factors)
                final_active[active_indices[top_idx]] = True
                active_factors = final_active
                n_active = np.sum(active_factors)
            if n_active < self.min_factors:
                top_idx = np.argsort(-pi_mean)[:self.min_factors]
                final_active = np.zeros_like(active_factors)
                final_active[top_idx] = True
                active_factors = final_active
                n_active = np.sum(active_factors)
            W_samples = self.trace.posterior['W'].values[:, :, :, active_factors]
            H_samples = self.trace.posterior['H'].values[:, :, active_factors, :]
            # Fallback: If Q-loss is high or n_active << max_factors/2, relax thresholds
            W_flat_tmp = W_samples.reshape(-1, W_samples.shape[-2], W_samples.shape[-1])
            H_flat_tmp = H_samples.reshape(-1, H_samples.shape[-2], H_samples.shape[-1])
            factor_contributions_tmp = np.mean(W_flat_tmp, axis=0)
            factor_profiles_tmp = np.mean(H_flat_tmp, axis=0)
            X_pred_tmp = np.dot(factor_contributions_tmp, factor_profiles_tmp)
            q_loss_tmp = np.sum(((X - X_pred_tmp) / (np.std(X) + 1e-8)) ** 2)
            W_full = self.trace.posterior['W'].values
            H_full = self.trace.posterior['H'].values
            W_flat_all = W_full.reshape(-1, W_full.shape[-2], W_full.shape[-1])
            H_flat_all = H_full.reshape(-1, H_full.shape[-2], H_full.shape[-1])
            factor_contributions_all = np.mean(W_flat_all, axis=0)
            factor_profiles_all = np.mean(H_flat_all, axis=0)
            X_pred_all = np.dot(factor_contributions_all, factor_profiles_all)
            q_loss_all = np.sum(((X - X_pred_all) / (np.std(X) + 1e-8)) ** 2)
            if (q_loss_tmp > 1.5 * q_loss_all) or (n_active < self.max_factors // 2):
                min_weight = 0.05
                active_factors = pi_mean > min_weight
                n_active = np.sum(active_factors)
                print(f"[Extract Fallback] Selected factors (pi_mean > {min_weight}): {np.where(active_factors)[0].tolist()}")
                print(f"[Extract Fallback] Selected weights: {[f'{float(pi_mean[i]):.3f}' for i in np.where(active_factors)[0]]}")
                if n_active < self.min_factors:
                    top_idx = np.argsort(-pi_mean)[:self.min_factors]
                    final_active = np.zeros_like(active_factors)
                    final_active[top_idx] = True
                    active_factors = final_active
                    n_active = np.sum(active_factors)
                W_samples = self.trace.posterior['W'].values[:, :, :, active_factors]
                H_samples = self.trace.posterior['H'].values[:, :, active_factors, :]
        # Flatten chain dimensions
        W_flat = W_samples.reshape(-1, W_samples.shape[-2], W_samples.shape[-1])
        H_flat = H_samples.reshape(-1, H_samples.shape[-2], H_samples.shape[-1])

        # Mean estimates
        self.factor_profiles = np.mean(H_flat, axis=0)
        self.factor_contributions = np.mean(W_flat, axis=0)
        n_active = self.factor_profiles.shape[0]
        factor_names = [f'Factor_{i + 1}' for i in range(n_active)]

        # Uncertainty quantification
        if self.uncertainty_estimation:
            W_lower = np.percentile(W_flat, 2.5, axis=0)
            W_upper = np.percentile(W_flat, 97.5, axis=0)
            H_lower = np.percentile(H_flat, 2.5, axis=0)
            H_upper = np.percentile(H_flat, 97.5, axis=0)

            self.uncertainty_bounds = {
                'profiles_lower': H_lower,
                'profiles_upper': H_upper,
                'contributions_lower': W_lower,
                'contributions_upper': W_upper
            }

        # Create DataFrames
        if species_names is None:
            species_names = [f'Species_{i}' for i in range(X.shape[1])]
        if sample_names is None:
            sample_names = [f'Sample_{i}' for i in range(X.shape[0])]

        self.profiles_df = pd.DataFrame(
            self.factor_profiles.T,
            index=species_names,
            columns=factor_names
        )

        self.contributions_df = pd.DataFrame(
            self.factor_contributions,
            index=sample_names,
            columns=factor_names
        )

        # Model diagnostics
        X_pred = np.dot(self.factor_contributions, self.factor_profiles)
        self.model_diagnostics = {
            'r2': r2_score(X.flatten(), X_pred.flatten()),
            'rmse': np.sqrt(mean_squared_error(X.flatten(), X_pred.flatten())),
            'optimal_factors': n_active,
            'approach': self.approach
        }

    def plot_factor_selection_results(self):
        """
        Plot factor selection results for different approaches
        """
        if self.selection_result is None:
            raise ValueError("No factor selection results available")

        if self.approach == "discrete_search" or self.approach == "nested_comparison":
            self._plot_model_comparison()
        elif self.approach == "ard":
            self._plot_ard_results()
        elif self.approach == "infinite":
            self._plot_infinite_results()

    def _plot_model_comparison(self):
        """
        Plot model comparison results for discrete search and nested comparison
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Model Evidence', 'Information Criteria',
                'Cross-Validation Scores', 'Stability Metrics'
            ]
        )

        factors = list(self.selection_result.model_evidence.keys())

        # Model evidence
        evidences = list(self.selection_result.model_evidence.values())
        fig.add_trace(go.Scatter(
            x=factors, y=evidences, mode='lines+markers',
            name='Log Evidence', line=dict(color='blue')
        ), row=1, col=1)

        # Information criteria
        if self.selection_result.information_criteria:
            waics = [self.selection_result.information_criteria[k]['waic'] for k in factors]
            loos = [self.selection_result.information_criteria[k]['loo'] for k in factors]
            bics = [self.selection_result.information_criteria[k]['bic'] for k in factors]

            fig.add_trace(go.Scatter(x=factors, y=waics, mode='lines+markers',
                                   name='WAIC'), row=1, col=2)
            fig.add_trace(go.Scatter(x=factors, y=loos, mode='lines+markers',
                                   name='LOO'), row=1, col=2)
            fig.add_trace(go.Scatter(x=factors, y=bics, mode='lines+markers',
                                   name='BIC'), row=1, col=2)

        # Cross-validation scores
        if self.selection_result.cross_validation_scores:
            cv_scores = list(self.selection_result.cross_validation_scores.values())
            fig.add_trace(go.Scatter(
                x=factors, y=cv_scores, mode='lines+markers',
                name='CV Score', line=dict(color='green')
            ), row=2, col=1)

        # Stability metrics
        if self.selection_result.stability_metrics:
            stabilities = list(self.selection_result.stability_metrics.values())
            fig.add_trace(go.Scatter(
                x=factors, y=stabilities, mode='lines+markers',
                name='Stability', line=dict(color='red')
            ), row=2, col=2)

        # Mark optimal
        fig.add_vline(x=self.optimal_factors, line_dash="dash",
                     annotation_text=f"Optimal: {self.optimal_factors}")

        fig.update_layout(
            title=f"Factor Selection Results ({self.approach})",
            height=600, width=1000, template='plotly_white'
        )
        fig.show()

    def _plot_ard_results(self):
        """
        Plot ARD results showing factor relevance
        """
        if 'lambda_W' not in self.trace.posterior:
            return

        lambda_W = self.trace.posterior['lambda_W'].values
        lambda_H = self.trace.posterior['lambda_H'].values

        lambda_W_mean = lambda_W.mean(axis=(0, 1))
        lambda_H_mean = lambda_H.mean(axis=(0, 1))

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Factor Relevance (W)', 'Factor Relevance (H)']
        )

        factors = list(range(1, len(lambda_W_mean) + 1))

        fig.add_trace(go.Bar(
            x=factors, y=1/lambda_W_mean, name='W Relevance',
            marker_color=['red' if self.selection_result.factor_probabilities[i]
                         else 'lightgray' for i in range(len(factors))]
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=factors, y=1/lambda_H_mean, name='H Relevance',
            marker_color=['blue' if self.selection_result.factor_probabilities[i]
                         else 'lightgray' for i in range(len(factors))]
        ), row=1, col=2)

        fig.update_layout(
            title="Automatic Relevance Determination Results",
            height=400, width=800, template='plotly_white'
        )
        fig.show()

    def _plot_infinite_results(self):
        """
        Plot infinite factor model results showing stick-breaking weights
        """
        pi_mean = self.selection_result.factor_probabilities

        fig = go.Figure()

        factors = list(range(1, len(pi_mean) + 1))
        colors = ['red' if pi_mean[i] > 0.01 else 'lightgray' for i in range(len(factors))]

        fig.add_trace(go.Bar(
            x=factors, y=pi_mean, name='Factor Weights',
            marker_color=colors
        ))

        fig.add_hline(y=0.01, line_dash="dash",
                     annotation_text="Relevance Threshold (1%)")

        fig.update_layout(
            title="Infinite Factor Model - Stick-Breaking Weights",
            xaxis_title="Factor Index",
            yaxis_title="Weight",
            height=400, width=800, template='plotly_white'
        )
        fig.show()

    def get_model_selection_summary(self) -> Dict:
        """
        Get summary of model selection results
        """
        summary = {
            'approach': self.approach,
            'optimal_factors': self.optimal_factors,
            'min_factors_considered': self.min_factors,
            'max_factors_considered': self.max_factors
        }

        if self.selection_result:
            summary.update({
                'model_evidence': self.selection_result.model_evidence,
                'information_criteria': self.selection_result.information_criteria,
                'stability_metrics': self.selection_result.stability_metrics,
                'cross_validation_scores': self.selection_result.cross_validation_scores
            })

        return summary


def compare_factor_selection_approaches(X, X_uncertainty, max_factors=15, species_names=None):
    """
    Compare different factor selection approaches on the same data
    """
    approaches = ["ard", "discrete_search", "nested_comparison", "infinite"]
    results = {}

    for approach in approaches:
        print(f"\n=== Testing {approach.upper()} approach ===")

        model = HierarchicalBayesianNMF(
            max_factors=max_factors,
            approach=approach,
            seed=42
        )

        try:
            model.fit(X, X_uncertainty, species_names=species_names, n_samples=1000, n_tune=500, n_chains=2)

            results[approach] = {
                'optimal_factors': model.optimal_factors,
                'model_diagnostics': model.model_diagnostics,
                'selection_summary': model.get_model_selection_summary()
            }

            print(f"Optimal factors: {model.optimal_factors}")

        except Exception as e:
            print(f"Failed to fit {approach}: {e}")
            results[approach] = {'error': str(e)}

    return results


# Demonstration
if __name__ == "__main__":
    # Generate synthetic data
    try:
        V, U, true_k, true_H, true_W, sim = generate_dataset(
            true_k=5, n_samples=200, n_features=15, seed=42
        )

        species_names = [f'Species_{i+1}' for i in range(V.shape[1])]

        print("=== Comparing Factor Selection Approaches ===")
        results = compare_factor_selection_approaches(
            V, U, max_factors=10, species_names=species_names
        )

        # Print summary
        print("\n=== RESULTS SUMMARY ===")
        for approach, result in results.items():
            if 'error' in result:
                print(f"{approach}: ERROR - {result['error']}")
            else:
                print(f"{approach}: {result['optimal_factors']} factors "
                      f"(True: {true_k})")

        # Detailed analysis with ARD approach
        print("\n=== Detailed ARD Analysis ===")
        model = HierarchicalBayesianNMF(approach="ard", max_factors=10)
        model.fit(V, U, species_names=species_names,
                 n_samples=2000, n_tune=1000, n_chains=4)

        model.plot_factor_selection_results()

        print(f"ARD selected {model.optimal_factors} factors (true: {true_k})")
        print(f"Model R²: {model.model_diagnostics['r2']:.3f}")

    except Exception as e:
        print(f"Demo failed: {e}")
        print("Please ensure utils.py with generate_dataset function is available")
