import os
import pickle
import warnings
import logging

import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import arviz as az
import pymc as pm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from esat.model.batch_sa import BatchSA
from utils import generate_dataset, q_loss

pio.renderers.default = "browser"
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class BayesianNMF:
    """
    Bayesian Non-negative Matrix Factorization for Source Apportionment
    Replicates EPA PMF functionality with uncertainty quantification
    """

    def __init__(self, n_factors=None, max_factors=15, auto_factor_selection=True,
                 uncertainty_estimation=True, vector_db_integration=True, seed: int = 42):
        self.n_factors = n_factors
        self.random_seed = seed
        self.min_factors = 2
        self.max_factors = max_factors
        self.auto_factor_selection = auto_factor_selection
        self.uncertainty_estimation = uncertainty_estimation
        self.vector_db_integration = vector_db_integration

        self.model = None
        self.trace = None
        self.optimal_factors = None
        self.factor_profiles = None
        self.factor_contributions = None
        self.uncertainty_bounds = None
        self.model_diagnostics = {}

    def _eval_range(self, V, U, min_factors, max_factors):
        """
        Evaluate a range of factor counts using BatchSA and return results.

        Parameters
        ----------
        V
        U
        min_factors
        max_factors

        Returns
        -------

        """
        logger.info(f"Evaluating factor range: {min_factors} to {max_factors} using Q(True) change")
        results = {}
        combined_metric = {}
        for n_factors in range(min_factors, max_factors + 1):
            logger.info(f"Evaluating {n_factors} factors")
            batch_sa = BatchSA(V=V, U=U, factors=n_factors, method="ls-nmf", verbose=False, max_iter=5000, models=10)
            batch_sa.train()
            # Collect AIC/BIC and H matrices
            aics, bics, Hs = [], [], []
            for i_sa in batch_sa.results:
                X_pred = i_sa.W @ i_sa.H
                resid = (V - X_pred) / U
                q_robust = np.sum(resid ** 2)
                log_likelihood = -0.5 * q_robust
                n_params = n_factors * (V.shape[0] + V.shape[1])
                aic = -2 * log_likelihood + 2 * n_params
                bic = -2 * log_likelihood + np.log(V.size) * n_params
                aics.append(aic)
                bics.append(bic)
                # Normalize H for stability
                Hs.append(i_sa.H / (i_sa.H.sum(axis=1, keepdims=True) + 1e-8))

            # Stability: mean pairwise cosine similarity between Hs
            stabilities = []
            for i in range(len(Hs)):
                for j in range(i + 1, len(Hs)):
                    sim = cosine_similarity(Hs[i], Hs[j]).mean()
                    stabilities.append(sim)
            mean_stability = np.mean(stabilities) if stabilities else 0

            results[n_factors] = {
                "Qtrue": np.mean([i_sa.Qtrue for i_sa in batch_sa.results]),
                "AIC": np.mean(aics),
                "BIC": np.mean(bics),
                "stability": mean_stability,
                "W": batch_sa.results[batch_sa.best_model].W,
                "H": batch_sa.results[batch_sa.best_model].H,
            }

            # Normalize AIC/BIC and stability for combination
        aic_vals = np.array([v["AIC"] for v in results.values()])
        stab_vals = np.array([v["stability"] for v in results.values()])
        aic_norm = (aic_vals - aic_vals.min()) / (aic_vals.max() - aic_vals.min() + 1e-8)
        stab_norm = (stab_vals - stab_vals.min()) / (stab_vals.max() - stab_vals.min() + 1e-8)

        for idx, n_factors in enumerate(range(min_factors, max_factors + 1)):
            # Example: lower AIC and higher stability are better, so invert AIC
            combined_score = (1 - aic_norm[idx]) * 0.5 + stab_norm[idx] * 0.5
            combined_metric[n_factors] = combined_score
            results[n_factors]["combined_metric"] = combined_score

        return results, combined_metric

    @staticmethod
    def run_factor_test(args):
        self_ref, n_factors, X, X_uncertainty, init_W, init_H = args
        try:
            return n_factors, self_ref._calculate_model_selection_criteria(X, X_uncertainty, n_factors, init_W=init_W, init_H=init_H)
        except Exception as e:
            print(f"Failed for {n_factors} factors: {e}")
            return n_factors, None

    def _calculate_model_selection_criteria(self, X, X_uncertainty, n_factors, init_W=None, init_H=None):
        """Calculate AIC, BIC, and Q_robust for model selection"""
        logger.info(f"Calculating model selection criteria for {n_factors} factors")
        X = np.ascontiguousarray(X, dtype=np.float64)
        X_uncertainty = np.ascontiguousarray(X_uncertainty, dtype=np.float64)
        assert X.ndim == 2 and X_uncertainty.ndim == 2
        assert X.shape == X_uncertainty.shape
        assert X.dtype != np.dtype('O') and X_uncertainty.dtype != np.dtype('O')

        if init_W is not None:
            init_W = np.ascontiguousarray(init_W, dtype=np.float64)
            assert init_W.shape == (X.shape[0], n_factors)
            assert init_W.dtype != np.dtype('O')
        if init_H is not None:
            init_H = np.ascontiguousarray(init_H, dtype=np.float64)
            assert init_H.shape == (n_factors, X.shape[1])
            assert init_H.dtype != np.dtype('O')

        with pm.Model() as model:
            # Factor profiles (H): species x factors
            # Using Exponential prior to enforce non-negativity
            if init_H is not None:
                H = pm.Exponential('H', lam=1.0, shape=(n_factors, X.shape[1]), initval=init_H)
            else:
                H = pm.Exponential('H', lam=1.0, shape=(n_factors, X.shape[1]))

            # Factor contributions (W): factors x samples
            if init_W is not None:
                W = pm.Exponential('W', lam=1.0, shape=(X.shape[0], n_factors), initval=init_W)
            else:
                W = pm.Exponential('W', lam=1.0, shape=(X.shape[0], n_factors))

            # Reconstruction
            X_recon = pm.math.dot(W, H)

            # Likelihood with heteroscedastic noise
            sigma = pm.HalfNormal('sigma', sigma=X_uncertainty)
            likelihood = pm.Normal('X_obs', mu=X_recon, sigma=sigma, observed=X)

            # Sample
            trace = pm.sample(1000, tune=750, cores=1, chains=4, progressbar=False, max_treedepth=12, return_inferencedata=False)

        # Calculate metrics
        W_samples = trace['W']  # shape: (n_samples, n_rows, n_factors)
        H_samples = trace['H']  # shape: (n_samples, n_factors, n_cols)
        X_pred_samples = np.matmul(W_samples, H_samples)  # shape: (n_samples, n_rows, n_cols)
        X_pred = np.mean(X_pred_samples, axis=0)  # shape: (n_rows, n_cols)

        # Q_robust (scaled residuals)
        residuals = (X - X_pred) / X_uncertainty
        q_robust = np.sum(residuals ** 2)

        # AIC/BIC approximations
        log_likelihood = -0.5 * q_robust
        n_params = n_factors * (X.shape[0] + X.shape[1])
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + np.log(X.size) * n_params

        return {'aic': aic, 'bic': bic, 'q_robust': q_robust, "W": W_samples.mean(axis=0), "H": H_samples.mean(axis=0)}

    def _determine_optimal_factors(self, V, U):
        """Automated factor count determination using model selection criteria"""
        print("Determining optimal number of factors...")

        min_factors = self.min_factors
        max_factors = self.max_factors
        surrogate_results, combined_results = self._eval_range(V, U, min_factors, max_factors)
        logger.info(f"Cutoff values for surrogate results: {combined_results}")
        max_factors = min(combined_results, key=combined_results.get) + 1
        logger.info(f"Best max cutoff metric found at {max_factors} factors (including +1 buffer)")
        results = {}

        args_list = [(self, n, V, U, surrogate_results[n]["W"], surrogate_results[n]["H"]) for n in range(min_factors, max_factors + 1)]

        with mp.Pool(processes=min(len(args_list), os.cpu_count() or 1)) as pool:
            pool_results = pool.map(BayesianNMF.run_factor_test, args_list)

        for n_factors, result in pool_results:
            if result is not None:
                results[n_factors] = result

        if not results:
            raise ValueError("No valid solutions found")

        if not results:
            raise ValueError("No valid solutions found")

        # Find optimal using combined criteria
        aic_scores = {k: v['aic'] for k, v in results.items()}
        bic_scores = {k: v['bic'] for k, v in results.items()}

        optimal_aic = min(aic_scores, key=aic_scores.get)
        optimal_bic = min(bic_scores, key=bic_scores.get)

        trace_W = results[optimal_bic]['W']  # Use BIC optimal W
        trace_H = results[optimal_bic]['H']  # Use BIC optimal H

        # Use BIC as primary criterion (more conservative)
        self.optimal_factors = optimal_bic
        self.model_diagnostics = {
            'factor_tests': results,
            'optimal_aic': optimal_aic,
            'optimal_bic': optimal_bic,
            'selected': self.optimal_factors
        }

        print(f"Optimal factors: {self.optimal_factors} (AIC: {optimal_aic}, BIC: {optimal_bic})")
        return self.optimal_factors, trace_W, trace_H

    def fit_test_nmf(self, X, X_uncertainty, max_factors=15, n_samples=2000, n_tune=1000, z_threshold=0.75, min_strength=1e-2, target_accept=0.95, alpha=1.0):
        """
        Fit Bayesian NMF with an ARD prior to infer the number of factors.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        X_uncertainty : array-like, shape (n_samples, n_features)
        max_factors : int
            Maximum number of latent factors to consider
        n_samples : int
            Posterior samples
        n_tune : int
            Tuning steps
        target_accept : float
            NUTS target_accept

        Returns
        -------
        n_active_factors : int
            Inferred number of active factors
        W_mean, H_mean : np.ndarray
            Mean factor matrices (using only active factors)
        trace : pm.InferenceData
            PyMC trace object
        """
        X = np.asarray(X, dtype=np.float64)
        X_uncertainty = np.asarray(X_uncertainty, dtype=np.float64)
        N, D = X.shape
        K = max_factors

        with pm.Model() as model:
            # Stick-breaking construction for DP
            v = pm.Beta('v', alpha=1, beta=alpha, shape=K)
            stick_segments = pm.Deterministic('stick_segments',
                                              v * pm.math.concatenate([[1], pm.math.cumprod(1 - v)[:-1]]))
            pi = pm.Deterministic('pi', stick_segments / pm.math.sum(stick_segments))  # Normalize

            # Factor matrices
            H = pm.Exponential('H', lam=1.0, shape=(K, D))
            W = pm.Exponential('W', lam=1.0, shape=(N, K))

            # Weight factors by stick-breaking proportions
            X_recon = pm.math.dot(W * pi, H)

            pm.Normal('X_obs', mu=X_recon, sigma=X_uncertainty, observed=X)

            trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                target_accept=target_accept,
                chains=4,
                cores=4,
                progressbar=True,
                return_inferencedata=True
            )

        # Post-hoc: prune factors with negligible stick weights
        pi_samples = trace.posterior['pi'].values.reshape(-1, K)
        pi_mean = pi_samples.mean(axis=0)
        active_factors = np.where(pi_mean > min_strength)[0]

        W_samples = trace.posterior['W'].values.reshape(-1, N, K)
        H_samples = trace.posterior['H'].values.reshape(-1, K, D)
        W_mean = W_samples.mean(axis=0)[:, active_factors]
        H_mean = H_samples.mean(axis=0)[active_factors, :]

        n_active_factors = len(active_factors)
        return n_active_factors, W_mean, H_mean, trace

    def fit(self, X, X_uncertainty=None, species_names=None, sample_names=None,
            n_samples=2000, n_tune=1000, target_accept=0.95, n_chains=None,
            init_W=None, init_H=None):
        """
        Fit Bayesian NMF model to concentration data

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_species)
            Concentration matrix
        X_uncertainty : array-like, shape (n_samples, n_species)
            Uncertainty matrix (standard deviations)
        init_W : array-like, shape (n_samples, n_factors), optional
            Initial value for W (from standard NMF)
        init_H : array-like, shape (n_factors, n_species), optional
            Initial value for H (from standard NMF)
        species_names : list
            Chemical species names
        sample_names : list
            Sample identifiers
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        assert X.ndim == 2, f"X must be 2D, got shape {X.shape}"
        assert X.shape == X_uncertainty.shape, "X and X_uncertainty must have the same shape"
        assert X.dtype != np.dtype('O'), "X must not be object dtype"

        if X_uncertainty is None:
            # Default uncertainty as 5% of concentration + detection limit proxy
            X_uncertainty = np.maximum(0.05 * X, 0.01 * np.mean(X, axis=0))

        X_uncertainty = np.ascontiguousarray(X_uncertainty, dtype=np.float64)
        assert X_uncertainty.ndim == 2, f"X_uncertainty must be 2D, got shape {X_uncertainty.shape}"
        assert X_uncertainty.dtype != np.dtype('O'), "X_uncertainty must not be object dtype"

        # Auto-determine factors if needed
        W_rate = 1.0
        H_rate = 1.0
        if self.auto_factor_selection and self.n_factors is None:
            self.n_factors, trace_W, trace_H, trace = self.fit_test_nmf(X, X_uncertainty, max_factors=self.max_factors, n_samples=1000, n_tune=1000, target_accept=0.95)
            # self.n_factors, trace_W, trace_H = self._determine_optimal_factors(X, X_uncertainty)
            W_rate = 1.0 / (trace_W + 1e-8)
            H_rate = 1.0 / (trace_H + 1e-8)
        elif self.n_factors is None:
            self.n_factors = 5  # Default

        print(f"Fitting Bayesian NMF with {self.n_factors} factors...")

        # Build Bayesian model
        with pm.Model() as self.model:
            # Factor profiles (H): species x factors
            # Using Exponential prior to enforce non-negativity
            if init_H is not None:
                H = pm.Exponential('H', lam=H_rate, shape=(self.n_factors, X.shape[1]), initval=init_H)
            else:
                H = pm.Exponential('H', lam=H_rate, shape=(self.n_factors, X.shape[1]))

            # Factor contributions (W): factors x samples
            if init_W is not None:
                lam_W = pm.Gamma('lam_W', alpha=2, beta=2, shape=self.n_factors)
                W = pm.Exponential('W', lam=lam_W[None, :], shape=(X.shape[0], self.n_factors), initval=init_W)
                # W = pm.Exponential('W', lam=W_rate, shape=(X.shape[0], self.n_factors), initval=init_W)
            else:
                lam_W = pm.Gamma('lam_W', alpha=2, beta=2, shape=self.n_factors)
                W = pm.Exponential('W', lam=lam_W[None, :], shape=(X.shape[0], self.n_factors))
                # W = pm.Exponential('W', lam=W_rate, shape=(X.shape[0], self.n_factors))

            # Add L2 regularization
            # W_reg = pm.Potential('W_reg', -0.01 * pm.math.sum(W ** 2))
            # H_reg = pm.Potential('H_reg', -0.01 * pm.math.sum(H ** 2))

            # Reconstruction
            X_recon = pm.math.dot(W, H)

            # Heteroscedastic likelihood
            likelihood = pm.Normal('X_obs', mu=X_recon, sigma=X_uncertainty, observed=X)

            # Sample posterior
            print("Sampling posterior...")
            self.trace = pm.sample(
                draws=n_samples,
                chains=n_chains if n_chains is not None else 2,
                tune=n_tune,
                random_seed=self.random_seed,
                cores=os.cpu_count() - 2,
                target_accept=target_accept,
                max_treedepth=20,
                progressbar=True,
                return_inferencedata=True,
            )

        # Extract results
        self._extract_results(X, species_names, sample_names)

        return self

    def _extract_results(self, X, species_names=None, sample_names=None):
        """Extract factor profiles, contributions, and uncertainties"""

        # Get posterior samples
        W_samples = self.trace.posterior['W'].values
        H_samples = self.trace.posterior['H'].values

        # Flatten chain dimensions
        W_flat = W_samples.reshape(-1, W_samples.shape[-2], W_samples.shape[-1])
        H_flat = H_samples.reshape(-1, H_samples.shape[-2], H_samples.shape[-1])

        # Mean estimates
        self.factor_profiles = np.mean(H_flat, axis=0)  # species x factors
        self.factor_contributions = np.mean(W_flat, axis=0)  # samples x factors

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

        # Create DataFrames for easier interpretation
        if species_names is None:
            species_names = [f'Species_{i}' for i in range(X.shape[1])]
        if sample_names is None:
            sample_names = [f'Sample_{i}' for i in range(X.shape[0])]

        factor_names = [f'Factor_{i + 1}' for i in range(self.n_factors)]

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

        # Calculate model fit statistics
        X_pred = np.dot(self.factor_contributions, self.factor_profiles)
        self.model_diagnostics.update({
            'r2': r2_score(X.flatten(), X_pred.flatten()),
            'rmse': np.sqrt(mean_squared_error(X.flatten(), X_pred.flatten())),
            'explained_variance': np.var(X_pred.flatten()) / np.var(X.flatten())
        })

    def get_factor_profiles(self, normalize=True):
        """Get factor profiles with optional normalization"""
        profiles = self.factor_profiles.copy()
        if normalize:
            profiles = profiles / profiles.sum(axis=0)
        return profiles

    def get_factor_contributions(self):
        """Get factor contributions"""
        return self.factor_contributions

    def get_uncertainty_intervals(self, confidence=0.95):
        """Get uncertainty intervals for profiles and contributions"""
        if not self.uncertainty_estimation or self.uncertainty_bounds is None:
            raise ValueError("Uncertainty estimation was not performed")

        alpha = 1 - confidence
        lower_pct = (alpha / 2) * 100
        upper_pct = (1 - alpha / 2) * 100

        return {
            'profiles_ci': (self.uncertainty_bounds['profiles_lower'],
                            self.uncertainty_bounds['profiles_upper']),
            'contributions_ci': (self.uncertainty_bounds['contributions_lower'],
                                 self.uncertainty_bounds['contributions_upper'])
        }

    def calculate_source_contributions(self, total_mass=None):
        """Calculate absolute source contributions"""
        if total_mass is None:
            # Use sum of all species as proxy for total mass
            total_mass = np.sum(self.factor_contributions, axis=1)

        return self.factor_contributions * total_mass[:, np.newaxis]

    def get_model_diagnostics(self):
        """Get comprehensive model diagnostics"""
        diagnostics = self.model_diagnostics.copy()

        if self.trace is not None:
            # Add MCMC diagnostics
            diagnostics.update({
                'rhat_W': az.rhat(self.trace, var_names=['W']).max().values,
                'rhat_H': az.rhat(self.trace, var_names=['H']).max().values,
                'ess_W': az.ess(self.trace, var_names=['W']).min().values,
                'ess_H': az.ess(self.trace, var_names=['H']).min().values,
                'mcse_W': az.mcse(self.trace, var_names=['W']).max().values,
                'mcse_H': az.mcse(self.trace, var_names=['H']).max().values,
            })

        return diagnostics

    def compare_with_speciate_profiles(self, speciate_profiles, similarity_threshold=0.8):
        """
        Compare extracted profiles with SPECIATE database profiles

        Parameters:
        -----------
        speciate_profiles : dict or DataFrame
            SPECIATE profiles for comparison
        similarity_threshold : float
            Cosine similarity threshold for profile matching
        """
        similarities = {}
        factor_profiles = self.get_factor_profiles(normalize=True)

        for factor_idx in range(self.n_factors):
            factor_profile = factor_profiles[:, factor_idx]
            max_sim = 0
            best_match = None

            for profile_name, speciate_profile in speciate_profiles.items():
                # Ensure same species alignment
                aligned_speciate = np.array([speciate_profile.get(species, 0)
                                             for species in self.profiles_df.index])
                aligned_speciate = aligned_speciate / aligned_speciate.sum()

                sim = cosine_similarity([factor_profile], [aligned_speciate])[0, 0]

                if sim > max_sim:
                    max_sim = sim
                    best_match = profile_name

            similarities[f'Factor_{factor_idx + 1}'] = {
                'best_match': best_match,
                'similarity': max_sim,
                'is_match': max_sim >= similarity_threshold
            }

        return similarities

    def save(self, filepath):
        """Save model state and trace to disk."""
        state = {
            'n_factors': self.n_factors,
            'max_factors': self.max_factors,
            'auto_factor_selection': self.auto_factor_selection,
            'uncertainty_estimation': self.uncertainty_estimation,
            'vector_db_integration': self.vector_db_integration,
            'optimal_factors': self.optimal_factors,
            'factor_profiles': self.factor_profiles,
            'factor_contributions': self.factor_contributions,
            'uncertainty_bounds': self.uncertainty_bounds,
            'model_diagnostics': self.model_diagnostics,
            'profiles_df': getattr(self, 'profiles_df', None),
            'contributions_df': getattr(self, 'contributions_df', None),
        }
        with open(filepath + '_state.pkl', 'wb') as f:
            pickle.dump(state, f)
        if self.trace is not None:
            az.to_netcdf(self.trace, filepath + '_trace.nc')

    @staticmethod
    def load(cls, filepath):
        """Load model state and trace from disk."""
        with open(filepath + '_state.pkl', 'rb') as f:
            state = pickle.load(f)
        model = cls(
            n_factors=state['n_factors'],
            max_factors=state['max_factors'],
            auto_factor_selection=state['auto_factor_selection'],
            uncertainty_estimation=state['uncertainty_estimation'],
            vector_db_integration=state['vector_db_integration'],
        )
        model.optimal_factors = state['optimal_factors']
        model.factor_profiles = state['factor_profiles']
        model.factor_contributions = state['factor_contributions']
        model.uncertainty_bounds = state['uncertainty_bounds']
        model.model_diagnostics = state['model_diagnostics']
        model.profiles_df = state['profiles_df']
        model.contributions_df = state['contributions_df']
        try:
            model.trace = az.from_netcdf(filepath + '_trace.nc')
        except Exception:
            model.trace = None
        return model

    def plot_trace(self, var_names=None):
        """Plot trace for the model's posterior samples using Plotly."""
        if self.trace is None:
            raise ValueError("No trace available. Run fit() first.")

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Get variables to plot
        if var_names is None:
            var_names = ['W', 'H']

        # Create subplots
        n_vars = len(var_names)
        fig = make_subplots(
            rows=n_vars, cols=1,
            subplot_titles=[f'{var} Trace' for var in var_names],
            vertical_spacing=0.1
        )

        for i, var_name in enumerate(var_names):
            if var_name in self.trace.posterior:
                # Get samples (flatten chain and draw dimensions)
                samples = self.trace.posterior[var_name].values
                if samples.ndim > 2:
                    # For multidimensional variables, plot first component
                    samples = samples.reshape(-1, samples.shape[-1])[:, 0]
                else:
                    samples = samples.flatten()

                fig.add_trace(
                    go.Scatter(
                        y=samples,
                        mode='lines',
                        name=f'{var_name} trace',
                        line=dict(width=1),
                        showlegend=False
                    ),
                    row=i + 1, col=1
                )

        fig.update_layout(
            title="MCMC Trace Plots",
            height=300 * n_vars,
            width=1000,
            template='plotly_white'
        )

        fig.update_xaxes(title_text="Iteration")
        fig.update_yaxes(title_text="Value")
        fig.show()

    def plot_posterior(self, var_names=None):
        """Plot posterior distributions using Plotly."""
        if self.trace is None:
            raise ValueError("No trace available. Run fit() first.")

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if var_names is None:
            var_names = ['W', 'H']

        n_vars = len(var_names)
        fig = make_subplots(
            rows=1, cols=n_vars,
            subplot_titles=[f'{var} Posterior' for var in var_names]
        )

        for i, var_name in enumerate(var_names):
            if var_name in self.trace.posterior:
                samples = self.trace.posterior[var_name].values
                if samples.ndim > 2:
                    samples = samples.reshape(-1, samples.shape[-1])[:, 0]
                else:
                    samples = samples.flatten()

                fig.add_trace(
                    go.Histogram(
                        x=samples,
                        name=f'{var_name}',
                        nbinsx=50,
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=1, col=i + 1
                )

        fig.update_layout(
            title="Posterior Distributions",
            height=400,
            width=400 * n_vars,
            template='plotly_white'
        )

        fig.show()

    def plot_pair(self, var_names=None):
        """Pair plot for selected variables."""
        if self.trace is None:
            raise ValueError("No trace available. Fit the model first.")
        az.plot_pair(self.trace, var_names=var_names, filter_vars="regex", divergences=True, textsize=18)
        plt.show()

    def plot_energy(self):
        """Plot energy plot for the trace."""
        if self.trace is None:
            raise ValueError("No trace available. Fit the model first.")
        az.plot_energy(self.trace)
        plt.show()

    def plot_uncertainty_intervals(self, confidence=0.95, figsize=(1200, 800)):
        """Plot uncertainty intervals for each factor: profile (boxplot) and contributions (lines per feature)"""
        if not self.uncertainty_estimation or self.trace is None:
            raise ValueError("Model was not fitted with uncertainty estimation enabled")

        # Get species names
        if hasattr(self, 'profiles_df') and self.profiles_df is not None:
            species_names = self.profiles_df.index.tolist()
        else:
            species_names = [f'Species_{i + 1}' for i in range(self.factor_profiles.shape[1])]

        n_factors = self.n_factors
        for factor_idx in range(n_factors):
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=[
                    f'Factor {factor_idx + 1} - Profile Distribution',
                    f'Factor {factor_idx + 1} - Contribution Time Series'
                ],
                vertical_spacing=0.25,
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
            )

            # --- Profile boxplots ---
            H_samples = self.trace.posterior['H'].values
            H_flat = H_samples.reshape(-1, H_samples.shape[-2], H_samples.shape[-1])
            H_normalized = H_flat / H_flat.sum(axis=-1, keepdims=True) * 100

            for species_idx, species_name in enumerate(species_names):
                species_samples = H_normalized[:, factor_idx, species_idx]
                fig.add_trace(go.Box(
                    name=species_name,
                    y=species_samples,
                    boxpoints='outliers',
                    notched=True,
                    marker_color='rgb(107,174,214)',
                    line_color='rgb(107,174,214)',
                    marker_size=4,
                    line_width=1,
                    showlegend=False
                ), row=1, col=1)

            # --- Contribution time series ---
            W_samples = self.trace.posterior['W'].values
            W_flat = W_samples.reshape(-1, W_samples.shape[-2], W_samples.shape[-1])
            n_samples, n_time, n_factors = W_flat.shape

            for species_idx, species_name in enumerate(species_names):
                contrib_samples = W_flat[:, :, factor_idx] * H_normalized[:, factor_idx, species_idx][:, np.newaxis]
                mean_contrib = np.mean(contrib_samples, axis=0)
                alpha = 1 - confidence
                lower = np.percentile(contrib_samples, (alpha / 2) * 100, axis=0)
                upper = np.percentile(contrib_samples, (1 - alpha / 2) * 100, axis=0)
                sample_indices = np.arange(len(mean_contrib))

                # Create a unique legend group for each species
                legend_group = f"group_{species_idx}"

                # Upper bound trace for hover
                fig.add_trace(go.Scatter(
                    x=sample_indices,
                    y=upper,
                    mode='lines',
                    line=dict(color='rgba(255,255,255,0)', width=0),
                    showlegend=False,
                    legendgroup=legend_group,
                    hovertemplate=f'Sample %{{x}} - {species_name}<br>Upper: %{{y:.2e}}<extra></extra>',
                    name=f'{species_name} Upper'
                ), row=2, col=1)
                # Mean line - this will be the main legend entry
                fig.add_trace(go.Scatter(
                    x=sample_indices,
                    y=mean_contrib,
                    mode='lines',
                    name=f'{species_name}',
                    line=dict(width=2),
                    legendgroup=legend_group,
                    showlegend=True,
                    hovertemplate=f'Mean: %{{y:.2e}}<br><extra></extra>'
                ), row=2, col=1)
                # Lower bound trace for hover
                fig.add_trace(go.Scatter(
                    x=sample_indices,
                    y=lower,
                    mode='lines',
                    line=dict(color='rgba(255,255,255,0)', width=0),
                    showlegend=False,
                    legendgroup=legend_group,
                    hovertemplate=f'Lower: %{{y:.2e}}<br>[{confidence * 100:.0f}% CI]<extra></extra>',
                    name=f'{species_name} Lower'
                ), row=2, col=1)

                # Uncertainty ribbon - visual only
                fig.add_trace(go.Scatter(
                    x=np.concatenate([sample_indices, sample_indices[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill='toself',
                    fillcolor='rgba(107,174,214,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    legendgroup=legend_group,
                    hoverinfo='skip'
                ), row=2, col=1)


            # Layout
            fig.update_layout(
                height=800,
                width=figsize[0],
                template='plotly_white',
                title_text=f"Uncertainty Intervals - Factor {factor_idx + 1}",
                title_x=0.5,
                showlegend=True,
                hovermode='x unified'
            )
            fig.update_xaxes(
                ticktext=species_names,
                tickvals=list(range(len(species_names))),
                tickangle=45,
                title_text='Feature',
                row=1, col=1
            )
            fig.update_yaxes(
                title_text='Percentage (%)',
                row=1, col=1
            )
            fig.update_xaxes(
                title_text='Sample Index',
                row=2, col=1
            )
            fig.update_yaxes(
                title_text='Contribution',
                row=2, col=1
            )
            fig.show()


def compare_factors_to_truth(true_W, true_H, estimated_W, estimated_H,
                           species_names=None, model_type="BatchSA",
                           uncertainty_intervals=None, confidence=0.95,
                           correlation_threshold=0.6):
    """
    Compare estimated factors (W, H) to true factors with detailed analysis and visualization.
    """
    n_true_factors = true_H.shape[0]
    n_est_factors = estimated_H.shape[0]
    n_species = true_H.shape[1]

    if species_names is None:
        species_names = [f'Species_{i+1}' for i in range(n_species)]

    # Normalize profiles for comparison
    true_H_norm = true_H / (true_H.sum(axis=1, keepdims=True) + 1e-8)
    est_H_norm = estimated_H / (estimated_H.sum(axis=1, keepdims=True) + 1e-8)

    # Find best factor mapping using profile correlations
    correlation_matrix = np.zeros((n_est_factors, n_true_factors))
    for i in range(n_est_factors):
        for j in range(n_true_factors):
            corr, _ = pearsonr(est_H_norm[i], true_H_norm[j])
            correlation_matrix[i, j] = corr**2  # R-squared

    # Find best matches
    factor_mapping = {}
    for i in range(n_est_factors):
        best_match_idx = np.argmax(correlation_matrix[i])
        best_corr = correlation_matrix[i, best_match_idx]
        factor_mapping[i] = {
            'true_factor': best_match_idx,
            'correlation': best_corr,
            'matched': best_corr >= correlation_threshold
        }

    # Calculate comprehensive metrics
    results = {
        'factor_mapping': factor_mapping,
        'correlation_matrix': correlation_matrix,
        'model_type': model_type,
        'factor_metrics': {}
    }

    # Plot comparison for each estimated factor (combined profile + contribution)
    for est_factor_idx in range(n_est_factors):
        true_factor_idx = factor_mapping[est_factor_idx]['true_factor']
        correlation = factor_mapping[est_factor_idx]['correlation']

        # Combined profile and contribution comparison
        fig_combined = _plot_profile_and_contribution_comparison(
            true_H_norm[true_factor_idx],
            est_H_norm[est_factor_idx],
            true_W[:, true_factor_idx],
            estimated_W[:, est_factor_idx],
            species_names,
            est_factor_idx,
            true_factor_idx,
            correlation,
            model_type,
            uncertainty_intervals,
            confidence
        )

        # Calculate detailed metrics
        profile_mse = np.mean((true_H_norm[true_factor_idx] - est_H_norm[est_factor_idx])**2)
        contrib_corr, _ = pearsonr(true_W[:, true_factor_idx], estimated_W[:, est_factor_idx])
        contrib_mse = np.mean((true_W[:, true_factor_idx] - estimated_W[:, est_factor_idx])**2)

        results['factor_metrics'][est_factor_idx] = {
            'mapped_to_true_factor': true_factor_idx,
            'profile_correlation': correlation,
            'profile_mse': profile_mse,
            'contribution_correlation': contrib_corr**2 if not np.isnan(contrib_corr) else 0,
            'contribution_mse': contrib_mse,
            'matched': factor_mapping[est_factor_idx]['matched']
        }

    # Summary plot showing all correlations
    _plot_correlation_matrix(correlation_matrix, n_est_factors, n_true_factors, correlation_threshold)

    return results

def _plot_profile_comparison(true_profile, est_profile, species_names, est_idx, true_idx,
                             correlation, model_type, uncertainty_intervals=None, confidence=0.95):
    """Plot profile and contribution comparison between true and estimated factors using subplots"""

    # Create subplots: profile on top, contribution on bottom
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f'Profile Comparison: Est. Factor {est_idx + 1} vs True Factor {true_idx + 1} (R² = {correlation:.3f})',
            f'Contribution Comparison: Est. Factor {est_idx + 1} vs True Factor {true_idx + 1}'
        ],
        vertical_spacing=0.15,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # Profile comparison (top subplot) - Bar plots
    x_pos_true = np.arange(len(species_names)) - 0.2
    x_pos_est = np.arange(len(species_names)) + 0.2

    # Add true profile bars
    fig.add_trace(go.Bar(
        x=x_pos_true,
        y=true_profile * 100,
        name=f'True Factor {true_idx + 1}',
        marker_color='red',
        opacity=0.7,
        width=0.35,
        hovertemplate='%{text}<br>True: %{y:.2f}%<extra></extra>',
        text=species_names
    ), row=1, col=1)

    # Add estimated profile bars
    fig.add_trace(go.Bar(
        x=x_pos_est,
        y=est_profile * 100,
        name=f'Est. Factor {est_idx + 1}',
        marker_color='blue',
        opacity=0.7,
        width=0.35,
        hovertemplate='%{text}<br>Est.: %{y:.2f}%<extra></extra>',
        text=species_names
    ), row=1, col=1)

    # Add uncertainty bands for Bayesian model (profile)
    if model_type == "Bayesian" and uncertainty_intervals is not None:
        try:
            # Get profile uncertainty intervals
            profile_lower = uncertainty_intervals['profiles_ci'][0][est_idx, :] * 100
            profile_upper = uncertainty_intervals['profiles_ci'][1][est_idx, :] * 100

            # Add error bars for estimated profile
            fig.add_trace(go.Scatter(
                x=x_pos_est,
                y=est_profile * 100,
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=profile_upper - est_profile * 100,
                    arrayminus=est_profile * 100 - profile_lower,
                    color='blue',
                    thickness=2,
                    width=3
                ),
                mode='markers',
                marker=dict(color='blue', size=0),
                showlegend=False,
                name=f'{confidence * 100:.0f}% CI',
                hovertemplate='CI: [%{customdata[0]:.2f}%, %{customdata[1]:.2f}%]<extra></extra>',
                customdata=np.column_stack([profile_lower, profile_upper])
            ), row=1, col=1)
        except (KeyError, IndexError):
            pass

    # Contribution comparison (bottom subplot) - Line plots with sample indices
    sample_indices = np.arange(len(true_profile))  # Assuming same length as profiles for demo

    # For demonstration, create mock contribution time series
    # In practice, you'd pass true_contrib and est_contrib as parameters
    np.random.seed(42)  # For reproducible demo
    true_contrib_demo = np.random.lognormal(2, 0.5, len(sample_indices))
    est_contrib_demo = true_contrib_demo * (0.8 + 0.4 * np.random.random(len(sample_indices)))

    # Add true contributions
    fig.add_trace(go.Scatter(
        x=sample_indices,
        y=true_contrib_demo,
        mode='lines',
        name=f'True Factor {true_idx + 1} Contrib.',
        line=dict(color='red', width=2),
        hovertemplate='Sample %{x}<br>True: %{y:.2e}<extra></extra>'
    ), row=2, col=1)

    # Add estimated contributions
    fig.add_trace(go.Scatter(
        x=sample_indices,
        y=est_contrib_demo,
        mode='lines',
        name=f'Est. Factor {est_idx + 1} Contrib.',
        line=dict(color='blue', width=2, dash='dash'),
        hovertemplate='Sample %{x}<br>Est.: %{y:.2e}<extra></extra>'
    ), row=2, col=1)

    # Add uncertainty bands for Bayesian model (contributions)
    if model_type == "Bayesian" and uncertainty_intervals is not None:
        try:
            # Get contribution uncertainty intervals
            contrib_lower = uncertainty_intervals['contributions_ci'][0][:, est_idx]
            contrib_upper = uncertainty_intervals['contributions_ci'][1][:, est_idx]

            # Add uncertainty band
            fig.add_trace(go.Scatter(
                x=np.concatenate([sample_indices, sample_indices[::-1]]),
                y=np.concatenate([contrib_upper, contrib_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name=f'{confidence * 100:.0f}% CI',
                hoverinfo='skip'
            ), row=2, col=1)
        except (KeyError, IndexError):
            pass

    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        template='plotly_white',
        showlegend=True,
        hovermode='closest'
    )

    # Update x-axes
    fig.update_xaxes(
        tickvals=np.arange(len(species_names)),
        ticktext=species_names,
        tickangle=45,
        title_text='Chemical Species',
        row=1, col=1
    )
    fig.update_xaxes(
        title_text='Sample Index',
        row=2, col=1
    )

    # Update y-axes
    fig.update_yaxes(
        title_text='Percentage (%)',
        row=1, col=1
    )
    fig.update_yaxes(
        title_text='Contribution',
        type='log',
        row=2, col=1
    )

    fig.show()
    return fig


def _plot_profile_and_contribution_comparison(true_profile, est_profile, true_contrib, est_contrib,
                                              species_names, est_idx, true_idx, correlation, model_type,
                                              uncertainty_intervals=None, confidence=0.95):
    """Plot combined profile (bar) and contribution (line) comparison between true and estimated factors"""
    # Create subplots: profile on top, contribution on bottom
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f'{model_type} - Profile Comparison: Est. Factor {est_idx + 1} vs True Factor {true_idx + 1} (R² = {correlation:.3f})',
            f'{model_type} - Contribution Comparison: Est. Factor {est_idx + 1} vs True Factor {true_idx + 1}'
        ],
        vertical_spacing=0.25,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # Profile comparison (top subplot) - Bar plots
    x_pos_true = np.arange(len(species_names)) - 0.2
    x_pos_est = np.arange(len(species_names)) + 0.2

    # Add true profile bars
    fig.add_trace(go.Bar(
        x=x_pos_true,
        y=true_profile * 100,
        name=f'True Factor {true_idx + 1}',
        marker_color='red',
        opacity=0.7,
        width=0.35,
        hovertemplate='%{text}<br>True: %{y:.2f}%<extra></extra>',
        text=species_names
    ), row=1, col=1)

    # Add estimated profile bars
    fig.add_trace(go.Bar(
        x=x_pos_est,
        y=est_profile * 100,
        name=f'Est. Factor {est_idx + 1}',
        marker_color='blue',
        opacity=0.7,
        width=0.35,
        hovertemplate='%{text}<br>Est.: %{y:.2f}%<extra></extra>',
        text=species_names
    ), row=1, col=1)

    # Add uncertainty bands for Bayesian model (profile)
    if model_type == "Bayesian" and uncertainty_intervals is not None:
        try:
            profile_lower = uncertainty_intervals['profiles_ci'][0][est_idx, :]
            profile_upper = uncertainty_intervals['profiles_ci'][1][est_idx, :]
            fig.add_trace(go.Scatter(
                x=x_pos_est,
                y=est_profile * 100,
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=profile_upper,
                    arrayminus=profile_lower,
                    color='blue',
                    thickness=2,
                    width=3
                ),
                mode='markers',
                marker=dict(color='blue', size=0),
                showlegend=False,
                name=f'{confidence * 100:.0f}% CI (Profile)'
            ), row=1, col=1)
        except (KeyError, IndexError) as e:
            print(f"Warning: Could not add profile uncertainty bands: {e}")

    # Contribution comparison (bottom subplot) - Line plots
    sample_indices = np.arange(len(true_contrib))

    # Add uncertainty bands for Bayesian model (contributions) - FIRST so it's behind the lines
    if model_type == "Bayesian" and uncertainty_intervals is not None:
        try:
            contrib_lower = uncertainty_intervals['contributions_ci'][0][:, est_idx]
            contrib_upper = uncertainty_intervals['contributions_ci'][1][:, est_idx]

            # Ensure we have valid data
            if len(contrib_lower) == len(sample_indices) and len(contrib_upper) == len(sample_indices):
                fig.add_trace(go.Scatter(
                    x=np.concatenate([sample_indices, sample_indices[::-1]]),
                    y=np.concatenate([contrib_upper, contrib_lower[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,100,200,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=True,
                    name=f'{confidence * 100:.0f}% CI',
                    hoverinfo='skip'
                ), row=2, col=1)
            else:
                print(f"Warning: Uncertainty interval dimensions don't match. "
                      f"Expected {len(sample_indices)}, got lower: {len(contrib_lower)}, upper: {len(contrib_upper)}")
        except (KeyError, IndexError, TypeError) as e:
            print(f"Warning: Could not add contribution uncertainty bands: {e}")

    # Add true contributions
    fig.add_trace(go.Scatter(
        x=sample_indices,
        y=true_contrib,
        mode='lines',
        name=f'True Factor {true_idx + 1}',
        line=dict(color='red', width=2),
        hovertemplate='Sample %{x}<br>True: %{y:.2e}<extra></extra>'
    ), row=2, col=1)

    # Add estimated contributions
    fig.add_trace(go.Scatter(
        x=sample_indices,
        y=est_contrib,
        mode='lines',
        name=f'Est. Factor {est_idx + 1}',
        line=dict(color='blue', width=2, dash='dash'),
        hovertemplate='Sample %{x}<br>Est.: %{y:.2e}<extra></extra>'
    ), row=2, col=1)

    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        template='plotly_white',
        showlegend=True,
        hovermode='x unified',
    )

    # Update x-axes
    fig.update_xaxes(
        tickvals=np.arange(len(species_names)),
        ticktext=species_names,
        tickangle=45,
        title_text='Feature',
        row=1, col=1
    )
    fig.update_xaxes(
        title_text='Sample Index',
        row=2, col=1
    )

    # Update y-axes
    fig.update_yaxes(
        title_text='Percentage (%)',
        row=1, col=1
    )
    fig.update_yaxes(
        title_text='Contribution',
        type='log',
        row=2, col=1
    )

    fig.show()
    return fig

def _plot_correlation_matrix(correlation_matrix, n_est, n_true, threshold):
    """Plot correlation matrix heatmap"""

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=[f'True {i + 1}' for i in range(n_true)],
        y=[f'Est. {i + 1}' for i in range(n_est)],
        colorscale='RdBu',
        zmid=threshold,
        colorbar=dict(title="R²"),
        text=np.round(correlation_matrix, 3),
        texttemplate="%{text}",
        textfont={"size": 12}
    ))

    fig.update_layout(
        title=f'Factor Correlation Matrix (R²)<br>Threshold = {threshold}',
        xaxis_title='True Factors',
        yaxis_title='Estimated Factors',
        width=600,
        height=500,
        template='plotly_white'
    )

    fig.show()

    return fig


# Demonstration with synthetic data
if __name__ == "__main__":

    n_samples = 500
    n_species = 20
    n_true_factors = 6
    random_seed = 42

    species_names = [f'Species_{i + 1}' for i in range(n_species)]

    V, U, true_k, true_H, true_W, sim = generate_dataset(true_k=n_true_factors, n_samples=n_samples, n_features=n_species, seed=random_seed)

    # Initialize and fit model
    model = BayesianNMF(n_factors=n_true_factors, auto_factor_selection=True, uncertainty_estimation=True)
    batch_sa = BatchSA(V=V, U=U, factors=n_true_factors, method="ls-nmf", verbose=False, max_iter=10000, models=20, seed=random_seed)
    batch_sa.train()
    init_W = batch_sa.results[batch_sa.best_model].W
    init_H = batch_sa.results[batch_sa.best_model].H

    print("Fitting Bayesian NMF model...")
    n_chains = int((mp.cpu_count() - 2)*2)
    model.fit(V, U, species_names=species_names, init_W=init_W, init_H=init_H, n_samples=5000, n_tune=3000, n_chains=6)

    # print(f"\nModel Diagnostics:")
    # diagnostics = model.get_model_diagnostics()
    # for key, value in diagnostics.items():
    #     if isinstance(value, (int, float)):
    #         print(f"{key}: {value:.4f}")

    # model.plot_trace()
    # model.plot_posterior()
    # model.plot_pair()
    # model.plot_energy()

    # batch_results = compare_factors_to_truth(
    #     true_W=true_W,
    #     true_H=true_H,
    #     estimated_W=batch_sa.results[batch_sa.best_model].W,
    #     estimated_H=batch_sa.results[batch_sa.best_model].H,
    #     species_names=species_names,
    #     model_type="BatchSA"
    # )
    # Compare Bayesian NMF results to truth (with uncertainty)
    uncertainty_intervals = model.get_uncertainty_intervals(confidence=0.95)

    bayes_results = compare_factors_to_truth(
        true_W=true_W,
        true_H=true_H,
        estimated_W=model.factor_contributions,
        estimated_H=model.factor_profiles,
        species_names=species_names,
        model_type="Bayesian",
        uncertainty_intervals=uncertainty_intervals,
        confidence=0.95
    )

    # Print summary
    # print("BatchSA Factor Matching:")
    # for factor, metrics in batch_results['factor_metrics'].items():
    #     print(f"  Est. Factor {factor + 1} -> True Factor {metrics['mapped_to_true_factor'] + 1}: "
    #           f"R² = {metrics['profile_correlation']:.3f}")

    print("\nBayesian Factor Matching:")
    for factor, metrics in bayes_results['factor_metrics'].items():
        print(f"  Est. Factor {factor + 1} -> True Factor {metrics['mapped_to_true_factor'] + 1}: "
              f"R² = {metrics['profile_correlation']:.3f}")

    print("\nPlot Uncertainty Intervals:")
    model.plot_uncertainty_intervals(confidence=0.95)

    q_loss = q_loss(V=V, U=U, W=model.factor_contributions, H=model.factor_profiles)
    print(f"\nBayesian NMF Q-loss: {q_loss:.4f}")
    print(f"SA Q-loss: {batch_sa.results[batch_sa.best_model].Qtrue:.4f}")
