class OnlineBayesianNMF(BayesianNMF):
    """
    Online/Incremental Bayesian NMF for continuous monitoring
    Handles streaming data without retraining from scratch
    """

    def __init__(self, window_size_hours=24, resolution_minutes=5,
                 update_frequency_hours=1, **kwargs):
        super().__init__(**kwargs)
        self.window_size = int(window_size_hours * 60 / resolution_minutes)  # samples in window
        self.resolution_minutes = resolution_minutes
        self.update_frequency = int(update_frequency_hours * 60 / resolution_minutes)

        # Online learning components
        self.data_buffer = None
        self.uncertainty_buffer = None
        self.sample_count = 0
        self.last_update = 0
        self.prior_params = None

        # Warm-up period (collect initial data before first model)
        self.warmup_samples = max(50, self.window_size // 4)
        self.is_warmed_up = False

        # Model stability tracking
        self.profile_history = []
        self.stability_threshold = 0.1  # cosine similarity threshold for profile drift

    def add_observation(self, x, x_uncertainty=None, timestamp=None):
        """
        Add single observation to rolling buffer

        Parameters:
        -----------
        x : array-like, shape (n_species,)
            Single observation vector
        x_uncertainty : array-like, shape (n_species,)
            Uncertainty for this observation
        timestamp : datetime
            Observation timestamp
        """
        x = np.array(x).reshape(1, -1)

        if x_uncertainty is None:
            x_uncertainty = np.maximum(0.05 * x, 0.01 * np.mean(x))
        else:
            x_uncertainty = np.array(x_uncertainty).reshape(1, -1)

        # Initialize buffers on first observation
        if self.data_buffer is None:
            self.data_buffer = np.zeros((self.window_size, x.shape[1]))
            self.uncertainty_buffer = np.zeros((self.window_size, x.shape[1]))

        # Rolling buffer update
        self.data_buffer = np.roll(self.data_buffer, -1, axis=0)
        self.uncertainty_buffer = np.roll(self.uncertainty_buffer, -1, axis=0)

        self.data_buffer[-1] = x
        self.uncertainty_buffer[-1] = x_uncertainty

        self.sample_count += 1

        # Check if update is needed
        if self._should_update():
            self._incremental_update()

    def _should_update(self):
        """Determine if model should be updated"""
        # Wait for warmup
        if not self.is_warmed_up and self.sample_count >= self.warmup_samples:
            self.is_warmed_up = True
            return True

        # Regular updates after warmup
        if (self.is_warmed_up and
                self.sample_count - self.last_update >= self.update_frequency):
            return True

        return False

    def _incremental_update(self):
        """
        Perform incremental model update using informative priors
        """
        print(f"Incremental update at sample {self.sample_count}")

        # Use current data window
        current_data = self.data_buffer.copy()
        current_uncertainty = self.uncertainty_buffer.copy()

        # Remove zeros from incomplete buffer
        if self.sample_count < self.window_size:
            n_valid = min(self.sample_count, self.window_size)
            current_data = current_data[-n_valid:]
            current_uncertainty = current_uncertainty[-n_valid:]

        if self.trace is None:
            # First fit - use default priors
            self.fit(current_data, current_uncertainty,
                     n_samples=1000, n_tune=500)
        else:
            # Incremental update with informative priors
            self._update_with_priors(current_data, current_uncertainty)

        self.last_update = self.sample_count

        # Track profile stability
        self._track_profile_stability()

    def _update_with_priors(self, X, X_uncertainty):
        """Update model using previous posterior as prior"""

        # Extract parameters from previous fit
        W_samples = self.trace.posterior['W'].values.reshape(-1, *self.trace.posterior['W'].values.shape[-2:])
        H_samples = self.trace.posterior['H'].values.reshape(-1, *self.trace.posterior['H'].values.shape[-2:])

        # Estimate prior parameters (exponential rate from previous posterior)
        W_mean = np.mean(W_samples, axis=0)
        H_mean = np.mean(H_samples, axis=0)

        W_rate = 1.0 / (W_mean + 1e-8)  # Rate parameter for exponential
        H_rate = 1.0 / (H_mean + 1e-8)

        # Fit with informative priors
        with pm.Model() as model:
            # Informative priors based on previous posterior
            W = pm.Exponential('W', lam=W_rate, shape=W_rate.shape)
            H = pm.Exponential('H', lam=H_rate, shape=H_rate.shape)

            # Reconstruction
            X_recon = pm.math.dot(W, H)

            # Likelihood
            likelihood = pm.Normal('X_obs', mu=X_recon, sigma=X_uncertainty, observed=X)

            # Sample with fewer iterations for speed
            trace = pm.sample(draws=800, tune=400, cores=1,
                              target_accept=0.9, progressbar=False,
                              return_inferencedata=True)

        self.model = model
        self.trace = trace
        self._extract_results(X)

    def _track_profile_stability(self):
        """Track profile evolution and detect significant changes"""
        from sklearn.metrics.pairwise import cosine_similarity

        current_profiles = self.get_factor_profiles(normalize=True)
        self.profile_history.append(current_profiles.copy())

        # Keep only recent history
        if len(self.profile_history) > 10:
            self.profile_history = self.profile_history[-10:]

        # Check stability if we have history
        if len(self.profile_history) >= 2:
            prev_profiles = self.profile_history[-2]
            similarities = []

            for i in range(current_profiles.shape[1]):
                sim = cosine_similarity([current_profiles[:, i]],
                                        [prev_profiles[:, i]])[0, 0]
                similarities.append(sim)

            min_similarity = min(similarities)

            if min_similarity < (1 - self.stability_threshold):
                print(f"Profile drift detected! Min similarity: {min_similarity:.3f}")
                # Could trigger alerts or model reinitialization

    def get_current_contributions(self, last_n_hours=1):
        """Get factor contributions for recent time period"""
        n_samples = int(last_n_hours * 60 / self.resolution_minutes)
        if self.factor_contributions is not None:
            return self.factor_contributions[-n_samples:]
        return None

    def predict_next_window(self, n_steps=12):  # 1 hour ahead at 5-min resolution
        """Predict future concentrations based on current model"""
        if self.trace is None:
            return None

        # Simple approach: use recent contribution trends
        recent_contributions = self.get_current_contributions(last_n_hours=2)
        if recent_contributions is None:
            return None

        # Linear extrapolation of trends
        trends = np.diff(recent_contributions[-24:], axis=0)  # Last 2 hours of trends
        mean_trend = np.mean(trends, axis=0)

        # Project forward
        last_contribution = recent_contributions[-1]
        future_contributions = []

        for step in range(n_steps):
            next_contrib = last_contribution + mean_trend * (step + 1)
            next_contrib = np.maximum(next_contrib, 0)  # Ensure non-negative
            future_contributions.append(next_contrib)

        future_contributions = np.array(future_contributions)

        # Convert to concentrations
        profiles = self.get_factor_profiles()
        future_concentrations = np.dot(future_contributions, profiles)

        return future_concentrations

    def save(self, filepath):
        """Save online model state and trace."""
        state = {
            'window_size': self.window_size,
            'resolution_minutes': self.resolution_minutes,
            'update_frequency': self.update_frequency,
            'data_buffer': self.data_buffer,
            'uncertainty_buffer': self.uncertainty_buffer,
            'sample_count': self.sample_count,
            'last_update': self.last_update,
            'prior_params': self.prior_params,
            'warmup_samples': self.warmup_samples,
            'is_warmed_up': self.is_warmed_up,
            'profile_history': self.profile_history,
            'stability_threshold': self.stability_threshold,
            # Inherit parent state
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
        """Load online model state and trace."""
        with open(filepath + '_state.pkl', 'rb') as f:
            state = pickle.load(f)
        model = cls(
            window_size_hours=state['window_size'] * state['resolution_minutes'] / 60,
            resolution_minutes=state['resolution_minutes'],
            update_frequency_hours=state['update_frequency'] * state['resolution_minutes'] / 60,
            n_factors=state['n_factors'],
            max_factors=state['max_factors'],
            auto_factor_selection=state['auto_factor_selection'],
            uncertainty_estimation=state['uncertainty_estimation'],
            vector_db_integration=state['vector_db_integration'],
        )
        # Restore state
        model.data_buffer = state['data_buffer']
        model.uncertainty_buffer = state['uncertainty_buffer']
        model.sample_count = state['sample_count']
        model.last_update = state['last_update']
        model.prior_params = state['prior_params']
        model.warmup_samples = state['warmup_samples']
        model.is_warmed_up = state['is_warmed_up']
        model.profile_history = state['profile_history']
        model.stability_threshold = state['stability_threshold']
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

