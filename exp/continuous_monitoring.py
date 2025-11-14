class ContinuousMonitoringFramework:
    """
    Complete framework for continuous EPA air quality monitoring
    """

    def __init__(self, site_ids, species_names, speciate_profiles=None):
        self.site_ids = site_ids
        self.species_names = species_names

        # One model per site for spatial heterogeneity
        self.site_models = {}
        for site_id in site_ids:
            self.site_models[site_id] = OnlineBayesianNMF(
                window_size_hours=24,
                resolution_minutes=5,
                update_frequency_hours=1,
                auto_factor_selection=True
            )

        # Shared vector database for profile matching
        self.vector_db = VectorDBIntegration()
        if speciate_profiles:
            self.vector_db.add_speciate_profiles(speciate_profiles)

        # Data logging
        self.observation_log = []

    def process_realtime_data(self, site_id, observation, uncertainty=None, timestamp=None):
        """Process single real-time observation"""
        if site_id not in self.site_models:
            raise ValueError(f"Unknown site_id: {site_id}")

        # Log observation
        self.observation_log.append({
            'site_id': site_id,
            'timestamp': timestamp,
            'data': observation.copy()
        })

        # Update site-specific model
        self.site_models[site_id].add_observation(observation, uncertainty, timestamp)

    def get_site_status(self, site_id):
        """Get current status of site model"""
        model = self.site_models[site_id]

        return {
            'is_warmed_up': model.is_warmed_up,
            'sample_count': model.sample_count,
            'last_update': model.last_update,
            'n_factors': model.n_factors,
            'model_diagnostics': model.get_model_diagnostics() if model.trace else None
        }

    def get_network_summary(self):
        """Get summary across all sites"""
        summary = {}
        for site_id in self.site_ids:
            summary[site_id] = self.get_site_status(site_id)
        return summary

    def cross_site_analysis(self):
        """Compare profiles across sites"""
        from sklearn.metrics.pairwise import cosine_similarity

        site_profiles = {}
        for site_id, model in self.site_models.items():
            if model.trace is not None:
                site_profiles[site_id] = model.get_factor_profiles(normalize=True)

        if len(site_profiles) < 2:
            return None

        # Pairwise site comparisons
        comparisons = {}
        sites = list(site_profiles.keys())

        for i, site1 in enumerate(sites):
            for site2 in sites[i + 1:]:
                profiles1 = site_profiles[site1]
                profiles2 = site_profiles[site2]

                # Match factors between sites (Hungarian algorithm could be used)
                similarities = cosine_similarity(profiles1.T, profiles2.T)
                comparisons[f"{site1}_vs_{site2}"] = similarities

        return comparisons