class VectorDBIntegration:
    """
    Vector database integration for SPECIATE profiles and continuous monitoring
    """

    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim
        self.profile_embeddings = {}
        self.speciate_profiles = {}

    def embed_profile(self, profile):
        """Create embedding vector from chemical profile"""
        # Simple embedding: normalize and pad/truncate to fixed dimension
        normalized = profile / np.sum(profile) if np.sum(profile) > 0 else profile

        if len(normalized) >= self.embedding_dim:
            return normalized[:self.embedding_dim]
        else:
            padded = np.zeros(self.embedding_dim)
            padded[:len(normalized)] = normalized
            return padded

    def add_speciate_profiles(self, speciate_data):
        """Add SPECIATE profiles to vector database"""
        for profile_id, profile_data in speciate_data.items():
            embedding = self.embed_profile(np.array(list(profile_data.values())))
            self.profile_embeddings[profile_id] = embedding
            self.speciate_profiles[profile_id] = profile_data

    def find_similar_profiles(self, query_profile, top_k=5):
        """Find most similar SPECIATE profiles"""
        from sklearn.metrics.pairwise import cosine_similarity

        query_embedding = self.embed_profile(query_profile)
        similarities = {}

        for profile_id, embedding in self.profile_embeddings.items():
            sim = cosine_similarity([query_embedding], [embedding])[0, 0]
            similarities[profile_id] = sim

        # Return top-k most similar
        sorted_profiles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_profiles[:top_k]


