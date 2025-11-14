import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from utils import generate_dataset, q_loss


@dataclass
class PIGTConfig:
    """Configuration for Physics-Informed Graph Transformer"""
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    max_factors: Dict[str, int] = None
    physics_weight: float = 0.1
    uncertainty_samples: int = 100

    def __post_init__(self):
        if self.max_factors is None:
            self.max_factors = {'regional': 5, 'urban': 8, 'local': 12}


class GraphTransformerLayer(nn.Module):
    """Graph Transformer layer with spatial-temporal attention"""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.edge_embedding = nn.Linear(3, embed_dim)  # spatial, temporal, chemical edges

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # Encode edge attributes: edge_emb shape [seq_len, seq_len, embed_dim]
        edge_emb = self.edge_embedding(edge_attr)
        # Aggregate edge embeddings for each node (sum over neighbors)
        # edge_emb.sum(dim=1) shape: [seq_len, embed_dim]
        node_edge_emb = edge_emb.sum(dim=1)
        # Add aggregated edge embedding to node embedding
        x_with_edge = x + node_edge_emb.unsqueeze(0)  # broadcast batch dim
        # Self-attention
        attn_out, _ = self.self_attn(x_with_edge, x_with_edge, x_with_edge)
        x = self.norm1(x_with_edge + attn_out)
        # Feed forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x


class PhysicsModule(nn.Module):
    """Physics constraints for atmospheric chemistry"""

    def __init__(self, n_species: int):
        super().__init__()
        self.n_species = n_species

        # Learnable physical parameters
        self.reaction_rates = nn.Parameter(torch.randn(n_species, n_species) * 0.01)
        self.transport_coeff = nn.Parameter(torch.ones(n_species))

    def mass_conservation_loss(self, concentrations: torch.Tensor,
                              sources: torch.Tensor) -> torch.Tensor:
        """Enforce mass conservation: input = output + accumulation"""
        # Simple mass balance check
        total_input = torch.sum(sources, dim=-1)
        total_conc = torch.sum(concentrations, dim=-1)

        return F.mse_loss(total_input, total_conc)

    def transport_loss(self, concentrations: torch.Tensor,
                      wind_data: torch.Tensor) -> torch.Tensor:
        """Basic advection-diffusion constraint"""
        # Simplified transport equation: ∂C/∂t + u∇C = S
        # This is a placeholder for full physics implementation

        spatial_grad = torch.gradient(concentrations, dim=1)[0]
        advection = torch.sum(wind_data.unsqueeze(-1) * spatial_grad, dim=1)

        return torch.mean(advection ** 2)  # Minimize unrealistic transport

    def chemical_kinetics_loss(self, concentrations: torch.Tensor) -> torch.Tensor:
        """Chemical reaction constraints"""
        # Simplified reaction kinetics
        reaction_term = torch.matmul(concentrations, torch.relu(self.reaction_rates))

        return F.mse_loss(reaction_term, torch.zeros_like(reaction_term))


class HierarchicalFactorModule(nn.Module):
    """Multi-scale hierarchical source factor discovery"""

    def __init__(self, embed_dim: int, max_factors: Dict[str, int]):
        super().__init__()
        self.scales = list(max_factors.keys())
        self.max_factors = max_factors

        # Factor discovery for each scale
        self.factor_networks = nn.ModuleDict({
            scale: nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, max_factors[scale])
            )
            for scale in self.scales
        })

        # Factor gates for automatic selection
        self.factor_gates = nn.ParameterDict({
            scale: nn.Parameter(torch.ones(max_factors[scale]))
            for scale in self.scales
        })

        # Cross-scale interaction
        self.scale_fusion = nn.Linear(sum(max_factors.values()), embed_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        factors = {}

        for scale in self.scales:
            # Extract factors for this scale
            raw_factors = self.factor_networks[scale](x)

            # Apply gates for automatic selection
            gates = torch.sigmoid(self.factor_gates[scale])
            factors[scale] = raw_factors * gates.unsqueeze(0).unsqueeze(0)

        # Fuse across scales
        all_factors = torch.cat(list(factors.values()), dim=-1)
        fused = self.scale_fusion(all_factors)

        return factors, fused


class BayesianLinear(nn.Module):
    """Bayesian linear layer for uncertainty quantification"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features))

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.randn(out_features))
        self.bias_logvar = nn.Parameter(torch.randn(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample weights and biases
        weight_std = torch.exp(0.5 * self.weight_logvar)
        weight = self.weight_mu + weight_std * torch.randn_like(weight_std)

        bias_std = torch.exp(0.5 * self.bias_logvar)
        bias = self.bias_mu + bias_std * torch.randn_like(bias_std)

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """KL divergence for variational inference"""
        weight_kl = 0.5 * torch.sum(
            self.weight_logvar.exp() + self.weight_mu.pow(2) - self.weight_logvar - 1
        )
        bias_kl = 0.5 * torch.sum(
            self.bias_logvar.exp() + self.bias_mu.pow(2) - self.bias_logvar - 1
        )
        return weight_kl + bias_kl


class PhysicsInformedGraphTransformer(nn.Module):
    """
    Complete Physics-Informed Graph Transformer for source apportionment

    Key improvements:
    1. Graph transformer for long-range dependencies
    2. Physics constraints from atmospheric chemistry
    3. Hierarchical multi-scale factor discovery
    4. Bayesian uncertainty quantification
    """

    def __init__(self, n_species: int, config: PIGTConfig):
        super().__init__()
        self.config = config
        self.n_species = n_species

        # Input embedding
        self.input_embedding = nn.Linear(n_species, config.embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1000, config.embed_dim))

        # Graph transformer layers
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(config.embed_dim, config.num_heads)
            for _ in range(config.num_layers)
        ])

        # Physics constraints
        self.physics = PhysicsModule(n_species)

        # Hierarchical factor discovery
        self.factor_module = HierarchicalFactorModule(
            config.embed_dim, config.max_factors
        )

        # Bayesian output layers
        self.source_profiles = BayesianLinear(
            config.embed_dim, sum(config.max_factors.values()) * n_species
        )
        self.factor_contributions = BayesianLinear(
            config.embed_dim, sum(config.max_factors.values())
        )

    def create_edges(self, coordinates: torch.Tensor,
                    timestamps: torch.Tensor) -> torch.Tensor:
        """Create spatial-temporal-chemical edge attributes"""
        n_nodes = coordinates.shape[0]
        edges = torch.zeros(n_nodes, n_nodes, 3)

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    # Spatial distance (normalized)
                    spatial_dist = torch.norm(coordinates[i] - coordinates[j])
                    edges[i, j, 0] = torch.exp(-spatial_dist / 50.0)  # 50km scale

                    # Temporal distance
                    temporal_dist = abs(timestamps[i] - timestamps[j])
                    edges[i, j, 1] = torch.exp(-temporal_dist / 24.0)  # 24hr scale

                    # Chemical similarity (placeholder)
                    edges[i, j, 2] = 1.0

        return edges

    def weighed_edges(self, coordinates: torch.Tensor, timestamps: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Create weighted spatial-temporal-chemical edge attributes using node weights.
        The weights are broadcasted or converted to match edge assignments.
        Args:
            coordinates: [n_nodes, 2] tensor
            timestamps: [n_nodes] tensor
            weights: [n_nodes] tensor (or [n_nodes, ...])
        Returns:
            edges: [n_nodes, n_nodes, 3] tensor with weights applied
        """
        n_nodes = coordinates.shape[0]
        edges = torch.zeros(n_nodes, n_nodes, 3)
        # Standard edge attributes
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    spatial_dist = torch.norm(coordinates[i] - coordinates[j])
                    edges[i, j, 0] = torch.exp(-spatial_dist / 50.0)
                    temporal_dist = abs(timestamps[i] - timestamps[j])
                    edges[i, j, 1] = torch.exp(-temporal_dist / 24.0)
                    edges[i, j, 2] = 1.0
        # Apply weights: expand weights to edge shape
        # If weights shape is [n_nodes], create [n_nodes, n_nodes, 1] by outer product
        if weights.dim() == 1:
            weight_matrix = torch.outer(weights, weights)  # [n_nodes, n_nodes]
            weight_matrix = weight_matrix.unsqueeze(-1)    # [n_nodes, n_nodes, 1]
        else:
            # If weights already have extra dims, try to broadcast
            weight_matrix = weights.unsqueeze(1) * weights.unsqueeze(0)
            weight_matrix = weight_matrix.mean(dim=-1, keepdim=True)  # [n_nodes, n_nodes, 1]
        # Multiply edge attributes by weights
        edges = edges * weight_matrix
        return edges

    def forward(self, x: torch.Tensor, coordinates: torch.Tensor,
                timestamps: torch.Tensor, U: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        # If uncertainty is provided, compute weights and apply to input
        weights = 1.0 / (U ** 2)
        # x = x * weights.unsqueeze(0)  # broadcast batch dim

        # Input embedding with positional encoding
        x_emb = self.input_embedding(x)
        x_emb += self.pos_encoding[:seq_len].unsqueeze(0)

        # Create edge attributes
        edge_attr = self.weighed_edges(coordinates, timestamps, weights)
        # edge_attr = self.create_edges(coordinates, timestamps)

        # Graph transformer layers
        for layer in self.transformer_layers:
            x_emb = layer(x_emb, edge_attr)

        # Hierarchical factor discovery
        factors, fused_factors = self.factor_module(x_emb)

        # Bayesian outputs
        source_profiles = self.source_profiles(fused_factors)
        factor_contribs = self.factor_contributions(fused_factors)

        # Reshape source profiles
        total_factors = sum(self.config.max_factors.values())
        source_profiles = source_profiles.view(
            batch_size, seq_len, total_factors, self.n_species
        )

        # Reconstruction
        reconstruction = torch.sum(
            factor_contribs.unsqueeze(-1) * source_profiles,
            dim=2
        )

        return {
            'reconstruction': reconstruction,
            'factors': factors,
            'source_profiles': source_profiles,
            'factor_contributions': factor_contribs,
            'embeddings': fused_factors
        }

    def physics_loss(self, outputs: Dict, x: torch.Tensor,
                    wind_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate physics-informed loss"""
        reconstruction = outputs['reconstruction']
        source_profiles = outputs['source_profiles']

        # Mass conservation
        mass_loss = self.physics.mass_conservation_loss(
            reconstruction, torch.sum(source_profiles, dim=2)
        )

        # Transport constraints
        transport_loss = 0
        if wind_data is not None:
            transport_loss = self.physics.transport_loss(reconstruction, wind_data)

        # Chemical kinetics
        kinetics_loss = self.physics.chemical_kinetics_loss(reconstruction)

        return mass_loss + transport_loss + kinetics_loss

    def uncertainty_forward(self, x: torch.Tensor, coordinates: torch.Tensor,
                          timestamps: torch.Tensor, n_samples: int = 100):
        """Forward pass with uncertainty quantification"""
        predictions = []

        for _ in range(n_samples):
            pred = self.forward(x, coordinates, timestamps)
            predictions.append(pred['reconstruction'])

        predictions = torch.stack(predictions)

        return {
            'mean': torch.mean(predictions, dim=0),
            'std': torch.std(predictions, dim=0),
            'samples': predictions
        }

    def torch_q_loss(self, V, U, W, H, uncertainty=True):
        """
        PyTorch version of q_loss for backpropagation.
        V: [samples, species] (torch.Tensor)
        U: [samples, species] (torch.Tensor)
        W: [samples, factors] (torch.Tensor)
        H: [factors, species] (torch.Tensor)
        """
        _wh = torch.matmul(W, H)
        residuals = V - _wh
        if uncertainty:
            residuals_u = residuals / U
            r2 = residuals_u ** 2
            _q = torch.sum(r2)
        else:
            _q = torch.sum(residuals ** 2)
        return _q

    def fit(self, V, U, epochs=100, lr=1e-1, verbose=True):
        """
        Fit the model to input data V and uncertainty U by minimizing q_loss.
        Uses torch tensors throughout for proper gradient flow.
        Shows progress with tqdm.
        """
        try:
            from tqdm import trange
        except ImportError:
            def trange(x):
                return range(x)
        V_tensor = torch.tensor(V, dtype=torch.float32) if not torch.is_tensor(V) else V.float()
        # Clip U to avoid very small values
        U = np.clip(U, 1e-2, None) if not torch.is_tensor(U) else torch.clamp(U, min=1e-2)
        U_tensor = torch.tensor(U, dtype=torch.float32) if not torch.is_tensor(U) else U.float()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        pbar = trange(epochs, desc="Training", leave=True)
        for epoch in pbar:
            optimizer.zero_grad()
            batch = V_tensor.unsqueeze(0)  # [1, samples, species]
            coords = torch.zeros(batch.shape[1], 2)
            times = torch.arange(batch.shape[1])
            outputs = self.forward(batch, coords, times, U=U_tensor)
            W = outputs['factor_contributions'][0, -1]  # [factors]
            H = outputs['source_profiles'][0, -1]       # [factors, species]
            W_exp = W.unsqueeze(0).expand(V_tensor.shape[0], -1)
            loss = self.torch_q_loss(V_tensor, U_tensor, W_exp, H)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            pbar.set_description(f"Epoch {epoch+1}/{epochs} | q_loss: {loss.item():.4e}")
        self.eval()

class StreamingSourceApportionment:
    """
    Streaming architecture for real-time EPA monitoring integration
    """

    def __init__(self, model: PhysicsInformedGraphTransformer):
        self.model = model
        self.model.eval()

        # Streaming buffers
        self.data_buffer = []
        self.max_buffer_size = 168  # 1 week of hourly data

    def process_stream(self, new_data: Dict) -> Dict:
        """Process streaming data from EPA monitors"""
        # Add to buffer
        self.data_buffer.append(new_data)

        # Maintain buffer size
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer.pop(0)

        # Convert to tensors
        concentrations = torch.FloatTensor([d['concentrations'] for d in self.data_buffer])
        coordinates = torch.FloatTensor([d['coordinates'] for d in self.data_buffer])
        timestamps = torch.FloatTensor([d['timestamp'] for d in self.data_buffer])

        # Inference with uncertainty
        with torch.no_grad():
            results = self.model.uncertainty_forward(
                concentrations.unsqueeze(0),
                coordinates,
                timestamps,
                n_samples=50
            )

        return {
            'source_contributions': results['mean'][-1].numpy(),  # Latest timestep
            'uncertainty': results['std'][-1].numpy(),
            'confidence_interval': torch.quantile(
                results['samples'][:, -1], torch.tensor([0.05, 0.95]), dim=0
            ).numpy()
        }


class CMAQISAMIntegration:
    """Integration with EPA's CMAQ-ISAM for validation and training"""

    def __init__(self, model: PhysicsInformedGraphTransformer):
        self.model = model

    def validate_against_isam(self, neural_results: Dict,
                             isam_results: Dict) -> Dict:
        """Validate neural predictions against CMAQ-ISAM"""

        # Calculate correlation between methods
        neural_contribs = neural_results['source_contributions']
        isam_contribs = isam_results['source_contributions']

        correlation = np.corrcoef(
            neural_contribs.flatten(),
            isam_contribs.flatten()
        )[0, 1]

        # Bias analysis
        bias = np.mean(neural_contribs - isam_contribs)
        rmse = np.sqrt(np.mean((neural_contribs - isam_contribs) ** 2))

        return {
            'correlation': correlation,
            'bias': bias,
            'rmse': rmse,
            'validation_score': correlation - abs(bias) - rmse
        }

    def transfer_learning_from_isam(self, isam_data: List[Dict]):
        """Pre-train model using CMAQ-ISAM outputs"""

        # Convert ISAM data to training format
        X = torch.FloatTensor([d['concentrations'] for d in isam_data])
        Y = torch.FloatTensor([d['source_contributions'] for d in isam_data])

        # Fine-tuning with ISAM supervision
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        for epoch in range(100):
            optimizer.zero_grad()

            outputs = self.model.forward(X, None, None)

            # Supervised loss using ISAM targets
            supervised_loss = F.mse_loss(
                outputs['factor_contributions'], Y
            )

            # Add KL divergence for Bayesian layers
            kl_loss = sum(
                layer.kl_divergence()
                for layer in self.model.modules()
                if isinstance(layer, BayesianLinear)
            )

            total_loss = supervised_loss + 0.01 * kl_loss
            total_loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Transfer learning epoch {epoch}: Loss = {total_loss:.4f}")


class SPECIATEIntegration:
    """Integration with EPA's SPECIATE database for profile matching"""

    def __init__(self, speciate_db_path: str, species_mapping: Dict[str, str]):
        self.species_mapping = species_mapping  # Model species -> SPECIATE codes
        self.speciate_profiles = self._load_speciate_db(speciate_db_path)

    def _load_speciate_db(self, db_path: str) -> Dict:
        """Load and preprocess SPECIATE database"""
        # In practice, load from EPA's SPECIATE database
        # This is a simplified structure
        return {
            'profile_ids': [],
            'source_types': [],
            'profiles': [],  # Normalized concentration profiles
            'metadata': []   # Source descriptions, references, etc.
        }

    def match_extracted_profiles(self, source_profiles: torch.Tensor,
                               confidence_threshold: float = 0.7) -> List[Dict]:
        """Match neural-extracted profiles against SPECIATE database"""
        matches = []

        for i, profile in enumerate(source_profiles):
            profile_np = profile.detach().cpu().numpy()

            # Normalize profile for comparison
            normalized_profile = profile_np / np.sum(profile_np)

            best_match = self._find_best_speciate_match(normalized_profile)

            matches.append({
                'factor_id': i,
                'speciate_id': best_match['id'],
                'source_type': best_match['source_type'],
                'correlation': best_match['correlation'],
                'confidence': 'High' if best_match['correlation'] > confidence_threshold else 'Medium',
                'is_novel': best_match['correlation'] < 0.5  # Potentially new source type
            })

        return matches

    def _find_best_speciate_match(self, profile: np.ndarray) -> Dict:
        """Find best matching SPECIATE profile using correlation"""
        best_correlation = 0
        best_match = {'id': 'NOVEL', 'source_type': 'Unknown', 'correlation': 0}

        for i, speciate_profile in enumerate(self.speciate_profiles['profiles']):
            # Map species to SPECIATE format
            mapped_profile = self._map_species_to_speciate(profile)

            # Calculate correlation
            correlation = np.corrcoef(mapped_profile, speciate_profile)[0, 1]

            if correlation > best_correlation:
                best_correlation = correlation
                best_match = {
                    'id': self.speciate_profiles['profile_ids'][i],
                    'source_type': self.speciate_profiles['source_types'][i],
                    'correlation': correlation
                }

        return best_match

    def _map_species_to_speciate(self, profile: np.ndarray) -> np.ndarray:
        """Map model species to SPECIATE species codes"""
        # Implementation depends on your species list and SPECIATE mapping
        return profile  # Simplified


class VectorProfileDatabase:
    """Vector database integration for fast profile similarity search"""

    def __init__(self, dimension: int, index_type: str = "IVF"):
        self.dimension = dimension

        # Initialize vector database (using Qdrant as example)
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self.client = QdrantClient(host="localhost", port=6333)

            # Create collection if not exists
            try:
                self.client.create_collection(
                    collection_name="source_profiles",
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=Distance.COSINE
                    )
                )
            except:
                pass  # Collection already exists

        except ImportError:
            print("Qdrant not available, using in-memory storage")
            self.client = None
            self.profiles = {}

    def store_profile(self, profile_id: str, profile_vector: np.ndarray,
                     metadata: Dict):
        """Store source profile in vector database"""
        if self.client:
            from qdrant_client.models import PointStruct

            point = PointStruct(
                id=profile_id,
                vector=profile_vector.tolist(),
                payload=metadata
            )

            self.client.upsert(
                collection_name="source_profiles",
                points=[point]
            )
        else:
            # In-memory fallback
            self.profiles[profile_id] = {
                'vector': profile_vector,
                'metadata': metadata
            }

    def search_similar_profiles(self, query_profile: np.ndarray,
                              top_k: int = 5, score_threshold: float = 0.8):
        """Search for similar profiles in vector database"""
        if self.client:
            results = self.client.search(
                collection_name="source_profiles",
                query_vector=query_profile.tolist(),
                limit=top_k,
                score_threshold=score_threshold
            )

            return [
                {
                    'id': hit.id,
                    'score': hit.score,
                    'metadata': hit.payload
                }
                for hit in results
            ]
        else:
            # In-memory search
            similarities = []
            for profile_id, data in self.profiles.items():
                similarity = np.dot(query_profile, data['vector']) / (
                    np.linalg.norm(query_profile) * np.linalg.norm(data['vector'])
                )
                if similarity >= score_threshold:
                    similarities.append({
                        'id': profile_id,
                        'score': similarity,
                        'metadata': data['metadata']
                    })

            return sorted(similarities, key=lambda x: x['score'], reverse=True)[:top_k]

    def store_neural_profiles(self, model_outputs: Dict, location_id: str,
                            timestamp: str, uncertainty_metrics: Dict):
        """Store newly discovered profiles from neural model"""
        source_profiles = model_outputs['source_profiles'][0, -1]  # Latest timestep

        for i, profile in enumerate(source_profiles):
            profile_id = f"{location_id}_{timestamp}_factor_{i}"

            metadata = {
                'location_id': location_id,
                'timestamp': timestamp,
                'factor_index': i,
                'method': 'Physics-Informed Graph Transformer',
                'uncertainty_std': uncertainty_metrics['std'][i].item(),
                'confidence_interval': uncertainty_metrics['confidence_interval'][i].tolist(),
                'physics_constrained': True
            }

            self.store_profile(profile_id, profile.detach().cpu().numpy(), metadata)


class IntegratedSourceApportionment:
    """Complete system integrating PIGT, SPECIATE, and vector database"""

    def __init__(self, model: PhysicsInformedGraphTransformer,
                 speciate_integration: SPECIATEIntegration,
                 vector_db: VectorProfileDatabase):
        self.model = model
        self.speciate = speciate_integration
        self.vector_db = vector_db

    def process_monitoring_data(self, data: Dict) -> Dict:
        """Complete workflow: inference -> matching -> storage"""

        # Neural inference with uncertainty
        concentrations = torch.FloatTensor(data['concentrations']).unsqueeze(0)
        coordinates = torch.FloatTensor(data['coordinates'])
        timestamps = torch.FloatTensor(data['timestamps'])

        # Get model predictions with uncertainty
        with torch.no_grad():
            outputs = self.model.forward(concentrations, coordinates, timestamps)
            uncertainty = self.model.uncertainty_forward(
                concentrations, coordinates, timestamps, n_samples=50
            )

        # Match against SPECIATE database
        source_profiles = outputs['source_profiles'][0, -1]  # Latest timestep
        speciate_matches = self.speciate.match_extracted_profiles(source_profiles)

        # Search for similar profiles in vector database
        vector_matches = []
        for i, profile in enumerate(source_profiles):
            profile_np = profile.detach().cpu().numpy()
            similar = self.vector_db.search_similar_profiles(profile_np)
            vector_matches.append(similar)

        # Store new profiles if novel or significantly different
        self._store_novel_profiles(outputs, uncertainty, data)

        return {
            'source_contributions': outputs['factor_contributions'][0, -1].detach().cpu().numpy(),
            'source_profiles': source_profiles.detach().cpu().numpy(),
            'speciate_matches': speciate_matches,
            'vector_matches': vector_matches,
            'uncertainty': {
                'mean': uncertainty['mean'][0, -1].numpy(),
                'std': uncertainty['std'][0, -1].numpy(),
                'confidence_interval': torch.quantile(
                    uncertainty['samples'][:, 0, -1],
                    torch.tensor([0.05, 0.95]), dim=0
                ).numpy()
            }
        }

    def _store_novel_profiles(self, outputs: Dict, uncertainty: Dict, data: Dict):
        """Store profiles that are novel or have low SPECIATE correlation"""
        source_profiles = outputs['source_profiles'][0, -1]

        for i, profile in enumerate(source_profiles):
            profile_np = profile.detach().cpu().numpy()

            # Check if this is a novel profile
            similar_profiles = self.vector_db.search_similar_profiles(
                profile_np, top_k=1, score_threshold=0.85
            )

            if len(similar_profiles) == 0:  # Novel profile
                self.vector_db.store_neural_profiles(
                    outputs,
                    data['location_id'],
                    data['timestamp'],
                    {
                        'std': uncertainty['std'][0, -1],
                        'confidence_interval': torch.quantile(
                            uncertainty['samples'][:, 0, -1],
                            torch.tensor([0.05, 0.95]), dim=0
                        )
                    }
                )


def run_physics_informed_gt_workflow(
    n_samples=500,
    n_species=20,
    n_factors=6,
    random_seed=42,
    model_kwargs=None
):
    """
    Workflow for running PhysicsInformedGraphTransformer for source apportionment.

    Returns:
        dict: {
            'W': np.ndarray,  # (n_samples, n_factors)
            'H': np.ndarray,  # (n_factors, n_species)
            'Qtrue': float,   # q_loss value
            'V': np.ndarray,  # input data
            'U': np.ndarray,  # uncertainty
            'model': PhysicsInformedGraphTransformer instance
        }
    """
    # Generate synthetic dataset
    V, U, true_k, true_H, true_W, sim = generate_dataset(
        true_k=n_factors,
        n_samples=n_samples,
        n_features=n_species,
        seed=random_seed
    )
    print(f"Generated dataset with {n_samples} samples, {n_species} species, {n_factors} true factors")
    # Initialize and fit PhysicsInformedGraphTransformer

    # Configuration
    config = PIGTConfig(
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        max_factors={'regional': 5, 'urban': 8, 'local': 12},
        physics_weight=0.1
    )

    gt_model = PhysicsInformedGraphTransformer(n_species=n_species, config=config)
    gt_model.fit(V, U)

    # Extract W and H from model output
    batch = torch.tensor(V, dtype=torch.float32).unsqueeze(0)
    coords = torch.zeros(batch.shape[1], 2)
    times = torch.arange(batch.shape[1])
    U_tensor = torch.tensor(U, dtype=torch.float32)
    outputs = gt_model.forward(batch, coords, times, U=U_tensor)
    W = outputs['factor_contributions'][0, -1].detach().cpu().numpy()  # (factors,)
    H = outputs['source_profiles'][0, -1].detach().cpu().numpy()       # (factors, species)

    # Evaluate q_loss
    Qtrue = q_loss(V=V, U=U, W=W, H=H)

    # Return results in SA-like structure
    return {
        'W': W,
        'H': H,
        'Qtrue': Qtrue,
        'V': V,
        'U': U,
        'model': gt_model
    }

# Example usage demonstrating complete integration
def main():
    """Complete workflow with SPECIATE and vector database integration"""

    # Configuration
    config = PIGTConfig(
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        max_factors={'regional': 5, 'urban': 8, 'local': 12},
        physics_weight=0.1
    )

    # Initialize components
    n_species = 25
    model = PhysicsInformedGraphTransformer(n_species, config)

    # SPECIATE integration
    species_mapping = {f'species_{i}': f'SPEC_{i:03d}' for i in range(n_species)}
    speciate = SPECIATEIntegration('path/to/speciate.db', species_mapping)

    # Vector database
    vector_db = VectorProfileDatabase(dimension=n_species)

    # Integrated system
    integrated_system = IntegratedSourceApportionment(model, speciate, vector_db)

    # Simulated EPA monitoring data
    monitoring_data = {
        'concentrations': np.random.lognormal(0, 1, (168, n_species)),  # 1 week
        'coordinates': np.random.uniform([30, -120], [45, -70], (168, 2)),
        'timestamps': np.arange(168),
        'location_id': 'EPA_SITE_001',
        'timestamp': '2024-06-01T00:00:00Z'
    }

    # Process data through complete workflow
    results = integrated_system.process_monitoring_data(monitoring_data)

    print(f"Identified {len(results['source_profiles'])} source factors")
    print(f"SPECIATE matches: {[m['source_type'] for m in results['speciate_matches']]}")
    print(f"Novel profiles stored in vector database")

    # Streaming integration
    streaming_system = StreamingSourceApportionment(model)

    # CMAQ-ISAM validation
    isam_integration = CMAQISAMIntegration(model)

    return integrated_system, streaming_system, isam_integration


if __name__ == "__main__":
    # model, streaming_system, isam_integration = main()
    results = run_physics_informed_gt_workflow()
    print(f"Qtrue: {results['Qtrue']:.4f} for synthetic dataset")
