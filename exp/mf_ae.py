import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import warnings

from utils import generate_dataset, q_loss

warnings.filterwarnings('ignore')


class GatedSAAutoencoder(nn.Module):
    """
    Autoencoder with gating mechanism for automatic factor selection ESAT/PMF for source apportionment
    """

    def __init__(self, f_features: int, max_factors: int = 15, dropout_rate: float = 0.1):
        super(GatedSAAutoencoder, self).__init__()

        self.f_features = f_features
        self.max_factors = max_factors

        # Encoder: maps observations to factor space (G matrix equivalent)
        self.encoder = nn.Sequential(
            nn.Linear(f_features, max_factors),
            nn.ReLU(),  # Ensure non-negativity like PMF
            nn.Dropout(dropout_rate)
        )

        # Gating mechanism for factor selection
        self.gate = nn.Sequential(
            nn.Linear(max_factors, max_factors),
            nn.Sigmoid()  # 0-1 weights for factor importance
        )

        # Decoder: maps factors to species (F matrix equivalent)
        self.decoder = nn.Sequential(
            nn.Linear(max_factors, f_features),
            nn.ReLU()  # Ensure non-negative outputs like PMF
        )

        # Initialize weights to be positive (PMF constraint)
        self._initialize_positive_weights()

    def _initialize_positive_weights(self):
        """Initialize weights to be positive for ESAT compatibility"""
        for layer in [self.encoder, self.decoder]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.uniform_(module.weight, 0.01, 1.0)
                    if module.bias is not None:
                        nn.init.uniform_(module.bias, 0.01, 0.1)

    def forward(self, x):
        # Encode to factor space (like G matrix in PMF)
        factors = self.encoder(x)

        # Apply gating for factor selection
        gate_weights = self.gate(factors)
        gated_factors = factors * gate_weights

        # Decode back to species space (like F matrix in PMF)
        reconstruction = self.decoder(gated_factors)

        return reconstruction, gated_factors, gate_weights


class AutoencoderESAT:
    """
    Main class for ESAT/PMF using autoencoders with automatic factor selection
    """

    def __init__(self, max_factors: int = 15, learning_rate: float = 0.001,
                 gate_sparsity: float = 0.01, uncertainty_weight: float = 1.0,
                 significance_threshold: float = 0.3):

        self.max_factors = max_factors
        self.learning_rate = learning_rate
        self.gate_sparsity = gate_sparsity
        self.uncertainty_weight = uncertainty_weight
        self.significance_threshold = significance_threshold

        self.model = None
        self.scaler = StandardScaler()
        self.training_history = []
        self.feature_names = None

    def prepare_data(self, concentrations: np.ndarray, uncertainties: np.ndarray = None,
                     feature_names: List[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare concentration and uncertainty data for PMF analysis

        Args:
            concentrations: (n_samples, n_species) concentration matrix
            uncertainties: (n_samples, n_species) uncertainty matrix
            feature_names: List of feature_names
        """

        self.feature_names = feature_names or [f"Feature_{i}" for i in range(concentrations.shape[1])]

        # Handle missing values (common in environmental data)
        concentrations = np.nan_to_num(concentrations, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure non-negative values (PMF constraint)
        concentrations = np.maximum(concentrations, 0.01)  # Small positive minimum

        # Scale data while preserving non-negativity
        concentrations_scaled = self.scaler.fit_transform(concentrations)
        concentrations_scaled = np.maximum(concentrations_scaled, 0.01)

        # Handle uncertainties
        if uncertainties is None:
            # Use Poisson uncertainty model (sqrt of concentration)
            uncertainties = np.sqrt(concentrations)

        uncertainties = np.nan_to_num(uncertainties, nan=1.0, posinf=1.0, neginf=1.0)
        uncertainties = np.maximum(uncertainties, 0.01)  # Avoid division by zero

        return (torch.FloatTensor(concentrations_scaled),
                torch.FloatTensor(uncertainties))

    def q_loss(self, reconstruction, target, uncertainties, gate_weights):
        """
        Q loss function with uncertainty weighting and gate sparsity
        """
        # Weighted reconstruction loss (EPA PMF objective)
        weighted_residuals = (reconstruction - target) / uncertainties
        reconstruction_loss = torch.mean(weighted_residuals ** 2)

        # Gate sparsity penalty (L1 regularization for factor selection)
        sparsity_loss = self.gate_sparsity * torch.mean(torch.abs(gate_weights))

        # Non-negativity penalty (soft constraint)
        nonnegativity_penalty = torch.mean(torch.relu(-reconstruction)) * 0.1

        total_loss = reconstruction_loss + sparsity_loss + nonnegativity_penalty

        return total_loss, reconstruction_loss, sparsity_loss

    def fit(self, concentrations: np.ndarray, uncertainties: np.ndarray = None,
            feature_names: List[str] = None, epochs: int = 1000,
            patience: int = 50, verbose: bool = True):
        """
        Fit the autoencoder source apportionment model
        """

        # Prepare data
        X_tensor, U_tensor = self.prepare_data(concentrations, uncertainties, feature_names)
        n_species = X_tensor.shape[1]

        # Initialize model
        self.model = GatedSAAutoencoder(n_species, self.max_factors)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

        # Training loop
        best_loss = float('inf')
        patience_counter = 0

        progress = tqdm.trange(epochs, desc="Training", disable=not verbose)
        for epoch in progress:
            self.model.train()

            # Forward pass
            reconstruction, gated_factors, gate_weights = self.model(X_tensor)

            # Calculate loss
            total_loss, recon_loss, sparsity_loss = self.q_loss(
                reconstruction, X_tensor, U_tensor, gate_weights
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step(total_loss)

            # Track training
            self.training_history.append({
                'epoch': epoch,
                'q_loss': total_loss.item(),
                'reconstruction_loss': recon_loss.item(),
                'sparsity_loss': sparsity_loss.item()
            })
            progress.set_postfix({'q_loss': total_loss.item()})

            # Early stopping
            # if total_loss.item() < best_loss:
            #     best_loss = total_loss.item()
            #     patience_counter = 0
            # else:
            #     patience_counter += 1

            # if patience_counter >= patience:
            #     if verbose:
            #         print(f"Early stopping at epoch {epoch}")
            #     break


    def get_source_profiles_and_contributions(self, concentrations: np.ndarray) -> Dict:
        """
        Extract source profiles (H matrix) and contributions (W matrix)
        """

        if self.model is None:
            raise ValueError("Model must be fitted first")

        self.model.eval()
        X_tensor, _ = self.prepare_data(concentrations, feature_names=self.feature_names)

        with torch.no_grad():
            reconstruction, gated_factors, gate_weights = self.model(X_tensor)

            # Determine active factors based on gate weights
            mean_gate_weights = torch.mean(gate_weights, dim=0)
            active_factors = (mean_gate_weights > self.significance_threshold).cpu().numpy()  # Threshold for significance
            n_active_factors = np.sum(active_factors)

            # Extract source profiles (F matrix equivalent)
            # Decoder weights represent how each factor contributes to each species
            source_profiles = self.model.decoder[0].weight.data.cpu().numpy()  # (n_species, n_factors)
            source_profiles = source_profiles[:, active_factors]  # Keep only active factors

            # Extract source contributions (G matrix equivalent)
            source_contributions = gated_factors.cpu().numpy()[:, active_factors]

            # Scale back to original units
            source_profiles_scaled = self.scaler.inverse_transform(
                np.maximum(source_profiles.T, 0)  # Ensure non-negative
            ).T

        return {
            'source_profiles': source_profiles_scaled,  # (n_species, n_factors)
            'source_contributions': source_contributions,  # (n_samples, n_factors)
            'gate_weights': mean_gate_weights.cpu().numpy(),
            'active_factors': active_factors,
            'n_active_factors': n_active_factors,
            'reconstruction': reconstruction.cpu().numpy(),
            'factor_importance': mean_gate_weights[active_factors].cpu().numpy()
        }

    def plot_results(self, results: Dict, figsize: Tuple[int, int] = (15, 10)):
        """
        Create EPA PMF-style plots for source apportionment results
        """

        fig = plt.figure(figsize=figsize)

        # 1. Source profiles (like EPA PMF F matrix)
        plt.subplot(2, 3, 1)
        source_profiles = results['source_profiles']
        im = plt.imshow(source_profiles, aspect='auto', cmap='viridis')
        plt.colorbar(im)
        plt.xlabel('Source Factor')
        plt.ylabel('Features')
        plt.title('Source Profiles (H Matrix)')
        if self.feature_names:
            plt.yticks(range(len(self.feature_names)), self.feature_names, fontsize=8)

        # 2. Gate weights (factor importance)
        plt.subplot(2, 3, 2)
        gate_weights = results['gate_weights']
        plt.bar(range(len(gate_weights)), gate_weights)
        plt.axhline(y=self.significance_threshold, color='r', linestyle='--', label='Significance threshold')
        plt.xlabel('Factor Index')
        plt.ylabel('Gate Weight')
        plt.title('Factor Importance (Gate Weights)')
        plt.legend()

        # 3. Training history
        plt.subplot(2, 3, 3)
        history_df = pd.DataFrame(self.training_history)
        plt.plot(history_df['epoch'], history_df['q_loss'], label='Q Loss')
        plt.plot(history_df['epoch'], history_df['reconstruction_loss'], label='Reconstruction')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.yscale('log')

        # 4. Source contributions time series
        plt.subplot(2, 3, 4)
        contributions = results['source_contributions']
        for i in range(contributions.shape[1]):
            plt.plot(contributions[:, i], label=f'Source {i + 1}')
        plt.xlabel('Sample Index')
        plt.ylabel('Contribution')
        plt.title('Source Contributions (W Matrix)')
        plt.legend()

        # 5. Reconstruction quality
        plt.subplot(2, 3, 5)
        reconstruction = results['reconstruction']
        # Show first few species as example
        n_show = min(3, len(self.feature_names))
        for i in range(n_show):
            original = self.scaler.inverse_transform(
                np.maximum(reconstruction, 0)
            )[:, i]  # This is simplified - you'd want the original data
            plt.plot(original, alpha=0.7, label=f'{self.feature_names[i]} (Reconstructed)')
        plt.xlabel('Sample Index')
        plt.ylabel('Concentration')
        plt.title('Reconstruction Quality')
        plt.legend()

        # 6. Factor loadings heatmap
        plt.subplot(2, 3, 6)
        active_profiles = source_profiles
        sns.heatmap(active_profiles, annot=False, cmap='viridis',
                    yticklabels=self.feature_names if self.feature_names else False)
        plt.xlabel('Active Source Factor')
        plt.ylabel('Chemical Species')
        plt.title('Active Source Profiles Heatmap')

        plt.tight_layout()
        plt.show()

        # Print summary
        print(f"\nSource Apportionment Summary:")
        print(f"Number of active factors: {results['n_active_factors']}")
        print(f"Factor importance scores: {results['factor_importance']}")


# Example usage and demonstration
def run_example_sa():
    """
    Example usage with synthetic environmental data
    """

    # Generate synthetic
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    n_true_sources = 6

    # Species names (typical PM2.5 components)
    feature_names = ['SO4', 'NO3', 'NH4', 'OC', 'EC', 'Si', 'Fe', 'Zn']

    V, U, true_k = generate_dataset(true_k=n_true_sources, n_samples=n_samples, n_features=n_features)

    print("Running Autoencoder ESAT for Source Apportionment...")
    print(f"Data shape: {V.shape}")
    print(f"Features: {feature_names}")

    # Initialize and fit the model
    ae_model = AutoencoderESAT(
        max_factors=12,  # Start with more factors than expected
        learning_rate=0.001,
        gate_sparsity=0.1,  # Encourage factor selection
        significance_threshold=0.2
    )

    # Fit the model
    ae_model.fit(
        concentrations=V,
        uncertainties=U,
        feature_names=feature_names,
        epochs=1000,
        verbose=True
    )

    # Get results
    results = ae_model.get_source_profiles_and_contributions(V)
    print(f"True Sources: {true_k} - Estimated Active Factors: {results['n_active_factors']}")
    W = results['source_contributions']
    H = results['source_profiles'].T
    # print(f"W: {W.shape}, H: {H.shape}")
    qtrue = q_loss(V, U, W, H)
    print(f"Q Loss (EPA PMF Objective): {float(qtrue):.4f}")

    # Plot results
    ae_model.plot_results(results)

    return ae_model, results, V


# Run the example
if __name__ == "__main__":
    model, results, data = run_example_sa()


