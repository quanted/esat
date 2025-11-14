import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from torch_geometric.data import Data
from utils import generate_dataset, q_loss
from esat.model.sa import SA
from esat.model.batch_sa import BatchSA
import numpy as np
import plotly.io as pio
from sklearn.decomposition import NMF
import torch_geometric.nn as pyg_nn

pio.renderers.default = "browser"



class _EncoderStack(nn.Module):
    def __init__(self, layers, gnn_type, dropout_rate):
        super().__init__()
        self.layers = layers
        self.gnn_type = gnn_type
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, edge_weight=None):
        x_current = x
        if self.gnn_type == 'GCN2Conv':
            # For GCN2Conv, x should already be projected to the correct dimension
            # and we use the projected x as both x_current and x_0
            x_0 = x_current  # Store the initial projected representation
            for layer in self.layers:
                x_current = layer(x_current, x_0, edge_index, edge_weight)
                x_current = F.relu(x_current)
                x_current = F.dropout(x_current, p=self.dropout_rate, training=self.training)
        else:
            for layer in self.layers:
                x_current = layer(x_current, edge_index, edge_weight)
                x_current = F.relu(x_current)
                x_current = F.dropout(x_current, p=self.dropout_rate, training=self.training)
        return x_current


class GNNSA(nn.Module):
    """
    Graph Neural Network for Source Apportionment (SA) using torch_geometric.
    Supports multiple GNN types via the gnn_type parameter and graph auto-encoder models via gae_type.
    """
    GNN_TYPE_MAP = {
        'GCNConv': {'class': pyg_nn.GCNConv, 'args': {}},
        'GATConv': {'class': pyg_nn.GATConv, 'args': {'heads': 1}},
        'SAGEConv': {'class': pyg_nn.SAGEConv, 'args': {}},
        'TAGConv': {'class': pyg_nn.TAGConv, 'args': {}},
        'SGConv': {'class': pyg_nn.SGConv, 'args': {}},
        'GCN2Conv': {'class': pyg_nn.GCN2Conv, 'args': {'alpha': 0.1, 'theta': 0.5, 'shared_weights': False, 'normalize': False}},
    }
    GAE_TYPE_MAP = ['none', 'GAE', 'VGAE', 'ARGVA']

    def __init__(self, num_features, factors, dropout_rate=0.0, reg_lambda=0.0, l1_lambda=0.0, loss_type='mse', huber_delta=1.0, min_uncertainty=1e-3, init_H='random', activation='softplus', V_for_init=None, gnn_type='GCNConv', num_gnn_layers=2, gae_type='none'):
        super(GNNSA, self).__init__()
        self.factors = factors
        self.num_features = num_features
        self.dropout_rate = dropout_rate
        self.reg_lambda = reg_lambda
        self.l1_lambda = l1_lambda
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.min_uncertainty = min_uncertainty
        self.activation = activation
        self.gnn_type = gnn_type
        self.num_gnn_layers = num_gnn_layers
        self.gae_type = gae_type
        hidden_dim = 128
        if gae_type not in self.GAE_TYPE_MAP:
            raise ValueError(f"Unsupported gae_type: {gae_type}. Supported types: {self.GAE_TYPE_MAP}")
        # Project input to hidden_dim for GCN2Conv
        if gnn_type == 'GCN2Conv':
            self.input_proj = nn.Linear(self.num_features + 1, hidden_dim)
        # Build encoder
        encoder_layers = self._build_encoder(hidden_dim)
        self.encoder = _EncoderStack(encoder_layers, self.gnn_type, self.dropout_rate)

        # H initialization and output layers (needed for both GAE and non-GAE modes)
        if init_H == 'nmf' and V_for_init is not None:
            nmf = NMF(n_components=factors, init='random', random_state=0)
            W_nmf = nmf.fit_transform(V_for_init)
            H_nmf = nmf.components_
            self.H = nn.Parameter(torch.tensor(H_nmf, dtype=torch.float32))
        else:
            self.H = nn.Parameter(torch.rand(factors, num_features))
        self.fc_w = nn.Linear(self._get_encoder_out_dim(hidden_dim), factors)

        # For GAE/VGAE/ARGVA, build the autoencoder wrapper (for auxiliary graph reconstruction loss)
        if gae_type == 'GAE':
            self.autoencoder = pyg_nn.GAE(self.encoder)
        elif gae_type == 'VGAE':
            self.autoencoder = pyg_nn.VGAE(self.encoder)
        elif gae_type == 'ARGVA':
            self.autoencoder = pyg_nn.ARGVA(self.encoder, discriminator=nn.Sequential(nn.Linear(factors, 32), nn.ReLU(), nn.Linear(32, 1)))

    def _build_encoder(self, hidden_dim):
        gnn_info = self.GNN_TYPE_MAP[self.gnn_type]
        gnn_class = gnn_info['class']
        gnn_args = gnn_info['args']
        layers = []
        if self.gnn_type == 'GCN2Conv':
            input_dim = self.num_features + 1
            for i in range(self.num_gnn_layers):
                out_dim = hidden_dim if i < self.num_gnn_layers - 1 else self.factors
                layers.append(gnn_class(input_dim, out_dim, **gnn_args))
                input_dim = out_dim if self.gnn_type != 'GATConv' else out_dim * gnn_args.get('heads', 1)
        else:
            input_dim = self.num_features + 1
            for i in range(self.num_gnn_layers):
                out_dim = hidden_dim if i < self.num_gnn_layers - 1 else self.factors
                if self.gnn_type == 'GATConv':
                    layers.append(gnn_class(input_dim, out_dim, **gnn_args))
                    input_dim = out_dim * gnn_args['heads']
                else:
                    layers.append(gnn_class(input_dim, out_dim, **gnn_args))
                    input_dim = out_dim
        return nn.ModuleList(layers)

    def _get_encoder_out_dim(self, hidden_dim):
        if self.gnn_type == 'GCN2Conv':
            return hidden_dim
        gnn_info = self.GNN_TYPE_MAP[self.gnn_type]
        gnn_args = gnn_info['args']
        input_dim = self.num_features + 1
        for i in range(self.num_gnn_layers):
            out_dim = hidden_dim if i < self.num_gnn_layers - 1 else self.factors
            if self.gnn_type == 'GATConv':
                input_dim = out_dim * gnn_args['heads']
            else:
                input_dim = out_dim
        return input_dim

    def _encode(self, x, edge_index, edge_weight):
        # Apply input projection for all GNN types to ensure consistent input dimensions
        # if hasattr(self, 'input_proj'):
        #     x = self.input_proj(x)

        # Process input through GNN layers
        if self.gnn_type == 'GCN2Conv':
            x_proj = self.input_proj(x)
            x_current = x_proj
            for i, gnn in enumerate(self.encoder.layers):
                x_0 = x_proj
                x_current = gnn(x_current, x_0, edge_index, edge_weight)
                x_current = F.relu(x_current)
                x_current = F.dropout(x_current, p=self.dropout_rate, training=self.training)
            return x_current
        else:
            x_current = x
            for i, gnn in enumerate(self.encoder.layers):
                x_current = gnn(x_current, edge_index, edge_weight)
                x_current = F.relu(x_current)
                x_current = F.dropout(x_current, p=self.dropout_rate, training=self.training)
            return x_current

    def forward(self, x, edge_index, edge_weight):
        # Always encode the input to get node embeddings
        x_enc = self._encode(x, edge_index, edge_weight)

        # Generate W and H matrices for source apportionment
        if self.activation == 'softplus':
            W = F.softplus(self.fc_w(x_enc))
            H = F.softplus(self.H)
        elif self.activation == 'relu':
            W = F.relu(self.fc_w(x_enc))
            H = F.relu(self.H)
        else:
            W = self.fc_w(x_enc)
            H = self.H

        if self.gae_type == 'none':
            return H, W
        elif self.gae_type in ['GAE', 'VGAE', 'ARGVA']:
            # For GAE models, use the same encoded representation for consistency
            # The autoencoder should use the same encoder we just used
            z = x_enc  # Use the already computed encoding

            # For GAE decode, we need to handle different output dimensions
            if hasattr(self.autoencoder, 'decode'):
                # Standard GAE decode expects edge_index for efficiency
                recon_adj = self.autoencoder.decode(z, edge_index)
            else:
                # Fallback: compute pairwise similarities
                recon_adj = torch.matmul(z, z.t())

            # Return both source apportionment matrices and graph reconstruction
            return H, W, z, recon_adj
        else:
            raise ValueError(f"Unsupported gae_type: {self.gae_type}")

    def fit(self, data, epochs=1000, lr=1e-3, verbose=True):
        if self.gae_type == 'none':
            return self._fit_gnn(data, epochs, lr, verbose)
        else:
            return self._fit_gae(data, epochs, lr, verbose)

    def _fit_gnn(self, data, epochs, lr, verbose):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        t = trange(epochs, desc=f'Training GNNSA')
        for epoch in t:
            optimizer.zero_grad()
            H, W = self.forward(data.x, data.edge_index, data.edge_weight)
            loss = self.loss(data.x[:, :-1], data.x[:, -1:], H, W)
            loss.backward()
            optimizer.step()
            t.set_description(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
        return H.detach(), W.detach()

    def _fit_gae(self, data, epochs, lr, verbose):
        # Use different optimizers for different components to improve stability
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.8)

        best_loss = float('inf')
        patience_counter = 0
        max_patience = 100

        t = trange(epochs, desc=f'Training GNNSA-AE')
        for epoch in t:
            optimizer.zero_grad()

            # Forward pass
            H, W, z, recon_adj = self.forward(data.x, data.edge_index, data.edge_weight)

            # Primary loss: Source apportionment reconstruction
            sa_loss = self.loss(data.x[:, :-1], data.x[:, -1:], H, W)

            # Auxiliary loss: Graph reconstruction (GAE loss)
            pos_edge_index = data.edge_index

            # Use proper GAE loss computation
            if hasattr(self.autoencoder, 'recon_loss'):
                # Use built-in reconstruction loss from torch_geometric
                gae_loss = self.autoencoder.recon_loss(z, pos_edge_index)

                # Add KL divergence for VGAE
                if self.gae_type == 'VGAE':
                    # Explicitly execute the VGAE forward pass to set internal attributes
                    self.autoencoder(z, pos_edge_index)  # Forward pass to set __mu__

                    kl_loss = (1 / data.x.size(0)) * self.autoencoder.kl_loss()
                    gae_loss = gae_loss + kl_loss

            else:
                # Efficient edge-based BCE loss instead of full adjacency matrix
                num_nodes = data.x.size(0)

                # Positive edges (existing edges)
                pos_score = torch.sigmoid(torch.sum(z[pos_edge_index[0]] * z[pos_edge_index[1]], dim=1))

                # Negative edges (random sampling for efficiency)
                neg_edge_index = torch.randint(0, num_nodes, pos_edge_index.size(), device=data.x.device)
                neg_score = torch.sigmoid(torch.sum(z[neg_edge_index[0]] * z[neg_edge_index[1]], dim=1))

                # BCE loss on edges only
                pos_loss = F.binary_cross_entropy(pos_score, torch.ones_like(pos_score))
                neg_loss = F.binary_cross_entropy(neg_score, torch.zeros_like(neg_score))
                gae_loss = pos_loss + neg_loss

            # Adaptive loss weighting based on relative magnitudes
            sa_magnitude = sa_loss.item()
            gae_magnitude = gae_loss.item()

            # Dynamic weighting to balance losses
            if gae_magnitude > 0:
                # gae_weight = min(0.5, sa_magnitude / gae_magnitude * 0.1)
                gae_weight = sa_magnitude / gae_magnitude
            else:
                gae_weight = 0.1

            # Combine losses with adaptive weighting
            total_loss = sa_loss + gae_weight * gae_loss

            # Backward pass with gradient clipping for stability
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            # Learning rate scheduling
            scheduler.step(total_loss)

            # Early stopping
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

            if verbose:
                t.set_description(f"Epoch {epoch+1}/{epochs} | Total: {total_loss.item():.4f} | SA: {sa_loss.item():.4f} | GAE: {gae_loss.item():.4f} | Weight: {gae_weight:.3f}")

        return H.detach(), W.detach()

    def loss(self, V, U, H, W):
        recon = torch.matmul(W, H)
        U_safe = torch.clamp(U, min=self.min_uncertainty)
        if U_safe.shape[1] == 1:
            U_safe = U_safe.expand_as(V)
        if self.loss_type == 'mse':
            weighted_loss = torch.mean(((V - recon)/ (U_safe + 1e-8)) ** 2)
        elif self.loss_type == 'huber':
            delta = self.huber_delta
            diff = (V - recon) / (U_safe + 1e-8)
            weighted_loss = torch.mean(torch.where(torch.abs(diff) < delta,
                                                   0.5 * diff ** 2,
                                                   delta * (torch.abs(diff) - 0.5 * delta)))
        else:
            raise ValueError("Invalid loss type. Choose 'mse' or 'huber'.")
        l2_reg = self.reg_lambda * (torch.norm(H, p=2) + torch.norm(W, p=2))
        l1_reg = self.l1_lambda * torch.norm(W, p=1)
        return weighted_loss + l2_reg + l1_reg

    def qtrue(self, V, H, W):
        recon = torch.matmul(W, H)
        qtrue = torch.sum((V - recon) ** 2)
        return qtrue


def bootstrap_gnnsa(V, U, true_factors, simulator, n_bootstrap=5, sample_frac=0.8, **model_kwargs):
    num_nodes = V.shape[0]
    results = []
    for b in range(n_bootstrap):
        idx = np.random.choice(num_nodes, int(num_nodes * sample_frac), replace=True)
        V_boot = V[idx]
        U_boot = U[idx]
        U_feat = U_boot.mean(axis=1, keepdims=True)
        node_features = np.concatenate([V_boot, U_feat], axis=1)
        x = torch.tensor(node_features, dtype=torch.float32)
        from sklearn.neighbors import NearestNeighbors
        k = true_factors
        nbrs = NearestNeighbors(n_neighbors=k*2, algorithm='auto').fit(V_boot)
        distances, indices = nbrs.kneighbors(V_boot)
        edge_index_list = []
        for i in range(len(idx)):
            for j in indices[i][1:]:
                edge_index_list.append([i, j])
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        U_tensor = torch.tensor(U_feat.squeeze(), dtype=torch.float32)
        edge_weight = 1.0 / (((U_tensor[edge_index[0]] + U_tensor[edge_index[1]]) / 2) ** 2 + 1e-8)
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
        model = GNNSA(num_features=V.shape[1], factors=true_factors, V_for_init=V_boot, **model_kwargs)
        H_pred, W_pred = model.fit(data, epochs=500, lr=1e-3, verbose=False)
        qtrue = q_loss(V=V_boot, U=U_boot, W=W_pred.detach().cpu().numpy(), H=H_pred.detach().cpu().numpy())
        results.append({'qtrue': qtrue, 'H': H_pred.detach().cpu().numpy(), 'W': W_pred.detach().cpu().numpy()})
        print(f"Bootstrap {b+1}/{n_bootstrap}: Qtrue={qtrue:.4f}")
    qtrue_vals = [r['qtrue'] for r in results]
    print(f"Bootstrap Qtrue mean: {np.mean(qtrue_vals):.4f}, std: {np.std(qtrue_vals):.4f}")
    return results


if __name__ == "__main__":
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Generate synthetic dataset
    seed = np.random.randint(0, 10000)
    V, U, true_factors, true_H, true_W, simulator = generate_dataset(seed=seed)
    print(f"Generated dataset with {V.shape[0]} samples, {V.shape[1]} features, {true_factors} true factors.")
    num_nodes, num_features = V.shape
    # Node features: concatenate V and U (uncertainty as extra feature)
    U_feat = U.mean(axis=1, keepdims=True)  # [num_nodes, 1] (mean uncertainty per node)
    node_features = np.concatenate([V, U_feat], axis=1)
    x = torch.tensor(node_features, dtype=torch.float32).to(device)

    # Build a k-nearest neighbors (KNN) graph instead of fully connected
    from sklearn.neighbors import NearestNeighbors
    k = true_factors  # Number of neighbors
    nbrs = NearestNeighbors(n_neighbors=k+2, algorithm='auto').fit(V)
    distances, indices = nbrs.kneighbors(V)
    # Build edge_index
    edge_index_list = []
    for i in range(num_nodes):
        for j in indices[i][1:]:  # skip self-loop
            edge_index_list.append([i, j])
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(device)
    # Compute edge weights: weights = 1/(U**2) using mean uncertainty of each node
    U_tensor = torch.tensor(U_feat.squeeze(), dtype=torch.float32).to(device)  # [num_nodes]
    edge_weight = 1.0 / (((U_tensor[edge_index[0]] + U_tensor[edge_index[1]]) / 2) ** 2 + 1e-8)
    edge_weight = edge_weight.to(device)
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    # Initialize GNNSA
    # "GCNConv"(Graph Convolutional Network)
    # "GATConv"(Graph Attention Network)
    # "SAGEConv"(SAGEConv in torch_geometric)
    # "TAGConv"(Topology Adaptive GCN)
    # "SGConv"(Simple Graph Convolution)

    # FLEXIBILITY OPTIONS FOR W AND H MATRICES:
    # 1. Reduce regularization: Lower reg_lambda and l1_lambda for more freedom
    # 2. Change activation: Use 'relu' instead of 'softplus' for less smoothing
    # 3. Reduce dropout: Lower dropout_rate for less constraint during training
    # 4. Increase model capacity: More GNN layers or higher learning rate

    model = GNNSA(
        num_features=num_features,
        factors=true_factors,
        dropout_rate=0.0,
        reg_lambda=0.0,
        l1_lambda=0.0,
        loss_type='huber',
        huber_delta=0.1,
        min_uncertainty=1e-4,
        init_H='nmf',
        activation='relu',  # Changed from 'softplus' - less smoothing, more flexibility
        V_for_init=V,
        gnn_type='GCNConv',
        gae_type='ARGVA',     # Options are: 'GAE', 'VGAE', 'ARGVA'
        num_gnn_layers=3
    ).to(device)  # Increased layers for more capacity

    # Train model
    H_pred, W_pred = model.fit(data, epochs=2000, lr=1e-2, verbose=True)
    qtrue = q_loss(V=V, U=U, W=W_pred.cpu().numpy(), H=H_pred.cpu().numpy())
    print(f"Final Qtrue: {qtrue:.4f}")
    # Compare true and predicted H
    sa = SA(V=V, U=U, factors=true_factors)
    sa.initialize()
    sa.H = H_pred.cpu().numpy()
    sa.W = W_pred.cpu().numpy()
    sa.WH = np.matmul(sa.W, sa.H)

    batch_sa = BatchSA(V=V, U=U, factors=true_factors, models=1)
    batch_sa.results.append(sa)

    simulator.compare(batch_sa)
    simulator.plot_profile_comparison(model_i=0)

    # Bootstrapping stability assessment
    # print("Running bootstrapping for stability assessment...")
    # bootstrap_gnnsa(V, U, true_factors, simulator, n_bootstrap=5, sample_frac=0.8,
    #                dropout_rate=0.2, reg_lambda=1e-6, l1_lambda=1e-6, loss_type='huber', huber_delta=1.0,
    #                min_uncertainty=1e-3, init_H='nmf', activation='relu', gnn_type='GCNConv', num_gnn_layers=2)
