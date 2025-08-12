import os
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from nnea.model.base import BaseModel
import warnings
warnings.filterwarnings('ignore')
import random
import logging

class NeuralUMAP(nn.Module):
    """
    Neural network-based UMAP implementation
    
    This implementation uses an encoder-decoder architecture to learn mappings from high-dimensional data to low-dimensional space,
    while preserving both local and global structure of the data.
    """
    
    def __init__(self, input_dim: int, embedding_dim: int = 2, 
                 hidden_dims: List[int] = [128, 64, 32], 
                 dropout: float = 0.1):
        """
        Initialize neural network UMAP model
        
        Args:
            input_dim: Input data dimension
            embedding_dim: Embedding space dimension (usually 2 for visualization)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout ratio
        """
        super(NeuralUMAP, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        
        # Encoder: high-dimensional -> low-dimensional
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        encoder_layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder: low-dimensional -> high-dimensional (optional, for reconstruction)
        decoder_layers = []
        prev_dim = embedding_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Reconstruction output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """Forward propagation"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def encode(self, x):
        """Encode only"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode only"""
        return self.decoder(z)


class UMAPLoss(nn.Module):
    """
    UMAP loss function implementation, using PCA to find nearest neighbor pairs (optimized version)
    
    Based on nn_umap.py implementation, with caching mechanism to improve training efficiency
    """
    
    def __init__(self, min_dist: float = 0.1, a: float = 1.0, b: float = 1.0, 
                 n_neighbors: int = 15, pca_components: int = 50, use_vectorized: bool = True, debug: bool = False):
        """
        Initialize UMAP loss function
        
        Args:
            min_dist: Minimum distance parameter
            a, b: UMAP a and b parameters
            n_neighbors: Number of neighbors
            pca_components: Number of PCA components
            use_vectorized: Whether to use vectorized implementation (more efficient)
            debug: Whether to enable debug mode
        """
        super(UMAPLoss, self).__init__()
        self.min_dist = min_dist
        self.a = a
        self.b = b
        self.n_neighbors = n_neighbors
        self.pca_components = pca_components
        self.use_vectorized = use_vectorized
        self.debug = debug
        self.pca = None
        self.original_data = None
        # Add cache variables
        self.positive_pairs = None
        self.negative_pairs = None
        self.is_fitted = False
        # Add global distance matrix cache
        self.global_distances = None
        # Add loss statistics
        self.loss_stats = {}
        # Add logger
        import logging
        self.logger = logging.getLogger(__name__)
        
    def set_umap_params(self, a: float = None, b: float = None, min_dist: float = None):
        """
        Set UMAP parameters
        
        Args:
            a: UMAP a parameter
            b: UMAP b parameter
            min_dist: Minimum distance parameter
        """
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        if min_dist is not None:
            self.min_dist = min_dist
            
        self.logger.info(f"UMAP parameters updated: a={self.a}, b={self.b}, min_dist={self.min_dist}")
        
    def get_loss_stats(self):
        """
        Get loss function statistics
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'a': self.a,
            'b': self.b,
            'min_dist': self.min_dist,
            'n_neighbors': self.n_neighbors,
            'pca_components': self.pca_components,
            'is_fitted': self.is_fitted,
            'has_global_distances': hasattr(self, 'global_distances') and self.global_distances is not None
        }
        
        if self.is_fitted:
            stats.update({
                'nbr_indices_shape': self.nbr_indices.shape if self.nbr_indices is not None else None,
                'original_data_shape': self.original_data.shape if self.original_data is not None else None,
                'global_distances_shape': self.global_distances.shape if self.global_distances is not None else None
            })
        
        return stats
        
    def fit_pca(self, X: np.ndarray, nadata=None):
        """
        Fit data using PCA and return positive and negative sample indices (computed only once)
        
        Args:
            X: Original high-dimensional data
            nadata: nadata object, if provided and contains pre-computed PCA data, use directly
            
        Returns:
            tuple: (pos_indices, neg_indices) Positive and negative sample index arrays
        """
        if self.is_fitted:
            return self.pos_indices, self.neg_indices
            
        # Check if pre-computed PCA data can be read from nadata.uns
        X_pca = None
        
        if nadata is not None and hasattr(nadata, 'uns'):
            # Check if there is pre-computed PCA data
            if 'pca' in nadata.uns:
                X_pca = nadata.uns['pca']
                self.logger.info("Reading pre-computed PCA data from nadata.uns")
                
                # Ensure PCA data shape is correct
                if X_pca.shape[0] != X.shape[0]:
                    self.logger.warning(f"PCA data sample count ({X_pca.shape[0]}) does not match input data sample count ({X.shape[0]}), recalculating")
                    X_pca = None
            else:
                self.logger.info("Pre-computed PCA data not found in nadata.uns")
        else:
            self.logger.info("nadata object is empty or has no uns attribute")
            
        # If no pre-computed PCA data, recalculate
        if X_pca is None:
            # Limit PCA component count to not exceed feature count
            n_components = min(self.pca_components, X.shape[1])
            self.pca = PCA(n_components=n_components)
            self.pca.fit(X)
            
            # Use PCA for dimensionality reduction
            X_pca = self.pca.transform(X)
            self.logger.info(f"Recalculating PCA, component count: {n_components}")
        else:
            self.logger.info(f"Using pre-computed PCA data, shape: {X_pca.shape}")
            # Create a virtual PCA object for compatibility
            self.pca = None
        
        # Use sklearn's NearestNeighbors to find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1, algorithm='auto').fit(X_pca)
        distances, nbr_indices = nbrs.kneighbors(X_pca)
        
        # Calculate and cache global distance matrix (for negative sample selection)
        self.global_distances = distances
        
        # Generate positive and negative sample indices
        pos_indices, neg_indices = self._generate_pos_neg_pairs(nbr_indices, X.shape[0])
        
        # Store results in nadata.uns
        if nadata is not None:
            if not hasattr(nadata, 'uns'):
                nadata.uns = {}
            nadata.uns['pca'] = X_pca
            nadata.uns['nbr_indices'] = nbr_indices
            nadata.uns['pos_indices'] = pos_indices
            nadata.uns['neg_indices'] = neg_indices
            nadata.uns['global_distances'] = self.global_distances
            self.logger.info("PCA, pos_indices, neg_indices and global_distances data have been stored in nadata.uns")
        
        # Cache results
        self.nbr_indices = nbr_indices
        self.pos_indices = pos_indices
        self.neg_indices = neg_indices
        self.original_data = X
        self.is_fitted = True
        
        return pos_indices, neg_indices
    
    def find_neighbors_pca(self, X: np.ndarray, nadata=None) -> np.ndarray:
        """
        Get cached neighbor indices (calculate first if not cached)
        
        Args:
            X: Original high-dimensional data
            nadata: nadata object, optional
            
        Returns:
            nbr_indices: Neighbor index array
        """
        if not self.is_fitted:
            self.fit_pca(X, nadata)
        
        return self.nbr_indices
    
    def _generate_pos_neg_pairs(self, nbr_indices: np.ndarray, n_samples: int):
        """
        Generate positive and negative sample pairs
        
        Args:
            nbr_indices: Neighbor index array, shape (n_samples, n_neighbors+1)
            n_samples: Number of samples
            
        Returns:
            tuple: (pos_indices, neg_indices) Positive and negative sample indices
        """
        # Positive samples: each sample with its neighbors (excluding self)
        pos_pairs = []
        for i in range(n_samples):
            # Get current sample's neighbors (excluding self)
            neighbors = nbr_indices[i][1:]  # Exclude the first one (self)
            for neighbor in neighbors:
                pos_pairs.append([i, neighbor])
        
        pos_indices = np.array(pos_pairs)
        
        # Negative samples: randomly select non-neighbor sample pairs
        neg_pairs = []
        n_neg_per_sample = min(self.n_neighbors, n_samples - self.n_neighbors - 1)  # Ensure not exceeding available sample count
        
        for i in range(n_samples):
            # Get current sample's neighbors
            neighbors = set(nbr_indices[i])
            
            # Randomly select non-neighbor samples as negative samples
            non_neighbors = [j for j in range(n_samples) if j not in neighbors and j != i]
            
            if len(non_neighbors) > 0:
                # Randomly select negative samples
                n_neg = min(n_neg_per_sample, len(non_neighbors))
                selected_neg = np.random.choice(non_neighbors, size=n_neg, replace=False)
                
                for neg_idx in selected_neg:
                    neg_pairs.append([i, neg_idx])
        
        neg_indices = np.array(neg_pairs) if neg_pairs else np.empty((0, 2), dtype=int)
        
        self.logger.info(f"Generated sample pairs: {len(pos_indices)} positive pairs, {len(neg_indices)} negative pairs")
        
        return pos_indices, neg_indices

    def forward(self, embeddings, original_data=None, nadata=None, batch_pos_indices=None, batch_neg_indices=None):
        """
        Calculate UMAP loss
        
        Args:
            embeddings: Embedding vectors, shape (batch_size, embedding_dim)
            original_data: Original data, used for reconstruction loss calculation
            nadata: nadata object
            batch_pos_indices: Batch positive sample indices, shape (batch_size, n_pos_pairs, 2)
            batch_neg_indices: Batch negative sample indices, shape (batch_size, n_neg_pairs, 2)
            
        Returns:
            total_loss: Total loss
        """
        if batch_pos_indices is None or batch_neg_indices is None:
            self.logger.warning("No positive or negative sample indices provided, returning zero loss")
            return torch.tensor(0.0, device=embeddings.device)
        
        # Validate input shapes
        batch_size = embeddings.shape[0]
        if batch_pos_indices.shape[0] != batch_size or batch_neg_indices.shape[0] != batch_size:
            self.logger.error(f"Batch size mismatch: embeddings={batch_size}, pos_indices={batch_pos_indices.shape[0]}, neg_indices={batch_neg_indices.shape[0]}")
            return torch.tensor(0.0, device=embeddings.device)
        
        # Ensure indices are torch tensors and moved to correct device
        if not torch.is_tensor(batch_pos_indices):
            batch_pos_indices = torch.tensor(batch_pos_indices, dtype=torch.long, device=embeddings.device)
        elif batch_pos_indices.device != embeddings.device:
            batch_pos_indices = batch_pos_indices.to(embeddings.device)
            
        if not torch.is_tensor(batch_neg_indices):
            batch_neg_indices = torch.tensor(batch_neg_indices, dtype=torch.long, device=embeddings.device)
        elif batch_neg_indices.device != embeddings.device:
            batch_neg_indices = batch_neg_indices.to(embeddings.device)
        
        # Choose between vectorized implementation or loop implementation
        if self.use_vectorized:
            # Use vectorized implementation (more efficient)
            pos_loss, neg_loss = self._compute_loss_vectorized(embeddings, batch_pos_indices, batch_neg_indices)
        else:
            # Use loop implementation (more intuitive)
            pos_loss = self._compute_positive_loss(embeddings, batch_pos_indices)
            neg_loss = self._compute_negative_loss(embeddings, batch_neg_indices)
        
        # Total loss
        total_loss = pos_loss + neg_loss
        
        # Record loss statistics
        if hasattr(self, 'loss_stats'):
            self.loss_stats['pos_loss'] = pos_loss.item()
            self.loss_stats['neg_loss'] = neg_loss.item()
            self.loss_stats['total_loss'] = total_loss.item()
        
        # Debug information (only shown when needed)
        if self.debug:
            self.logger.info(f"UMAP loss - Positive: {pos_loss.item():.6f}, Negative: {neg_loss.item():.6f}, Total: {total_loss.item():.6f}")
        
        return total_loss
    
    def _compute_positive_loss(self, embeddings, pos_indices):
        """
        Calculate positive sample loss
        
        Args:
            embeddings: Embedding vectors, shape (batch_size, embedding_dim)
            pos_indices: Positive sample index pairs, shape (batch_size, n_pos_pairs, 2)
            
        Returns:
            pos_loss: Positive sample loss
        """
        if pos_indices.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Process batch data
        batch_size = embeddings.shape[0]
        pos_pairs = []
        
        # Iterate through positive sample pairs for each sample
        for i in range(batch_size):
            sample_pos_pairs = pos_indices[i]  # Shape (n_pos_pairs, 2)
            for pair in sample_pos_pairs:
                idx1, idx2 = pair[0].item(), pair[1].item()
                # Ensure indices are within batch range
                if 0 <= idx1 < batch_size and 0 <= idx2 < batch_size:
                    pos_pairs.append([embeddings[idx1], embeddings[idx2]])
        
        if not pos_pairs:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Use more efficient tensor stacking
        pos_pairs = torch.stack([torch.stack(pair) for pair in pos_pairs])
        
        # Calculate distances between positive sample pairs
        pos_distances = torch.norm(pos_pairs[:, 0] - pos_pairs[:, 1], dim=1)
        
        # UMAP positive sample loss: use cross-entropy loss
        # Goal: positive sample pairs should be close
        pos_targets = torch.ones(len(pos_distances), device=embeddings.device)
        
        # Use sigmoid to convert distance to probability
        pos_probs = torch.sigmoid(-pos_distances / self.min_dist)
        pos_loss = F.binary_cross_entropy(pos_probs, pos_targets)
        
        return pos_loss
    
    def _compute_negative_loss(self, embeddings, neg_indices):
        """
        Calculate negative sample loss
        
        Args:
            embeddings: Embedding vectors, shape (batch_size, embedding_dim)
            neg_indices: Negative sample index pairs, shape (batch_size, n_neg_pairs, 2)
            
        Returns:
            neg_loss: Negative sample loss
        """
        if neg_indices.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Process batch data
        batch_size = embeddings.shape[0]
        neg_pairs = []
        
        # Iterate through negative sample pairs for each sample
        for i in range(batch_size):
            sample_neg_pairs = neg_indices[i]  # Shape (n_neg_pairs, 2)
            for pair in sample_neg_pairs:
                idx1, idx2 = pair[0].item(), pair[1].item()
                # Ensure indices are within batch range
                if 0 <= idx1 < batch_size and 0 <= idx2 < batch_size:
                    neg_pairs.append([embeddings[idx1], embeddings[idx2]])
        
        if not neg_pairs:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Use more efficient tensor stacking
        neg_pairs = torch.stack([torch.stack(pair) for pair in neg_pairs])
        
        # Calculate distances between negative sample pairs
        neg_distances = torch.norm(neg_pairs[:, 0] - neg_pairs[:, 1], dim=1)
        
        # UMAP negative sample loss: use cross-entropy loss
        # Goal: negative sample pairs should be far apart
        neg_targets = torch.zeros(len(neg_distances), device=embeddings.device)
        
        # Use sigmoid to convert distance to probability
        neg_probs = torch.sigmoid(-neg_distances / self.min_dist)
        neg_loss = F.binary_cross_entropy(neg_probs, neg_targets)
        
        return neg_loss
    
    def _compute_loss_vectorized(self, embeddings, pos_indices, neg_indices):
        """
        Vectorized UMAP loss calculation (more efficient implementation)
        
        Args:
            embeddings: Embedding vectors, shape (batch_size, embedding_dim)
            pos_indices: Positive sample index pairs, shape (batch_size, n_pos_pairs, 2)
            neg_indices: Negative sample index pairs, shape (batch_size, n_neg_pairs, 2)
            
        Returns:
            pos_loss, neg_loss: Positive sample loss and negative sample loss
        """
        batch_size = embeddings.shape[0]
        
        # Vectorized processing of positive sample pairs
        pos_loss = torch.tensor(0.0, device=embeddings.device)
        if pos_indices.numel() > 0:
            # Reshape indices for vectorized processing
            pos_indices_flat = pos_indices.view(-1, 2)
            
            # Filter valid index pairs (within batch range)
            valid_mask = (pos_indices_flat[:, 0] >= 0) & (pos_indices_flat[:, 0] < batch_size) & \
                        (pos_indices_flat[:, 1] >= 0) & (pos_indices_flat[:, 1] < batch_size)
            
            if valid_mask.any():
                valid_pos_indices = pos_indices_flat[valid_mask]
                pos_embeddings1 = embeddings[valid_pos_indices[:, 0]]
                pos_embeddings2 = embeddings[valid_pos_indices[:, 1]]
                
                # Calculate distances
                pos_distances = torch.norm(pos_embeddings1 - pos_embeddings2, dim=1)
                
                # Calculate loss
                pos_targets = torch.ones(len(pos_distances), device=embeddings.device)
                pos_probs = torch.sigmoid(-pos_distances / self.min_dist)
                pos_loss = F.binary_cross_entropy(pos_probs, pos_targets)
        
        # Vectorized processing of negative sample pairs
        neg_loss = torch.tensor(0.0, device=embeddings.device)
        if neg_indices.numel() > 0:
            # Reshape indices for vectorized processing
            neg_indices_flat = neg_indices.view(-1, 2)
            
            # Filter valid index pairs (within batch range)
            valid_mask = (neg_indices_flat[:, 0] >= 0) & (neg_indices_flat[:, 0] < batch_size) & \
                        (neg_indices_flat[:, 1] >= 0) & (neg_indices_flat[:, 1] < batch_size)
            
            if valid_mask.any():
                valid_neg_indices = neg_indices_flat[valid_mask]
                neg_embeddings1 = embeddings[valid_neg_indices[:, 0]]
                neg_embeddings2 = embeddings[valid_neg_indices[:, 1]]
                
                # Calculate distances
                neg_distances = torch.norm(neg_embeddings1 - neg_embeddings2, dim=1)
                
                # Calculate loss
                neg_targets = torch.zeros(len(neg_distances), device=embeddings.device)
                neg_probs = torch.sigmoid(-neg_distances / self.min_dist)
                neg_loss = F.binary_cross_entropy(neg_probs, neg_targets)
        
        return pos_loss, neg_loss


class NNEAUMAP(BaseModel):
    """
    NNEA UMAP dimensionality reducer
    Implements a neural network-based UMAP dimensionality reduction, providing interpretable dimensionality reduction results
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize NNEA UMAP dimensionality reducer
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.task = 'umap'

    def build(self, nadata) -> None:
        """
        Build the model
        
        Args:
            nadata: nadata object
        """
        if nadata is None:
            raise ValueError("nadata object cannot be empty")
        
        # Get input dimension
        if hasattr(nadata, 'X') and nadata.X is not None:
            input_dim = nadata.X.shape[1]  # Number of genes
        else:
            raise ValueError("Expression matrix not loaded")
        
        # Get UMAP configuration
        umap_config = self.config.get('umap', {})
        embedding_dim = umap_config.get('embedding_dim', 2)
        hidden_dims = umap_config.get('hidden_dims', [128, 64, 32])
        dropout = umap_config.get('dropout', 0.1)
        
        # Update configuration
        self.config['input_dim'] = input_dim
        self.config['embedding_dim'] = embedding_dim
        self.config['device'] = str(self.device)
        
        # Create model
        self.model = NeuralUMAP(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        self.model.to(self.device)
        
        # Create UMAP loss function (using optimized PCA version)
        n_neighbors = umap_config.get('n_neighbors', 15)
        min_dist = umap_config.get('min_dist', 0.1)
        pca_components = umap_config.get('pca_components', 50)
        
        self.umap_loss = UMAPLoss(
            min_dist=min_dist,
            n_neighbors=n_neighbors,
            pca_components=pca_components
        ).to(self.device)
        
        self.logger.info(f"NNEA UMAP dimensionality reducer built: input dimension={input_dim}, embedding dimension={embedding_dim}")
        self.logger.info(f"Hidden layer dimensions: {hidden_dims}")
        self.logger.info(f"UMAP parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, pca_components={pca_components}")
    
    def train(self, nadata, verbose: int = 1, max_epochs: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            nadata: nadata object
            verbose: Verbosity level
                0=Only show progress bar
                1=Show training loss
                2=Show training loss and reconstruction loss
            max_epochs: Maximum number of epochs, if None, use epochs from config
            **kwargs: Additional parameters
            
        Returns:
            Training results dictionary
        """
        if self.model is None:
            raise ValueError("Model not built")
        
        # Prepare data
        X = nadata.X

        # Complete PCA and neighbor calculation before training (only once)
        self.logger.info("Calculating PCA and neighbor relationships...")
        pos_indices, neg_indices = self.umap_loss.fit_pca(X, nadata)
        self.logger.info("PCA and neighbor calculation completed!")
        
        # Training parameters
        training_config = self.config.get('training', {})
        if max_epochs is None:
            epochs = training_config.get('epochs', 100)
        else:
            epochs = max_epochs
        learning_rate = training_config.get('learning_rate', 0.001)
        batch_size = training_config.get('batch_size', 32)
        test_size = training_config.get('test_size', 0.2)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Custom dataset class
        class UMAPDataset(torch.utils.data.Dataset):
            def __init__(self, X, pos_indices, neg_indices):
                self.X = X
                self.pos_indices = pos_indices
                self.neg_indices = neg_indices
                
                # Create index mapping for each sample
                self.sample_to_pos = {}
                self.sample_to_neg = {}
                
                # Build sample to positive pair mapping
                for i, (idx1, idx2) in enumerate(pos_indices):
                    if idx1 not in self.sample_to_pos:
                        self.sample_to_pos[idx1] = []
                    self.sample_to_pos[idx1].append(idx2)
                    
                    if idx2 not in self.sample_to_pos:
                        self.sample_to_pos[idx2] = []
                    self.sample_to_pos[idx2].append(idx1)
                
                # Build sample to negative pair mapping
                for i, (idx1, idx2) in enumerate(neg_indices):
                    if idx1 not in self.sample_to_neg:
                        self.sample_to_neg[idx1] = []
                    self.sample_to_neg[idx1].append(idx2)
                    
                    if idx2 not in self.sample_to_neg:
                        self.sample_to_neg[idx2] = []
                    self.sample_to_neg[idx2].append(idx1)
                
                # Calculate maximum positive and negative sample count for padding
                self.max_pos_pairs = 0
                self.max_neg_pairs = 0
                for i in range(len(X)):
                    pos_count = len(self.sample_to_pos.get(i, []))
                    neg_count = len(self.sample_to_neg.get(i, []))
                    self.max_pos_pairs = max(self.max_pos_pairs, pos_count)
                    self.max_neg_pairs = max(self.max_neg_pairs, neg_count)
                
                # Ensure at least one sample pair
                self.max_pos_pairs = max(self.max_pos_pairs, 1)
                self.max_neg_pairs = max(self.max_neg_pairs, 1)
                
                # Record maximum sample pair count
                print(f"Dataset statistics: max positive sample pairs={self.max_pos_pairs}, max negative sample pairs={self.max_neg_pairs}")
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                # Get positive sample indices for the current sample
                pos_neighbors = self.sample_to_pos.get(idx, [])
                if len(pos_neighbors) == 0:
                    pos_neighbors = [idx]  # If no positive samples, use self
                
                # Get negative sample indices for the current sample
                neg_neighbors = self.sample_to_neg.get(idx, [])
                if len(neg_neighbors) == 0:
                    neg_neighbors = [idx]  # If no negative samples, use self
                
                # Create positive sample pairs indices and pad to fixed size
                pos_pairs = [[idx, neighbor] for neighbor in pos_neighbors]
                while len(pos_pairs) < self.max_pos_pairs:
                    pos_pairs.append([idx, idx])  # Pad with self
                
                # Create negative sample pairs indices and pad to fixed size
                neg_pairs = [[idx, neighbor] for neighbor in neg_neighbors]
                while len(neg_pairs) < self.max_neg_pairs:
                    neg_pairs.append([idx, idx])  # Pad with self
                
                # Convert original indices to batch indices (relative position)
                # We return original indices, which will be converted in DataLoader's collate_fn
                return (self.X[idx], 
                       torch.tensor(pos_pairs, dtype=torch.long),
                       torch.tensor(neg_pairs, dtype=torch.long))
        
        # Build full dataset
        full_dataset = UMAPDataset(X_tensor, pos_indices, neg_indices)
        
        # Split dataset using random_split
        n_samples = X.shape[0]
        train_size = int(n_samples * (1 - test_size))
        test_size_split = n_samples - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size_split]
        )
        
        self.logger.info(f"Dataset split: {len(train_dataset)} samples in training set, {len(test_dataset)} samples in test set")
        
        # Store training indices for later use (obtained from random_split)
        self.train_indices = train_dataset.indices
        
        # Define collate function to convert original indices to batch indices
        def umap_collate_fn(batch):
            """
            Convert original indices in batch data to batch indices
            """
            batch_X = []
            batch_pos_indices = []
            batch_neg_indices = []
            
            # Get original indices of all samples in the current batch from the train_dataset.indices
            # Since random_split re-indexes, we need to get original indices from train_dataset.indices
            batch_original_indices = [train_dataset.indices[i] for i in range(len(batch))]
            
            # Create a mapping from original indices to batch indices
            original_to_batch = {orig_idx: batch_idx for batch_idx, orig_idx in enumerate(batch_original_indices)}
            
            for i, (X_item, pos_pairs, neg_pairs) in enumerate(batch):
                batch_X.append(X_item)
                
                # Convert original indices to batch indices
                pos_pairs_batch = pos_pairs.clone()
                neg_pairs_batch = neg_pairs.clone()
                
                # Convert positive sample pair indices
                for j in range(pos_pairs_batch.shape[0]):
                    orig_idx1, orig_idx2 = pos_pairs_batch[j]
                    if orig_idx1 in original_to_batch and orig_idx2 in original_to_batch:
                        pos_pairs_batch[j, 0] = original_to_batch[orig_idx1]
                        pos_pairs_batch[j, 1] = original_to_batch[orig_idx2]
                    else:
                        # If index is not in the current batch, use self index
                        pos_pairs_batch[j, 0] = i
                        pos_pairs_batch[j, 1] = i
                
                # Convert negative sample pair indices
                for j in range(neg_pairs_batch.shape[0]):
                    orig_idx1, orig_idx2 = neg_pairs_batch[j]
                    if orig_idx1 in original_to_batch and orig_idx2 in original_to_batch:
                        neg_pairs_batch[j, 0] = original_to_batch[orig_idx1]
                        neg_pairs_batch[j, 1] = original_to_batch[orig_idx2]
                    else:
                        # If index is not in the current batch, use self index
                        neg_pairs_batch[j, 0] = i
                        neg_pairs_batch[j, 1] = i
                
                batch_pos_indices.append(pos_pairs_batch)
                batch_neg_indices.append(neg_pairs_batch)
            
            # Stack batch data
            batch_X = torch.stack(batch_X)
            batch_pos_indices = torch.stack(batch_pos_indices)
            batch_neg_indices = torch.stack(batch_neg_indices)
            
            return batch_X, batch_pos_indices, batch_neg_indices
        
        # Create data loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=umap_collate_fn
        )
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Early stopping parameters
        patience = training_config.get('patience', 10)
        min_delta = 1e-6
        
        # Early stopping variable initialization
        best_loss = float('inf')
        patience_counter = 0
        early_stopped = False
        
        # Training loop
        train_losses = []
        
        if verbose >= 1:
            self.logger.info("Starting NNEA UMAP model training...")
            self.logger.info(f"Early stopping configuration: patience={patience}, min_delta={min_delta}")
        
        # Import tqdm for progress bar
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
        
        # Create progress bar (only shown when verbose=0)
        if verbose == 0 and use_tqdm:
            pbar = tqdm(range(epochs), desc="Training progress")
        else:
            pbar = range(epochs)
        
        for epoch in pbar:
            # Training mode
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # Train with batch data using data loader
            for batch_idx, (batch_X, batch_pos_indices, batch_neg_indices) in enumerate(train_loader):

                # Move data to device
                batch_X = batch_X.to(self.device)
                
                optimizer.zero_grad()
                
                try:
                    # Forward pass
                    encoded, decoded = self.model(batch_X)

                    # Debug information: show indices count
                    if verbose >= 2 and batch_idx == 0:
                        self.logger.info(f"Epoch {epoch}, Batch {batch_idx}: batch_pos_indices={batch_pos_indices.shape}, batch_neg_indices={batch_neg_indices.shape}")


                    umap_loss = self.umap_loss(encoded, X, nadata, batch_pos_indices, batch_neg_indices)
                    
                    # Calculate reconstruction loss (optional)
                    recon_loss = F.mse_loss(decoded, batch_X)
                    
                    # Total loss
                    total_loss = umap_loss + 0.1 * recon_loss
                    
                    # Backward pass
                    total_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.error(f"Epoch {epoch}, Batch {batch_idx}: Error during training: {e}")
                    continue
            
            # Calculate average loss
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                train_losses.append(avg_loss)
                
                # Show training loss when verbose=1
                if verbose >= 1:
                    self.logger.info(f"Epoch {epoch}: Train Loss={avg_loss:.4f}")
                
                # Early stopping check
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Check if early stopping is triggered
                if patience_counter >= patience:
                    early_stopped = True
                    self.logger.info(f"🛑 Epoch {epoch}: Early stopping triggered! Loss did not improve for {patience} epochs")
                    break
        
        # Training completed
        self.is_trained = True
        
        # Log early stopping info
        if early_stopped:
            self.logger.info(f"📊 Training ended due to early stopping, trained for {epoch+1} epochs")
        else:
            self.logger.info(f"📊 Training completed, trained for {epochs} epochs")
        
        # Show cache info
        cache_info = self.get_cache_info()
        self.logger.info("Cache info:")
        self.logger.info(f"- PCA fitted: {cache_info['pca_fitted']}")
        self.logger.info(f"- Neighbor index shape: {cache_info['nbr_indices_shape']}")
        self.logger.info(f"- PCA components: {cache_info['pca_components']}")
        self.logger.info(f"- Original data shape: {cache_info['original_data_shape']}")
        self.logger.info(f"- Global distances shape: {cache_info['global_distances_shape']}")
        self.logger.info(f"- Smart negative sampling: {cache_info['smart_negative_sampling']}")
        self.logger.info("✅ Improvement: Using global distance information for smart negative sample selection, improving UMAP quality")
        
        # Return training results
        results = {
            'train_losses': train_losses,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'epochs_trained': epoch + 1 if early_stopped else epochs,
            'early_stopped': early_stopped,
            'best_loss': best_loss
        }
        
        return results

    def predict(self, nadata) -> np.ndarray:
        """
        Model prediction (dimensionality reduction)
        
        Args:
            nadata: nadata object
            
        Returns:
            Embedding results after dimensionality reduction
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        self.model.eval()
        with torch.no_grad():
            X = nadata.X
            
            # Data standardization
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            encoded, _ = self.model(X_tensor)
            return encoded.cpu().numpy()
    
    def evaluate(self, nadata, split='test') -> Dict[str, float]:
        """
        Model evaluation
        
        Args:
            nadata: nadata object
            split: Data set split for evaluation
            
        Returns:
            Evaluation metrics dictionary
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Get data indices
        indices = nadata.Model.get_indices(split)
        if indices is None:
            raise ValueError(f"Indices for {split} set not found")
        
        # Get data based on indices
        X = nadata.X[indices]
        
        # Get embedding results
        embeddings = self.predict(nadata)
        embeddings_split = embeddings[indices]
        
        # Calculate dimensionality reduction quality metrics
        try:
            # Reconstruction error
            X_tensor = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                encoded, decoded = self.model(X_tensor)
                reconstruction_error = F.mse_loss(decoded, X_tensor).item()
            
            # If there are associated labels, calculate clustering metrics
            if hasattr(nadata, 'Meta') and nadata.Meta is not None:
                target_col = self.config.get('dataset', {}).get('target_column', 'target')
                if target_col in nadata.Meta.columns:
                    labels = nadata.Meta.iloc[indices][target_col].values
                    
                    # Calculate clustering metrics
                    silhouette = silhouette_score(embeddings_split, labels)
                    calinski_harabasz = calinski_harabasz_score(embeddings_split, labels)
                    davies_bouldin = davies_bouldin_score(embeddings_split, labels)
                    
                    results = {
                        'reconstruction_error': reconstruction_error,
                        'silhouette_score': silhouette,
                        'calinski_harabasz_score': calinski_harabasz,
                        'davies_bouldin_score': davies_bouldin
                    }
                else:
                    results = {
                        'reconstruction_error': reconstruction_error
                    }
            else:
                results = {
                    'reconstruction_error': reconstruction_error
                }
            
        except Exception as e:
            self.logger.error(f"Error calculating evaluation metrics: {e}")
            results = {
                'reconstruction_error': float('inf')
            }
        
        # Save evaluation results to Model container
        eval_results = nadata.Model.get_metadata('evaluation_results') or {}
        eval_results[split] = results
        nadata.Model.add_metadata('evaluation_results', eval_results)
        
        self.logger.info(f"Model evaluation completed - {split} set:")
        for metric, value in results.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def explain(self, nadata, method='importance') -> Dict[str, Any]:
        """
        Model explanation
        
        Args:
            nadata: nadata object
            method: Explanation method
            
        Returns:
            Explanation results dictionary
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        if method == 'importance':
            try:
                # Get embedding results
                embeddings = self.predict(nadata)
                
                # Calculate feature importance (based on reconstruction error)
                feature_importance = self._calculate_feature_importance(nadata)
                
                # Sort and get top 20 important features
                top_indices = np.argsort(feature_importance)[::-1][:20]
                top_features = [nadata.Var.iloc[i]['Gene'] for i in top_indices]
                top_scores = feature_importance[top_indices]
                
                # Print top 20 features
                self.logger.info(f"  - Top 20 important genes:")
                self.logger.info(f"    {'Rank':<4} {'Gene Name':<15} {'Importance Score':<12}")
                self.logger.info(f"    {'-'*4} {'-'*15} {'-'*12}")
                for i, (gene, score) in enumerate(zip(top_features, top_scores)):
                    self.logger.info(f"    {i+1:<4} {gene:<15} {score:<12.4f}")
                
                explain_results = {
                    'importance': {
                        'top_features': top_features,
                        'importance_scores': top_scores.tolist(),
                        'embeddings': embeddings.tolist(),
                        'feature_importance': feature_importance.tolist()
                    }
                }
                
                # Save explanation results
                nadata.uns['nnea_umap_explain'] = explain_results
                
                self.logger.info(f"Model explanation completed:")
                return explain_results
                
            except Exception as e:
                self.logger.error(f"Model explanation failed: {e}")
                return {}
        else:
            raise ValueError(f"Unsupported explanation method: {method}")
    
    def _calculate_feature_importance(self, nadata) -> np.ndarray:
        """
        Calculate feature importance
        
        Args:
            nadata: nadata object
            
        Returns:
            Feature importance array
        """
        X = nadata.X
        feature_importance = np.zeros(X.shape[1])
        
        # Use reconstruction error as importance metric
        for i in range(X.shape[1]):
            # Create perturbed data
            X_perturbed = X.copy()
            X_perturbed[:, i] = 0  # Set the i-th feature to zero
            
            # Calculate reconstruction error
            X_tensor = torch.FloatTensor(X_perturbed).to(self.device)
            with torch.no_grad():
                encoded, decoded = self.model(X_tensor)
                reconstruction_error = F.mse_loss(decoded, X_tensor).item()
            
            feature_importance[i] = reconstruction_error
        
        return feature_importance
    
    def save_model(self, save_path: str) -> None:
        """
        Save model state
        
        Args:
            save_path: Save path
        """
        if self.model is None:
            raise ValueError("Model not built")
        
        # Save model state dictionary
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'umap_loss_state_dict': self.umap_loss.state_dict(),
            'config': self.config,
            'device': self.device,
            'is_trained': self.is_trained
        }, save_path)
        
        self.logger.info(f"Model saved to: {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """
        Load model state
        
        Args:
            load_path: Load path
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        # Load model state dictionary
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # Load model parameters
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.umap_loss.load_state_dict(checkpoint['umap_loss_state_dict'])
        
        # Update other attributes
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        if 'is_trained' in checkpoint:
            self.is_trained = checkpoint['is_trained']
        
        self.logger.info(f"Model loaded from {load_path}")
    
    def plot_umap_results(self, nadata, title: str = "NNEA UMAP Visualization", 
                         figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Visualize UMAP results
        
        Args:
            nadata: nadata object
            title: Chart title
            figsize: Chart size
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Get embedding results
        embeddings = self.predict(nadata)
        
        # Get labels (if any)
        labels = None
        if hasattr(nadata, 'Meta') and nadata.Meta is not None:
            target_col = self.config.get('dataset', {}).get('target_column', 'target')
            if target_col in nadata.Meta.columns:
                labels = nadata.Meta[target_col].values
        
        # Create visualization
        plt.figure(figsize=figsize)
        
        if labels is not None:
            # Case with labels
            unique_labels = np.unique(labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                           c=[colors[i]], label=f'Class {label}', alpha=0.7)
        else:
            # Case without labels
            plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.7)
        
        plt.title(title)
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        
        if labels is not None:
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        self.logger.info(f"UMAP visualization completed: {title}")

    def get_cache_info(self):
        """
        Get cache information
        
        Returns:
            Cache information dictionary
        """
        if hasattr(self.umap_loss, 'is_fitted') and self.umap_loss.is_fitted:
            return {
                'pca_fitted': True,
                'nbr_indices_shape': self.umap_loss.nbr_indices.shape if self.umap_loss.nbr_indices is not None else None,
                'pca_components': self.umap_loss.pca.n_components_ if self.umap_loss.pca else 0,
                'original_data_shape': self.umap_loss.original_data.shape if self.umap_loss.original_data is not None else None,
                'global_distances_shape': self.umap_loss.global_distances.shape if self.umap_loss.global_distances is not None else None,
                'smart_negative_sampling': True
            }
        else:
            return {
                'pca_fitted': False,
                'nbr_indices_shape': None,
                'pca_components': 0,
                'original_data_shape': None,
                'global_distances_shape': None,
                'smart_negative_sampling': False
            }
