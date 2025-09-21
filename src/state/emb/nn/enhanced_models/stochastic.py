"""
Stochastic Cellular Behavior Model

Implements stochastic cellular behavior and noise modeling to capture
the inherent variability in biological systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
from .base_enhanced import BaseEnhancedModel


class StochasticCellularModel(BaseEnhancedModel):
    """
    Stochastic cellular model that implements:
    1. Intrinsic noise modeling (molecular stochasticity)
    2. Extrinsic noise modeling (environmental variability)
    3. Heterogeneous response generation
    4. Uncertainty quantification
    """
    
    def __init__(
        self,
        token_dim: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        output_dim: int,
        dropout: float = 0.0,
        warmup_steps: int = 0,
        compiled: bool = False,
        max_lr: float = 4e-4,
        emb_cnt: int = 145469,
        emb_size: int = 5120,
        cfg: Optional[Dict[str, Any]] = None,
        collater: Optional[Any] = None,
        # Stochastic-specific parameters
        noise_dim: int = 64,
        intrinsic_noise_scale: float = 0.1,
        extrinsic_noise_scale: float = 0.05,
        num_samples: int = 5,
        **kwargs
    ):
        super().__init__(
            token_dim=token_dim,
            d_model=d_model,
            nhead=nhead,
            d_hid=d_hid,
            nlayers=nlayers,
            output_dim=output_dim,
            dropout=dropout,
            warmup_steps=warmup_steps,
            compiled=compiled,
            max_lr=max_lr,
            emb_cnt=emb_cnt,
            emb_size=emb_size,
            cfg=cfg,
            collater=collater,
            noise_dim=noise_dim,
            **kwargs
        )
        
        self.intrinsic_noise_scale = intrinsic_noise_scale
        self.extrinsic_noise_scale = extrinsic_noise_scale
        self.num_samples = num_samples
        
        self._init_stochastic_components()
    
    def _init_stochastic_components(self):
        """Initialize stochastic processing components."""
        # Intrinsic noise encoder (molecular stochasticity)
        self.intrinsic_noise_encoder = nn.Sequential(
            nn.Linear(self.d_model, self.noise_dim * 2),  # Mean and variance
            nn.SiLU(),
            nn.Linear(self.noise_dim * 2, self.noise_dim * 2)
        )
        
        # Extrinsic noise encoder (environmental variability)
        self.extrinsic_noise_encoder = nn.Sequential(
            nn.Linear(self.d_model, self.noise_dim * 2),  # Mean and variance
            nn.SiLU(),
            nn.Linear(self.noise_dim * 2, self.noise_dim * 2)
        )
        
        # Noise fusion mechanism
        self.noise_fusion = nn.Sequential(
            nn.Linear(self.noise_dim * 2, self.noise_dim),
            nn.SiLU(),
            nn.Linear(self.noise_dim, self.noise_dim),
            nn.LayerNorm(self.noise_dim)
        )
        
        # Stochastic decoder
        self.stochastic_decoder = nn.Sequential(
            nn.Linear(self.d_model + self.noise_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.output_dim)
        )
        
        # Uncertainty quantifier
        self.uncertainty_quantifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
        # Heterogeneity generator
        self.heterogeneity_generator = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Noise attention mechanism
        self.noise_attention = nn.MultiheadAttention(
            self.noise_dim,
            num_heads=4,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Stochastic activation
        self.stochastic_activation = nn.SiLU()
        
        # Noise regularization
        self.noise_regularization = nn.Parameter(torch.tensor(0.1))
    
    def generate_intrinsic_noise(
        self, 
        gene_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate intrinsic noise (molecular stochasticity).
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
        
        Returns:
            Tuple of (noise, noise_parameters)
        """
        # Encode noise parameters
        noise_params = self.intrinsic_noise_encoder(gene_embeddings)
        mean, log_var = noise_params.chunk(2, dim=-1)
        
        # Generate noise
        noise = torch.randn_like(mean) * torch.exp(0.5 * log_var) + mean
        
        # Scale by intrinsic noise scale
        noise = noise * self.intrinsic_noise_scale
        
        return noise, (mean, log_var)
    
    def generate_extrinsic_noise(
        self, 
        gene_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate extrinsic noise (environmental variability).
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
        
        Returns:
            Tuple of (noise, noise_parameters)
        """
        # Encode noise parameters
        noise_params = self.extrinsic_noise_encoder(gene_embeddings)
        mean, log_var = noise_params.chunk(2, dim=-1)
        
        # Generate noise
        noise = torch.randn_like(mean) * torch.exp(0.5 * log_var) + mean
        
        # Scale by extrinsic noise scale
        noise = noise * self.extrinsic_noise_scale
        
        return noise, (mean, log_var)
    
    def fuse_noise(
        self, 
        intrinsic_noise: torch.Tensor,
        extrinsic_noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse intrinsic and extrinsic noise.
        
        Args:
            intrinsic_noise: Intrinsic noise [batch, seq_len, noise_dim]
            extrinsic_noise: Extrinsic noise [batch, seq_len, noise_dim]
        
        Returns:
            Fused noise [batch, seq_len, noise_dim]
        """
        # Combine noise sources
        combined_noise = torch.cat([intrinsic_noise, extrinsic_noise], dim=-1)
        
        # Fuse noise
        fused_noise = self.noise_fusion(combined_noise)
        
        return fused_noise
    
    def apply_stochastic_processing(
        self, 
        gene_embeddings: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply stochastic processing to gene embeddings.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
            noise: Fused noise [batch, seq_len, noise_dim]
        
        Returns:
            Stochastic processed embeddings
        """
        # Combine embeddings with noise
        stochastic_input = torch.cat([gene_embeddings, noise], dim=-1)
        
        # Apply stochastic decoder
        stochastic_output = self.stochastic_decoder(stochastic_input)
        
        # Apply stochastic activation
        stochastic_output = self.stochastic_activation(stochastic_output)
        
        return stochastic_output
    
    def generate_heterogeneous_responses(
        self, 
        gene_embeddings: torch.Tensor,
        num_samples: int = 5
    ) -> List[torch.Tensor]:
        """
        Generate heterogeneous responses for the same input.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
            num_samples: Number of heterogeneous samples to generate
        
        Returns:
            List of heterogeneous responses
        """
        responses = []
        
        for _ in range(num_samples):
            # Generate intrinsic noise
            intrinsic_noise, _ = self.generate_intrinsic_noise(gene_embeddings)
            
            # Generate extrinsic noise
            extrinsic_noise, _ = self.generate_extrinsic_noise(gene_embeddings)
            
            # Fuse noise
            fused_noise = self.fuse_noise(intrinsic_noise, extrinsic_noise)
            
            # Apply stochastic processing
            stochastic_output = self.apply_stochastic_processing(gene_embeddings, fused_noise)
            
            # Apply heterogeneity generator
            heterogeneous_output = self.heterogeneity_generator(stochastic_output)
            
            responses.append(heterogeneous_output)
        
        return responses
    
    def compute_uncertainty(
        self, 
        gene_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute uncertainty for predictions.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
        
        Returns:
            Uncertainty scores [batch, seq_len, 1]
        """
        # Compute uncertainty
        uncertainty = self.uncertainty_quantifier(gene_embeddings)
        
        return uncertainty
    
    def forward_stochastic(
        self, 
        src: torch.Tensor, 
        mask: torch.Tensor,
        num_samples: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Stochastic forward pass.
        
        Args:
            src: Input tensor [batch, seq_len, token_dim]
            mask: Attention mask [batch, seq_len]
            num_samples: Number of stochastic samples to generate
        
        Returns:
            Tuple of (gene_output, embedding, dataset_emb, stochastic_info)
        """
        if num_samples is None:
            num_samples = self.num_samples
        
        # Initial encoding
        src = self.encoder(src) * math.sqrt(self.d_model)
        
        # Add count processing if available
        if 'counts' in kwargs and kwargs['counts'] is not None:
            counts = kwargs['counts']
            counts = counts.unsqueeze(-1)
            
            bin_weights = self.count_encoder(counts)
            bin_weights = F.softmax(bin_weights, dim=-1)
            
            bin_indices = torch.arange(10, device=self.device)
            bin_embeddings = self.bin_encoder(bin_indices)
            count_emb = torch.matmul(bin_weights, bin_embeddings)
            
            if self.dataset_token is not None:
                dataset_count_emb = torch.zeros(count_emb.size(0), 1, count_emb.size(2), device=self.device)
                count_emb = torch.cat((count_emb, dataset_count_emb), dim=1)
            
            src = src + count_emb
        
        # Generate intrinsic noise
        intrinsic_noise, intrinsic_params = self.generate_intrinsic_noise(src)
        
        # Generate extrinsic noise
        extrinsic_noise, extrinsic_params = self.generate_extrinsic_noise(src)
        
        # Fuse noise
        fused_noise = self.fuse_noise(intrinsic_noise, extrinsic_noise)
        
        # Apply stochastic processing
        stochastic_src = self.apply_stochastic_processing(src, fused_noise)
        
        # Apply transformer encoder
        output = self.transformer_encoder(stochastic_src, src_key_padding_mask=None)
        
        # Decode with stochastic information
        gene_output = self.decoder(output)
        
        # Extract embeddings
        embedding = gene_output[:, 0, :]  # CLS token
        embedding = F.normalize(embedding, dim=1)
        
        # Dataset embedding (if available)
        dataset_emb = None
        if self.dataset_token is not None:
            dataset_emb = gene_output[:, -1, :]
        
        # Compute uncertainty
        uncertainty = self.compute_uncertainty(stochastic_src)
        
        # Generate heterogeneous responses
        heterogeneous_responses = self.generate_heterogeneous_responses(
            stochastic_src, num_samples
        )
        
        # Prepare stochastic information
        stochastic_info = {
            'intrinsic_noise': intrinsic_noise,
            'extrinsic_noise': extrinsic_noise,
            'fused_noise': fused_noise,
            'intrinsic_params': intrinsic_params,
            'extrinsic_params': extrinsic_params,
            'uncertainty': uncertainty,
            'heterogeneous_responses': heterogeneous_responses
        }
        
        return gene_output, embedding, dataset_emb, stochastic_info
    
    def forward(self, src: torch.Tensor, mask: torch.Tensor, **kwargs):
        """Override forward to use stochastic processing."""
        num_samples = kwargs.get('num_samples', None)
        
        return self.forward_stochastic(src, mask, num_samples, **kwargs)
    
    def sample_predictions(
        self, 
        src: torch.Tensor, 
        mask: torch.Tensor,
        num_samples: int = 10
    ) -> List[torch.Tensor]:
        """
        Sample multiple predictions for uncertainty quantification.
        
        Args:
            src: Input tensor [batch, seq_len, token_dim]
            mask: Attention mask [batch, seq_len]
            num_samples: Number of samples to generate
        
        Returns:
            List of sampled predictions
        """
        predictions = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                gene_output, _, _ = self.forward_stochastic(src, mask)
                predictions.append(gene_output)
        
        return predictions
    
    def compute_prediction_uncertainty(
        self, 
        predictions: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute prediction uncertainty from multiple samples.
        
        Args:
            predictions: List of predictions [num_samples, batch, seq_len, output_dim]
        
        Returns:
            Uncertainty scores [batch, seq_len, output_dim]
        """
        # Stack predictions
        predictions_tensor = torch.stack(predictions, dim=0)  # [num_samples, batch, seq_len, output_dim]
        
        # Compute mean and variance
        mean_pred = predictions_tensor.mean(dim=0)
        var_pred = predictions_tensor.var(dim=0)
        
        # Uncertainty is the standard deviation
        uncertainty = torch.sqrt(var_pred + 1e-8)
        
        return uncertainty
    
    def compute_stochastic_loss(
        self, 
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: torch.Tensor,
        stochastic_info: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute stochastic loss with uncertainty weighting.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            uncertainty: Uncertainty scores
            stochastic_info: Stochastic information from forward pass
        
        Returns:
            Stochastic loss
        """
        # Base prediction loss
        base_loss = F.mse_loss(predictions, targets)
        
        # Uncertainty-weighted loss
        uncertainty_weighted_loss = base_loss / (uncertainty + 1e-8)
        
        # Noise regularization
        intrinsic_noise = stochastic_info['intrinsic_noise']
        extrinsic_noise = stochastic_info['extrinsic_noise']
        
        noise_regularization = (
            self.noise_regularization * 
            (intrinsic_noise.norm(2) + extrinsic_noise.norm(2))
        )
        
        # Total loss
        total_loss = uncertainty_weighted_loss.mean() + noise_regularization
        
        return total_loss
    
    def get_noise_statistics(
        self, 
        stochastic_info: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Get statistics about the noise generation.
        
        Args:
            stochastic_info: Stochastic information from forward pass
        
        Returns:
            Dictionary containing noise statistics
        """
        intrinsic_noise = stochastic_info['intrinsic_noise']
        extrinsic_noise = stochastic_info['extrinsic_noise']
        fused_noise = stochastic_info['fused_noise']
        
        return {
            'intrinsic_noise_mean': intrinsic_noise.mean(),
            'intrinsic_noise_std': intrinsic_noise.std(),
            'extrinsic_noise_mean': extrinsic_noise.mean(),
            'extrinsic_noise_std': extrinsic_noise.std(),
            'fused_noise_mean': fused_noise.mean(),
            'fused_noise_std': fused_noise.std(),
            'noise_regularization': self.noise_regularization
        }
