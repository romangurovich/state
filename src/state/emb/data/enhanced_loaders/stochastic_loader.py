"""
Stochastic Data Loader

Enhanced data loader for stochastic cellular behavior models with
noise-aware normalization and variability modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from .base_enhanced_loader import BaseEnhancedLoader


class StochasticNormalization(nn.Module):
    """Stochastic normalization for cellular variability modeling."""
    
    def __init__(
        self,
        d_model: int,
        noise_dim: int = 64,
        num_noise_types: int = 3
    ):
        super().__init__()
        self.d_model = d_model
        self.noise_dim = noise_dim
        self.num_noise_types = num_noise_types
        
        # Noise type embeddings
        self.noise_type_embeddings = nn.Embedding(num_noise_types, noise_dim)
        
        # Noise encoders for different types
        self.noise_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, noise_dim),
                nn.SiLU(),
                nn.Linear(noise_dim, noise_dim * 2)  # mean and log_var
            ) for _ in range(num_noise_types)
        ])
        
        # Noise decoders
        self.noise_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(noise_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model)
            ) for _ in range(num_noise_types)
        ])
        
        # Noise type classifier
        self.noise_type_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_noise_types)
        )
        
        # Variability predictor
        self.variability_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        noise_types: Optional[torch.Tensor] = None,
        noise_strength: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply stochastic normalization."""
        batch_size, seq_len, d_model = x.shape
        
        # Classify noise types if not provided
        if noise_types is None:
            noise_types = self.noise_type_classifier(x)
            noise_types = F.softmax(noise_types, dim=-1)
        
        # Predict noise strength if not provided
        if noise_strength is None:
            noise_strength = self.variability_predictor(x)
        
        # Apply stochastic normalization
        stochastic_outputs = []
        noise_samples = []
        noise_means = []
        noise_log_vars = []
        
        for i in range(batch_size):
            for j in range(seq_len):
                # Get noise type
                noise_type = torch.argmax(noise_types[i, j]).item()
                
                # Encode noise parameters
                noise_params = self.noise_encoders[noise_type](x[i, j])
                mean, log_var = noise_params.chunk(2, dim=-1)
                
                # Sample noise
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                noise = eps * std + mean
                
                # Decode noise
                noise_decoded = self.noise_decoders[noise_type](noise)
                
                # Apply noise strength
                noise_strength_ij = noise_strength[i, j]
                stochastic_output = x[i, j] + noise_strength_ij * noise_decoded
                
                stochastic_outputs.append(stochastic_output)
                noise_samples.append(noise)
                noise_means.append(mean)
                noise_log_vars.append(log_var)
        
        # Reshape outputs
        stochastic_output = torch.stack(stochastic_outputs).view(batch_size, seq_len, d_model)
        noise_samples = torch.stack(noise_samples).view(batch_size, seq_len, noise_dim)
        noise_means = torch.stack(noise_means).view(batch_size, seq_len, noise_dim)
        noise_log_vars = torch.stack(noise_log_vars).view(batch_size, seq_len, noise_dim)
        
        return stochastic_output, noise_samples, noise_means, noise_log_vars


class IntrinsicNoiseNormalization(nn.Module):
    """Intrinsic noise normalization for cellular variability."""
    
    def __init__(self, d_model: int, noise_dim: int = 32):
        super().__init__()
        self.d_model = d_model
        self.noise_dim = noise_dim
        
        # Intrinsic noise encoder
        self.intrinsic_noise_encoder = nn.Sequential(
            nn.Linear(d_model, noise_dim),
            nn.SiLU(),
            nn.Linear(noise_dim, noise_dim * 2)  # mean and log_var
        )
        
        # Intrinsic noise decoder
        self.intrinsic_noise_decoder = nn.Sequential(
            nn.Linear(noise_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Noise strength controller
        self.noise_strength_controller = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply intrinsic noise normalization."""
        batch_size, seq_len, d_model = x.shape
        
        # Encode intrinsic noise parameters
        noise_params = self.intrinsic_noise_encoder(x)
        mean, log_var = noise_params.chunk(2, dim=-1)
        
        # Sample intrinsic noise
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        intrinsic_noise = eps * std + mean
        
        # Decode intrinsic noise
        noise_decoded = self.intrinsic_noise_decoder(intrinsic_noise)
        
        # Control noise strength
        noise_strength = self.noise_strength_controller(x)
        stochastic_output = x + noise_strength * noise_decoded
        
        return stochastic_output, mean, log_var


class ExtrinsicNoiseNormalization(nn.Module):
    """Extrinsic noise normalization for environmental variability."""
    
    def __init__(self, d_model: int, noise_dim: int = 32):
        super().__init__()
        self.d_model = d_model
        self.noise_dim = noise_dim
        
        # Extrinsic noise encoder
        self.extrinsic_noise_encoder = nn.Sequential(
            nn.Linear(d_model, noise_dim),
            nn.SiLU(),
            nn.Linear(noise_dim, noise_dim * 2)  # mean and log_var
        )
        
        # Extrinsic noise decoder
        self.extrinsic_noise_decoder = nn.Sequential(
            nn.Linear(noise_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Environmental context encoder
        self.environmental_context = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, noise_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        environmental_context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply extrinsic noise normalization."""
        batch_size, seq_len, d_model = x.shape
        
        # Get environmental context
        if environmental_context is None:
            environmental_context = self.environmental_context(x)
        
        # Encode extrinsic noise parameters
        noise_params = self.extrinsic_noise_encoder(x)
        mean, log_var = noise_params.chunk(2, dim=-1)
        
        # Sample extrinsic noise
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        extrinsic_noise = eps * std + mean
        
        # Decode extrinsic noise
        noise_decoded = self.extrinsic_noise_decoder(extrinsic_noise)
        
        # Apply environmental context
        environmental_effect = environmental_context.unsqueeze(-1) * noise_decoded
        stochastic_output = x + environmental_effect
        
        return stochastic_output, mean, log_var


class StochasticDataLoader(BaseEnhancedLoader):
    """Enhanced data loader for stochastic cellular behavior models."""
    
    def __init__(
        self,
        cfg,
        valid_gene_mask=None,
        ds_emb_mapping_inference=None,
        is_train=True,
        precision=None,
        # Stochastic-specific parameters
        noise_dim: int = 64,
        num_noise_types: int = 3,
        intrinsic_noise_dim: int = 32,
        extrinsic_noise_dim: int = 32,
        noise_strength_range: Tuple[float, float] = (0.1, 0.9),
        **kwargs
    ):
        super().__init__(
            cfg, valid_gene_mask, ds_emb_mapping_inference, is_train, precision,
            **kwargs
        )
        
        self.noise_dim = noise_dim
        self.num_noise_types = num_noise_types
        self.intrinsic_noise_dim = intrinsic_noise_dim
        self.extrinsic_noise_dim = extrinsic_noise_dim
        self.noise_strength_range = noise_strength_range
        
        # Initialize stochastic normalization modules
        self.stochastic_normalizer = StochasticNormalization(
            d_model=512,  # Will be updated
            noise_dim=noise_dim,
            num_noise_types=num_noise_types
        )
        
        self.intrinsic_noise_normalizer = IntrinsicNoiseNormalization(
            d_model=512,  # Will be updated
            noise_dim=intrinsic_noise_dim
        )
        
        self.extrinsic_noise_normalizer = ExtrinsicNoiseNormalization(
            d_model=512,  # Will be updated
            noise_dim=extrinsic_noise_dim
        )
        
        # Stochastic statistics
        self.stochastic_stats = {
            'noise_types': [],
            'noise_strengths': [],
            'intrinsic_noise_means': [],
            'intrinsic_noise_stds': [],
            'extrinsic_noise_means': [],
            'extrinsic_noise_stds': [],
            'variability_scores': []
        }
    
    def generate_stochastic_context(
        self,
        counts: torch.Tensor,
        gene_names: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate stochastic context for given counts."""
        batch_size, seq_len = counts.shape
        
        # Generate noise types
        noise_types = torch.randint(0, self.num_noise_types, (batch_size, seq_len))
        
        # Generate noise strength
        noise_strength = torch.rand(batch_size, seq_len) * (
            self.noise_strength_range[1] - self.noise_strength_range[0]
        ) + self.noise_strength_range[0]
        
        # Generate environmental context
        environmental_context = torch.randn(batch_size, seq_len, self.extrinsic_noise_dim)
        
        return noise_types, noise_strength, environmental_context
    
    def stochastic_count_processing(
        self,
        counts: torch.Tensor,
        gene_names: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process counts with stochastic awareness."""
        # Generate stochastic context
        noise_types, noise_strength, environmental_context = self.generate_stochastic_context(
            counts, gene_names
        )
        
        # Apply base normalization
        normalized_counts = self.enhanced_count_processing(counts)
        
        # Reshape for stochastic processing
        counts_reshaped = normalized_counts.unsqueeze(-1).expand(-1, -1, 512)
        
        # Apply stochastic normalization
        stochastic_normalized, noise_samples, noise_means, noise_log_vars = self.stochastic_normalizer(
            counts_reshaped, noise_types, noise_strength
        )
        
        # Apply intrinsic noise normalization
        intrinsic_normalized, intrinsic_means, intrinsic_log_vars = self.intrinsic_noise_normalizer(
            stochastic_normalized
        )
        
        # Apply extrinsic noise normalization
        extrinsic_normalized, extrinsic_means, extrinsic_log_vars = self.extrinsic_noise_normalizer(
            intrinsic_normalized, environmental_context
        )
        
        # Update stochastic statistics
        self._update_stochastic_stats(
            counts, noise_types, noise_strength, noise_means, noise_log_vars,
            intrinsic_means, intrinsic_log_vars, extrinsic_means, extrinsic_log_vars
        )
        
        return (
            extrinsic_normalized.squeeze(-1), noise_types, noise_strength,
            intrinsic_means, extrinsic_means
        )
    
    def _update_stochastic_stats(
        self,
        counts: torch.Tensor,
        noise_types: torch.Tensor,
        noise_strength: torch.Tensor,
        noise_means: torch.Tensor,
        noise_log_vars: torch.Tensor,
        intrinsic_means: torch.Tensor,
        intrinsic_log_vars: torch.Tensor,
        extrinsic_means: torch.Tensor,
        extrinsic_log_vars: torch.Tensor
    ):
        """Update stochastic statistics."""
        # Update noise types
        noise_type_counts = torch.bincount(noise_types.flatten(), minlength=self.num_noise_types)
        self.stochastic_stats['noise_types'].append(noise_type_counts.cpu())
        
        # Update noise strengths
        self.stochastic_stats['noise_strengths'].append(noise_strength.mean().cpu())
        
        # Update intrinsic noise statistics
        self.stochastic_stats['intrinsic_noise_means'].append(intrinsic_means.mean().cpu())
        self.stochastic_stats['intrinsic_noise_stds'].append(torch.exp(0.5 * intrinsic_log_vars).mean().cpu())
        
        # Update extrinsic noise statistics
        self.stochastic_stats['extrinsic_noise_means'].append(extrinsic_means.mean().cpu())
        self.stochastic_stats['extrinsic_noise_stds'].append(torch.exp(0.5 * extrinsic_log_vars).mean().cpu())
        
        # Update variability scores
        variability_score = counts.std() / (counts.mean() + 1e-8)
        self.stochastic_stats['variability_scores'].append(variability_score.cpu())
    
    def get_stochastic_stats(self) -> Dict[str, Any]:
        """Get stochastic statistics."""
        stats = {}
        
        # Noise type statistics
        if self.stochastic_stats['noise_types']:
            noise_type_counts = torch.stack(self.stochastic_stats['noise_types'])
            stats['noise_type_distribution'] = noise_type_counts.mean(dim=0)
        
        # Noise strength statistics
        if self.stochastic_stats['noise_strengths']:
            stats['avg_noise_strength'] = torch.stack(self.stochastic_stats['noise_strengths']).mean()
            stats['max_noise_strength'] = torch.stack(self.stochastic_stats['noise_strengths']).max()
        
        # Intrinsic noise statistics
        if self.stochastic_stats['intrinsic_noise_means']:
            stats['avg_intrinsic_noise_mean'] = torch.stack(self.stochastic_stats['intrinsic_noise_means']).mean()
            stats['avg_intrinsic_noise_std'] = torch.stack(self.stochastic_stats['intrinsic_noise_stds']).mean()
        
        # Extrinsic noise statistics
        if self.stochastic_stats['extrinsic_noise_means']:
            stats['avg_extrinsic_noise_mean'] = torch.stack(self.stochastic_stats['extrinsic_noise_means']).mean()
            stats['avg_extrinsic_noise_std'] = torch.stack(self.stochastic_stats['extrinsic_noise_stds']).mean()
        
        # Variability statistics
        if self.stochastic_stats['variability_scores']:
            stats['avg_variability_score'] = torch.stack(self.stochastic_stats['variability_scores']).mean()
            stats['max_variability_score'] = torch.stack(self.stochastic_stats['variability_scores']).max()
        
        return stats
    
    def reset_stochastic_stats(self):
        """Reset stochastic statistics."""
        self.stochastic_stats = {
            'noise_types': [],
            'noise_strengths': [],
            'intrinsic_noise_means': [],
            'intrinsic_noise_stds': [],
            'extrinsic_noise_means': [],
            'extrinsic_noise_stds': [],
            'variability_scores': []
        }
    
    def sample_cell_sentences_stochastic(
        self,
        counts_raw,
        dataset,
        shared_genes=None,
        valid_gene_mask=None,
        downsample_frac=None,
        gene_names=None
    ):
        """Stochastic cell sentence sampling."""
        # Get gene names if not provided
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(counts_raw.size(1))]
        
        # Apply stochastic processing
        counts_processed, noise_types, noise_strength, intrinsic_means, extrinsic_means = self.stochastic_count_processing(
            counts_raw, gene_names
        )
        
        # Call original sampling method with processed counts
        result = self.sample_cell_sentences(
            counts_processed,
            dataset,
            shared_genes,
            valid_gene_mask,
            downsample_frac
        )
        
        # Add stochastic context information
        enhanced_result = result + (noise_types, noise_strength, intrinsic_means, extrinsic_means)
        
        return enhanced_result
    
    def __call__(self, batch):
        """Enhanced collate function with stochastic normalization."""
        # Call parent collate function
        result = super().__call__(batch)
        
        # Extract stochastic information
        if len(result) > 8:  # Enhanced result with stochastic context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums,
                pathway_ids, compartment_ids, cell_type_ids, time_steps
            ) = result
        else:
            # Standard result, create dummy stochastic context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums
            ) = result
            pathway_ids = None
            compartment_ids = None
            cell_type_ids = None
            time_steps = None
        
        # Apply stochastic normalization
        if hasattr(self, 'stochastic_normalizer'):
            # Generate dummy stochastic context if not available
            batch_size, seq_len = Xs.shape[:2]
            noise_types = torch.randint(0, self.num_noise_types, (batch_size, seq_len))
            noise_strength = torch.rand(batch_size, seq_len)
            environmental_context = torch.randn(batch_size, seq_len, self.extrinsic_noise_dim)
            
            Xs, noise_samples, noise_means, noise_log_vars = self.stochastic_normalizer(
                Xs, noise_types, noise_strength
            )
            
            Xs, intrinsic_means, intrinsic_log_vars = self.intrinsic_noise_normalizer(Xs)
            Xs, extrinsic_means, extrinsic_log_vars = self.extrinsic_noise_normalizer(Xs, environmental_context)
        
        # Return enhanced result
        return (
            batch_sentences, Xs, Ys, idxs, batch_weights, masks,
            total_counts_all, batch_sentences_counts, dataset_nums,
            pathway_ids, compartment_ids, cell_type_ids, time_steps
        )
