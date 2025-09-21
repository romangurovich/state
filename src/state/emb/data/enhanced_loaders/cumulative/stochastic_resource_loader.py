"""
Stochastic Resource Loader

Combined loader for StochasticResourceModel that provides both
stochastic and resource normalization strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from ..stochastic_loader import StochasticDataLoader
from ..resource_loader import ResourceDataLoader


class StochasticResourceLoader(StochasticDataLoader, ResourceDataLoader):
    """
    Combined loader for StochasticResourceModel.
    Provides both stochastic and resource normalization strategies.
    """
    
    def __init__(
        self,
        cfg,
        valid_gene_mask=None,
        ds_emb_mapping_inference=None,
        is_train=True,
        precision=None,
        # Stochastic parameters
        noise_dim: int = 64,
        num_noise_types: int = 3,
        intrinsic_noise_dim: int = 32,
        extrinsic_noise_dim: int = 32,
        noise_strength_range: Tuple[float, float] = (0.1, 0.9),
        # Resource parameters
        resource_dim: int = 32,
        num_resource_types: int = 5,
        energy_dim: int = 16,
        nutrient_dim: int = 16,
        resource_efficiency_range: Tuple[float, float] = (0.1, 1.0),
        **kwargs
    ):
        # Initialize both parent classes
        StochasticDataLoader.__init__(
            self, cfg, valid_gene_mask, ds_emb_mapping_inference, is_train, precision,
            noise_dim=noise_dim,
            num_noise_types=num_noise_types,
            intrinsic_noise_dim=intrinsic_noise_dim,
            extrinsic_noise_dim=extrinsic_noise_dim,
            noise_strength_range=noise_strength_range,
            **kwargs
        )
        
        ResourceDataLoader.__init__(
            self, cfg, valid_gene_mask, ds_emb_mapping_inference, is_train, precision,
            resource_dim=resource_dim,
            num_resource_types=num_resource_types,
            energy_dim=energy_dim,
            nutrient_dim=nutrient_dim,
            resource_efficiency_range=resource_efficiency_range,
            **kwargs
        )
        
        # Combined normalization parameters
        self.combined_normalization = True
        
        # Combined statistics
        self.combined_stats = {
            'noise_resource_correlations': [],
            'stochastic_resource_efficiency': [],
            'variability_resource_utilization': []
        }
    
    def stochastic_resource_count_processing(
        self,
        counts: torch.Tensor,
        gene_names: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Process counts with both stochastic and resource awareness."""
        
        # Apply stochastic processing
        stochastic_processed, noise_types, noise_strength, intrinsic_means, extrinsic_means = self.stochastic_count_processing(
            counts, gene_names
        )
        
        # Apply resource processing
        resource_processed, resource_state, resource_types, energy_state, nutrient_state, resource_allocation = self.resource_count_processing(
            stochastic_processed, gene_names
        )
        
        # Combine context
        context = {
            'noise_types': noise_types,
            'noise_strength': noise_strength,
            'intrinsic_means': intrinsic_means,
            'extrinsic_means': extrinsic_means,
            'resource_state': resource_state,
            'resource_types': resource_types,
            'energy_state': energy_state,
            'nutrient_state': nutrient_state,
            'resource_allocation': resource_allocation
        }
        
        # Update combined statistics
        self._update_combined_stats(
            counts, noise_strength, intrinsic_means, extrinsic_means,
            resource_state, resource_allocation
        )
        
        return resource_processed, context
    
    def _update_combined_stats(
        self,
        counts: torch.Tensor,
        noise_strength: torch.Tensor,
        intrinsic_means: torch.Tensor,
        extrinsic_means: torch.Tensor,
        resource_state: torch.Tensor,
        resource_allocation: torch.Tensor
    ):
        """Update combined stochastic-resource statistics."""
        # Calculate noise-resource correlations
        try:
            noise_resource_correlation = torch.corrcoef(torch.stack([
                noise_strength.flatten(), 
                resource_state.mean(dim=1).repeat_interleave(noise_strength.size(1))
            ]))[0, 1]
            self.combined_stats['noise_resource_correlations'].append(noise_resource_correlation.cpu())
        except:
            pass  # Skip if correlation cannot be calculated
        
        # Calculate stochastic-resource efficiency
        stochastic_efficiency = (intrinsic_means.mean() + extrinsic_means.mean()) / 2
        resource_efficiency = resource_allocation.mean()
        combined_efficiency = stochastic_efficiency * resource_efficiency
        self.combined_stats['stochastic_resource_efficiency'].append(combined_efficiency.cpu())
        
        # Calculate variability-resource utilization
        variability = counts.std() / (counts.mean() + 1e-8)
        resource_utilization = resource_allocation.mean()
        variability_utilization = variability * resource_utilization
        self.combined_stats['variability_resource_utilization'].append(variability_utilization.cpu())
    
    def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined stochastic-resource statistics."""
        stats = {}
        
        # Noise-resource correlations
        if self.combined_stats['noise_resource_correlations']:
            stats['avg_noise_resource_correlation'] = torch.stack(
                self.combined_stats['noise_resource_correlations']
            ).mean()
        
        # Stochastic-resource efficiency
        if self.combined_stats['stochastic_resource_efficiency']:
            stats['avg_stochastic_resource_efficiency'] = torch.stack(
                self.combined_stats['stochastic_resource_efficiency']
            ).mean()
        
        # Variability-resource utilization
        if self.combined_stats['variability_resource_utilization']:
            stats['avg_variability_resource_utilization'] = torch.stack(
                self.combined_stats['variability_resource_utilization']
            ).mean()
        
        # Combine parent statistics
        stochastic_stats = self.get_stochastic_stats()
        resource_stats = self.get_resource_stats()
        
        stats.update(stochastic_stats)
        stats.update(resource_stats)
        
        return stats
    
    def reset_combined_stats(self):
        """Reset combined statistics."""
        self.combined_stats = {
            'noise_resource_correlations': [],
            'stochastic_resource_efficiency': [],
            'variability_resource_utilization': []
        }
        self.reset_stochastic_stats()
        self.reset_resource_stats()
    
    def sample_cell_sentences_stochastic_resource(
        self,
        counts_raw,
        dataset,
        shared_genes=None,
        valid_gene_mask=None,
        downsample_frac=None,
        gene_names=None
    ):
        """Stochastic-resource cell sentence sampling."""
        # Get gene names if not provided
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(counts_raw.size(1))]
        
        # Apply combined processing
        counts_processed, context = self.stochastic_resource_count_processing(
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
        
        # Add combined context information
        enhanced_result = result + (
            context['noise_types'],
            context['noise_strength'],
            context['intrinsic_means'],
            context['extrinsic_means'],
            context['resource_state'],
            context['resource_types'],
            context['energy_state'],
            context['nutrient_state'],
            context['resource_allocation']
        )
        
        return enhanced_result
    
    def __call__(self, batch):
        """Enhanced collate function with stochastic-resource normalization."""
        # Call parent collate function
        result = super().__call__(batch)
        
        # Extract stochastic-resource information
        if len(result) > 8:  # Enhanced result with stochastic-resource context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums,
                pathway_ids, compartment_ids, cell_type_ids, time_steps
            ) = result
        else:
            # Standard result, create dummy stochastic-resource context
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
        
        # Apply resource normalization
        if hasattr(self, 'resource_normalizer'):
            # Generate dummy resource context if not available
            batch_size, seq_len = Xs.shape[:2]
            resource_state = torch.rand(batch_size, self.resource_dim)
            resource_types = torch.randint(0, self.num_resource_types, (batch_size, seq_len))
            energy_state = torch.rand(batch_size, self.energy_dim)
            nutrient_state = torch.rand(batch_size, self.nutrient_dim)
            
            Xs, resource_allocation, resource_types = self.resource_normalizer(
                Xs, resource_state, resource_types
            )
            
            Xs, energy_allocation, energy_consumption = self.energy_normalizer(Xs, energy_state)
            Xs, nutrient_allocation, nutrient_consumption = self.nutrient_normalizer(Xs, nutrient_state)
        
        # Return enhanced result
        return (
            batch_sentences, Xs, Ys, idxs, batch_weights, masks,
            total_counts_all, batch_sentences_counts, dataset_nums,
            pathway_ids, compartment_ids, cell_type_ids, time_steps
        )
