"""
Resource Data Loader

Enhanced data loader for resource-constrained models with
resource-aware normalization and energy modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from .base_enhanced_loader import BaseEnhancedLoader


class ResourceAwareNormalization(nn.Module):
    """Resource-aware normalization for cellular resource constraints."""
    
    def __init__(
        self,
        d_model: int,
        resource_dim: int = 32,
        num_resource_types: int = 5
    ):
        super().__init__()
        self.d_model = d_model
        self.resource_dim = resource_dim
        self.num_resource_types = num_resource_types
        
        # Resource type embeddings
        self.resource_type_embeddings = nn.Embedding(num_resource_types, resource_dim)
        
        # Resource state encoder
        self.resource_state_encoder = nn.Sequential(
            nn.Linear(d_model, resource_dim),
            nn.SiLU(),
            nn.Linear(resource_dim, resource_dim)
        )
        
        # Resource allocation mechanism
        self.resource_allocator = nn.Sequential(
            nn.Linear(resource_dim * 2, resource_dim),
            nn.SiLU(),
            nn.Linear(resource_dim, d_model),
            nn.Sigmoid()
        )
        
        # Resource consumption predictor
        self.resource_consumption_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_resource_types)
        )
        
        # Resource efficiency controller
        self.resource_efficiency_controller = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        resource_state: torch.Tensor,
        resource_types: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply resource-aware normalization."""
        batch_size, seq_len, d_model = x.shape
        
        # Predict resource consumption if not provided
        if resource_types is None:
            resource_types = self.resource_consumption_predictor(x)
            resource_types = F.softmax(resource_types, dim=-1)
        
        # Encode resource state
        resource_state_encoded = self.resource_state_encoder(x)
        
        # Get resource type embeddings
        resource_type_embs = self.resource_type_embeddings(
            torch.argmax(resource_types, dim=-1)
        )  # [batch_size, seq_len, resource_dim]
        
        # Combine resource state and type embeddings
        combined_resource = torch.cat([resource_state_encoded, resource_type_embs], dim=-1)
        
        # Allocate resources
        resource_allocation = self.resource_allocator(combined_resource)
        
        # Control resource efficiency
        efficiency = self.resource_efficiency_controller(x)
        
        # Apply resource-aware normalization
        x_normalized = x * resource_allocation * efficiency
        
        return x_normalized, resource_allocation, resource_types


class EnergyConstraintNormalization(nn.Module):
    """Energy constraint normalization for cellular energy modeling."""
    
    def __init__(self, d_model: int, energy_dim: int = 16):
        super().__init__()
        self.d_model = d_model
        self.energy_dim = energy_dim
        
        # Energy state encoder
        self.energy_encoder = nn.Sequential(
            nn.Linear(d_model, energy_dim),
            nn.SiLU(),
            nn.Linear(energy_dim, energy_dim)
        )
        
        # Energy consumption predictor
        self.energy_consumption_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Energy allocation mechanism
        self.energy_allocator = nn.Sequential(
            nn.Linear(energy_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Energy efficiency controller
        self.energy_efficiency_controller = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        energy_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply energy constraint normalization."""
        batch_size, seq_len, d_model = x.shape
        
        # Encode energy state
        energy_encoded = self.energy_encoder(x)
        
        # Predict energy consumption
        energy_consumption = self.energy_consumption_predictor(x)
        
        # Allocate energy
        energy_allocation = self.energy_allocator(energy_encoded)
        
        # Control energy efficiency
        efficiency = self.energy_efficiency_controller(x)
        
        # Apply energy constraint normalization
        x_normalized = x * energy_allocation * efficiency * energy_consumption
        
        return x_normalized, energy_allocation, energy_consumption


class NutrientConstraintNormalization(nn.Module):
    """Nutrient constraint normalization for cellular nutrient modeling."""
    
    def __init__(self, d_model: int, nutrient_dim: int = 16):
        super().__init__()
        self.d_model = d_model
        self.nutrient_dim = nutrient_dim
        
        # Nutrient state encoder
        self.nutrient_encoder = nn.Sequential(
            nn.Linear(d_model, nutrient_dim),
            nn.SiLU(),
            nn.Linear(nutrient_dim, nutrient_dim)
        )
        
        # Nutrient consumption predictor
        self.nutrient_consumption_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Nutrient allocation mechanism
        self.nutrient_allocator = nn.Sequential(
            nn.Linear(nutrient_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Nutrient efficiency controller
        self.nutrient_efficiency_controller = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        nutrient_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply nutrient constraint normalization."""
        batch_size, seq_len, d_model = x.shape
        
        # Encode nutrient state
        nutrient_encoded = self.nutrient_encoder(x)
        
        # Predict nutrient consumption
        nutrient_consumption = self.nutrient_consumption_predictor(x)
        
        # Allocate nutrients
        nutrient_allocation = self.nutrient_allocator(nutrient_encoded)
        
        # Control nutrient efficiency
        efficiency = self.nutrient_efficiency_controller(x)
        
        # Apply nutrient constraint normalization
        x_normalized = x * nutrient_allocation * efficiency * nutrient_consumption
        
        return x_normalized, nutrient_allocation, nutrient_consumption


class ResourceDataLoader(BaseEnhancedLoader):
    """Enhanced data loader for resource-constrained models."""
    
    def __init__(
        self,
        cfg,
        valid_gene_mask=None,
        ds_emb_mapping_inference=None,
        is_train=True,
        precision=None,
        # Resource-specific parameters
        resource_dim: int = 32,
        num_resource_types: int = 5,
        energy_dim: int = 16,
        nutrient_dim: int = 16,
        resource_efficiency_range: Tuple[float, float] = (0.1, 1.0),
        **kwargs
    ):
        super().__init__(
            cfg, valid_gene_mask, ds_emb_mapping_inference, is_train, precision,
            **kwargs
        )
        
        self.resource_dim = resource_dim
        self.num_resource_types = num_resource_types
        self.energy_dim = energy_dim
        self.nutrient_dim = nutrient_dim
        self.resource_efficiency_range = resource_efficiency_range
        
        # Initialize resource normalization modules
        self.resource_normalizer = ResourceAwareNormalization(
            d_model=512,  # Will be updated
            resource_dim=resource_dim,
            num_resource_types=num_resource_types
        )
        
        self.energy_normalizer = EnergyConstraintNormalization(
            d_model=512,  # Will be updated
            energy_dim=energy_dim
        )
        
        self.nutrient_normalizer = NutrientConstraintNormalization(
            d_model=512,  # Will be updated
            nutrient_dim=nutrient_dim
        )
        
        # Resource statistics
        self.resource_stats = {
            'resource_types': [],
            'resource_allocations': [],
            'energy_consumptions': [],
            'nutrient_consumptions': [],
            'efficiency_scores': [],
            'resource_utilization': []
        }
    
    def generate_resource_context(
        self,
        counts: torch.Tensor,
        gene_names: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate resource context for given counts."""
        batch_size, seq_len = counts.shape
        
        # Generate resource state
        resource_state = torch.rand(batch_size, self.resource_dim)
        
        # Generate resource types
        resource_types = torch.randint(0, self.num_resource_types, (batch_size, seq_len))
        
        # Generate energy state
        energy_state = torch.rand(batch_size, self.energy_dim)
        
        # Generate nutrient state
        nutrient_state = torch.rand(batch_size, self.nutrient_dim)
        
        return resource_state, resource_types, energy_state, nutrient_state
    
    def resource_count_processing(
        self,
        counts: torch.Tensor,
        gene_names: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process counts with resource awareness."""
        # Generate resource context
        resource_state, resource_types, energy_state, nutrient_state = self.generate_resource_context(
            counts, gene_names
        )
        
        # Apply base normalization
        normalized_counts = self.enhanced_count_processing(counts)
        
        # Reshape for resource processing
        counts_reshaped = normalized_counts.unsqueeze(-1).expand(-1, -1, 512)
        
        # Apply resource normalization
        resource_normalized, resource_allocation, resource_types = self.resource_normalizer(
            counts_reshaped, resource_state, resource_types
        )
        
        # Apply energy constraint normalization
        energy_normalized, energy_allocation, energy_consumption = self.energy_normalizer(
            resource_normalized, energy_state
        )
        
        # Apply nutrient constraint normalization
        nutrient_normalized, nutrient_allocation, nutrient_consumption = self.nutrient_normalizer(
            energy_normalized, nutrient_state
        )
        
        # Update resource statistics
        self._update_resource_stats(
            counts, resource_types, resource_allocation, energy_consumption,
            nutrient_consumption, resource_state, energy_state, nutrient_state
        )
        
        return (
            nutrient_normalized.squeeze(-1), resource_state, resource_types,
            energy_state, nutrient_state, resource_allocation
        )
    
    def _update_resource_stats(
        self,
        counts: torch.Tensor,
        resource_types: torch.Tensor,
        resource_allocation: torch.Tensor,
        energy_consumption: torch.Tensor,
        nutrient_consumption: torch.Tensor,
        resource_state: torch.Tensor,
        energy_state: torch.Tensor,
        nutrient_state: torch.Tensor
    ):
        """Update resource statistics."""
        # Update resource types
        resource_type_counts = torch.bincount(resource_types.flatten(), minlength=self.num_resource_types)
        self.resource_stats['resource_types'].append(resource_type_counts.cpu())
        
        # Update resource allocations
        self.resource_stats['resource_allocations'].append(resource_allocation.mean().cpu())
        
        # Update energy consumptions
        self.resource_stats['energy_consumptions'].append(energy_consumption.mean().cpu())
        
        # Update nutrient consumptions
        self.resource_stats['nutrient_consumptions'].append(nutrient_consumption.mean().cpu())
        
        # Update efficiency scores
        efficiency_score = (resource_state.mean() + energy_state.mean() + nutrient_state.mean()) / 3
        self.resource_stats['efficiency_scores'].append(efficiency_score.cpu())
        
        # Update resource utilization
        resource_utilization = (resource_allocation * energy_consumption * nutrient_consumption).mean()
        self.resource_stats['resource_utilization'].append(resource_utilization.cpu())
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource statistics."""
        stats = {}
        
        # Resource type statistics
        if self.resource_stats['resource_types']:
            resource_type_counts = torch.stack(self.resource_stats['resource_types'])
            stats['resource_type_distribution'] = resource_type_counts.mean(dim=0)
        
        # Resource allocation statistics
        if self.resource_stats['resource_allocations']:
            stats['avg_resource_allocation'] = torch.stack(self.resource_stats['resource_allocations']).mean()
            stats['max_resource_allocation'] = torch.stack(self.resource_stats['resource_allocations']).max()
        
        # Energy consumption statistics
        if self.resource_stats['energy_consumptions']:
            stats['avg_energy_consumption'] = torch.stack(self.resource_stats['energy_consumptions']).mean()
            stats['max_energy_consumption'] = torch.stack(self.resource_stats['energy_consumptions']).max()
        
        # Nutrient consumption statistics
        if self.resource_stats['nutrient_consumptions']:
            stats['avg_nutrient_consumption'] = torch.stack(self.resource_stats['nutrient_consumptions']).mean()
            stats['max_nutrient_consumption'] = torch.stack(self.resource_stats['nutrient_consumptions']).max()
        
        # Efficiency statistics
        if self.resource_stats['efficiency_scores']:
            stats['avg_efficiency_score'] = torch.stack(self.resource_stats['efficiency_scores']).mean()
            stats['max_efficiency_score'] = torch.stack(self.resource_stats['efficiency_scores']).max()
        
        # Resource utilization statistics
        if self.resource_stats['resource_utilization']:
            stats['avg_resource_utilization'] = torch.stack(self.resource_stats['resource_utilization']).mean()
            stats['max_resource_utilization'] = torch.stack(self.resource_stats['resource_utilization']).max()
        
        return stats
    
    def reset_resource_stats(self):
        """Reset resource statistics."""
        self.resource_stats = {
            'resource_types': [],
            'resource_allocations': [],
            'energy_consumptions': [],
            'nutrient_consumptions': [],
            'efficiency_scores': [],
            'resource_utilization': []
        }
    
    def sample_cell_sentences_resource(
        self,
        counts_raw,
        dataset,
        shared_genes=None,
        valid_gene_mask=None,
        downsample_frac=None,
        gene_names=None
    ):
        """Resource-aware cell sentence sampling."""
        # Get gene names if not provided
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(counts_raw.size(1))]
        
        # Apply resource processing
        counts_processed, resource_state, resource_types, energy_state, nutrient_state, resource_allocation = self.resource_count_processing(
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
        
        # Add resource context information
        enhanced_result = result + (resource_state, resource_types, energy_state, nutrient_state, resource_allocation)
        
        return enhanced_result
    
    def __call__(self, batch):
        """Enhanced collate function with resource normalization."""
        # Call parent collate function
        result = super().__call__(batch)
        
        # Extract resource information
        if len(result) > 8:  # Enhanced result with resource context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums,
                pathway_ids, compartment_ids, cell_type_ids, time_steps
            ) = result
        else:
            # Standard result, create dummy resource context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums
            ) = result
            pathway_ids = None
            compartment_ids = None
            cell_type_ids = None
            time_steps = None
        
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
