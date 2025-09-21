"""
Multi-Scale Data Loader

Enhanced data loader for multi-scale models with
scale-aware normalization and hierarchical processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from .base_enhanced_loader import BaseEnhancedLoader


class MultiScaleNormalization(nn.Module):
    """Multi-scale normalization for hierarchical cellular processing."""
    
    def __init__(
        self,
        d_model: int,
        molecular_scale_dim: int = 128,
        pathway_scale_dim: int = 256,
        cellular_scale_dim: int = 512
    ):
        super().__init__()
        self.d_model = d_model
        self.molecular_scale_dim = molecular_scale_dim
        self.pathway_scale_dim = pathway_scale_dim
        self.cellular_scale_dim = cellular_scale_dim
        
        # Molecular scale processor
        self.molecular_processor = nn.Sequential(
            nn.Linear(d_model, molecular_scale_dim),
            nn.SiLU(),
            nn.Linear(molecular_scale_dim, molecular_scale_dim),
            nn.LayerNorm(molecular_scale_dim)
        )
        
        # Pathway scale processor
        self.pathway_processor = nn.Sequential(
            nn.Linear(d_model, pathway_scale_dim),
            nn.SiLU(),
            nn.Linear(pathway_scale_dim, pathway_scale_dim),
            nn.LayerNorm(pathway_scale_dim)
        )
        
        # Cellular scale processor
        self.cellular_processor = nn.Sequential(
            nn.Linear(d_model, cellular_scale_dim),
            nn.SiLU(),
            nn.Linear(cellular_scale_dim, cellular_scale_dim),
            nn.LayerNorm(cellular_scale_dim)
        )
        
        # Cross-scale attention
        self.cross_scale_attention = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True
        )
        
        # Scale integration
        self.scale_integration = nn.Sequential(
            nn.Linear(molecular_scale_dim + pathway_scale_dim + cellular_scale_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Scale weights
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)
    
    def forward(
        self,
        x: torch.Tensor,
        scale_types: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply multi-scale normalization."""
        batch_size, seq_len, d_model = x.shape
        
        # Process at different scales
        molecular_outputs = []
        pathway_outputs = []
        cellular_outputs = []
        
        for i in range(batch_size):
            for j in range(seq_len):
                scale_type = scale_types[i, j].item()
                
                if scale_type == 0:  # Molecular scale
                    molecular_output = self.molecular_processor(x[i, j])
                    molecular_outputs.append(molecular_output)
                    pathway_outputs.append(torch.zeros(self.pathway_scale_dim, device=x.device))
                    cellular_outputs.append(torch.zeros(self.cellular_scale_dim, device=x.device))
                elif scale_type == 1:  # Pathway scale
                    pathway_output = self.pathway_processor(x[i, j])
                    molecular_outputs.append(torch.zeros(self.molecular_scale_dim, device=x.device))
                    pathway_outputs.append(pathway_output)
                    cellular_outputs.append(torch.zeros(self.cellular_scale_dim, device=x.device))
                elif scale_type == 2:  # Cellular scale
                    cellular_output = self.cellular_processor(x[i, j])
                    molecular_outputs.append(torch.zeros(self.molecular_scale_dim, device=x.device))
                    pathway_outputs.append(torch.zeros(self.pathway_scale_dim, device=x.device))
                    cellular_outputs.append(cellular_output)
        
        # Reshape outputs
        molecular_outputs = torch.stack(molecular_outputs).view(batch_size, seq_len, self.molecular_scale_dim)
        pathway_outputs = torch.stack(pathway_outputs).view(batch_size, seq_len, self.pathway_scale_dim)
        cellular_outputs = torch.stack(cellular_outputs).view(batch_size, seq_len, self.cellular_scale_dim)
        
        # Integrate scales
        integrated_output = torch.cat([molecular_outputs, pathway_outputs, cellular_outputs], dim=-1)
        integrated_output = self.scale_integration(integrated_output)
        
        # Apply cross-scale attention
        attended_output, attention_weights = self.cross_scale_attention(
            integrated_output, integrated_output, integrated_output
        )
        
        # Apply scale weights
        scale_weights = F.softmax(self.scale_weights, dim=0)
        final_output = integrated_output + scale_weights[0] * attended_output
        
        return final_output, molecular_outputs, pathway_outputs, cellular_outputs


class HierarchicalScaleNormalization(nn.Module):
    """Hierarchical scale normalization for nested cellular structures."""
    
    def __init__(self, d_model: int, num_hierarchy_levels: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_hierarchy_levels = num_hierarchy_levels
        
        # Hierarchy level processors
        self.hierarchy_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model)
            ) for _ in range(num_hierarchy_levels)
        ])
        
        # Hierarchy attention
        self.hierarchy_attention = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True
        )
        
        # Hierarchy integration
        self.hierarchy_integration = nn.Sequential(
            nn.Linear(d_model * num_hierarchy_levels, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        hierarchy_levels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply hierarchical scale normalization."""
        batch_size, seq_len, d_model = x.shape
        
        # Process at different hierarchy levels
        hierarchy_outputs = []
        
        for i in range(batch_size):
            for j in range(seq_len):
                hierarchy_level = hierarchy_levels[i, j].item()
                
                if hierarchy_level < self.num_hierarchy_levels:
                    hierarchy_output = self.hierarchy_processors[hierarchy_level](x[i, j])
                    hierarchy_outputs.append(hierarchy_output)
                else:
                    hierarchy_outputs.append(torch.zeros(d_model, device=x.device))
        
        # Reshape outputs
        hierarchy_outputs = torch.stack(hierarchy_outputs).view(batch_size, seq_len, d_model)
        
        # Apply hierarchy attention
        attended_output, attention_weights = self.hierarchy_attention(
            hierarchy_outputs, hierarchy_outputs, hierarchy_outputs
        )
        
        # Integrate hierarchy levels
        integrated_output = torch.cat([hierarchy_outputs, attended_output], dim=-1)
        integrated_output = self.hierarchy_integration(integrated_output)
        
        return integrated_output, attention_weights


class CrossScaleInteractionNormalization(nn.Module):
    """Cross-scale interaction normalization for inter-scale communication."""
    
    def __init__(self, d_model: int, num_scales: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        
        # Cross-scale interaction matrices
        self.cross_scale_matrices = nn.Parameter(
            torch.randn(num_scales, num_scales, d_model, d_model) * 0.1
        )
        
        # Scale interaction attention
        self.scale_interaction_attention = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True
        )
        
        # Interaction integration
        self.interaction_integration = nn.Sequential(
            nn.Linear(d_model * num_scales, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        scale_types: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-scale interaction normalization."""
        batch_size, seq_len, d_model = x.shape
        
        # Compute cross-scale interactions
        scale_interactions = []
        
        for i in range(batch_size):
            for j in range(seq_len):
                scale_type = scale_types[i, j].item()
                
                # Get interactions with other scales
                interactions = []
                for other_scale in range(self.num_scales):
                    if other_scale != scale_type:
                        interaction_matrix = self.cross_scale_matrices[scale_type, other_scale]
                        interaction = torch.matmul(x[i, j], interaction_matrix)
                        interactions.append(interaction)
                
                # Pad interactions if needed
                while len(interactions) < self.num_scales - 1:
                    interactions.append(torch.zeros(d_model, device=x.device))
                
                # Concatenate interactions
                scale_interaction = torch.cat(interactions, dim=-1)
                scale_interactions.append(scale_interaction)
        
        # Reshape interactions
        scale_interactions = torch.stack(scale_interactions).view(batch_size, seq_len, -1)
        
        # Apply interaction attention
        attended_interactions, attention_weights = self.scale_interaction_attention(
            scale_interactions, scale_interactions, scale_interactions
        )
        
        # Integrate interactions
        integrated_output = self.interaction_integration(attended_interactions)
        
        return integrated_output, attention_weights


class MultiScaleDataLoader(BaseEnhancedLoader):
    """Enhanced data loader for multi-scale models."""
    
    def __init__(
        self,
        cfg,
        valid_gene_mask=None,
        ds_emb_mapping_inference=None,
        is_train=True,
        precision=None,
        # Multi-scale specific parameters
        molecular_scale_dim: int = 128,
        pathway_scale_dim: int = 256,
        cellular_scale_dim: int = 512,
        num_hierarchy_levels: int = 3,
        num_scales: int = 3,
        **kwargs
    ):
        super().__init__(
            cfg, valid_gene_mask, ds_emb_mapping_inference, is_train, precision,
            **kwargs
        )
        
        self.molecular_scale_dim = molecular_scale_dim
        self.pathway_scale_dim = pathway_scale_dim
        self.cellular_scale_dim = cellular_scale_dim
        self.num_hierarchy_levels = num_hierarchy_levels
        self.num_scales = num_scales
        
        # Initialize multi-scale normalization modules
        self.multiscale_normalizer = MultiScaleNormalization(
            d_model=512,  # Will be updated
            molecular_scale_dim=molecular_scale_dim,
            pathway_scale_dim=pathway_scale_dim,
            cellular_scale_dim=cellular_scale_dim
        )
        
        self.hierarchical_normalizer = HierarchicalScaleNormalization(
            d_model=512,  # Will be updated
            num_hierarchy_levels=num_hierarchy_levels
        )
        
        self.cross_scale_normalizer = CrossScaleInteractionNormalization(
            d_model=512,  # Will be updated
            num_scales=num_scales
        )
        
        # Multi-scale statistics
        self.multiscale_stats = {
            'scale_types': [],
            'hierarchy_levels': [],
            'molecular_outputs': [],
            'pathway_outputs': [],
            'cellular_outputs': [],
            'cross_scale_interactions': []
        }
    
    def generate_multiscale_context(
        self,
        counts: torch.Tensor,
        gene_names: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate multi-scale context for given counts."""
        batch_size, seq_len = counts.shape
        
        # Generate scale types (0=molecular, 1=pathway, 2=cellular)
        scale_types = torch.randint(0, self.num_scales, (batch_size, seq_len))
        
        # Generate hierarchy levels
        hierarchy_levels = torch.randint(0, self.num_hierarchy_levels, (batch_size, seq_len))
        
        return scale_types, hierarchy_levels
    
    def multiscale_count_processing(
        self,
        counts: torch.Tensor,
        gene_names: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process counts with multi-scale awareness."""
        # Generate multi-scale context
        scale_types, hierarchy_levels = self.generate_multiscale_context(counts, gene_names)
        
        # Apply base normalization
        normalized_counts = self.enhanced_count_processing(counts)
        
        # Reshape for multi-scale processing
        counts_reshaped = normalized_counts.unsqueeze(-1).expand(-1, -1, 512)
        
        # Apply multi-scale normalization
        multiscale_normalized, molecular_outputs, pathway_outputs, cellular_outputs = self.multiscale_normalizer(
            counts_reshaped, scale_types
        )
        
        # Apply hierarchical normalization
        hierarchical_normalized, hierarchy_attention = self.hierarchical_normalizer(
            multiscale_normalized, hierarchy_levels
        )
        
        # Apply cross-scale interaction normalization
        cross_scale_normalized, cross_scale_attention = self.cross_scale_normalizer(
            hierarchical_normalized, scale_types
        )
        
        # Update multi-scale statistics
        self._update_multiscale_stats(
            counts, scale_types, hierarchy_levels, molecular_outputs,
            pathway_outputs, cellular_outputs, cross_scale_attention
        )
        
        return (
            cross_scale_normalized.squeeze(-1), scale_types, hierarchy_levels,
            molecular_outputs, pathway_outputs, cellular_outputs
        )
    
    def _update_multiscale_stats(
        self,
        counts: torch.Tensor,
        scale_types: torch.Tensor,
        hierarchy_levels: torch.Tensor,
        molecular_outputs: torch.Tensor,
        pathway_outputs: torch.Tensor,
        cellular_outputs: torch.Tensor,
        cross_scale_attention: torch.Tensor
    ):
        """Update multi-scale statistics."""
        # Update scale types
        scale_type_counts = torch.bincount(scale_types.flatten(), minlength=self.num_scales)
        self.multiscale_stats['scale_types'].append(scale_type_counts.cpu())
        
        # Update hierarchy levels
        hierarchy_level_counts = torch.bincount(hierarchy_levels.flatten(), minlength=self.num_hierarchy_levels)
        self.multiscale_stats['hierarchy_levels'].append(hierarchy_level_counts.cpu())
        
        # Update molecular outputs
        self.multiscale_stats['molecular_outputs'].append(molecular_outputs.mean().cpu())
        
        # Update pathway outputs
        self.multiscale_stats['pathway_outputs'].append(pathway_outputs.mean().cpu())
        
        # Update cellular outputs
        self.multiscale_stats['cellular_outputs'].append(cellular_outputs.mean().cpu())
        
        # Update cross-scale interactions
        self.multiscale_stats['cross_scale_interactions'].append(cross_scale_attention.mean().cpu())
    
    def get_multiscale_stats(self) -> Dict[str, Any]:
        """Get multi-scale statistics."""
        stats = {}
        
        # Scale type statistics
        if self.multiscale_stats['scale_types']:
            scale_type_counts = torch.stack(self.multiscale_stats['scale_types'])
            stats['scale_type_distribution'] = scale_type_counts.mean(dim=0)
        
        # Hierarchy level statistics
        if self.multiscale_stats['hierarchy_levels']:
            hierarchy_level_counts = torch.stack(self.multiscale_stats['hierarchy_levels'])
            stats['hierarchy_level_distribution'] = hierarchy_level_counts.mean(dim=0)
        
        # Molecular output statistics
        if self.multiscale_stats['molecular_outputs']:
            stats['avg_molecular_output'] = torch.stack(self.multiscale_stats['molecular_outputs']).mean()
            stats['max_molecular_output'] = torch.stack(self.multiscale_stats['molecular_outputs']).max()
        
        # Pathway output statistics
        if self.multiscale_stats['pathway_outputs']:
            stats['avg_pathway_output'] = torch.stack(self.multiscale_stats['pathway_outputs']).mean()
            stats['max_pathway_output'] = torch.stack(self.multiscale_stats['pathway_outputs']).max()
        
        # Cellular output statistics
        if self.multiscale_stats['cellular_outputs']:
            stats['avg_cellular_output'] = torch.stack(self.multiscale_stats['cellular_outputs']).mean()
            stats['max_cellular_output'] = torch.stack(self.multiscale_stats['cellular_outputs']).max()
        
        # Cross-scale interaction statistics
        if self.multiscale_stats['cross_scale_interactions']:
            stats['avg_cross_scale_interaction'] = torch.stack(self.multiscale_stats['cross_scale_interactions']).mean()
            stats['max_cross_scale_interaction'] = torch.stack(self.multiscale_stats['cross_scale_interactions']).max()
        
        return stats
    
    def reset_multiscale_stats(self):
        """Reset multi-scale statistics."""
        self.multiscale_stats = {
            'scale_types': [],
            'hierarchy_levels': [],
            'molecular_outputs': [],
            'pathway_outputs': [],
            'cellular_outputs': [],
            'cross_scale_interactions': []
        }
    
    def sample_cell_sentences_multiscale(
        self,
        counts_raw,
        dataset,
        shared_genes=None,
        valid_gene_mask=None,
        downsample_frac=None,
        gene_names=None
    ):
        """Multi-scale cell sentence sampling."""
        # Get gene names if not provided
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(counts_raw.size(1))]
        
        # Apply multi-scale processing
        counts_processed, scale_types, hierarchy_levels, molecular_outputs, pathway_outputs, cellular_outputs = self.multiscale_count_processing(
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
        
        # Add multi-scale context information
        enhanced_result = result + (scale_types, hierarchy_levels, molecular_outputs, pathway_outputs, cellular_outputs)
        
        return enhanced_result
    
    def __call__(self, batch):
        """Enhanced collate function with multi-scale normalization."""
        # Call parent collate function
        result = super().__call__(batch)
        
        # Extract multi-scale information
        if len(result) > 8:  # Enhanced result with multi-scale context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums,
                pathway_ids, compartment_ids, cell_type_ids, time_steps
            ) = result
        else:
            # Standard result, create dummy multi-scale context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums
            ) = result
            pathway_ids = None
            compartment_ids = None
            cell_type_ids = None
            time_steps = None
        
        # Apply multi-scale normalization
        if hasattr(self, 'multiscale_normalizer'):
            # Generate dummy multi-scale context if not available
            batch_size, seq_len = Xs.shape[:2]
            scale_types = torch.randint(0, self.num_scales, (batch_size, seq_len))
            hierarchy_levels = torch.randint(0, self.num_hierarchy_levels, (batch_size, seq_len))
            
            Xs, molecular_outputs, pathway_outputs, cellular_outputs = self.multiscale_normalizer(
                Xs, scale_types
            )
            
            Xs, hierarchy_attention = self.hierarchical_normalizer(Xs, hierarchy_levels)
            Xs, cross_scale_attention = self.cross_scale_normalizer(Xs, scale_types)
        
        # Return enhanced result
        return (
            batch_sentences, Xs, Ys, idxs, batch_weights, masks,
            total_counts_all, batch_sentences_counts, dataset_nums,
            pathway_ids, compartment_ids, cell_type_ids, time_steps
        )
