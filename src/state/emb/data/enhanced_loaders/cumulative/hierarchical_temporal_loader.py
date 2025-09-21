"""
Hierarchical Temporal Loader

Combined loader for HierarchicalTemporalModel that provides both
hierarchical and temporal normalization strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from ..hierarchical_loader import HierarchicalDataLoader
from ..temporal_loader import TemporalDataLoader


class HierarchicalTemporalLoader(HierarchicalDataLoader, TemporalDataLoader):
    """
    Combined loader for HierarchicalTemporalModel.
    Provides both hierarchical and temporal normalization strategies.
    """
    
    def __init__(
        self,
        cfg,
        valid_gene_mask=None,
        ds_emb_mapping_inference=None,
        is_train=True,
        precision=None,
        # Hierarchical parameters
        num_pathways: int = 1000,
        num_compartments: int = 5,
        pathway_annotation_file: Optional[str] = None,
        compartment_annotation_file: Optional[str] = None,
        # Temporal parameters
        time_steps: int = 5,
        max_sequence_length: int = 100,
        temporal_sampling_strategy: str = "uniform",
        **kwargs
    ):
        # Initialize both parent classes
        HierarchicalDataLoader.__init__(
            self, cfg, valid_gene_mask, ds_emb_mapping_inference, is_train, precision,
            num_pathways=num_pathways,
            num_compartments=num_compartments,
            pathway_annotation_file=pathway_annotation_file,
            compartment_annotation_file=compartment_annotation_file,
            **kwargs
        )
        
        TemporalDataLoader.__init__(
            self, cfg, valid_gene_mask, ds_emb_mapping_inference, is_train, precision,
            time_steps=time_steps,
            max_sequence_length=max_sequence_length,
            temporal_sampling_strategy=temporal_sampling_strategy,
            **kwargs
        )
        
        # Combined normalization parameters
        self.combined_normalization = True
        
        # Combined statistics
        self.combined_stats = {
            'pathway_temporal_correlations': [],
            'compartment_temporal_correlations': [],
            'hierarchical_temporal_attention': []
        }
    
    def hierarchical_temporal_count_processing(
        self,
        counts: torch.Tensor,
        gene_names: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Process counts with both hierarchical and temporal awareness."""
        
        # Apply hierarchical processing
        hierarchical_processed, pathway_ids, compartment_ids = self.pathway_aware_count_processing(
            counts, gene_names
        )
        
        # Apply temporal processing
        temporal_processed, time_steps, response_types = self.temporal_count_processing(
            hierarchical_processed, gene_names
        )
        
        # Combine context
        context = {
            'pathway_ids': pathway_ids,
            'compartment_ids': compartment_ids,
            'time_steps': time_steps,
            'response_types': response_types
        }
        
        # Update combined statistics
        self._update_combined_stats(
            counts, pathway_ids, compartment_ids, time_steps, response_types
        )
        
        return temporal_processed, context
    
    def _update_combined_stats(
        self,
        counts: torch.Tensor,
        pathway_ids: torch.Tensor,
        compartment_ids: torch.Tensor,
        time_steps: torch.Tensor,
        response_types: torch.Tensor
    ):
        """Update combined hierarchical-temporal statistics."""
        # Calculate pathway-temporal correlations
        for pathway_id in range(self.num_pathways):
            pathway_mask = (pathway_ids == pathway_id)
            if pathway_mask.any():
                pathway_time_steps = time_steps[pathway_mask]
                pathway_counts = counts[pathway_mask]
                try:
                    correlation = torch.corrcoef(torch.stack([
                        pathway_time_steps.float(), pathway_counts.float()
                    ]))[0, 1]
                    self.combined_stats['pathway_temporal_correlations'].append(correlation.cpu())
                except:
                    pass  # Skip if correlation cannot be calculated
        
        # Calculate compartment-temporal correlations
        for compartment_id in range(self.num_compartments):
            compartment_mask = (compartment_ids == compartment_id)
            if compartment_mask.any():
                compartment_time_steps = time_steps[compartment_mask]
                compartment_counts = counts[compartment_mask]
                try:
                    correlation = torch.corrcoef(torch.stack([
                        compartment_time_steps.float(), compartment_counts.float()
                    ]))[0, 1]
                    self.combined_stats['compartment_temporal_correlations'].append(correlation.cpu())
                except:
                    pass  # Skip if correlation cannot be calculated
    
    def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined hierarchical-temporal statistics."""
        stats = {}
        
        # Pathway-temporal correlations
        if self.combined_stats['pathway_temporal_correlations']:
            stats['avg_pathway_temporal_correlation'] = torch.stack(
                self.combined_stats['pathway_temporal_correlations']
            ).mean()
        
        # Compartment-temporal correlations
        if self.combined_stats['compartment_temporal_correlations']:
            stats['avg_compartment_temporal_correlation'] = torch.stack(
                self.combined_stats['compartment_temporal_correlations']
            ).mean()
        
        # Combine parent statistics
        hierarchical_stats = self.get_pathway_stats()
        temporal_stats = self.get_temporal_stats()
        
        stats.update(hierarchical_stats)
        stats.update(temporal_stats)
        
        return stats
    
    def reset_combined_stats(self):
        """Reset combined statistics."""
        self.combined_stats = {
            'pathway_temporal_correlations': [],
            'compartment_temporal_correlations': [],
            'hierarchical_temporal_attention': []
        }
        self.reset_pathway_stats()
        self.reset_temporal_stats()
    
    def sample_cell_sentences_hierarchical_temporal(
        self,
        counts_raw,
        dataset,
        shared_genes=None,
        valid_gene_mask=None,
        downsample_frac=None,
        gene_names=None
    ):
        """Hierarchical-temporal cell sentence sampling."""
        # Get gene names if not provided
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(counts_raw.size(1))]
        
        # Apply combined processing
        counts_processed, context = self.hierarchical_temporal_count_processing(
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
            context['pathway_ids'],
            context['compartment_ids'],
            context['time_steps'],
            context['response_types']
        )
        
        return enhanced_result
    
    def __call__(self, batch):
        """Enhanced collate function with hierarchical-temporal normalization."""
        # Call parent collate function
        result = super().__call__(batch)
        
        # Extract hierarchical-temporal information
        if len(result) > 8:  # Enhanced result with hierarchical-temporal context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums,
                pathway_ids, compartment_ids, cell_type_ids, time_steps
            ) = result
        else:
            # Standard result, create dummy hierarchical-temporal context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums
            ) = result
            pathway_ids = None
            compartment_ids = None
            cell_type_ids = None
            time_steps = None
        
        # Apply hierarchical normalization
        if pathway_ids is not None and hasattr(self, 'pathway_normalizer'):
            Xs = self.pathway_normalizer(Xs, pathway_ids)
        
        # Apply compartment normalization
        if compartment_ids is not None and hasattr(self, 'compartment_normalizer'):
            Xs = self.compartment_normalizer(Xs, compartment_ids)
        
        # Apply temporal normalization
        if time_steps is not None and hasattr(self, 'temporal_normalizer'):
            Xs = self.temporal_normalizer(Xs, time_steps)
        
        # Apply fast/slow response normalization
        if hasattr(self, 'fast_slow_normalizer'):
            Xs, response_types = self.fast_slow_normalizer(Xs)
        
        # Apply feedback loop normalization
        if hasattr(self, 'feedback_normalizer'):
            Xs = self.feedback_normalizer(Xs)
        
        # Return enhanced result
        return (
            batch_sentences, Xs, Ys, idxs, batch_weights, masks,
            total_counts_all, batch_sentences_counts, dataset_nums,
            pathway_ids, compartment_ids, cell_type_ids, time_steps
        )
