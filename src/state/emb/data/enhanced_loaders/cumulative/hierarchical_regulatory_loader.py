"""
Hierarchical Regulatory Loader

Combined loader for HierarchicalRegulatoryModel that provides both
hierarchical and regulatory normalization strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from ..hierarchical_loader import HierarchicalDataLoader
from ..regulatory_loader import RegulatoryDataLoader


class HierarchicalRegulatoryLoader(HierarchicalDataLoader, RegulatoryDataLoader):
    """
    Combined loader for HierarchicalRegulatoryModel.
    Provides both hierarchical and regulatory normalization strategies.
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
        # Regulatory parameters
        interaction_matrix_file: Optional[str] = None,
        gene_type_annotation_file: Optional[str] = None,
        regulatory_strength_threshold: float = 0.5,
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
        
        RegulatoryDataLoader.__init__(
            self, cfg, valid_gene_mask, ds_emb_mapping_inference, is_train, precision,
            interaction_matrix_file=interaction_matrix_file,
            gene_type_annotation_file=gene_type_annotation_file,
            regulatory_strength_threshold=regulatory_strength_threshold,
            **kwargs
        )
        
        # Combined normalization parameters
        self.combined_normalization = True
        
        # Combined statistics
        self.combined_stats = {
            'pathway_regulatory_interactions': [],
            'compartment_regulatory_interactions': [],
            'hierarchical_regulatory_attention': []
        }
    
    def hierarchical_regulatory_count_processing(
        self,
        counts: torch.Tensor,
        gene_names: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Process counts with both hierarchical and regulatory awareness."""
        
        # Apply hierarchical processing
        hierarchical_processed, pathway_ids, compartment_ids = self.pathway_aware_count_processing(
            counts, gene_names
        )
        
        # Apply regulatory processing
        regulatory_processed, gene_types, interaction_mask = self.regulatory_count_processing(
            hierarchical_processed, gene_names
        )
        
        # Combine context
        context = {
            'pathway_ids': pathway_ids,
            'compartment_ids': compartment_ids,
            'gene_types': gene_types,
            'interaction_mask': interaction_mask
        }
        
        # Update combined statistics
        self._update_combined_stats(
            counts, pathway_ids, compartment_ids, gene_types, interaction_mask
        )
        
        return regulatory_processed, context
    
    def _update_combined_stats(
        self,
        counts: torch.Tensor,
        pathway_ids: torch.Tensor,
        compartment_ids: torch.Tensor,
        gene_types: torch.Tensor,
        interaction_mask: torch.Tensor
    ):
        """Update combined hierarchical-regulatory statistics."""
        # Calculate pathway-regulatory interactions
        for pathway_id in range(self.num_pathways):
            pathway_mask = (pathway_ids == pathway_id)
            if pathway_mask.any():
                pathway_gene_types = gene_types[pathway_mask]
                pathway_interactions = interaction_mask[pathway_mask]
                interaction_strength = pathway_interactions.sum() / len(pathway_gene_types)
                self.combined_stats['pathway_regulatory_interactions'].append(interaction_strength.cpu())
        
        # Calculate compartment-regulatory interactions
        for compartment_id in range(self.num_compartments):
            compartment_mask = (compartment_ids == compartment_id)
            if compartment_mask.any():
                compartment_gene_types = gene_types[compartment_mask]
                compartment_interactions = interaction_mask[compartment_mask]
                interaction_strength = compartment_interactions.sum() / len(compartment_gene_types)
                self.combined_stats['compartment_regulatory_interactions'].append(interaction_strength.cpu())
    
    def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined hierarchical-regulatory statistics."""
        stats = {}
        
        # Pathway-regulatory interactions
        if self.combined_stats['pathway_regulatory_interactions']:
            stats['avg_pathway_regulatory_interaction'] = torch.stack(
                self.combined_stats['pathway_regulatory_interactions']
            ).mean()
        
        # Compartment-regulatory interactions
        if self.combined_stats['compartment_regulatory_interactions']:
            stats['avg_compartment_regulatory_interaction'] = torch.stack(
                self.combined_stats['compartment_regulatory_interactions']
            ).mean()
        
        # Combine parent statistics
        hierarchical_stats = self.get_pathway_stats()
        regulatory_stats = self.get_regulatory_stats()
        
        stats.update(hierarchical_stats)
        stats.update(regulatory_stats)
        
        return stats
    
    def reset_combined_stats(self):
        """Reset combined statistics."""
        self.combined_stats = {
            'pathway_regulatory_interactions': [],
            'compartment_regulatory_interactions': [],
            'hierarchical_regulatory_attention': []
        }
        self.reset_pathway_stats()
        self.reset_regulatory_stats()
    
    def sample_cell_sentences_hierarchical_regulatory(
        self,
        counts_raw,
        dataset,
        shared_genes=None,
        valid_gene_mask=None,
        downsample_frac=None,
        gene_names=None
    ):
        """Hierarchical-regulatory cell sentence sampling."""
        # Get gene names if not provided
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(counts_raw.size(1))]
        
        # Apply combined processing
        counts_processed, context = self.hierarchical_regulatory_count_processing(
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
            context['gene_types'],
            context['interaction_mask']
        )
        
        return enhanced_result
    
    def __call__(self, batch):
        """Enhanced collate function with hierarchical-regulatory normalization."""
        # Call parent collate function
        result = super().__call__(batch)
        
        # Extract hierarchical-regulatory information
        if len(result) > 8:  # Enhanced result with hierarchical-regulatory context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums,
                pathway_ids, compartment_ids, cell_type_ids, time_steps
            ) = result
        else:
            # Standard result, create dummy hierarchical-regulatory context
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
        
        # Apply regulatory normalization
        if hasattr(self, 'regulatory_norm'):
            # Generate dummy gene types and interaction mask if not available
            batch_size, seq_len = Xs.shape[:2]
            gene_types = torch.randint(0, 2, (batch_size, seq_len))
            interaction_mask = torch.rand(seq_len, seq_len) > 0.5
            
            Xs = self.regulatory_norm(Xs, gene_types, interaction_mask)
        
        # Return enhanced result
        return (
            batch_sentences, Xs, Ys, idxs, batch_weights, masks,
            total_counts_all, batch_sentences_counts, dataset_nums,
            pathway_ids, compartment_ids, cell_type_ids, time_steps
        )
