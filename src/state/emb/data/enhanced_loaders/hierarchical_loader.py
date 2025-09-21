"""
Hierarchical Data Loader

Enhanced data loader for hierarchical gene organization models with
pathway and compartment-aware normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from .base_enhanced_loader import BaseEnhancedLoader


class PathwayAwareNormalization(nn.Module):
    """Pathway-aware normalization for hierarchical models."""
    
    def __init__(self, d_model: int, num_pathways: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.num_pathways = num_pathways
        
        # Pathway-specific normalization
        self.pathway_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_pathways)
        ])
        
        # Pathway interaction normalization
        self.pathway_interaction_norm = nn.Linear(d_model, d_model)
        
        # Cross-pathway attention normalization
        self.cross_pathway_norm = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True
        )
    
    def forward(
        self,
        x: torch.Tensor,
        pathway_ids: torch.Tensor,
        pathway_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply pathway-aware normalization."""
        batch_size, seq_len, d_model = x.shape
        
        # Apply pathway-specific normalization
        for i in range(batch_size):
            for j in range(seq_len):
                pathway_id = pathway_ids[i, j].item()
                if pathway_id < self.num_pathways:
                    x[i, j] = self.pathway_norms[pathway_id](x[i, j])
        
        # Apply pathway interaction normalization
        x = x + self.pathway_interaction_norm(x)
        
        # Apply cross-pathway attention normalization
        x_attended, _ = self.cross_pathway_norm(x, x, x)
        x = x + x_attended
        
        return x


class CompartmentAwareNormalization(nn.Module):
    """Compartment-aware normalization for hierarchical models."""
    
    def __init__(self, d_model: int, num_compartments: int = 5):
        super().__init__()
        self.d_model = d_model
        self.num_compartments = num_compartments
        
        # Compartment-specific normalization
        self.compartment_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_compartments)
        ])
        
        # Compartment transition normalization
        self.compartment_transition_norm = nn.Linear(d_model, d_model)
        
        # Compartment-specific scaling
        self.compartment_scaling = nn.Parameter(torch.ones(num_compartments))
    
    def forward(
        self,
        x: torch.Tensor,
        compartment_ids: torch.Tensor
    ) -> torch.Tensor:
        """Apply compartment-aware normalization."""
        batch_size, seq_len, d_model = x.shape
        
        # Apply compartment-specific normalization
        for i in range(batch_size):
            for j in range(seq_len):
                compartment_id = compartment_ids[i, j].item()
                if compartment_id < self.num_compartments:
                    x[i, j] = self.compartment_norms[compartment_id](x[i, j])
                    # Apply compartment-specific scaling
                    x[i, j] = x[i, j] * self.compartment_scaling[compartment_id]
        
        # Apply compartment transition normalization
        x = x + self.compartment_transition_norm(x)
        
        return x


class HierarchicalDataLoader(BaseEnhancedLoader):
    """Enhanced data loader for hierarchical gene organization models."""
    
    def __init__(
        self,
        cfg,
        valid_gene_mask=None,
        ds_emb_mapping_inference=None,
        is_train=True,
        precision=None,
        # Hierarchical-specific parameters
        num_pathways: int = 1000,
        num_compartments: int = 5,
        pathway_annotation_file: Optional[str] = None,
        compartment_annotation_file: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            cfg, valid_gene_mask, ds_emb_mapping_inference, is_train, precision,
            biological_normalization=True,
            **kwargs
        )
        
        self.num_pathways = num_pathways
        self.num_compartments = num_compartments
        self.pathway_annotation_file = pathway_annotation_file
        self.compartment_annotation_file = compartment_annotation_file
        
        # Load pathway and compartment annotations
        self.pathway_annotations = self._load_pathway_annotations()
        self.compartment_annotations = self._load_compartment_annotations()
        
        # Initialize hierarchical normalization modules
        self.pathway_normalizer = PathwayAwareNormalization(
            d_model=512,  # Will be updated
            num_pathways=num_pathways
        )
        
        self.compartment_normalizer = CompartmentAwareNormalization(
            d_model=512,  # Will be updated
            num_compartments=num_compartments
        )
        
        # Pathway-specific statistics
        self.pathway_stats = {
            'pathway_means': {},
            'pathway_stds': {},
            'pathway_counts': {}
        }
        
        # Compartment-specific statistics
        self.compartment_stats = {
            'compartment_means': {},
            'compartment_stds': {},
            'compartment_counts': {}
        }
    
    def _load_pathway_annotations(self) -> Optional[Dict[str, int]]:
        """Load pathway annotations from file."""
        if self.pathway_annotation_file is None:
            return None
        
        try:
            # Load pathway annotations (gene_name -> pathway_id)
            pathway_annotations = {}
            with open(self.pathway_annotation_file, 'r') as f:
                for line in f:
                    gene_name, pathway_id = line.strip().split('\t')
                    pathway_annotations[gene_name] = int(pathway_id)
            return pathway_annotations
        except Exception as e:
            print(f"Warning: Could not load pathway annotations: {e}")
            return None
    
    def _load_compartment_annotations(self) -> Optional[Dict[str, int]]:
        """Load compartment annotations from file."""
        if self.compartment_annotation_file is None:
            return None
        
        try:
            # Load compartment annotations (gene_name -> compartment_id)
            compartment_annotations = {}
            with open(self.compartment_annotation_file, 'r') as f:
                for line in f:
                    gene_name, compartment_id = line.strip().split('\t')
                    compartment_annotations[gene_name] = int(compartment_id)
            return compartment_annotations
        except Exception as e:
            print(f"Warning: Could not load compartment annotations: {e}")
            return None
    
    def get_pathway_ids(self, gene_names: List[str]) -> torch.Tensor:
        """Get pathway IDs for given gene names."""
        if self.pathway_annotations is None:
            # Return random pathway IDs if no annotations
            return torch.randint(0, self.num_pathways, (len(gene_names),))
        
        pathway_ids = []
        for gene_name in gene_names:
            pathway_id = self.pathway_annotations.get(gene_name, 0)
            pathway_ids.append(pathway_id)
        
        return torch.tensor(pathway_ids, dtype=torch.long)
    
    def get_compartment_ids(self, gene_names: List[str]) -> torch.Tensor:
        """Get compartment IDs for given gene names."""
        if self.compartment_annotations is None:
            # Return random compartment IDs if no annotations
            return torch.randint(0, self.num_compartments, (len(gene_names),))
        
        compartment_ids = []
        for gene_name in gene_names:
            compartment_id = self.compartment_annotations.get(gene_name, 0)
            compartment_ids.append(compartment_id)
        
        return torch.tensor(compartment_ids, dtype=torch.long)
    
    def pathway_aware_count_processing(
        self,
        counts: torch.Tensor,
        gene_names: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process counts with pathway and compartment awareness."""
        # Get pathway and compartment IDs
        pathway_ids = self.get_pathway_ids(gene_names)
        compartment_ids = self.get_compartment_ids(gene_names)
        
        # Apply base normalization
        normalized_counts = self.enhanced_count_processing(counts)
        
        # Apply pathway-specific normalization
        pathway_normalized = self._apply_pathway_normalization(
            normalized_counts, pathway_ids
        )
        
        # Apply compartment-specific normalization
        compartment_normalized = self._apply_compartment_normalization(
            pathway_normalized, compartment_ids
        )
        
        return compartment_normalized, pathway_ids, compartment_ids
    
    def _apply_pathway_normalization(
        self,
        counts: torch.Tensor,
        pathway_ids: torch.Tensor
    ) -> torch.Tensor:
        """Apply pathway-specific normalization."""
        # Group counts by pathway
        pathway_groups = {}
        for i, pathway_id in enumerate(pathway_ids):
            pathway_id = pathway_id.item()
            if pathway_id not in pathway_groups:
                pathway_groups[pathway_id] = []
            pathway_groups[pathway_id].append(i)
        
        # Apply pathway-specific normalization
        normalized_counts = counts.clone()
        for pathway_id, gene_indices in pathway_groups.items():
            if len(gene_indices) > 1:  # Only normalize if multiple genes in pathway
                pathway_counts = counts[:, gene_indices]
                
                # Pathway-specific statistics
                pathway_mean = pathway_counts.mean(dim=0, keepdim=True)
                pathway_std = pathway_counts.std(dim=0, keepdim=True) + 1e-8
                
                # Normalize within pathway
                pathway_normalized = (pathway_counts - pathway_mean) / pathway_std
                normalized_counts[:, gene_indices] = pathway_normalized
                
                # Update pathway statistics
                self._update_pathway_stats(pathway_id, pathway_mean, pathway_std)
        
        return normalized_counts
    
    def _apply_compartment_normalization(
        self,
        counts: torch.Tensor,
        compartment_ids: torch.Tensor
    ) -> torch.Tensor:
        """Apply compartment-specific normalization."""
        # Group counts by compartment
        compartment_groups = {}
        for i, compartment_id in enumerate(compartment_ids):
            compartment_id = compartment_id.item()
            if compartment_id not in compartment_groups:
                compartment_groups[compartment_id] = []
            compartment_groups[compartment_id].append(i)
        
        # Apply compartment-specific normalization
        normalized_counts = counts.clone()
        for compartment_id, gene_indices in compartment_groups.items():
            if len(gene_indices) > 1:  # Only normalize if multiple genes in compartment
                compartment_counts = counts[:, gene_indices]
                
                # Compartment-specific statistics
                compartment_mean = compartment_counts.mean(dim=0, keepdim=True)
                compartment_std = compartment_counts.std(dim=0, keepdim=True) + 1e-8
                
                # Normalize within compartment
                compartment_normalized = (compartment_counts - compartment_mean) / compartment_std
                normalized_counts[:, gene_indices] = compartment_normalized
                
                # Update compartment statistics
                self._update_compartment_stats(compartment_id, compartment_mean, compartment_std)
        
        return normalized_counts
    
    def _update_pathway_stats(
        self,
        pathway_id: int,
        pathway_mean: torch.Tensor,
        pathway_std: torch.Tensor
    ):
        """Update pathway-specific statistics."""
        if pathway_id not in self.pathway_stats['pathway_means']:
            self.pathway_stats['pathway_means'][pathway_id] = []
            self.pathway_stats['pathway_stds'][pathway_id] = []
            self.pathway_stats['pathway_counts'][pathway_id] = 0
        
        self.pathway_stats['pathway_means'][pathway_id].append(pathway_mean.cpu())
        self.pathway_stats['pathway_stds'][pathway_id].append(pathway_std.cpu())
        self.pathway_stats['pathway_counts'][pathway_id] += 1
    
    def _update_compartment_stats(
        self,
        compartment_id: int,
        compartment_mean: torch.Tensor,
        compartment_std: torch.Tensor
    ):
        """Update compartment-specific statistics."""
        if compartment_id not in self.compartment_stats['compartment_means']:
            self.compartment_stats['compartment_means'][compartment_id] = []
            self.compartment_stats['compartment_stds'][compartment_id] = []
            self.compartment_stats['compartment_counts'][compartment_id] = 0
        
        self.compartment_stats['compartment_means'][compartment_id].append(compartment_mean.cpu())
        self.compartment_stats['compartment_stds'][compartment_id].append(compartment_std.cpu())
        self.compartment_stats['compartment_counts'][compartment_id] += 1
    
    def get_pathway_stats(self) -> Dict[str, Any]:
        """Get pathway-specific statistics."""
        stats = {}
        for pathway_id in self.pathway_stats['pathway_means']:
            if self.pathway_stats['pathway_means'][pathway_id]:
                stats[f'pathway_{pathway_id}_mean'] = torch.stack(
                    self.pathway_stats['pathway_means'][pathway_id]
                ).mean(dim=0)
                stats[f'pathway_{pathway_id}_std'] = torch.stack(
                    self.pathway_stats['pathway_stds'][pathway_id]
                ).mean(dim=0)
                stats[f'pathway_{pathway_id}_count'] = self.pathway_stats['pathway_counts'][pathway_id]
        return stats
    
    def get_compartment_stats(self) -> Dict[str, Any]:
        """Get compartment-specific statistics."""
        stats = {}
        for compartment_id in self.compartment_stats['compartment_means']:
            if self.compartment_stats['compartment_means'][compartment_id]:
                stats[f'compartment_{compartment_id}_mean'] = torch.stack(
                    self.compartment_stats['compartment_means'][compartment_id]
                ).mean(dim=0)
                stats[f'compartment_{compartment_id}_std'] = torch.stack(
                    self.compartment_stats['compartment_stds'][compartment_id]
                ).mean(dim=0)
                stats[f'compartment_{compartment_id}_count'] = self.compartment_stats['compartment_counts'][compartment_id]
        return stats
    
    def sample_cell_sentences_hierarchical(
        self,
        counts_raw,
        dataset,
        shared_genes=None,
        valid_gene_mask=None,
        downsample_frac=None,
        gene_names=None
    ):
        """Hierarchical cell sentence sampling with pathway and compartment awareness."""
        # Get gene names if not provided
        if gene_names is None:
            # Extract gene names from dataset (this would need to be implemented)
            gene_names = [f"gene_{i}" for i in range(counts_raw.size(1))]
        
        # Apply pathway and compartment aware processing
        counts_processed, pathway_ids, compartment_ids = self.pathway_aware_count_processing(
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
        
        # Add hierarchical context information
        enhanced_result = result + (pathway_ids, compartment_ids)
        
        return enhanced_result
    
    def __call__(self, batch):
        """Enhanced collate function with hierarchical normalization."""
        # Call parent collate function
        result = super().__call__(batch)
        
        # Extract hierarchical information
        if len(result) > 8:  # Enhanced result with hierarchical context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums,
                pathway_ids, compartment_ids, cell_type_ids, time_steps
            ) = result
        else:
            # Standard result, create dummy hierarchical context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums
            ) = result
            pathway_ids = None
            compartment_ids = None
            cell_type_ids = None
            time_steps = None
        
        # Apply pathway-aware normalization
        if pathway_ids is not None and hasattr(self, 'pathway_normalizer'):
            Xs = self.pathway_normalizer(Xs, pathway_ids)
        
        # Apply compartment-aware normalization
        if compartment_ids is not None and hasattr(self, 'compartment_normalizer'):
            Xs = self.compartment_normalizer(Xs, compartment_ids)
        
        # Return enhanced result
        return (
            batch_sentences, Xs, Ys, idxs, batch_weights, masks,
            total_counts_all, batch_sentences_counts, dataset_nums,
            pathway_ids, compartment_ids, cell_type_ids, time_steps
        )
