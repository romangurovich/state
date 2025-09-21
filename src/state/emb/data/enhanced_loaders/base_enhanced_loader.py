"""
Base Enhanced Data Loader

Provides common normalization functionality and base classes for enhanced data loaders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from ..loader import VCIDatasetSentenceCollator, FilteredGenesCounts
from ... import utils


class NormalizationModule(nn.Module):
    """Base normalization module with various normalization strategies."""
    
    def __init__(
        self,
        normalization_type: str = "z_score",
        per_gene: bool = True,
        per_cell: bool = True,
        robust: bool = False,
        quantile_normalize: bool = False
    ):
        super().__init__()
        self.normalization_type = normalization_type
        self.per_gene = per_gene
        self.per_cell = per_cell
        self.robust = robust
        self.quantile_normalize = quantile_normalize
        
        # Statistics for normalization
        self.register_buffer('gene_mean', None)
        self.register_buffer('gene_std', None)
        self.register_buffer('cell_mean', None)
        self.register_buffer('cell_std', None)
        self.register_buffer('quantiles', None)
        
        self.fitted = False
    
    def fit(self, counts: torch.Tensor):
        """Fit normalization parameters."""
        if self.per_gene:
            if self.robust:
                # Robust statistics (median and MAD)
                self.gene_mean = torch.median(counts, dim=0)[0]
                mad = torch.median(torch.abs(counts - self.gene_mean), dim=0)[0]
                self.gene_std = 1.4826 * mad + 1e-8
            else:
                # Standard statistics
                self.gene_mean = counts.mean(dim=0)
                self.gene_std = counts.std(dim=0) + 1e-8
        
        if self.per_cell:
            if self.robust:
                # Robust statistics for cells
                self.cell_mean = torch.median(counts, dim=1, keepdim=True)[0]
                mad = torch.median(torch.abs(counts - self.cell_mean), dim=1, keepdim=True)[0]
                self.cell_std = 1.4826 * mad + 1e-8
            else:
                # Standard statistics for cells
                self.cell_mean = counts.mean(dim=1, keepdim=True)
                self.cell_std = counts.std(dim=1, keepdim=True) + 1e-8
        
        if self.quantile_normalize:
            # Compute quantiles for quantile normalization
            self.quantiles = torch.quantile(counts, torch.linspace(0, 1, 100, device=counts.device), dim=0)
        
        self.fitted = True
    
    def transform(self, counts: torch.Tensor) -> torch.Tensor:
        """Apply normalization transformation."""
        if not self.fitted:
            raise ValueError("Normalization not fitted. Call fit() first.")
        
        normalized_counts = counts.clone()
        
        if self.per_gene and self.gene_mean is not None:
            normalized_counts = (normalized_counts - self.gene_mean) / self.gene_std
        
        if self.per_cell and self.cell_mean is not None:
            normalized_counts = (normalized_counts - self.cell_mean) / self.cell_std
        
        if self.quantile_normalize and self.quantiles is not None:
            # Quantile normalization
            sorted_indices = torch.argsort(normalized_counts, dim=0)
            sorted_values = torch.gather(normalized_counts, 0, sorted_indices)
            quantile_values = torch.quantile(sorted_values, torch.linspace(0, 1, sorted_values.size(0), device=counts.device), dim=0)
            normalized_counts = torch.gather(quantile_values, 0, torch.argsort(sorted_indices, dim=0))
        
        return normalized_counts
    
    def fit_transform(self, counts: torch.Tensor) -> torch.Tensor:
        """Fit normalization parameters and transform data."""
        self.fit(counts)
        return self.transform(counts)


class BatchAwareNormalization(nn.Module):
    """Batch-aware normalization for handling batch effects."""
    
    def __init__(self, d_model: int, num_batches: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.num_batches = num_batches
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(d_model, affine=True)
        
        # Dataset-specific normalization
        self.dataset_norm = nn.LayerNorm(d_model)
        
        # Batch effect correction
        if num_batches is not None:
            self.batch_correction = nn.Linear(num_batches, d_model)
        else:
            self.batch_correction = None
    
    def forward(self, x: torch.Tensor, batch_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply batch-aware normalization."""
        # Apply batch normalization
        if x.dim() == 3:  # [batch, seq_len, d_model]
            x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        else:  # [batch, d_model]
            x = self.batch_norm(x)
        
        # Apply dataset normalization
        x = self.dataset_norm(x)
        
        # Apply batch correction if batch IDs provided
        if batch_ids is not None and self.batch_correction is not None:
            batch_one_hot = F.one_hot(batch_ids, num_classes=self.num_batches).float()
            batch_effect = self.batch_correction(batch_one_hot)
            x = x + batch_effect
        
        return x


class BiologicalNormalization(nn.Module):
    """Biological context-aware normalization."""
    
    def __init__(
        self,
        d_model: int,
        pathway_dim: Optional[int] = None,
        compartment_dim: Optional[int] = None,
        cell_type_dim: Optional[int] = None
    ):
        super().__init__()
        self.d_model = d_model
        
        # Pathway-specific normalization
        if pathway_dim:
            self.pathway_norm = nn.ModuleDict({
                f'pathway_{i}': nn.LayerNorm(d_model) 
                for i in range(pathway_dim)
            })
        
        # Compartment-specific normalization
        if compartment_dim:
            self.compartment_norm = nn.ModuleDict({
                f'compartment_{i}': nn.LayerNorm(d_model) 
                for i in range(compartment_dim)
            })
        
        # Cell type-specific normalization
        if cell_type_dim:
            self.cell_type_norm = nn.ModuleDict({
                f'cell_type_{i}': nn.LayerNorm(d_model) 
                for i in range(cell_type_dim)
            })
        
        # Adaptive normalization
        self.adaptive_norm = nn.Linear(d_model, d_model)
        self.norm_gate = nn.Sigmoid()
    
    def forward(
        self,
        x: torch.Tensor,
        pathway_ids: Optional[torch.Tensor] = None,
        compartment_ids: Optional[torch.Tensor] = None,
        cell_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply biological context-aware normalization."""
        # Apply pathway-specific normalization
        if pathway_ids is not None and hasattr(self, 'pathway_norm'):
            for i in range(x.size(0)):
                pathway_id = pathway_ids[i].item()
                if f'pathway_{pathway_id}' in self.pathway_norm:
                    x[i] = self.pathway_norm[f'pathway_{pathway_id}'](x[i])
        
        # Apply compartment-specific normalization
        if compartment_ids is not None and hasattr(self, 'compartment_norm'):
            for i in range(x.size(0)):
                compartment_id = compartment_ids[i].item()
                if f'compartment_{compartment_id}' in self.compartment_norm:
                    x[i] = self.compartment_norm[f'compartment_{compartment_id}'](x[i])
        
        # Apply cell type-specific normalization
        if cell_type_ids is not None and hasattr(self, 'cell_type_norm'):
            for i in range(x.size(0)):
                cell_type_id = cell_type_ids[i].item()
                if f'cell_type_{cell_type_id}' in self.cell_type_norm:
                    x[i] = self.cell_type_norm[f'cell_type_{cell_type_id}'](x[i])
        
        # Adaptive normalization
        gate = self.norm_gate(self.adaptive_norm(x))
        x = x * gate + x * (1 - gate)  # Residual connection
        
        return x


class BaseEnhancedLoader(VCIDatasetSentenceCollator):
    """Base enhanced data loader with common normalization functionality."""
    
    def __init__(
        self,
        cfg,
        valid_gene_mask=None,
        ds_emb_mapping_inference=None,
        is_train=True,
        precision=None,
        # Enhanced normalization parameters
        normalization_type: str = "z_score",
        per_gene_normalization: bool = True,
        per_cell_normalization: bool = True,
        robust_normalization: bool = False,
        quantile_normalization: bool = False,
        batch_aware_normalization: bool = True,
        biological_normalization: bool = True,
        **kwargs
    ):
        super().__init__(
            cfg, valid_gene_mask, ds_emb_mapping_inference, is_train, precision
        )
        
        # Enhanced normalization parameters
        self.normalization_type = normalization_type
        self.per_gene_normalization = per_gene_normalization
        self.per_cell_normalization = per_cell_normalization
        self.robust_normalization = robust_normalization
        self.quantile_normalization = quantile_normalization
        self.batch_aware_normalization = batch_aware_normalization
        self.biological_normalization = biological_normalization
        
        # Initialize normalization modules
        self.count_normalizer = NormalizationModule(
            normalization_type=normalization_type,
            per_gene=per_gene_normalization,
            per_cell=per_cell_normalization,
            robust=robust_normalization,
            quantile_normalize=quantile_normalization
        )
        
        # Batch-aware normalization
        if batch_aware_normalization:
            num_datasets = len(self.dataset_to_protein_embeddings) if hasattr(self, 'dataset_to_protein_embeddings') else None
            self.batch_normalizer = BatchAwareNormalization(
                d_model=512,  # Default d_model, will be updated
                num_batches=num_datasets
            )
        
        # Biological normalization
        if biological_normalization:
            self.biological_normalizer = BiologicalNormalization(
                d_model=512,  # Default d_model, will be updated
                pathway_dim=kwargs.get('pathway_dim', None),
                compartment_dim=kwargs.get('compartment_dim', None),
                cell_type_dim=kwargs.get('cell_type_dim', None)
            )
        
        # Statistics tracking
        self.normalization_stats = {
            'gene_means': [],
            'gene_stds': [],
            'cell_means': [],
            'cell_stds': [],
            'quantiles': []
        }
    
    def enhanced_count_processing(self, counts: torch.Tensor) -> torch.Tensor:
        """Enhanced count processing with multiple normalization strategies."""
        # Store original counts for reference
        original_counts = counts.clone()
        
        # Apply log1p transformation if needed
        if self.is_raw_integer_counts(counts):
            counts = torch.log1p(counts)
        
        # Apply normalization
        if not self.count_normalizer.fitted:
            counts = self.count_normalizer.fit_transform(counts)
        else:
            counts = self.count_normalizer.transform(counts)
        
        # Store normalization statistics
        self._update_normalization_stats(original_counts, counts)
        
        return counts
    
    def _update_normalization_stats(self, original_counts: torch.Tensor, normalized_counts: torch.Tensor):
        """Update normalization statistics."""
        if self.per_gene_normalization:
            self.normalization_stats['gene_means'].append(original_counts.mean(dim=0).cpu())
            self.normalization_stats['gene_stds'].append(original_counts.std(dim=0).cpu())
        
        if self.per_cell_normalization:
            self.normalization_stats['cell_means'].append(original_counts.mean(dim=1).cpu())
            self.normalization_stats['cell_stds'].append(original_counts.std(dim=1).cpu())
        
        if self.quantile_normalization:
            self.normalization_stats['quantiles'].append(
                torch.quantile(original_counts, torch.linspace(0, 1, 100), dim=0).cpu()
            )
    
    def get_normalization_stats(self) -> Dict[str, torch.Tensor]:
        """Get normalization statistics."""
        stats = {}
        
        if self.normalization_stats['gene_means']:
            stats['gene_mean'] = torch.stack(self.normalization_stats['gene_means']).mean(dim=0)
            stats['gene_std'] = torch.stack(self.normalization_stats['gene_stds']).mean(dim=0)
        
        if self.normalization_stats['cell_means']:
            stats['cell_mean'] = torch.stack(self.normalization_stats['cell_means']).mean(dim=0)
            stats['cell_std'] = torch.stack(self.normalization_stats['cell_stds']).mean(dim=0)
        
        if self.normalization_stats['quantiles']:
            stats['quantiles'] = torch.stack(self.normalization_stats['quantiles']).mean(dim=0)
        
        return stats
    
    def reset_normalization_stats(self):
        """Reset normalization statistics."""
        for key in self.normalization_stats:
            self.normalization_stats[key] = []
    
    def sample_cell_sentences_enhanced(
        self,
        counts_raw,
        dataset,
        shared_genes=None,
        valid_gene_mask=None,
        downsample_frac=None,
        pathway_ids=None,
        compartment_ids=None,
        cell_type_ids=None,
        time_steps=None
    ):
        """Enhanced cell sentence sampling with biological context."""
        # Apply enhanced count processing
        counts_processed = self.enhanced_count_processing(counts_raw)
        
        # Call original sampling method with processed counts
        result = self.sample_cell_sentences(
            counts_processed,
            dataset,
            shared_genes,
            valid_gene_mask,
            downsample_frac
        )
        
        # Add biological context information
        enhanced_result = result + (
            pathway_ids,
            compartment_ids,
            cell_type_ids,
            time_steps
        )
        
        return enhanced_result
    
    def __call__(self, batch):
        """Enhanced collate function with normalization."""
        # Call parent collate function
        result = super().__call__(batch)
        
        # Extract additional information if available
        if len(result) > 8:  # Enhanced result with biological context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums,
                pathway_ids, compartment_ids, cell_type_ids, time_steps
            ) = result
        else:
            # Standard result, create dummy biological context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums
            ) = result
            pathway_ids = None
            compartment_ids = None
            cell_type_ids = None
            time_steps = None
        
        # Apply batch-aware normalization if enabled
        if self.batch_aware_normalization and hasattr(self, 'batch_normalizer'):
            Xs = self.batch_normalizer(Xs, dataset_nums)
        
        # Apply biological normalization if enabled
        if self.biological_normalization and hasattr(self, 'biological_normalizer'):
            Xs = self.biological_normalizer(
                Xs, pathway_ids, compartment_ids, cell_type_ids
            )
        
        # Return enhanced result
        return (
            batch_sentences, Xs, Ys, idxs, batch_weights, masks,
            total_counts_all, batch_sentences_counts, dataset_nums,
            pathway_ids, compartment_ids, cell_type_ids, time_steps
        )
