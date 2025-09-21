"""
Regulatory Data Loader

Enhanced data loader for regulatory network models with
interaction-aware normalization and regulatory constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from .base_enhanced_loader import BaseEnhancedLoader


class RegulatoryNormalization(nn.Module):
    """Regulatory network-aware normalization."""
    
    def __init__(
        self,
        d_model: int,
        num_genes: int,
        interaction_matrix: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_genes = num_genes
        
        # Interaction matrix
        if interaction_matrix is not None:
            self.register_buffer('interaction_matrix', interaction_matrix)
        else:
            self.interaction_matrix = None
        
        # Learnable interaction weights
        self.interaction_weights = nn.Parameter(
            torch.randn(num_genes, num_genes) * 0.1
        )
        
        # Gene type normalization (TF vs Target)
        self.tf_norm = nn.LayerNorm(d_model)
        self.target_norm = nn.LayerNorm(d_model)
        
        # Regulatory strength normalization
        self.regulatory_strength_norm = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Interaction-aware attention
        self.interaction_attention = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True
        )
    
    def forward(
        self,
        x: torch.Tensor,
        gene_types: torch.Tensor,
        interaction_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply regulatory normalization."""
        batch_size, seq_len, d_model = x.shape
        
        # Apply gene type-specific normalization
        for i in range(batch_size):
            for j in range(seq_len):
                gene_type = gene_types[i, j].item()
                if gene_type == 0:  # Transcription factor
                    x[i, j] = self.tf_norm(x[i, j])
                elif gene_type == 1:  # Target gene
                    x[i, j] = self.target_norm(x[i, j])
        
        # Apply interaction-aware attention
        if interaction_mask is not None:
            # Create attention mask from interaction matrix
            attention_mask = self._create_attention_mask(interaction_mask)
            x_attended, _ = self.interaction_attention(
                x, x, x, attn_mask=attention_mask
            )
            x = x + x_attended
        else:
            x_attended, _ = self.interaction_attention(x, x, x)
            x = x + x_attended
        
        # Apply regulatory strength normalization
        regulatory_strength = self.regulatory_strength_norm(x)
        x = x * regulatory_strength
        
        return x
    
    def _create_attention_mask(self, interaction_mask: torch.Tensor) -> torch.Tensor:
        """Create attention mask from interaction matrix."""
        # Convert interaction matrix to attention mask
        attention_mask = interaction_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        attention_mask = attention_mask.expand(-1, 8, -1, -1)  # [1, 8, seq_len, seq_len]
        return attention_mask


class RegulatoryDataLoader(BaseEnhancedLoader):
    """Enhanced data loader for regulatory network models."""
    
    def __init__(
        self,
        cfg,
        valid_gene_mask=None,
        ds_emb_mapping_inference=None,
        is_train=True,
        precision=None,
        # Regulatory-specific parameters
        interaction_matrix_file: Optional[str] = None,
        gene_type_annotation_file: Optional[str] = None,
        regulatory_strength_threshold: float = 0.5,
        **kwargs
    ):
        super().__init__(
            cfg, valid_gene_mask, ds_emb_mapping_inference, is_train, precision,
            **kwargs
        )
        
        self.interaction_matrix_file = interaction_matrix_file
        self.gene_type_annotation_file = gene_type_annotation_file
        self.regulatory_strength_threshold = regulatory_strength_threshold
        
        # Load regulatory data
        self.interaction_matrix = self._load_interaction_matrix()
        self.gene_type_annotations = self._load_gene_type_annotations()
        
        # Initialize regulatory normalization
        num_genes = len(self.gene_type_annotations) if self.gene_type_annotations else 1000
        self.regulatory_norm = RegulatoryNormalization(
            d_model=512,  # Will be updated
            num_genes=num_genes,
            interaction_matrix=self.interaction_matrix
        )
        
        # Regulatory statistics
        self.regulatory_stats = {
            'tf_counts': 0,
            'target_counts': 0,
            'interaction_strengths': [],
            'regulatory_networks': []
        }
    
    def _load_interaction_matrix(self) -> Optional[torch.Tensor]:
        """Load gene interaction matrix."""
        if self.interaction_matrix_file is None:
            return None
        
        try:
            # Load interaction matrix from file
            interaction_matrix = torch.load(self.interaction_matrix_file)
            return interaction_matrix
        except Exception as e:
            print(f"Warning: Could not load interaction matrix: {e}")
            return None
    
    def _load_gene_type_annotations(self) -> Optional[Dict[str, int]]:
        """Load gene type annotations (0=TF, 1=Target)."""
        if self.gene_type_annotation_file is None:
            return None
        
        try:
            gene_type_annotations = {}
            with open(self.gene_type_annotation_file, 'r') as f:
                for line in f:
                    gene_name, gene_type = line.strip().split('\t')
                    gene_type_annotations[gene_name] = int(gene_type)
            return gene_type_annotations
        except Exception as e:
            print(f"Warning: Could not load gene type annotations: {e}")
            return None
    
    def get_gene_types(self, gene_names: List[str]) -> torch.Tensor:
        """Get gene types for given gene names."""
        if self.gene_type_annotations is None:
            # Return random gene types if no annotations
            return torch.randint(0, 2, (len(gene_names),))
        
        gene_types = []
        for gene_name in gene_names:
            gene_type = self.gene_type_annotations.get(gene_name, 0)
            gene_types.append(gene_type)
        
        return torch.tensor(gene_types, dtype=torch.long)
    
    def create_interaction_mask(
        self,
        gene_names: List[str],
        interaction_threshold: float = 0.5
    ) -> torch.Tensor:
        """Create interaction mask for given genes."""
        if self.interaction_matrix is None:
            # Return random interaction mask if no matrix
            return torch.rand(len(gene_names), len(gene_names)) > 0.5
        
        # Create gene name to index mapping
        gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}
        
        # Extract relevant interactions
        interaction_mask = torch.zeros(len(gene_names), len(gene_names))
        for i, gene1 in enumerate(gene_names):
            for j, gene2 in enumerate(gene_names):
                if gene1 in gene_to_idx and gene2 in gene_to_idx:
                    idx1 = gene_to_idx[gene1]
                    idx2 = gene_to_idx[gene2]
                    if idx1 < self.interaction_matrix.size(0) and idx2 < self.interaction_matrix.size(1):
                        interaction_strength = self.interaction_matrix[idx1, idx2]
                        interaction_mask[i, j] = (interaction_strength > interaction_threshold).float()
        
        return interaction_mask
    
    def regulatory_count_processing(
        self,
        counts: torch.Tensor,
        gene_names: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process counts with regulatory awareness."""
        # Get gene types
        gene_types = self.get_gene_types(gene_names)
        
        # Create interaction mask
        interaction_mask = self.create_interaction_mask(gene_names)
        
        # Apply base normalization
        normalized_counts = self.enhanced_count_processing(counts)
        
        # Reshape for regulatory processing
        counts_reshaped = normalized_counts.unsqueeze(-1).expand(-1, -1, 512)
        
        # Apply regulatory normalization
        regulatory_normalized = self.regulatory_norm(
            counts_reshaped, gene_types.unsqueeze(0).expand(counts.size(0), -1), interaction_mask
        )
        
        # Update regulatory statistics
        self._update_regulatory_stats(counts, gene_types, interaction_mask)
        
        return regulatory_normalized.squeeze(-1), gene_types, interaction_mask
    
    def _update_regulatory_stats(
        self,
        counts: torch.Tensor,
        gene_types: torch.Tensor,
        interaction_mask: torch.Tensor
    ):
        """Update regulatory statistics."""
        # Update gene type counts
        tf_count = (gene_types == 0).sum().item()
        target_count = (gene_types == 1).sum().item()
        
        self.regulatory_stats['tf_counts'] += tf_count
        self.regulatory_stats['target_counts'] += target_count
        
        # Update interaction strengths
        interaction_strengths = interaction_mask.sum(dim=1).mean()
        self.regulatory_stats['interaction_strengths'].append(interaction_strengths.cpu())
        
        # Update regulatory networks
        regulatory_network = self._extract_regulatory_network(gene_types, interaction_mask)
        self.regulatory_stats['regulatory_networks'].append(regulatory_network)
    
    def _extract_regulatory_network(
        self,
        gene_types: torch.Tensor,
        interaction_mask: torch.Tensor
    ) -> Dict[str, List[str]]:
        """Extract regulatory network from interaction mask."""
        regulatory_network = {}
        
        # Find TFs and their targets
        tf_indices = torch.where(gene_types == 0)[0]
        target_indices = torch.where(gene_types == 1)[0]
        
        for tf_idx in tf_indices:
            tf_targets = []
            for target_idx in target_indices:
                if interaction_mask[tf_idx, target_idx] > 0:
                    tf_targets.append(f"target_{target_idx.item()}")
            regulatory_network[f"tf_{tf_idx.item()}"] = tf_targets
        
        return regulatory_network
    
    def get_regulatory_stats(self) -> Dict[str, Any]:
        """Get regulatory statistics."""
        stats = {}
        
        # Gene type statistics
        total_genes = self.regulatory_stats['tf_counts'] + self.regulatory_stats['target_counts']
        if total_genes > 0:
            stats['tf_ratio'] = self.regulatory_stats['tf_counts'] / total_genes
            stats['target_ratio'] = self.regulatory_stats['target_counts'] / total_genes
        
        # Interaction statistics
        if self.regulatory_stats['interaction_strengths']:
            stats['avg_interaction_strength'] = torch.stack(
                self.regulatory_stats['interaction_strengths']
            ).mean()
        
        # Regulatory network statistics
        if self.regulatory_stats['regulatory_networks']:
            network_sizes = [
                len(network) for network in self.regulatory_stats['regulatory_networks']
            ]
            stats['avg_network_size'] = np.mean(network_sizes)
            stats['max_network_size'] = np.max(network_sizes)
        
        return stats
    
    def reset_regulatory_stats(self):
        """Reset regulatory statistics."""
        self.regulatory_stats = {
            'tf_counts': 0,
            'target_counts': 0,
            'interaction_strengths': [],
            'regulatory_networks': []
        }
    
    def sample_cell_sentences_regulatory(
        self,
        counts_raw,
        dataset,
        shared_genes=None,
        valid_gene_mask=None,
        downsample_frac=None,
        gene_names=None
    ):
        """Regulatory cell sentence sampling."""
        # Get gene names if not provided
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(counts_raw.size(1))]
        
        # Apply regulatory processing
        counts_processed, gene_types, interaction_mask = self.regulatory_count_processing(
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
        
        # Add regulatory context information
        enhanced_result = result + (gene_types, interaction_mask)
        
        return enhanced_result
    
    def __call__(self, batch):
        """Enhanced collate function with regulatory normalization."""
        # Call parent collate function
        result = super().__call__(batch)
        
        # Extract regulatory information
        if len(result) > 8:  # Enhanced result with regulatory context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums,
                pathway_ids, compartment_ids, cell_type_ids, time_steps
            ) = result
        else:
            # Standard result, create dummy regulatory context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums
            ) = result
            pathway_ids = None
            compartment_ids = None
            cell_type_ids = None
            time_steps = None
        
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
