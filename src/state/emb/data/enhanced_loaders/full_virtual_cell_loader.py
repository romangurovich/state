"""
Full Virtual Cell Data Loader

Comprehensive data loader that combines all normalization strategies
for the full virtual cell model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from .base_enhanced_loader import BaseEnhancedLoader
from .hierarchical_loader import HierarchicalDataLoader
from .temporal_loader import TemporalDataLoader


class ComprehensiveNormalization(nn.Module):
    """Comprehensive normalization combining all strategies."""
    
    def __init__(
        self,
        d_model: int,
        num_pathways: int = 1000,
        num_compartments: int = 5,
        time_steps: int = 5,
        memory_dim: int = 512,
        noise_dim: int = 64,
        resource_dim: int = 32
    ):
        super().__init__()
        self.d_model = d_model
        
        # Hierarchical normalization
        self.hierarchical_norm = nn.ModuleDict({
            'pathway_norms': nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(num_pathways)
            ]),
            'compartment_norms': nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(num_compartments)
            ])
        })
        
        # Temporal normalization
        self.temporal_norm = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(time_steps)
        ])
        
        # Memory normalization
        self.memory_norm = nn.Sequential(
            nn.Linear(memory_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Stochastic normalization
        self.stochastic_norm = nn.Sequential(
            nn.Linear(noise_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Resource normalization
        self.resource_norm = nn.Sequential(
            nn.Linear(resource_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Multi-scale normalization
        self.multiscale_norm = nn.ModuleDict({
            'molecular_norm': nn.LayerNorm(d_model),
            'pathway_norm': nn.LayerNorm(d_model),
            'cellular_norm': nn.LayerNorm(d_model)
        })
        
        # Integration normalization
        self.integration_norm = nn.Sequential(
            nn.Linear(d_model * 6, d_model),  # 6 different normalization sources
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Adaptive weighting
        self.adaptive_weights = nn.Parameter(torch.ones(6) / 6)
    
    def forward(
        self,
        x: torch.Tensor,
        pathway_ids: Optional[torch.Tensor] = None,
        compartment_ids: Optional[torch.Tensor] = None,
        time_steps: Optional[torch.Tensor] = None,
        memory_state: Optional[torch.Tensor] = None,
        noise_state: Optional[torch.Tensor] = None,
        resource_state: Optional[torch.Tensor] = None,
        scale_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply comprehensive normalization."""
        batch_size, seq_len, d_model = x.shape
        
        # Initialize normalized representations
        normalized_reprs = []
        
        # 1. Hierarchical normalization
        hierarchical_x = x.clone()
        if pathway_ids is not None:
            for i in range(batch_size):
                for j in range(seq_len):
                    pathway_id = pathway_ids[i, j].item()
                    if pathway_id < len(self.hierarchical_norm['pathway_norms']):
                        hierarchical_x[i, j] = self.hierarchical_norm['pathway_norms'][pathway_id](hierarchical_x[i, j])
        
        if compartment_ids is not None:
            for i in range(batch_size):
                for j in range(seq_len):
                    compartment_id = compartment_ids[i, j].item()
                    if compartment_id < len(self.hierarchical_norm['compartment_norms']):
                        hierarchical_x[i, j] = self.hierarchical_norm['compartment_norms'][compartment_id](hierarchical_x[i, j])
        
        normalized_reprs.append(hierarchical_x)
        
        # 2. Temporal normalization
        temporal_x = x.clone()
        if time_steps is not None:
            for i in range(batch_size):
                for j in range(seq_len):
                    time_step = time_steps[i, j].item()
                    if time_step < len(self.temporal_norm):
                        temporal_x[i, j] = self.temporal_norm[time_step](temporal_x[i, j])
        
        normalized_reprs.append(temporal_x)
        
        # 3. Memory normalization
        memory_x = x.clone()
        if memory_state is not None:
            memory_effect = self.memory_norm(memory_state)
            memory_x = memory_x + memory_effect.unsqueeze(1).expand(-1, seq_len, -1)
        
        normalized_reprs.append(memory_x)
        
        # 4. Stochastic normalization
        stochastic_x = x.clone()
        if noise_state is not None:
            noise_effect = self.stochastic_norm(noise_state)
            stochastic_x = stochastic_x + noise_effect.unsqueeze(1).expand(-1, seq_len, -1)
        
        normalized_reprs.append(stochastic_x)
        
        # 5. Resource normalization
        resource_x = x.clone()
        if resource_state is not None:
            resource_effect = self.resource_norm(resource_state)
            resource_x = resource_x + resource_effect.unsqueeze(1).expand(-1, seq_len, -1)
        
        normalized_reprs.append(resource_x)
        
        # 6. Multi-scale normalization
        multiscale_x = x.clone()
        if scale_type is not None:
            for i in range(batch_size):
                for j in range(seq_len):
                    scale = scale_type[i, j].item()
                    if scale == 0:  # Molecular
                        multiscale_x[i, j] = self.multiscale_norm['molecular_norm'](multiscale_x[i, j])
                    elif scale == 1:  # Pathway
                        multiscale_x[i, j] = self.multiscale_norm['pathway_norm'](multiscale_x[i, j])
                    elif scale == 2:  # Cellular
                        multiscale_x[i, j] = self.multiscale_norm['cellular_norm'](multiscale_x[i, j])
        
        normalized_reprs.append(multiscale_x)
        
        # Apply adaptive weighting
        weighted_reprs = []
        for i, repr in enumerate(normalized_reprs):
            weight = F.softmax(self.adaptive_weights, dim=0)[i]
            weighted_reprs.append(weight * repr)
        
        # Integrate all normalized representations
        integrated_x = torch.cat(weighted_reprs, dim=-1)
        final_x = self.integration_norm(integrated_x)
        
        return final_x


class FullVirtualCellLoader(BaseEnhancedLoader):
    """Comprehensive data loader for the full virtual cell model."""
    
    def __init__(
        self,
        cfg,
        valid_gene_mask=None,
        ds_emb_mapping_inference=None,
        is_train=True,
        precision=None,
        # Comprehensive parameters
        num_pathways: int = 1000,
        num_compartments: int = 5,
        time_steps: int = 5,
        memory_dim: int = 512,
        noise_dim: int = 64,
        resource_dim: int = 32,
        pathway_annotation_file: Optional[str] = None,
        compartment_annotation_file: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            cfg, valid_gene_mask, ds_emb_mapping_inference, is_train, precision,
            biological_normalization=True,
            batch_aware_normalization=True,
            **kwargs
        )
        
        # Comprehensive parameters
        self.num_pathways = num_pathways
        self.num_compartments = num_compartments
        self.time_steps = time_steps
        self.memory_dim = memory_dim
        self.noise_dim = noise_dim
        self.resource_dim = resource_dim
        self.pathway_annotation_file = pathway_annotation_file
        self.compartment_annotation_file = compartment_annotation_file
        
        # Initialize comprehensive normalization
        self.comprehensive_norm = ComprehensiveNormalization(
            d_model=512,  # Will be updated
            num_pathways=num_pathways,
            num_compartments=num_compartments,
            time_steps=time_steps,
            memory_dim=memory_dim,
            noise_dim=noise_dim,
            resource_dim=resource_dim
        )
        
        # Load annotations
        self.pathway_annotations = self._load_pathway_annotations()
        self.compartment_annotations = self._load_compartment_annotations()
        
        # Comprehensive statistics
        self.comprehensive_stats = {
            'pathway_stats': {},
            'compartment_stats': {},
            'temporal_stats': {},
            'memory_stats': {},
            'stochastic_stats': {},
            'resource_stats': {},
            'multiscale_stats': {}
        }
        
        # State tracking
        self.memory_states = []
        self.noise_states = []
        self.resource_states = []
        self.previous_states = []
    
    def _load_pathway_annotations(self) -> Optional[Dict[str, int]]:
        """Load pathway annotations."""
        if self.pathway_annotation_file is None:
            return None
        
        try:
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
        """Load compartment annotations."""
        if self.compartment_annotation_file is None:
            return None
        
        try:
            compartment_annotations = {}
            with open(self.compartment_annotation_file, 'r') as f:
                for line in f:
                    gene_name, compartment_id = line.strip().split('\t')
                    compartment_annotations[gene_name] = int(compartment_id)
            return compartment_annotations
        except Exception as e:
            print(f"Warning: Could not load compartment annotations: {e}")
            return None
    
    def generate_comprehensive_context(
        self,
        counts: torch.Tensor,
        gene_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Generate comprehensive biological context."""
        batch_size, seq_len = counts.shape
        
        # Generate pathway IDs
        if self.pathway_annotations:
            pathway_ids = torch.tensor([
                self.pathway_annotations.get(gene_name, 0) for gene_name in gene_names
            ], dtype=torch.long)
        else:
            pathway_ids = torch.randint(0, self.num_pathways, (seq_len,))
        
        # Generate compartment IDs
        if self.compartment_annotations:
            compartment_ids = torch.tensor([
                self.compartment_annotations.get(gene_name, 0) for gene_name in gene_names
            ], dtype=torch.long)
        else:
            compartment_ids = torch.randint(0, self.num_compartments, (seq_len,))
        
        # Generate time steps
        time_steps = torch.randint(0, self.time_steps, (batch_size, seq_len))
        
        # Generate memory state
        memory_state = torch.randn(batch_size, self.memory_dim)
        
        # Generate noise state
        noise_state = torch.randn(batch_size, self.noise_dim)
        
        # Generate resource state
        resource_state = torch.rand(batch_size, self.resource_dim)
        
        # Generate scale type (0=molecular, 1=pathway, 2=cellular)
        scale_type = torch.randint(0, 3, (batch_size, seq_len))
        
        return {
            'pathway_ids': pathway_ids.unsqueeze(0).expand(batch_size, -1),
            'compartment_ids': compartment_ids.unsqueeze(0).expand(batch_size, -1),
            'time_steps': time_steps,
            'memory_state': memory_state,
            'noise_state': noise_state,
            'resource_state': resource_state,
            'scale_type': scale_type
        }
    
    def comprehensive_count_processing(
        self,
        counts: torch.Tensor,
        gene_names: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Process counts with comprehensive normalization."""
        # Generate comprehensive context
        context = self.generate_comprehensive_context(counts, gene_names)
        
        # Apply base normalization
        normalized_counts = self.enhanced_count_processing(counts)
        
        # Reshape for comprehensive processing
        counts_reshaped = normalized_counts.unsqueeze(-1).expand(-1, -1, 512)
        
        # Apply comprehensive normalization
        comprehensive_normalized = self.comprehensive_norm(
            counts_reshaped,
            context['pathway_ids'],
            context['compartment_ids'],
            context['time_steps'],
            context['memory_state'],
            context['noise_state'],
            context['resource_state'],
            context['scale_type']
        )
        
        # Update comprehensive statistics
        self._update_comprehensive_stats(counts, context)
        
        return comprehensive_normalized.squeeze(-1), context
    
    def _update_comprehensive_stats(
        self,
        counts: torch.Tensor,
        context: Dict[str, torch.Tensor]
    ):
        """Update comprehensive statistics."""
        # Update pathway statistics
        pathway_ids = context['pathway_ids']
        for pathway_id in range(self.num_pathways):
            pathway_mask = (pathway_ids == pathway_id)
            if pathway_mask.any():
                pathway_counts = counts[pathway_mask]
                if pathway_id not in self.comprehensive_stats['pathway_stats']:
                    self.comprehensive_stats['pathway_stats'][pathway_id] = []
                self.comprehensive_stats['pathway_stats'][pathway_id].append(
                    pathway_counts.mean().cpu()
                )
        
        # Update compartment statistics
        compartment_ids = context['compartment_ids']
        for compartment_id in range(self.num_compartments):
            compartment_mask = (compartment_ids == compartment_id)
            if compartment_mask.any():
                compartment_counts = counts[compartment_mask]
                if compartment_id not in self.comprehensive_stats['compartment_stats']:
                    self.comprehensive_stats['compartment_stats'][compartment_id] = []
                self.comprehensive_stats['compartment_stats'][compartment_id].append(
                    compartment_counts.mean().cpu()
                )
        
        # Update temporal statistics
        time_steps = context['time_steps']
        for time_step in range(self.time_steps):
            time_mask = (time_steps == time_step)
            if time_mask.any():
                time_counts = counts[time_mask]
                if time_step not in self.comprehensive_stats['temporal_stats']:
                    self.comprehensive_stats['temporal_stats'][time_step] = []
                self.comprehensive_stats['temporal_stats'][time_step].append(
                    time_counts.mean().cpu()
                )
        
        # Update memory statistics
        memory_state = context['memory_state']
        self.comprehensive_stats['memory_stats'].append(memory_state.mean().cpu())
        
        # Update stochastic statistics
        noise_state = context['noise_state']
        self.comprehensive_stats['stochastic_stats'].append(noise_state.std().cpu())
        
        # Update resource statistics
        resource_state = context['resource_state']
        self.comprehensive_stats['resource_stats'].append(resource_state.mean().cpu())
        
        # Update multiscale statistics
        scale_type = context['scale_type']
        for scale in range(3):
            scale_mask = (scale_type == scale)
            if scale_mask.any():
                scale_counts = counts[scale_mask]
                if scale not in self.comprehensive_stats['multiscale_stats']:
                    self.comprehensive_stats['multiscale_stats'][scale] = []
                self.comprehensive_stats['multiscale_stats'][scale].append(
                    scale_counts.mean().cpu()
                )
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {}
        
        # Pathway statistics
        for pathway_id in self.comprehensive_stats['pathway_stats']:
            if self.comprehensive_stats['pathway_stats'][pathway_id]:
                stats[f'pathway_{pathway_id}_mean'] = torch.stack(
                    self.comprehensive_stats['pathway_stats'][pathway_id]
                ).mean()
        
        # Compartment statistics
        for compartment_id in self.comprehensive_stats['compartment_stats']:
            if self.comprehensive_stats['compartment_stats'][compartment_id]:
                stats[f'compartment_{compartment_id}_mean'] = torch.stack(
                    self.comprehensive_stats['compartment_stats'][compartment_id]
                ).mean()
        
        # Temporal statistics
        for time_step in self.comprehensive_stats['temporal_stats']:
            if self.comprehensive_stats['temporal_stats'][time_step]:
                stats[f'time_step_{time_step}_mean'] = torch.stack(
                    self.comprehensive_stats['temporal_stats'][time_step]
                ).mean()
        
        # Memory statistics
        if self.comprehensive_stats['memory_stats']:
            stats['memory_mean'] = torch.stack(self.comprehensive_stats['memory_stats']).mean()
        
        # Stochastic statistics
        if self.comprehensive_stats['stochastic_stats']:
            stats['noise_std'] = torch.stack(self.comprehensive_stats['stochastic_stats']).mean()
        
        # Resource statistics
        if self.comprehensive_stats['resource_stats']:
            stats['resource_mean'] = torch.stack(self.comprehensive_stats['resource_stats']).mean()
        
        # Multiscale statistics
        for scale in self.comprehensive_stats['multiscale_stats']:
            if self.comprehensive_stats['multiscale_stats'][scale]:
                stats[f'scale_{scale}_mean'] = torch.stack(
                    self.comprehensive_stats['multiscale_stats'][scale]
                ).mean()
        
        return stats
    
    def reset_comprehensive_stats(self):
        """Reset comprehensive statistics."""
        self.comprehensive_stats = {
            'pathway_stats': {},
            'compartment_stats': {},
            'temporal_stats': {},
            'memory_stats': [],
            'stochastic_stats': [],
            'resource_stats': [],
            'multiscale_stats': {}
        }
        self.memory_states = []
        self.noise_states = []
        self.resource_states = []
        self.previous_states = []
    
    def sample_cell_sentences_comprehensive(
        self,
        counts_raw,
        dataset,
        shared_genes=None,
        valid_gene_mask=None,
        downsample_frac=None,
        gene_names=None
    ):
        """Comprehensive cell sentence sampling."""
        # Get gene names if not provided
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(counts_raw.size(1))]
        
        # Apply comprehensive processing
        counts_processed, context = self.comprehensive_count_processing(
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
        
        # Add comprehensive context information
        enhanced_result = result + (
            context['pathway_ids'],
            context['compartment_ids'],
            context['time_steps'],
            context['memory_state'],
            context['noise_state'],
            context['resource_state'],
            context['scale_type']
        )
        
        return enhanced_result
    
    def __call__(self, batch):
        """Enhanced collate function with comprehensive normalization."""
        # Call parent collate function
        result = super().__call__(batch)
        
        # Extract comprehensive information
        if len(result) > 8:  # Enhanced result with comprehensive context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums,
                pathway_ids, compartment_ids, cell_type_ids, time_steps
            ) = result
        else:
            # Standard result, create dummy comprehensive context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums
            ) = result
            pathway_ids = None
            compartment_ids = None
            cell_type_ids = None
            time_steps = None
        
        # Apply comprehensive normalization
        if hasattr(self, 'comprehensive_norm'):
            # Generate dummy context if not available
            if pathway_ids is None:
                batch_size, seq_len = Xs.shape[:2]
                pathway_ids = torch.randint(0, self.num_pathways, (batch_size, seq_len))
                compartment_ids = torch.randint(0, self.num_compartments, (batch_size, seq_len))
                time_steps = torch.randint(0, self.time_steps, (batch_size, seq_len))
                memory_state = torch.randn(batch_size, self.memory_dim)
                noise_state = torch.randn(batch_size, self.noise_dim)
                resource_state = torch.rand(batch_size, self.resource_dim)
                scale_type = torch.randint(0, 3, (batch_size, seq_len))
            else:
                memory_state = None
                noise_state = None
                resource_state = None
                scale_type = None
            
            Xs = self.comprehensive_norm(
                Xs, pathway_ids, compartment_ids, time_steps,
                memory_state, noise_state, resource_state, scale_type
            )
        
        # Return enhanced result
        return (
            batch_sentences, Xs, Ys, idxs, batch_weights, masks,
            total_counts_all, batch_sentences_counts, dataset_nums,
            pathway_ids, compartment_ids, cell_type_ids, time_steps
        )
