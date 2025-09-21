"""
Temporal Memory Loader

Combined loader for TemporalMemoryModel that provides both
temporal and memory normalization strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from ..temporal_loader import TemporalDataLoader
from ..memory_loader import MemoryDataLoader


class TemporalMemoryLoader(TemporalDataLoader, MemoryDataLoader):
    """
    Combined loader for TemporalMemoryModel.
    Provides both temporal and memory normalization strategies.
    """
    
    def __init__(
        self,
        cfg,
        valid_gene_mask=None,
        ds_emb_mapping_inference=None,
        is_train=True,
        precision=None,
        # Temporal parameters
        time_steps: int = 5,
        max_sequence_length: int = 100,
        temporal_sampling_strategy: str = "uniform",
        # Memory parameters
        memory_dim: int = 512,
        memory_size: int = 1000,
        num_memory_types: int = 5,
        num_epigenetic_states: int = 3,
        memory_decay_rate: float = 0.9,
        **kwargs
    ):
        # Initialize both parent classes
        TemporalDataLoader.__init__(
            self, cfg, valid_gene_mask, ds_emb_mapping_inference, is_train, precision,
            time_steps=time_steps,
            max_sequence_length=max_sequence_length,
            temporal_sampling_strategy=temporal_sampling_strategy,
            **kwargs
        )
        
        MemoryDataLoader.__init__(
            self, cfg, valid_gene_mask, ds_emb_mapping_inference, is_train, precision,
            memory_dim=memory_dim,
            memory_size=memory_size,
            num_memory_types=num_memory_types,
            num_epigenetic_states=num_epigenetic_states,
            memory_decay_rate=memory_decay_rate,
            **kwargs
        )
        
        # Combined normalization parameters
        self.combined_normalization = True
        
        # Combined statistics
        self.combined_stats = {
            'temporal_memory_correlations': [],
            'memory_temporal_attention': [],
            'epigenetic_temporal_states': []
        }
    
    def temporal_memory_count_processing(
        self,
        counts: torch.Tensor,
        gene_names: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Process counts with both temporal and memory awareness."""
        
        # Apply temporal processing
        temporal_processed, time_steps, response_types = self.temporal_count_processing(
            counts, gene_names
        )
        
        # Apply memory processing
        memory_processed, memory_types, epigenetic_states, memory_state = self.memory_count_processing(
            temporal_processed, gene_names
        )
        
        # Combine context
        context = {
            'time_steps': time_steps,
            'response_types': response_types,
            'memory_types': memory_types,
            'epigenetic_states': epigenetic_states,
            'memory_state': memory_state
        }
        
        # Update combined statistics
        self._update_combined_stats(
            counts, time_steps, response_types, memory_types, epigenetic_states, memory_state
        )
        
        return memory_processed, context
    
    def _update_combined_stats(
        self,
        counts: torch.Tensor,
        time_steps: torch.Tensor,
        response_types: torch.Tensor,
        memory_types: torch.Tensor,
        epigenetic_states: torch.Tensor,
        memory_state: torch.Tensor
    ):
        """Update combined temporal-memory statistics."""
        # Calculate temporal-memory correlations
        try:
            temporal_memory_correlation = torch.corrcoef(torch.stack([
                time_steps.float().flatten(), 
                memory_state.mean(dim=1).repeat_interleave(time_steps.size(1))
            ]))[0, 1]
            self.combined_stats['temporal_memory_correlations'].append(temporal_memory_correlation.cpu())
        except:
            pass  # Skip if correlation cannot be calculated
        
        # Calculate epigenetic-temporal states
        for time_step in range(self.time_steps):
            time_mask = (time_steps == time_step)
            if time_mask.any():
                time_epigenetic_states = epigenetic_states[time_mask]
                epigenetic_distribution = torch.bincount(
                    time_epigenetic_states.flatten(), minlength=self.num_epigenetic_states
                )
                self.combined_stats['epigenetic_temporal_states'].append(epigenetic_distribution.cpu())
    
    def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined temporal-memory statistics."""
        stats = {}
        
        # Temporal-memory correlations
        if self.combined_stats['temporal_memory_correlations']:
            stats['avg_temporal_memory_correlation'] = torch.stack(
                self.combined_stats['temporal_memory_correlations']
            ).mean()
        
        # Epigenetic-temporal states
        if self.combined_stats['epigenetic_temporal_states']:
            epigenetic_states = torch.stack(self.combined_stats['epigenetic_temporal_states'])
            stats['epigenetic_temporal_distribution'] = epigenetic_states.mean(dim=0)
        
        # Combine parent statistics
        temporal_stats = self.get_temporal_stats()
        memory_stats = self.get_memory_stats()
        
        stats.update(temporal_stats)
        stats.update(memory_stats)
        
        return stats
    
    def reset_combined_stats(self):
        """Reset combined statistics."""
        self.combined_stats = {
            'temporal_memory_correlations': [],
            'memory_temporal_attention': [],
            'epigenetic_temporal_states': []
        }
        self.reset_temporal_stats()
        self.reset_memory_stats()
    
    def sample_cell_sentences_temporal_memory(
        self,
        counts_raw,
        dataset,
        shared_genes=None,
        valid_gene_mask=None,
        downsample_frac=None,
        gene_names=None
    ):
        """Temporal-memory cell sentence sampling."""
        # Get gene names if not provided
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(counts_raw.size(1))]
        
        # Apply combined processing
        counts_processed, context = self.temporal_memory_count_processing(
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
            context['time_steps'],
            context['response_types'],
            context['memory_types'],
            context['epigenetic_states'],
            context['memory_state']
        )
        
        return enhanced_result
    
    def __call__(self, batch):
        """Enhanced collate function with temporal-memory normalization."""
        # Call parent collate function
        result = super().__call__(batch)
        
        # Extract temporal-memory information
        if len(result) > 8:  # Enhanced result with temporal-memory context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums,
                pathway_ids, compartment_ids, cell_type_ids, time_steps
            ) = result
        else:
            # Standard result, create dummy temporal-memory context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums
            ) = result
            pathway_ids = None
            compartment_ids = None
            cell_type_ids = None
            time_steps = None
        
        # Apply temporal normalization
        if time_steps is not None and hasattr(self, 'temporal_normalizer'):
            Xs = self.temporal_normalizer(Xs, time_steps)
        
        # Apply fast/slow response normalization
        if hasattr(self, 'fast_slow_normalizer'):
            Xs, response_types = self.fast_slow_normalizer(Xs)
        
        # Apply feedback loop normalization
        if hasattr(self, 'feedback_normalizer'):
            Xs = self.feedback_normalizer(Xs)
        
        # Apply memory normalization
        if hasattr(self, 'memory_normalizer'):
            # Generate dummy memory context if not available
            batch_size, seq_len = Xs.shape[:2]
            memory_types = torch.randint(0, self.num_memory_types, (batch_size, seq_len))
            epigenetic_states = torch.randint(0, self.num_epigenetic_states, (batch_size, seq_len))
            memory_state = torch.randn(batch_size, self.memory_dim)
            
            Xs, updated_memory_state, memory_types = self.memory_normalizer(
                Xs, memory_state, memory_types
            )
            
            Xs, epigenetic_states = self.epigenetic_normalizer(Xs, epigenetic_states)
        
        # Return enhanced result
        return (
            batch_sentences, Xs, Ys, idxs, batch_weights, masks,
            total_counts_all, batch_sentences_counts, dataset_nums,
            pathway_ids, compartment_ids, cell_type_ids, time_steps
        )
