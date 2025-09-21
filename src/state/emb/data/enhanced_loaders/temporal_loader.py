"""
Temporal Data Loader

Enhanced data loader for temporal dynamics models with
time-aware normalization and temporal sequence processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from .base_enhanced_loader import BaseEnhancedLoader


class TemporalNormalization(nn.Module):
    """Temporal normalization for time series data."""
    
    def __init__(self, d_model: int, time_steps: int = 5):
        super().__init__()
        self.d_model = d_model
        self.time_steps = time_steps
        
        # Time-step specific normalization
        self.temporal_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(time_steps)
        ])
        
        # Temporal shift normalization
        self.temporal_shift = nn.Linear(d_model, d_model)
        
        # Temporal attention normalization
        self.temporal_attention = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True
        )
        
        # Temporal decay factors
        self.temporal_decay = nn.Parameter(torch.ones(time_steps))
    
    def forward(
        self,
        x: torch.Tensor,
        time_steps: torch.Tensor,
        previous_states: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Apply temporal normalization."""
        batch_size, seq_len, d_model = x.shape
        
        # Apply time-step specific normalization
        for i in range(batch_size):
            for j in range(seq_len):
                time_step = time_steps[i, j].item()
                if time_step < self.time_steps:
                    x[i, j] = self.temporal_norms[time_step](x[i, j])
                    # Apply temporal decay
                    x[i, j] = x[i, j] * self.temporal_decay[time_step]
        
        # Apply temporal shift normalization
        x = x + self.temporal_shift(x)
        
        # Apply temporal attention normalization
        x_attended, _ = self.temporal_attention(x, x, x)
        x = x + x_attended
        
        return x


class FastSlowResponseNormalization(nn.Module):
    """Normalization for fast and slow response kinetics."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Fast response normalization
        self.fast_response_norm = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Slow response normalization
        self.slow_response_norm = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Response type classifier
        self.response_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 2)  # Fast (0) or Slow (1)
        )
        
        # Response fusion
        self.response_fusion = nn.Linear(d_model * 2, d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        response_types: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply fast/slow response normalization."""
        # Classify response types if not provided
        if response_types is None:
            response_types = self.response_classifier(x)
            response_types = F.softmax(response_types, dim=-1)
        
        # Apply fast response normalization
        fast_response = self.fast_response_norm(x)
        
        # Apply slow response normalization
        slow_response = self.slow_response_norm(x)
        
        # Weight by response type
        fast_weight = response_types[:, :, 0:1]  # Fast response weight
        slow_weight = response_types[:, :, 1:2]  # Slow response weight
        
        weighted_fast = fast_response * fast_weight
        weighted_slow = slow_response * slow_weight
        
        # Fuse responses
        fused_response = self.response_fusion(
            torch.cat([weighted_fast, weighted_slow], dim=-1)
        )
        
        return fused_response, response_types


class FeedbackLoopNormalization(nn.Module):
    """Normalization for feedback loops and regulatory mechanisms."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Feedback loop normalization
        self.feedback_norm = nn.LSTM(
            d_model, d_model, num_layers=2, batch_first=True
        )
        
        # Feedback gate
        self.feedback_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # Feedback strength predictor
        self.feedback_strength = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        previous_output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply feedback loop normalization."""
        # Apply feedback loop processing
        feedback_output, _ = self.feedback_norm(x)
        
        # Apply feedback gate
        if previous_output is not None:
            gate_input = torch.cat([x, previous_output], dim=-1)
            gate = self.feedback_gate(gate_input)
            feedback_output = feedback_output * gate
        
        # Apply feedback strength
        strength = self.feedback_strength(x)
        feedback_output = feedback_output * strength
        
        return feedback_output


class TemporalDataLoader(BaseEnhancedLoader):
    """Enhanced data loader for temporal dynamics models."""
    
    def __init__(
        self,
        cfg,
        valid_gene_mask=None,
        ds_emb_mapping_inference=None,
        is_train=True,
        precision=None,
        # Temporal-specific parameters
        time_steps: int = 5,
        max_sequence_length: int = 100,
        temporal_sampling_strategy: str = "uniform",
        **kwargs
    ):
        super().__init__(
            cfg, valid_gene_mask, ds_emb_mapping_inference, is_train, precision,
            **kwargs
        )
        
        self.time_steps = time_steps
        self.max_sequence_length = max_sequence_length
        self.temporal_sampling_strategy = temporal_sampling_strategy
        
        # Initialize temporal normalization modules
        self.temporal_normalizer = TemporalNormalization(
            d_model=512,  # Will be updated
            time_steps=time_steps
        )
        
        self.fast_slow_normalizer = FastSlowResponseNormalization(
            d_model=512  # Will be updated
        )
        
        self.feedback_normalizer = FeedbackLoopNormalization(
            d_model=512  # Will be updated
        )
        
        # Temporal statistics
        self.temporal_stats = {
            'time_step_means': {},
            'time_step_stds': {},
            'response_type_counts': {'fast': 0, 'slow': 0},
            'feedback_strengths': []
        }
        
        # Previous states for feedback loops
        self.previous_states = []
    
    def generate_time_steps(
        self,
        batch_size: int,
        seq_len: int,
        strategy: str = "uniform"
    ) -> torch.Tensor:
        """Generate time steps for temporal processing."""
        if strategy == "uniform":
            # Uniform sampling across time steps
            time_steps = torch.randint(0, self.time_steps, (batch_size, seq_len))
        elif strategy == "sequential":
            # Sequential time steps
            time_steps = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            time_steps = time_steps % self.time_steps
        elif strategy == "exponential":
            # Exponential decay sampling
            time_steps = torch.exponential(torch.ones(self.time_steps))
            time_steps = (time_steps / time_steps.sum() * seq_len).round().long()
            time_steps = time_steps.unsqueeze(0).expand(batch_size, -1)
        else:
            raise ValueError(f"Unknown temporal sampling strategy: {strategy}")
        
        return time_steps
    
    def temporal_count_processing(
        self,
        counts: torch.Tensor,
        time_steps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process counts with temporal awareness."""
        batch_size, seq_len = counts.shape
        
        # Generate time steps if not provided
        if time_steps is None:
            time_steps = self.generate_time_steps(
                batch_size, seq_len, self.temporal_sampling_strategy
            )
        
        # Apply base normalization
        normalized_counts = self.enhanced_count_processing(counts)
        
        # Apply temporal normalization
        temporal_normalized = self._apply_temporal_normalization(
            normalized_counts, time_steps
        )
        
        # Apply fast/slow response normalization
        response_normalized, response_types = self._apply_response_normalization(
            temporal_normalized
        )
        
        # Apply feedback loop normalization
        feedback_normalized = self._apply_feedback_normalization(
            response_normalized
        )
        
        return feedback_normalized, time_steps, response_types
    
    def _apply_temporal_normalization(
        self,
        counts: torch.Tensor,
        time_steps: torch.Tensor
    ) -> torch.Tensor:
        """Apply temporal normalization."""
        # Reshape for temporal processing
        counts_reshaped = counts.unsqueeze(-1).expand(-1, -1, 512)  # Add d_model dimension
        
        # Apply temporal normalization
        temporal_normalized = self.temporal_normalizer(counts_reshaped, time_steps)
        
        # Update temporal statistics
        self._update_temporal_stats(counts, time_steps)
        
        return temporal_normalized.squeeze(-1)  # Remove d_model dimension
    
    def _apply_response_normalization(
        self,
        counts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply fast/slow response normalization."""
        # Reshape for response processing
        counts_reshaped = counts.unsqueeze(-1).expand(-1, -1, 512)  # Add d_model dimension
        
        # Apply response normalization
        response_normalized, response_types = self.fast_slow_normalizer(counts_reshaped)
        
        # Update response statistics
        self._update_response_stats(response_types)
        
        return response_normalized.squeeze(-1), response_types.squeeze(-1)
    
    def _apply_feedback_normalization(
        self,
        counts: torch.Tensor
    ) -> torch.Tensor:
        """Apply feedback loop normalization."""
        # Reshape for feedback processing
        counts_reshaped = counts.unsqueeze(-1).expand(-1, -1, 512)  # Add d_model dimension
        
        # Get previous output if available
        previous_output = None
        if self.previous_states:
            previous_output = self.previous_states[-1]
        
        # Apply feedback normalization
        feedback_normalized = self.feedback_normalizer(counts_reshaped, previous_output)
        
        # Update previous states
        self.previous_states.append(feedback_normalized.detach())
        if len(self.previous_states) > self.max_sequence_length:
            self.previous_states.pop(0)
        
        return feedback_normalized.squeeze(-1)
    
    def _update_temporal_stats(
        self,
        counts: torch.Tensor,
        time_steps: torch.Tensor
    ):
        """Update temporal statistics."""
        for time_step in range(self.time_steps):
            time_mask = (time_steps == time_step)
            if time_mask.any():
                time_counts = counts[time_mask]
                
                if time_step not in self.temporal_stats['time_step_means']:
                    self.temporal_stats['time_step_means'][time_step] = []
                    self.temporal_stats['time_step_stds'][time_step] = []
                
                self.temporal_stats['time_step_means'][time_step].append(time_counts.mean().cpu())
                self.temporal_stats['time_step_stds'][time_step].append(time_counts.std().cpu())
    
    def _update_response_stats(self, response_types: torch.Tensor):
        """Update response type statistics."""
        fast_count = (response_types[:, :, 0] > 0.5).sum().item()
        slow_count = (response_types[:, :, 1] > 0.5).sum().item()
        
        self.temporal_stats['response_type_counts']['fast'] += fast_count
        self.temporal_stats['response_type_counts']['slow'] += slow_count
    
    def get_temporal_stats(self) -> Dict[str, Any]:
        """Get temporal statistics."""
        stats = {}
        
        # Time step statistics
        for time_step in self.temporal_stats['time_step_means']:
            if self.temporal_stats['time_step_means'][time_step]:
                stats[f'time_step_{time_step}_mean'] = torch.stack(
                    self.temporal_stats['time_step_means'][time_step]
                ).mean()
                stats[f'time_step_{time_step}_std'] = torch.stack(
                    self.temporal_stats['time_step_stds'][time_step]
                ).mean()
        
        # Response type statistics
        total_responses = sum(self.temporal_stats['response_type_counts'].values())
        if total_responses > 0:
            stats['fast_response_ratio'] = (
                self.temporal_stats['response_type_counts']['fast'] / total_responses
            )
            stats['slow_response_ratio'] = (
                self.temporal_stats['response_type_counts']['slow'] / total_responses
            )
        
        return stats
    
    def reset_temporal_stats(self):
        """Reset temporal statistics."""
        self.temporal_stats = {
            'time_step_means': {},
            'time_step_stds': {},
            'response_type_counts': {'fast': 0, 'slow': 0},
            'feedback_strengths': []
        }
        self.previous_states = []
    
    def sample_cell_sentences_temporal(
        self,
        counts_raw,
        dataset,
        shared_genes=None,
        valid_gene_mask=None,
        downsample_frac=None,
        time_steps=None
    ):
        """Temporal cell sentence sampling with time awareness."""
        # Apply temporal processing
        counts_processed, time_steps, response_types = self.temporal_count_processing(
            counts_raw, time_steps
        )
        
        # Call original sampling method with processed counts
        result = self.sample_cell_sentences(
            counts_processed,
            dataset,
            shared_genes,
            valid_gene_mask,
            downsample_frac
        )
        
        # Add temporal context information
        enhanced_result = result + (time_steps, response_types)
        
        return enhanced_result
    
    def __call__(self, batch):
        """Enhanced collate function with temporal normalization."""
        # Call parent collate function
        result = super().__call__(batch)
        
        # Extract temporal information
        if len(result) > 8:  # Enhanced result with temporal context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums,
                pathway_ids, compartment_ids, cell_type_ids, time_steps
            ) = result
        else:
            # Standard result, create dummy temporal context
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
        
        # Return enhanced result
        return (
            batch_sentences, Xs, Ys, idxs, batch_weights, masks,
            total_counts_all, batch_sentences_counts, dataset_nums,
            pathway_ids, compartment_ids, cell_type_ids, time_steps
        )
