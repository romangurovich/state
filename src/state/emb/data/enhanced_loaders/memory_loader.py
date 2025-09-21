"""
Memory Data Loader

Enhanced data loader for cellular memory models with
memory-aware normalization and epigenetic state tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from .base_enhanced_loader import BaseEnhancedLoader


class MemoryAwareNormalization(nn.Module):
    """Memory-aware normalization for cellular memory models."""
    
    def __init__(
        self,
        d_model: int,
        memory_dim: int = 512,
        memory_size: int = 1000,
        num_memory_types: int = 5
    ):
        super().__init__()
        self.d_model = d_model
        self.memory_dim = memory_dim
        self.memory_size = memory_size
        self.num_memory_types = num_memory_types
        
        # Memory bank
        self.memory_bank = nn.Parameter(torch.randn(memory_size, memory_dim))
        
        # Memory type embeddings
        self.memory_type_embeddings = nn.Embedding(num_memory_types, memory_dim)
        
        # Memory retrieval mechanism
        self.memory_retrieval = nn.MultiheadAttention(
            memory_dim, num_heads=8, batch_first=True
        )
        
        # Memory update mechanism
        self.memory_update = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, memory_dim),
            nn.LayerNorm(memory_dim)
        )
        
        # Memory-aware normalization
        self.memory_norm = nn.Sequential(
            nn.Linear(memory_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Memory decay factors
        self.memory_decay = nn.Parameter(torch.ones(memory_size))
        
        # Memory type classifier
        self.memory_type_classifier = nn.Sequential(
            nn.Linear(d_model, memory_dim // 2),
            nn.SiLU(),
            nn.Linear(memory_dim // 2, num_memory_types)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
        memory_types: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply memory-aware normalization."""
        batch_size, seq_len, d_model = x.shape
        
        # Classify memory types if not provided
        if memory_types is None:
            memory_types = self.memory_type_classifier(x)
            memory_types = F.softmax(memory_types, dim=-1)
        
        # Retrieve relevant memories
        if memory_state is not None:
            # Use provided memory state
            retrieved_memories = memory_state.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            # Retrieve from memory bank
            retrieved_memories = self._retrieve_memories(x, memory_types)
        
        # Apply memory-aware normalization
        memory_effect = self.memory_norm(retrieved_memories)
        x_normalized = x + memory_effect
        
        # Update memory state
        updated_memory_state = self._update_memory_state(x, retrieved_memories)
        
        return x_normalized, updated_memory_state, memory_types
    
    def _retrieve_memories(
        self,
        x: torch.Tensor,
        memory_types: torch.Tensor
    ) -> torch.Tensor:
        """Retrieve relevant memories from memory bank."""
        batch_size, seq_len, d_model = x.shape
        
        # Get memory type embeddings
        memory_type_embs = self.memory_type_embeddings(
            torch.argmax(memory_types, dim=-1)
        )  # [batch_size, seq_len, memory_dim]
        
        # Retrieve memories using attention
        retrieved_memories, _ = self.memory_retrieval(
            memory_type_embs, self.memory_bank.unsqueeze(0), self.memory_bank.unsqueeze(0)
        )
        
        return retrieved_memories
    
    def _update_memory_state(
        self,
        x: torch.Tensor,
        retrieved_memories: torch.Tensor
    ) -> torch.Tensor:
        """Update memory state based on current input."""
        # Combine current input with retrieved memories
        combined_input = torch.cat([x, retrieved_memories], dim=-1)
        
        # Update memory state
        updated_memory = self.memory_update(combined_input)
        
        # Apply memory decay
        decay_factors = self.memory_decay.unsqueeze(0).unsqueeze(0)
        updated_memory = updated_memory * decay_factors
        
        return updated_memory.mean(dim=1)  # Average over sequence length


class EpigeneticNormalization(nn.Module):
    """Epigenetic state-aware normalization."""
    
    def __init__(self, d_model: int, num_epigenetic_states: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_epigenetic_states = num_epigenetic_states
        
        # Epigenetic state embeddings
        self.epigenetic_embeddings = nn.Embedding(num_epigenetic_states, d_model)
        
        # Epigenetic state normalization
        self.epigenetic_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_epigenetic_states)
        ])
        
        # Epigenetic state transition
        self.epigenetic_transition = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Epigenetic state predictor
        self.epigenetic_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_epigenetic_states)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        epigenetic_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply epigenetic normalization."""
        batch_size, seq_len, d_model = x.shape
        
        # Predict epigenetic states if not provided
        if epigenetic_states is None:
            epigenetic_states = self.epigenetic_predictor(x)
            epigenetic_states = F.softmax(epigenetic_states, dim=-1)
        
        # Apply epigenetic state-specific normalization
        for i in range(batch_size):
            for j in range(seq_len):
                state_id = torch.argmax(epigenetic_states[i, j]).item()
                if state_id < len(self.epigenetic_norms):
                    x[i, j] = self.epigenetic_norms[state_id](x[i, j])
        
        # Apply epigenetic state transition
        epigenetic_embs = self.epigenetic_embeddings(
            torch.argmax(epigenetic_states, dim=-1)
        )
        x = x + self.epigenetic_transition(torch.cat([x, epigenetic_embs], dim=-1))
        
        return x, epigenetic_states


class MemoryDataLoader(BaseEnhancedLoader):
    """Enhanced data loader for cellular memory models."""
    
    def __init__(
        self,
        cfg,
        valid_gene_mask=None,
        ds_emb_mapping_inference=None,
        is_train=True,
        precision=None,
        # Memory-specific parameters
        memory_dim: int = 512,
        memory_size: int = 1000,
        num_memory_types: int = 5,
        num_epigenetic_states: int = 3,
        memory_decay_rate: float = 0.9,
        **kwargs
    ):
        super().__init__(
            cfg, valid_gene_mask, ds_emb_mapping_inference, is_train, precision,
            **kwargs
        )
        
        self.memory_dim = memory_dim
        self.memory_size = memory_size
        self.num_memory_types = num_memory_types
        self.num_epigenetic_states = num_epigenetic_states
        self.memory_decay_rate = memory_decay_rate
        
        # Initialize memory normalization modules
        self.memory_normalizer = MemoryAwareNormalization(
            d_model=512,  # Will be updated
            memory_dim=memory_dim,
            memory_size=memory_size,
            num_memory_types=num_memory_types
        )
        
        self.epigenetic_normalizer = EpigeneticNormalization(
            d_model=512,  # Will be updated
            num_epigenetic_states=num_epigenetic_states
        )
        
        # Memory state tracking
        self.memory_states = []
        self.epigenetic_states = []
        self.memory_history = []
        
        # Memory statistics
        self.memory_stats = {
            'memory_usage': [],
            'memory_types': [],
            'epigenetic_states': [],
            'memory_decay': []
        }
    
    def generate_memory_context(
        self,
        counts: torch.Tensor,
        gene_names: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate memory context for given counts."""
        batch_size, seq_len = counts.shape
        
        # Generate memory types
        memory_types = torch.randint(0, self.num_memory_types, (batch_size, seq_len))
        
        # Generate epigenetic states
        epigenetic_states = torch.randint(0, self.num_epigenetic_states, (batch_size, seq_len))
        
        # Generate memory state
        memory_state = torch.randn(batch_size, self.memory_dim)
        
        return memory_types, epigenetic_states, memory_state
    
    def memory_count_processing(
        self,
        counts: torch.Tensor,
        gene_names: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process counts with memory awareness."""
        # Generate memory context
        memory_types, epigenetic_states, memory_state = self.generate_memory_context(
            counts, gene_names
        )
        
        # Apply base normalization
        normalized_counts = self.enhanced_count_processing(counts)
        
        # Reshape for memory processing
        counts_reshaped = normalized_counts.unsqueeze(-1).expand(-1, -1, 512)
        
        # Apply memory normalization
        memory_normalized, updated_memory_state, memory_types = self.memory_normalizer(
            counts_reshaped, memory_state, memory_types
        )
        
        # Apply epigenetic normalization
        epigenetic_normalized, epigenetic_states = self.epigenetic_normalizer(
            memory_normalized, epigenetic_states
        )
        
        # Update memory states
        self.memory_states.append(updated_memory_state.detach())
        self.epigenetic_states.append(epigenetic_states.detach())
        
        # Update memory statistics
        self._update_memory_stats(counts, memory_types, epigenetic_states, updated_memory_state)
        
        return epigenetic_normalized.squeeze(-1), memory_types, epigenetic_states, updated_memory_state
    
    def _update_memory_stats(
        self,
        counts: torch.Tensor,
        memory_types: torch.Tensor,
        epigenetic_states: torch.Tensor,
        memory_state: torch.Tensor
    ):
        """Update memory statistics."""
        # Update memory usage
        memory_usage = memory_state.norm(dim=1).mean()
        self.memory_stats['memory_usage'].append(memory_usage.cpu())
        
        # Update memory types
        memory_type_counts = torch.bincount(memory_types.flatten(), minlength=self.num_memory_types)
        self.memory_stats['memory_types'].append(memory_type_counts.cpu())
        
        # Update epigenetic states
        epigenetic_state_counts = torch.bincount(epigenetic_states.flatten(), minlength=self.num_epigenetic_states)
        self.memory_stats['epigenetic_states'].append(epigenetic_state_counts.cpu())
        
        # Update memory decay
        memory_decay = self.memory_decay_rate ** len(self.memory_states)
        self.memory_stats['memory_decay'].append(memory_decay)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {}
        
        # Memory usage statistics
        if self.memory_stats['memory_usage']:
            stats['avg_memory_usage'] = torch.stack(self.memory_stats['memory_usage']).mean()
            stats['max_memory_usage'] = torch.stack(self.memory_stats['memory_usage']).max()
        
        # Memory type statistics
        if self.memory_stats['memory_types']:
            memory_type_counts = torch.stack(self.memory_stats['memory_types'])
            stats['memory_type_distribution'] = memory_type_counts.mean(dim=0)
        
        # Epigenetic state statistics
        if self.memory_stats['epigenetic_states']:
            epigenetic_state_counts = torch.stack(self.memory_stats['epigenetic_states'])
            stats['epigenetic_state_distribution'] = epigenetic_state_counts.mean(dim=0)
        
        # Memory decay statistics
        if self.memory_stats['memory_decay']:
            stats['avg_memory_decay'] = np.mean(self.memory_stats['memory_decay'])
        
        return stats
    
    def reset_memory_stats(self):
        """Reset memory statistics."""
        self.memory_stats = {
            'memory_usage': [],
            'memory_types': [],
            'epigenetic_states': [],
            'memory_decay': []
        }
        self.memory_states = []
        self.epigenetic_states = []
        self.memory_history = []
    
    def sample_cell_sentences_memory(
        self,
        counts_raw,
        dataset,
        shared_genes=None,
        valid_gene_mask=None,
        downsample_frac=None,
        gene_names=None
    ):
        """Memory-aware cell sentence sampling."""
        # Get gene names if not provided
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(counts_raw.size(1))]
        
        # Apply memory processing
        counts_processed, memory_types, epigenetic_states, memory_state = self.memory_count_processing(
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
        
        # Add memory context information
        enhanced_result = result + (memory_types, epigenetic_states, memory_state)
        
        return enhanced_result
    
    def __call__(self, batch):
        """Enhanced collate function with memory normalization."""
        # Call parent collate function
        result = super().__call__(batch)
        
        # Extract memory information
        if len(result) > 8:  # Enhanced result with memory context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums,
                pathway_ids, compartment_ids, cell_type_ids, time_steps
            ) = result
        else:
            # Standard result, create dummy memory context
            (
                batch_sentences, Xs, Ys, idxs, batch_weights, masks,
                total_counts_all, batch_sentences_counts, dataset_nums
            ) = result
            pathway_ids = None
            compartment_ids = None
            cell_type_ids = None
            time_steps = None
        
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
