"""
Cellular Memory System Model

Implements epigenetic memory and cellular state memory mechanisms
that allow the model to remember previous perturbations and states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
from .base_enhanced import BaseEnhancedModel


class CellularMemoryModel(BaseEnhancedModel):
    """
    Cellular memory model that implements:
    1. Epigenetic memory bank for storing previous perturbations
    2. Memory retrieval mechanisms based on similarity
    3. Memory update mechanisms for learning from new experiences
    4. Cellular state memory for maintaining context
    """
    
    def __init__(
        self,
        token_dim: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        output_dim: int,
        dropout: float = 0.0,
        warmup_steps: int = 0,
        compiled: bool = False,
        max_lr: float = 4e-4,
        emb_cnt: int = 145469,
        emb_size: int = 5120,
        cfg: Optional[Dict[str, Any]] = None,
        collater: Optional[Any] = None,
        # Memory-specific parameters
        memory_size: int = 1000,
        memory_dim: int = 512,
        memory_heads: int = 8,
        memory_update_rate: float = 0.1,
        memory_decay_rate: float = 0.95,
        **kwargs
    ):
        super().__init__(
            token_dim=token_dim,
            d_model=d_model,
            nhead=nhead,
            d_hid=d_hid,
            nlayers=nlayers,
            output_dim=output_dim,
            dropout=dropout,
            warmup_steps=warmup_steps,
            compiled=compiled,
            max_lr=max_lr,
            emb_cnt=emb_cnt,
            emb_size=emb_size,
            cfg=cfg,
            collater=collater,
            memory_size=memory_size,
            **kwargs
        )
        
        self.memory_dim = memory_dim
        self.memory_heads = memory_heads
        self.memory_update_rate = memory_update_rate
        self.memory_decay_rate = memory_decay_rate
        
        self._init_memory_components()
    
    def _init_memory_components(self):
        """Initialize memory system components."""
        # Epigenetic memory bank
        self.memory_bank = nn.Parameter(
            torch.randn(self.memory_size, self.memory_dim) * 0.1
        )
        
        # Memory retrieval attention
        self.memory_attention = nn.MultiheadAttention(
            self.memory_dim,
            num_heads=self.memory_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Memory update mechanism
        self.memory_update = nn.Sequential(
            nn.Linear(self.memory_dim * 2, self.memory_dim),
            nn.SiLU(),
            nn.Linear(self.memory_dim, self.memory_dim),
            nn.LayerNorm(self.memory_dim)
        )
        
        # Memory importance scoring
        self.memory_importance = nn.Sequential(
            nn.Linear(self.memory_dim, self.memory_dim // 2),
            nn.SiLU(),
            nn.Linear(self.memory_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Memory decay mechanism
        self.memory_decay = nn.Parameter(torch.ones(self.memory_size) * self.memory_decay_rate)
        
        # Cellular state memory
        self.state_memory = nn.LSTM(
            self.d_model,
            self.memory_dim,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout
        )
        
        # Memory integration
        self.memory_integration = nn.Sequential(
            nn.Linear(self.d_model + self.memory_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Perturbation memory encoder
        self.perturbation_encoder = nn.Sequential(
            nn.Linear(self.d_model, self.memory_dim),
            nn.SiLU(),
            nn.Linear(self.memory_dim, self.memory_dim),
            nn.LayerNorm(self.memory_dim)
        )
        
        # Memory similarity computation
        self.memory_similarity = nn.CosineSimilarity(dim=-1)
        
        # Memory retrieval gates
        self.retrieval_gate = nn.Sequential(
            nn.Linear(self.memory_dim * 2, self.memory_dim),
            nn.Sigmoid()
        )
    
    def retrieve_memory(
        self, 
        query: torch.Tensor,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve relevant memories from the memory bank.
        
        Args:
            query: Query embedding [batch, d_model]
            top_k: Number of top memories to retrieve
        
        Returns:
            Tuple of (retrieved_memories, attention_weights, memory_indices)
        """
        # Encode query for memory retrieval
        query_encoded = self.perturbation_encoder(query)
        
        # Compute similarity with memory bank
        similarities = self.memory_similarity(
            query_encoded.unsqueeze(1),  # [batch, 1, memory_dim]
            self.memory_bank.unsqueeze(0)  # [1, memory_size, memory_dim]
        )  # [batch, memory_size]
        
        # Get top-k most similar memories
        top_k_similarities, top_k_indices = torch.topk(similarities, top_k, dim=-1)
        
        # Retrieve memories
        batch_size = query.size(0)
        retrieved_memories = self.memory_bank[top_k_indices]  # [batch, top_k, memory_dim]
        
        # Apply attention to retrieved memories
        attended_memories, attention_weights = self.memory_attention(
            query_encoded.unsqueeze(1),  # [batch, 1, memory_dim]
            retrieved_memories,  # [batch, top_k, memory_dim]
            retrieved_memories  # [batch, top_k, memory_dim]
        )
        
        return attended_memories, attention_weights, top_k_indices
    
    def update_memory(
        self, 
        new_experience: torch.Tensor,
        memory_indices: torch.Tensor,
        importance_scores: torch.Tensor
    ):
        """
        Update memory bank with new experience.
        
        Args:
            new_experience: New experience embedding [batch, memory_dim]
            memory_indices: Indices of memories to update [batch, top_k]
            importance_scores: Importance scores for new experience [batch, 1]
        """
        # Encode new experience
        encoded_experience = self.perturbation_encoder(new_experience)
        
        # Update selected memories
        for batch_idx in range(encoded_experience.size(0)):
            for k_idx in range(memory_indices.size(1)):
                memory_idx = memory_indices[batch_idx, k_idx]
                
                # Get current memory
                current_memory = self.memory_bank[memory_idx]
                
                # Compute update
                memory_update = self.memory_update(
                    torch.cat([current_memory, encoded_experience[batch_idx]], dim=-1)
                )
                
                # Apply importance-weighted update
                update_weight = importance_scores[batch_idx] * self.memory_update_rate
                
                # Update memory
                self.memory_bank.data[memory_idx] = (
                    (1 - update_weight) * current_memory + 
                    update_weight * memory_update
                )
    
    def compute_memory_importance(
        self, 
        experience: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute importance score for new experience.
        
        Args:
            experience: Experience embedding [batch, d_model]
        
        Returns:
            Importance scores [batch, 1]
        """
        # Encode experience
        encoded_experience = self.perturbation_encoder(experience)
        
        # Compute importance
        importance = self.memory_importance(encoded_experience)
        
        return importance
    
    def apply_memory_decay(self):
        """Apply decay to all memories."""
        # Apply decay to memory bank
        self.memory_bank.data *= self.memory_decay.unsqueeze(0)
        
        # Update decay rates (gradual forgetting)
        self.memory_decay.data *= 0.999
    
    def integrate_memory(
        self, 
        current_state: torch.Tensor,
        retrieved_memories: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate retrieved memories with current state.
        
        Args:
            current_state: Current cellular state [batch, d_model]
            retrieved_memories: Retrieved memories [batch, memory_dim]
        
        Returns:
            Memory-integrated state [batch, d_model]
        """
        # Combine current state with retrieved memories
        combined = torch.cat([current_state, retrieved_memories.squeeze(1)], dim=-1)
        
        # Integrate memories
        integrated_state = self.memory_integration(combined)
        
        return integrated_state
    
    def update_state_memory(
        self, 
        new_state: torch.Tensor,
        previous_states: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Update cellular state memory.
        
        Args:
            new_state: New cellular state [batch, d_model]
            previous_states: Previous states for context
        
        Returns:
            Updated state memory
        """
        if previous_states is not None:
            # Use previous states as context
            state_sequence = torch.stack(previous_states + [new_state], dim=1)
        else:
            # Use only current state
            state_sequence = new_state.unsqueeze(1)
        
        # Update state memory
        state_memory, _ = self.state_memory(state_sequence)
        
        return state_memory[:, -1, :]  # Return latest state
    
    def forward_memory(
        self, 
        src: torch.Tensor, 
        mask: torch.Tensor,
        previous_states: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Memory-enhanced forward pass.
        
        Args:
            src: Input tensor [batch, seq_len, token_dim]
            mask: Attention mask [batch, seq_len]
            previous_states: Previous cellular states
        
        Returns:
            Tuple of (gene_output, embedding, dataset_emb, memory_info)
        """
        # Initial encoding
        src = self.encoder(src) * math.sqrt(self.d_model)
        
        # Add count processing if available
        if 'counts' in kwargs and kwargs['counts'] is not None:
            counts = kwargs['counts']
            counts = counts.unsqueeze(-1)
            
            bin_weights = self.count_encoder(counts)
            bin_weights = F.softmax(bin_weights, dim=-1)
            
            bin_indices = torch.arange(10, device=self.device)
            bin_embeddings = self.bin_encoder(bin_indices)
            count_emb = torch.matmul(bin_weights, bin_embeddings)
            
            if self.dataset_token is not None:
                dataset_count_emb = torch.zeros(count_emb.size(0), 1, count_emb.size(2), device=self.device)
                count_emb = torch.cat((count_emb, dataset_count_emb), dim=1)
            
            src = src + count_emb
        
        # Get current cellular state (CLS token)
        current_state = src[:, 0, :]  # [batch, d_model]
        
        # Retrieve relevant memories
        retrieved_memories, memory_attention_weights, memory_indices = self.retrieve_memory(
            current_state, top_k=10
        )
        
        # Integrate memories with current state
        memory_integrated_state = self.integrate_memory(current_state, retrieved_memories)
        
        # Update state memory
        updated_state_memory = self.update_state_memory(
            memory_integrated_state, previous_states
        )
        
        # Replace CLS token with memory-integrated state
        src_with_memory = src.clone()
        src_with_memory[:, 0, :] = updated_state_memory
        
        # Apply transformer encoder
        output = self.transformer_encoder(src_with_memory, src_key_padding_mask=None)
        
        # Decode with memory information
        gene_output = self.decoder(output)
        
        # Extract embeddings
        embedding = gene_output[:, 0, :]  # CLS token
        embedding = F.normalize(embedding, dim=1)
        
        # Dataset embedding (if available)
        dataset_emb = None
        if self.dataset_token is not None:
            dataset_emb = gene_output[:, -1, :]
        
        # Compute importance for memory update
        importance_scores = self.compute_memory_importance(current_state)
        
        # Update memory bank
        self.update_memory(
            current_state, memory_indices, importance_scores
        )
        
        # Apply memory decay
        self.apply_memory_decay()
        
        # Prepare memory information
        memory_info = {
            'retrieved_memories': retrieved_memories,
            'memory_attention_weights': memory_attention_weights,
            'memory_indices': memory_indices,
            'importance_scores': importance_scores,
            'updated_state_memory': updated_state_memory
        }
        
        return gene_output, embedding, dataset_emb, memory_info
    
    def forward(self, src: torch.Tensor, mask: torch.Tensor, **kwargs):
        """Override forward to use memory processing."""
        previous_states = kwargs.get('previous_states', None)
        
        return self.forward_memory(src, mask, previous_states, **kwargs)
    
    def get_memory_statistics(self) -> Dict[str, torch.Tensor]:
        """
        Get statistics about the memory bank.
        
        Returns:
            Dictionary containing memory statistics
        """
        # Compute memory importance scores
        memory_importance = self.memory_importance(self.memory_bank)
        
        # Compute memory diversity (pairwise distances)
        memory_distances = torch.cdist(self.memory_bank, self.memory_bank)
        memory_diversity = memory_distances.mean()
        
        # Compute memory utilization
        memory_utilization = (memory_importance > 0.1).float().mean()
        
        return {
            'memory_importance': memory_importance,
            'memory_diversity': memory_diversity,
            'memory_utilization': memory_utilization,
            'memory_decay_rates': self.memory_decay
        }
    
    def clear_memory(self):
        """Clear the memory bank."""
        self.memory_bank.data.zero_()
        self.memory_decay.data.fill_(self.memory_decay_rate)
    
    def save_memory(self, filepath: str):
        """Save memory bank to file."""
        torch.save({
            'memory_bank': self.memory_bank.data,
            'memory_decay': self.memory_decay.data
        }, filepath)
    
    def load_memory(self, filepath: str):
        """Load memory bank from file."""
        checkpoint = torch.load(filepath)
        self.memory_bank.data = checkpoint['memory_bank']
        self.memory_decay.data = checkpoint['memory_decay']
