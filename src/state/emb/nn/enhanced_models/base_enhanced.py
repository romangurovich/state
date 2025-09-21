"""
Base Enhanced Model Class

Provides a foundation for all enhanced virtual cell models with common functionality
and biological-inspired components.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Optional, Dict, Any, Tuple
from ..model import StateEmbeddingModel, SkipBlock


class BaseEnhancedModel(StateEmbeddingModel):
    """
    Base class for enhanced virtual cell models.
    
    Extends the base StateEmbeddingModel with common biological enhancements
    and provides a foundation for more specialized models.
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
        # Enhanced model parameters
        pathway_dim: int = 64,
        compartment_dim: int = 32,
        memory_size: int = 1000,
        noise_dim: int = 64,
        resource_dim: int = 32,
        time_steps: int = 5,
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
            **kwargs
        )
        
        # Enhanced model parameters
        self.pathway_dim = pathway_dim
        self.compartment_dim = compartment_dim
        self.memory_size = memory_size
        self.noise_dim = noise_dim
        self.resource_dim = resource_dim
        self.time_steps = time_steps
        
        # Common enhanced components
        self._init_enhanced_components()
    
    def _init_enhanced_components(self):
        """Initialize common enhanced components."""
        # Pathway embeddings
        self.pathway_embeddings = nn.Embedding(1000, self.pathway_dim)  # 1000 pathways max
        
        # Compartment embeddings (nucleus, cytoplasm, membrane, mitochondria, ER)
        self.compartment_embeddings = nn.Embedding(5, self.compartment_dim)
        
        # Enhanced attention mechanisms
        self.enhanced_attention = nn.MultiheadAttention(
            self.d_model, 
            num_heads=self.nhead, 
            dropout=self.dropout,
            batch_first=True
        )
        
        # Biological activation functions
        self.biological_activation = nn.SiLU()  # Smooth ReLU for biological processes
        
        # Enhanced normalization
        self.enhanced_norm = nn.LayerNorm(self.d_model)
    
    def get_pathway_embeddings(self, pathway_ids: torch.Tensor) -> torch.Tensor:
        """Get pathway embeddings for given pathway IDs."""
        return self.pathway_embeddings(pathway_ids)
    
    def get_compartment_embeddings(self, compartment_ids: torch.Tensor) -> torch.Tensor:
        """Get compartment embeddings for given compartment IDs."""
        return self.compartment_embeddings(compartment_ids)
    
    def apply_biological_constraints(
        self, 
        attention_weights: torch.Tensor, 
        gene_interactions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply biological constraints to attention weights.
        
        Args:
            attention_weights: Raw attention weights [batch, heads, seq_len, seq_len]
            gene_interactions: Known gene interaction matrix [seq_len, seq_len]
        
        Returns:
            Constrained attention weights
        """
        if gene_interactions is not None:
            # Apply interaction mask
            mask = gene_interactions.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            attention_weights = attention_weights * mask
        
        # Apply sparsity constraint (most genes don't interact)
        sparsity_threshold = 0.1
        attention_weights = torch.where(
            attention_weights < sparsity_threshold,
            torch.zeros_like(attention_weights),
            attention_weights
        )
        
        return attention_weights
    
    def compute_biological_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        pathway_consistency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute biological consistency loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            pathway_consistency: Pathway consistency scores
        
        Returns:
            Biological loss component
        """
        # Base prediction loss
        base_loss = F.mse_loss(predictions, targets)
        
        # Pathway consistency loss (if available)
        pathway_loss = 0.0
        if pathway_consistency is not None:
            pathway_loss = F.mse_loss(predictions, pathway_consistency)
        
        return base_loss + 0.1 * pathway_loss
    
    def forward_enhanced(
        self, 
        src: torch.Tensor, 
        mask: torch.Tensor,
        pathway_ids: Optional[torch.Tensor] = None,
        compartment_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Enhanced forward pass with biological components.
        
        Args:
            src: Input tensor [batch, seq_len, token_dim]
            mask: Attention mask [batch, seq_len]
            pathway_ids: Pathway IDs for each gene [batch, seq_len]
            compartment_ids: Compartment IDs for each gene [batch, seq_len]
        
        Returns:
            Tuple of (gene_output, embedding, dataset_emb)
        """
        # Get pathway and compartment embeddings
        pathway_emb = None
        compartment_emb = None
        
        if pathway_ids is not None:
            pathway_emb = self.get_pathway_embeddings(pathway_ids)
        
        if compartment_ids is not None:
            compartment_emb = self.get_compartment_embeddings(compartment_ids)
        
        # Enhanced encoding
        src = self.encoder(src) * math.sqrt(self.d_model)
        
        # Add pathway and compartment information
        if pathway_emb is not None:
            src = src + F.linear(pathway_emb, torch.randn(self.d_model, self.pathway_dim))
        
        if compartment_emb is not None:
            src = src + F.linear(compartment_emb, torch.randn(self.d_model, self.compartment_dim))
        
        # Apply enhanced attention
        src = self.enhanced_norm(src)
        src = self.biological_activation(src)
        
        # Continue with transformer processing
        output = self.transformer_encoder(src, src_key_padding_mask=None)
        gene_output = self.decoder(output)
        
        # Extract embeddings
        embedding = gene_output[:, 0, :]  # CLS token
        embedding = F.normalize(embedding, dim=1)
        
        # Dataset embedding (if available)
        dataset_emb = None
        if self.dataset_token is not None:
            dataset_emb = gene_output[:, -1, :]
        
        return gene_output, embedding, dataset_emb
    
    def forward(self, src: torch.Tensor, mask: torch.Tensor, **kwargs):
        """Override forward to use enhanced processing."""
        return self.forward_enhanced(src, mask, **kwargs)
