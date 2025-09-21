"""
Hierarchical Gene Organization Model

Implements hierarchical processing of genes organized by biological pathways
and cellular compartments, mirroring the organization of cellular processes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from .base_enhanced import BaseEnhancedModel


class HierarchicalGeneModel(BaseEnhancedModel):
    """
    Hierarchical gene organization model that processes genes at multiple levels:
    1. Pathway level: Groups genes by biological pathways
    2. Gene level: Individual gene processing within pathways
    3. Compartment level: Considers cellular location
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
        # Hierarchical-specific parameters
        num_pathways: int = 1000,
        num_compartments: int = 5,
        pathway_attention_heads: int = 8,
        gene_attention_heads: int = 8,
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
        
        self.num_pathways = num_pathways
        self.num_compartments = num_compartments
        self.pathway_attention_heads = pathway_attention_heads
        self.gene_attention_heads = gene_attention_heads
        
        self._init_hierarchical_components()
    
    def _init_hierarchical_components(self):
        """Initialize hierarchical processing components."""
        # Pathway-level processing
        self.pathway_attention = nn.MultiheadAttention(
            self.d_model,
            num_heads=self.pathway_attention_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Gene-level processing within pathways
        self.gene_attention = nn.MultiheadAttention(
            self.d_model,
            num_heads=self.gene_attention_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Pathway aggregation
        self.pathway_aggregator = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Compartment-aware processing
        self.compartment_processor = nn.Sequential(
            nn.Linear(self.d_model + self.compartment_dim, self.d_model),
            nn.SiLU(),
            nn.LayerNorm(self.d_model)
        )
        
        # Cross-pathway attention
        self.cross_pathway_attention = nn.MultiheadAttention(
            self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Pathway-specific decoders
        self.pathway_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.SiLU(),
                nn.Linear(self.d_model, self.output_dim)
            ) for _ in range(self.num_pathways)
        ])
    
    def process_pathway_level(
        self, 
        gene_embeddings: torch.Tensor, 
        pathway_ids: torch.Tensor,
        compartment_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Process genes at the pathway level.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
            pathway_ids: Pathway IDs for each gene [batch, seq_len]
            compartment_ids: Compartment IDs for each gene [batch, seq_len]
        
        Returns:
            Pathway-processed embeddings
        """
        batch_size, seq_len, d_model = gene_embeddings.shape
        
        # Get compartment embeddings
        compartment_emb = self.get_compartment_embeddings(compartment_ids)
        
        # Add compartment information to gene embeddings
        enhanced_embeddings = self.compartment_processor(
            torch.cat([gene_embeddings, compartment_emb], dim=-1)
        )
        
        # Group genes by pathway
        pathway_embeddings = []
        pathway_masks = []
        
        for pathway_id in range(self.num_pathways):
            # Find genes belonging to this pathway
            pathway_mask = (pathway_ids == pathway_id)
            
            if pathway_mask.any():
                # Extract genes for this pathway
                pathway_genes = enhanced_embeddings[pathway_mask]
                
                # Apply pathway-level attention
                pathway_attended, _ = self.pathway_attention(
                    pathway_genes, pathway_genes, pathway_genes
                )
                
                # Aggregate pathway information
                pathway_aggregated = self.pathway_aggregator(pathway_attended)
                
                # Store pathway embedding (mean of genes in pathway)
                pathway_emb = pathway_aggregated.mean(dim=0, keepdim=True)
                pathway_embeddings.append(pathway_emb)
                pathway_masks.append(pathway_mask)
            else:
                # Empty pathway
                pathway_embeddings.append(torch.zeros(1, d_model, device=gene_embeddings.device))
                pathway_masks.append(torch.zeros(batch_size, seq_len, dtype=torch.bool, device=gene_embeddings.device))
        
        # Stack pathway embeddings
        pathway_embeddings = torch.cat(pathway_embeddings, dim=0)  # [num_pathways, d_model]
        
        return pathway_embeddings, pathway_masks
    
    def process_gene_level(
        self, 
        gene_embeddings: torch.Tensor, 
        pathway_embeddings: torch.Tensor,
        pathway_masks: list
    ) -> torch.Tensor:
        """
        Process genes at the individual level within pathway context.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
            pathway_embeddings: Pathway embeddings [num_pathways, d_model]
            pathway_masks: List of pathway masks
        
        Returns:
            Gene-level processed embeddings
        """
        batch_size, seq_len, d_model = gene_embeddings.shape
        processed_embeddings = gene_embeddings.clone()
        
        # Process each gene within its pathway context
        for gene_idx in range(seq_len):
            # Find which pathway this gene belongs to
            gene_pathway = None
            for pathway_id, mask in enumerate(pathway_masks):
                if mask[:, gene_idx].any():
                    gene_pathway = pathway_id
                    break
            
            if gene_pathway is not None:
                # Get pathway context
                pathway_context = pathway_embeddings[gene_pathway].unsqueeze(0).expand(batch_size, -1)
                
                # Combine gene embedding with pathway context
                gene_with_pathway = gene_embeddings[:, gene_idx:gene_idx+1, :] + pathway_context.unsqueeze(1)
                
                # Apply gene-level attention
                gene_attended, _ = self.gene_attention(
                    gene_with_pathway, gene_with_pathway, gene_with_pathway
                )
                
                processed_embeddings[:, gene_idx, :] = gene_attended.squeeze(1)
        
        return processed_embeddings
    
    def apply_cross_pathway_attention(
        self, 
        gene_embeddings: torch.Tensor, 
        pathway_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply cross-pathway attention to capture interactions between pathways.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
            pathway_embeddings: Pathway embeddings [num_pathways, d_model]
        
        Returns:
            Cross-pathway attended embeddings
        """
        batch_size, seq_len, d_model = gene_embeddings.shape
        
        # Expand pathway embeddings to match gene sequence
        pathway_context = pathway_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply cross-pathway attention
        cross_attended, _ = self.cross_pathway_attention(
            gene_embeddings, pathway_context, pathway_context
        )
        
        return cross_attended
    
    def forward_hierarchical(
        self, 
        src: torch.Tensor, 
        mask: torch.Tensor,
        pathway_ids: torch.Tensor,
        compartment_ids: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Hierarchical forward pass.
        
        Args:
            src: Input tensor [batch, seq_len, token_dim]
            mask: Attention mask [batch, seq_len]
            pathway_ids: Pathway IDs for each gene [batch, seq_len]
            compartment_ids: Compartment IDs for each gene [batch, seq_len]
        
        Returns:
            Tuple of (gene_output, embedding, dataset_emb)
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
        
        # Hierarchical processing
        # Step 1: Pathway-level processing
        pathway_embeddings, pathway_masks = self.process_pathway_level(
            src, pathway_ids, compartment_ids
        )
        
        # Step 2: Gene-level processing within pathway context
        gene_processed = self.process_gene_level(
            src, pathway_embeddings, pathway_masks
        )
        
        # Step 3: Cross-pathway attention
        cross_pathway_attended = self.apply_cross_pathway_attention(
            gene_processed, pathway_embeddings
        )
        
        # Apply transformer encoder
        output = self.transformer_encoder(cross_pathway_attended, src_key_padding_mask=None)
        
        # Decode with pathway-specific information
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
        """Override forward to use hierarchical processing."""
        # Extract pathway and compartment information
        pathway_ids = kwargs.get('pathway_ids', torch.zeros(src.size(0), src.size(1), dtype=torch.long, device=src.device))
        compartment_ids = kwargs.get('compartment_ids', torch.zeros(src.size(0), src.size(1), dtype=torch.long, device=src.device))
        
        return self.forward_hierarchical(src, mask, pathway_ids, compartment_ids, **kwargs)
    
    def get_pathway_attention_weights(self, pathway_ids: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for pathway-level processing.
        
        Args:
            pathway_ids: Pathway IDs for each gene [batch, seq_len]
        
        Returns:
            Attention weights [batch, heads, seq_len, seq_len]
        """
        # This would be implemented to extract attention weights
        # from the pathway attention mechanism for interpretability
        pass
    
    def compute_pathway_consistency_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        pathway_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pathway consistency loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            pathway_ids: Pathway IDs for each gene
        
        Returns:
            Pathway consistency loss
        """
        # Compute loss within each pathway
        pathway_losses = []
        
        for pathway_id in range(self.num_pathways):
            pathway_mask = (pathway_ids == pathway_id)
            
            if pathway_mask.any():
                pathway_pred = predictions[pathway_mask]
                pathway_target = targets[pathway_mask]
                
                if len(pathway_pred) > 1:
                    # Compute consistency within pathway
                    pathway_loss = F.mse_loss(pathway_pred, pathway_target)
                    pathway_losses.append(pathway_loss)
        
        if pathway_losses:
            return torch.stack(pathway_losses).mean()
        else:
            return torch.tensor(0.0, device=predictions.device)
