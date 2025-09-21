"""
Multi-Scale Biological Processing Model

Implements multi-scale processing to capture biological processes
at different scales: molecular, pathway, and cellular levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
from .base_enhanced import BaseEnhancedModel


class MultiScaleModel(BaseEnhancedModel):
    """
    Multi-scale model that processes information at:
    1. Molecular scale (individual genes)
    2. Pathway scale (gene sets and pathways)
    3. Cellular scale (global cellular state)
    4. Cross-scale integration and information flow
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
        # Multi-scale specific parameters
        molecular_scale_dim: int = 256,
        pathway_scale_dim: int = 512,
        cellular_scale_dim: int = 1024,
        cross_scale_heads: int = 8,
        pathway_group_size: int = 10,
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
        
        self.molecular_scale_dim = molecular_scale_dim
        self.pathway_scale_dim = pathway_scale_dim
        self.cellular_scale_dim = cellular_scale_dim
        self.cross_scale_heads = cross_scale_heads
        self.pathway_group_size = pathway_group_size
        
        self._init_multiscale_components()
    
    def _init_multiscale_components(self):
        """Initialize multi-scale processing components."""
        # Molecular scale processor
        self.molecular_processor = nn.Sequential(
            nn.Linear(self.d_model, self.molecular_scale_dim),
            nn.SiLU(),
            nn.Linear(self.molecular_scale_dim, self.molecular_scale_dim),
            nn.LayerNorm(self.molecular_scale_dim)
        )
        
        # Pathway scale processor
        self.pathway_processor = nn.Sequential(
            nn.Linear(self.d_model, self.pathway_scale_dim),
            nn.SiLU(),
            nn.Linear(self.pathway_scale_dim, self.pathway_scale_dim),
            nn.LayerNorm(self.pathway_scale_dim)
        )
        
        # Cellular scale processor
        self.cellular_processor = nn.Sequential(
            nn.Linear(self.d_model, self.cellular_scale_dim),
            nn.SiLU(),
            nn.Linear(self.cellular_scale_dim, self.cellular_scale_dim),
            nn.LayerNorm(self.cellular_scale_dim)
        )
        
        # Cross-scale attention mechanisms
        self.molecular_to_pathway_attention = nn.MultiheadAttention(
            self.molecular_scale_dim,
            num_heads=self.cross_scale_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        self.pathway_to_cellular_attention = nn.MultiheadAttention(
            self.pathway_scale_dim,
            num_heads=self.cross_scale_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        self.cellular_to_molecular_attention = nn.MultiheadAttention(
            self.cellular_scale_dim,
            num_heads=self.cross_scale_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Scale-specific transformers
        self.molecular_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.molecular_scale_dim,
                nhead=self.nhead,
                dim_feedforward=self.d_hid,
                dropout=self.dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        self.pathway_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.pathway_scale_dim,
                nhead=self.nhead,
                dim_feedforward=self.d_hid,
                dropout=self.dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        self.cellular_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.cellular_scale_dim,
                nhead=self.nhead,
                dim_feedforward=self.d_hid,
                dropout=self.dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Cross-scale integration
        self.cross_scale_integration = nn.Sequential(
            nn.Linear(
                self.molecular_scale_dim + self.pathway_scale_dim + self.cellular_scale_dim,
                self.d_model
            ),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Scale-specific decoders
        self.molecular_decoder = nn.Sequential(
            nn.Linear(self.molecular_scale_dim, self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )
        
        self.pathway_decoder = nn.Sequential(
            nn.Linear(self.pathway_scale_dim, self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )
        
        self.cellular_decoder = nn.Sequential(
            nn.Linear(self.cellular_scale_dim, self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )
        
        # Scale fusion weights
        self.scale_fusion_weights = nn.Parameter(torch.ones(3) / 3)  # Equal weights initially
    
    def process_molecular_scale(
        self, 
        gene_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Process genes at the molecular scale.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
        
        Returns:
            Molecular scale embeddings [batch, seq_len, molecular_scale_dim]
        """
        # Process at molecular scale
        molecular_embeddings = self.molecular_processor(gene_embeddings)
        
        # Apply molecular transformer
        molecular_processed = self.molecular_transformer(molecular_embeddings)
        
        return molecular_processed
    
    def process_pathway_scale(
        self, 
        gene_embeddings: torch.Tensor,
        pathway_groups: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process genes at the pathway scale.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
            pathway_groups: Pathway group assignments [batch, seq_len]
        
        Returns:
            Pathway scale embeddings [batch, num_pathways, pathway_scale_dim]
        """
        # Process at pathway scale
        pathway_embeddings = self.pathway_processor(gene_embeddings)
        
        # Group genes by pathways if pathway_groups provided
        if pathway_groups is not None:
            # Group genes by pathway
            pathway_processed = []
            for pathway_id in range(pathway_groups.max().item() + 1):
                pathway_mask = (pathway_groups == pathway_id)
                if pathway_mask.any():
                    pathway_genes = pathway_embeddings[pathway_mask]
                    pathway_aggregated = pathway_genes.mean(dim=0, keepdim=True)
                    pathway_processed.append(pathway_aggregated)
            
            if pathway_processed:
                pathway_processed = torch.cat(pathway_processed, dim=0)
            else:
                pathway_processed = pathway_embeddings.mean(dim=1, keepdim=True)
        else:
            # Use all genes as single pathway
            pathway_processed = pathway_embeddings.mean(dim=1, keepdim=True)
        
        # Apply pathway transformer
        pathway_processed = self.pathway_transformer(pathway_processed)
        
        return pathway_processed
    
    def process_cellular_scale(
        self, 
        gene_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Process genes at the cellular scale.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
        
        Returns:
            Cellular scale embeddings [batch, 1, cellular_scale_dim]
        """
        # Process at cellular scale
        cellular_embeddings = self.cellular_processor(gene_embeddings)
        
        # Aggregate to cellular level
        cellular_aggregated = cellular_embeddings.mean(dim=1, keepdim=True)
        
        # Apply cellular transformer
        cellular_processed = self.cellular_transformer(cellular_aggregated)
        
        return cellular_processed
    
    def apply_cross_scale_attention(
        self, 
        molecular_embeddings: torch.Tensor,
        pathway_embeddings: torch.Tensor,
        cellular_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply cross-scale attention mechanisms.
        
        Args:
            molecular_embeddings: Molecular scale embeddings
            pathway_embeddings: Pathway scale embeddings
            cellular_embeddings: Cellular scale embeddings
        
        Returns:
            Tuple of (attended_molecular, attended_pathway, attended_cellular)
        """
        # Molecular to pathway attention
        molecular_to_pathway, _ = self.molecular_to_pathway_attention(
            molecular_embeddings, pathway_embeddings, pathway_embeddings
        )
        
        # Pathway to cellular attention
        pathway_to_cellular, _ = self.pathway_to_cellular_attention(
            pathway_embeddings, cellular_embeddings, cellular_embeddings
        )
        
        # Cellular to molecular attention
        cellular_to_molecular, _ = self.cellular_to_molecular_attention(
            cellular_embeddings, molecular_embeddings, molecular_embeddings
        )
        
        return molecular_to_pathway, pathway_to_cellular, cellular_to_molecular
    
    def integrate_scales(
        self, 
        molecular_embeddings: torch.Tensor,
        pathway_embeddings: torch.Tensor,
        cellular_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate information from all scales.
        
        Args:
            molecular_embeddings: Molecular scale embeddings
            pathway_embeddings: Pathway scale embeddings
            cellular_embeddings: Cellular scale embeddings
        
        Returns:
            Integrated embeddings
        """
        # Ensure all embeddings have the same sequence length
        batch_size = molecular_embeddings.size(0)
        seq_len = molecular_embeddings.size(1)
        
        # Expand pathway and cellular embeddings to match molecular length
        pathway_expanded = pathway_embeddings.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        cellular_expanded = cellular_embeddings.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        
        # Combine all scales
        combined_embeddings = torch.cat([
            molecular_embeddings,
            pathway_expanded,
            cellular_expanded
        ], dim=-1)
        
        # Integrate scales
        integrated_embeddings = self.cross_scale_integration(combined_embeddings)
        
        return integrated_embeddings
    
    def forward_multiscale(
        self, 
        src: torch.Tensor, 
        mask: torch.Tensor,
        pathway_groups: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Multi-scale forward pass.
        
        Args:
            src: Input tensor [batch, seq_len, token_dim]
            mask: Attention mask [batch, seq_len]
            pathway_groups: Pathway group assignments [batch, seq_len]
        
        Returns:
            Tuple of (gene_output, embedding, dataset_emb, multiscale_info)
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
        
        # Process at different scales
        molecular_embeddings = self.process_molecular_scale(src)
        pathway_embeddings = self.process_pathway_scale(src, pathway_groups)
        cellular_embeddings = self.process_cellular_scale(src)
        
        # Apply cross-scale attention
        attended_molecular, attended_pathway, attended_cellular = self.apply_cross_scale_attention(
            molecular_embeddings, pathway_embeddings, cellular_embeddings
        )
        
        # Integrate scales
        integrated_embeddings = self.integrate_scales(
            attended_molecular, attended_pathway, attended_cellular
        )
        
        # Apply transformer encoder
        output = self.transformer_encoder(integrated_embeddings, src_key_padding_mask=None)
        
        # Decode with multi-scale information
        gene_output = self.decoder(output)
        
        # Extract embeddings
        embedding = gene_output[:, 0, :]  # CLS token
        embedding = F.normalize(embedding, dim=1)
        
        # Dataset embedding (if available)
        dataset_emb = None
        if self.dataset_token is not None:
            dataset_emb = gene_output[:, -1, :]
        
        # Prepare multi-scale information
        multiscale_info = {
            'molecular_embeddings': molecular_embeddings,
            'pathway_embeddings': pathway_embeddings,
            'cellular_embeddings': cellular_embeddings,
            'attended_molecular': attended_molecular,
            'attended_pathway': attended_pathway,
            'attended_cellular': attended_cellular,
            'integrated_embeddings': integrated_embeddings
        }
        
        return gene_output, embedding, dataset_emb, multiscale_info
    
    def forward(self, src: torch.Tensor, mask: torch.Tensor, **kwargs):
        """Override forward to use multi-scale processing."""
        pathway_groups = kwargs.get('pathway_groups', None)
        
        return self.forward_multiscale(src, mask, pathway_groups, **kwargs)
    
    def get_scale_specific_predictions(
        self, 
        multiscale_info: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Get predictions from each scale.
        
        Args:
            multiscale_info: Multi-scale information from forward pass
        
        Returns:
            Dictionary containing scale-specific predictions
        """
        molecular_embeddings = multiscale_info['molecular_embeddings']
        pathway_embeddings = multiscale_info['pathway_embeddings']
        cellular_embeddings = multiscale_info['cellular_embeddings']
        
        # Get predictions from each scale
        molecular_predictions = self.molecular_decoder(molecular_embeddings)
        pathway_predictions = self.pathway_decoder(pathway_embeddings)
        cellular_predictions = self.cellular_decoder(cellular_embeddings)
        
        return {
            'molecular_predictions': molecular_predictions,
            'pathway_predictions': pathway_predictions,
            'cellular_predictions': cellular_predictions
        }
    
    def compute_multiscale_loss(
        self, 
        predictions: torch.Tensor,
        targets: torch.Tensor,
        multiscale_info: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute multi-scale loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            multiscale_info: Multi-scale information from forward pass
        
        Returns:
            Multi-scale loss
        """
        # Base prediction loss
        base_loss = F.mse_loss(predictions, targets)
        
        # Get scale-specific predictions
        scale_predictions = self.get_scale_specific_predictions(multiscale_info)
        
        # Compute scale-specific losses
        molecular_loss = F.mse_loss(scale_predictions['molecular_predictions'], targets)
        pathway_loss = F.mse_loss(scale_predictions['pathway_predictions'], targets)
        cellular_loss = F.mse_loss(scale_predictions['cellular_predictions'], targets)
        
        # Weighted combination
        scale_loss = (
            self.scale_fusion_weights[0] * molecular_loss +
            self.scale_fusion_weights[1] * pathway_loss +
            self.scale_fusion_weights[2] * cellular_loss
        )
        
        # Total loss
        total_loss = base_loss + 0.1 * scale_loss
        
        return total_loss
    
    def get_scale_attention_weights(
        self, 
        multiscale_info: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention weights from each scale.
        
        Args:
            multiscale_info: Multi-scale information from forward pass
        
        Returns:
            Dictionary containing attention weights
        """
        # This would be implemented to extract attention weights
        # from each scale for interpretability
        return {
            'molecular_attention': None,
            'pathway_attention': None,
            'cellular_attention': None
        }
    
    def update_scale_fusion_weights(
        self, 
        new_weights: torch.Tensor
    ):
        """
        Update scale fusion weights.
        
        Args:
            new_weights: New fusion weights [3]
        """
        self.scale_fusion_weights.data = new_weights
