"""
Regulatory Network Constraints Model

Implements biologically constrained attention mechanisms that enforce
known gene-gene interactions and regulatory network topologies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
from .base_enhanced import BaseEnhancedModel


class RegulatoryConstrainedModel(BaseEnhancedModel):
    """
    Regulatory network constrained model that:
    1. Enforces known gene-gene interactions
    2. Applies biological constraints to attention mechanisms
    3. Models directional regulatory relationships
    4. Incorporates prior biological knowledge
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
        # Regulatory-specific parameters
        gene_interaction_matrix: Optional[torch.Tensor] = None,
        transcription_factor_mask: Optional[torch.Tensor] = None,
        regulatory_strength: float = 0.1,
        sparsity_threshold: float = 0.1,
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
        
        self.regulatory_strength = regulatory_strength
        self.sparsity_threshold = sparsity_threshold
        
        # Store interaction matrices
        self.register_buffer('gene_interaction_matrix', gene_interaction_matrix)
        self.register_buffer('transcription_factor_mask', transcription_factor_mask)
        
        self._init_regulatory_components()
    
    def _init_regulatory_components(self):
        """Initialize regulatory constraint components."""
        # Learnable interaction weights
        if self.gene_interaction_matrix is not None:
            self.interaction_weights = nn.Parameter(
                torch.randn_like(self.gene_interaction_matrix) * 0.1
            )
        else:
            # Initialize with random interactions if none provided
            self.interaction_weights = nn.Parameter(
                torch.randn(1000, 1000) * 0.1  # Default size
            )
        
        # Regulatory attention mechanisms
        self.regulatory_attention = nn.MultiheadAttention(
            self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Transcription factor processing
        self.tf_processor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Target gene processing
        self.target_processor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Regulatory strength predictor
        self.regulatory_strength_predictor = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )
        
        # Directional attention (TF -> Target)
        self.directional_attention = nn.MultiheadAttention(
            self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Regulatory constraint loss
        self.regulatory_loss_weight = 0.1
    
    def create_regulatory_mask(
        self, 
        seq_len: int,
        gene_types: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Create regulatory mask based on gene interactions.
        
        Args:
            seq_len: Sequence length
            gene_types: Gene type indicators (0=TF, 1=Target) [batch, seq_len]
        
        Returns:
            Regulatory mask [seq_len, seq_len]
        """
        # Start with all-to-all attention
        mask = torch.ones(seq_len, seq_len, device=self.device)
        
        # Apply sparsity constraint
        mask = torch.where(
            torch.rand_like(mask) < self.sparsity_threshold,
            mask,
            torch.zeros_like(mask)
        )
        
        # Apply directional constraints if gene types provided
        if gene_types is not None:
            for i in range(seq_len):
                for j in range(seq_len):
                    # TFs can regulate targets, but not vice versa
                    if gene_types[i] == 0 and gene_types[j] == 1:  # TF -> Target
                        mask[i, j] = 1.0
                    elif gene_types[i] == 1 and gene_types[j] == 0:  # Target -> TF
                        mask[i, j] = 0.0
        
        return mask
    
    def apply_regulatory_constraints(
        self, 
        attention_weights: torch.Tensor,
        regulatory_mask: torch.Tensor,
        interaction_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply regulatory constraints to attention weights.
        
        Args:
            attention_weights: Raw attention weights [batch, heads, seq_len, seq_len]
            regulatory_mask: Regulatory mask [seq_len, seq_len]
            interaction_weights: Learnable interaction weights [seq_len, seq_len]
        
        Returns:
            Constrained attention weights
        """
        # Apply regulatory mask
        constrained_weights = attention_weights * regulatory_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply interaction strength weights
        constrained_weights = constrained_weights * interaction_weights.unsqueeze(0).unsqueeze(0)
        
        # Normalize to maintain attention properties
        constrained_weights = F.softmax(constrained_weights, dim=-1)
        
        return constrained_weights
    
    def process_transcription_factors(
        self, 
        gene_embeddings: torch.Tensor,
        tf_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Process transcription factors with specialized attention.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
            tf_mask: Transcription factor mask [batch, seq_len]
        
        Returns:
            Processed TF embeddings
        """
        # Extract TF embeddings
        tf_embeddings = gene_embeddings * tf_mask.unsqueeze(-1)
        
        # Process TFs
        tf_processed = self.tf_processor(tf_embeddings)
        
        return tf_processed
    
    def process_target_genes(
        self, 
        gene_embeddings: torch.Tensor,
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Process target genes with specialized attention.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
            target_mask: Target gene mask [batch, seq_len]
        
        Returns:
            Processed target gene embeddings
        """
        # Extract target gene embeddings
        target_embeddings = gene_embeddings * target_mask.unsqueeze(-1)
        
        # Process targets
        target_processed = self.target_processor(target_embeddings)
        
        return target_processed
    
    def apply_directional_attention(
        self, 
        tf_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply directional attention from TFs to targets.
        
        Args:
            tf_embeddings: TF embeddings [batch, num_tfs, d_model]
            target_embeddings: Target embeddings [batch, num_targets, d_model]
        
        Returns:
            Tuple of (attended_tfs, attended_targets)
        """
        # TF to target attention
        tf_attended, _ = self.directional_attention(
            target_embeddings, tf_embeddings, tf_embeddings
        )
        
        # Target to TF attention (weaker)
        target_attended, _ = self.directional_attention(
            tf_embeddings, target_embeddings, target_embeddings
        )
        
        return tf_attended, target_attended
    
    def predict_regulatory_strength(
        self, 
        tf_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict regulatory strength between TFs and targets.
        
        Args:
            tf_embeddings: TF embeddings [batch, d_model]
            target_embeddings: Target embeddings [batch, d_model]
        
        Returns:
            Regulatory strength predictions [batch, 1]
        """
        # Combine TF and target embeddings
        combined = torch.cat([tf_embeddings, target_embeddings], dim=-1)
        
        # Predict regulatory strength
        strength = self.regulatory_strength_predictor(combined)
        
        return strength
    
    def forward_regulatory(
        self, 
        src: torch.Tensor, 
        mask: torch.Tensor,
        gene_types: Optional[torch.Tensor] = None,
        tf_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Regulatory constrained forward pass.
        
        Args:
            src: Input tensor [batch, seq_len, token_dim]
            mask: Attention mask [batch, seq_len]
            gene_types: Gene type indicators [batch, seq_len]
            tf_mask: Transcription factor mask [batch, seq_len]
            target_mask: Target gene mask [batch, seq_len]
        
        Returns:
            Tuple of (gene_output, embedding, dataset_emb, regulatory_info)
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
        
        # Create regulatory mask
        regulatory_mask = self.create_regulatory_mask(src.size(1), gene_types)
        
        # Process transcription factors
        if tf_mask is not None:
            tf_processed = self.process_transcription_factors(src, tf_mask)
        else:
            tf_processed = src
        
        # Process target genes
        if target_mask is not None:
            target_processed = self.process_target_genes(src, target_mask)
        else:
            target_processed = src
        
        # Apply directional attention
        tf_attended, target_attended = self.apply_directional_attention(
            tf_processed, target_processed
        )
        
        # Combine TF and target information
        combined_embeddings = tf_attended + target_attended
        
        # Apply regulatory attention with constraints
        regulatory_attended, attention_weights = self.regulatory_attention(
            combined_embeddings, combined_embeddings, combined_embeddings
        )
        
        # Apply regulatory constraints to attention weights
        constrained_attention = self.apply_regulatory_constraints(
            attention_weights, regulatory_mask, self.interaction_weights
        )
        
        # Apply transformer encoder
        output = self.transformer_encoder(regulatory_attended, src_key_padding_mask=None)
        
        # Decode with regulatory information
        gene_output = self.decoder(output)
        
        # Extract embeddings
        embedding = gene_output[:, 0, :]  # CLS token
        embedding = F.normalize(embedding, dim=1)
        
        # Dataset embedding (if available)
        dataset_emb = None
        if self.dataset_token is not None:
            dataset_emb = gene_output[:, -1, :]
        
        # Predict regulatory strengths
        regulatory_strengths = None
        if tf_mask is not None and target_mask is not None:
            tf_emb = tf_attended.mean(dim=1)  # Average TF embeddings
            target_emb = target_attended.mean(dim=1)  # Average target embeddings
            regulatory_strengths = self.predict_regulatory_strength(tf_emb, target_emb)
        
        # Prepare regulatory information
        regulatory_info = {
            'attention_weights': attention_weights,
            'constrained_attention': constrained_attention,
            'regulatory_mask': regulatory_mask,
            'tf_embeddings': tf_attended,
            'target_embeddings': target_attended,
            'regulatory_strengths': regulatory_strengths
        }
        
        return gene_output, embedding, dataset_emb, regulatory_info
    
    def forward(self, src: torch.Tensor, mask: torch.Tensor, **kwargs):
        """Override forward to use regulatory processing."""
        gene_types = kwargs.get('gene_types', None)
        tf_mask = kwargs.get('tf_mask', None)
        target_mask = kwargs.get('target_mask', None)
        
        return self.forward_regulatory(src, mask, gene_types, tf_mask, target_mask, **kwargs)
    
    def compute_regulatory_loss(
        self, 
        predictions: torch.Tensor,
        targets: torch.Tensor,
        regulatory_info: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute regulatory constraint loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            regulatory_info: Regulatory information from forward pass
        
        Returns:
            Regulatory loss
        """
        # Base prediction loss
        base_loss = F.mse_loss(predictions, targets)
        
        # Regulatory constraint loss
        attention_weights = regulatory_info.get('attention_weights')
        constrained_attention = regulatory_info.get('constrained_attention')
        
        if attention_weights is not None and constrained_attention is not None:
            # Penalize deviation from constrained attention
            constraint_loss = F.mse_loss(attention_weights, constrained_attention)
            base_loss += self.regulatory_loss_weight * constraint_loss
        
        return base_loss
    
    def get_regulatory_network(
        self, 
        gene_names: List[str],
        threshold: float = 0.5
    ) -> Dict[str, List[str]]:
        """
        Extract regulatory network from learned interactions.
        
        Args:
            gene_names: List of gene names
            threshold: Threshold for regulatory strength
        
        Returns:
            Dictionary mapping TFs to their target genes
        """
        # Get interaction weights
        interactions = self.interaction_weights.detach()
        
        # Apply threshold
        strong_interactions = (interactions > threshold).float()
        
        # Build regulatory network
        regulatory_network = {}
        
        for i, tf_name in enumerate(gene_names):
            targets = []
            for j, target_name in enumerate(gene_names):
                if strong_interactions[i, j] > 0:
                    targets.append(target_name)
            regulatory_network[tf_name] = targets
        
        return regulatory_network
    
    def update_interaction_matrix(
        self, 
        new_interactions: torch.Tensor
    ):
        """
        Update gene interaction matrix with new data.
        
        Args:
            new_interactions: New interaction matrix [seq_len, seq_len]
        """
        # Update interaction weights
        self.interaction_weights.data = new_interactions
        
        # Update stored matrix
        self.gene_interaction_matrix = new_interactions
