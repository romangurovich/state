"""
Resource-Constrained Cellular Model

Implements energy and resource constraint modeling to capture
the limitations of cellular resources and metabolic state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
from .base_enhanced import BaseEnhancedModel


class ResourceConstrainedModel(BaseEnhancedModel):
    """
    Resource-constrained model that implements:
    1. Energy state tracking (ATP, NADH, etc.)
    2. Resource allocation mechanisms
    3. Resource-aware attention mechanisms
    4. Metabolic constraint enforcement
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
        # Resource-specific parameters
        resource_dim: int = 32,
        energy_dim: int = 16,
        resource_heads: int = 8,
        energy_efficiency: float = 0.8,
        resource_decay_rate: float = 0.95,
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
            resource_dim=resource_dim,
            **kwargs
        )
        
        self.energy_dim = energy_dim
        self.resource_heads = resource_heads
        self.energy_efficiency = energy_efficiency
        self.resource_decay_rate = resource_decay_rate
        
        self._init_resource_components()
    
    def _init_resource_components(self):
        """Initialize resource constraint components."""
        # Energy state tracker
        self.energy_tracker = nn.Sequential(
            nn.Linear(self.d_model, self.energy_dim * 2),  # ATP, NADH, etc.
            nn.SiLU(),
            nn.Linear(self.energy_dim * 2, self.energy_dim),
            nn.Sigmoid()  # Energy levels between 0 and 1
        )
        
        # Resource state tracker
        self.resource_tracker = nn.Sequential(
            nn.Linear(self.d_model, self.resource_dim),
            nn.SiLU(),
            nn.Linear(self.resource_dim, self.resource_dim),
            nn.Sigmoid()  # Resource levels between 0 and 1
        )
        
        # Resource allocation mechanism
        self.resource_allocator = nn.Sequential(
            nn.Linear(self.d_model + self.resource_dim + self.energy_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Resource-aware attention
        self.resource_attention = nn.MultiheadAttention(
            self.d_model,
            num_heads=self.resource_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Energy cost predictor
        self.energy_cost_predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, 1),
            nn.Softplus()  # Energy cost is always positive
        )
        
        # Resource competition mechanism
        self.resource_competition = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()  # Resource allocation probability
        )
        
        # Metabolic state encoder
        self.metabolic_encoder = nn.Sequential(
            nn.Linear(self.energy_dim + self.resource_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Resource efficiency optimizer
        self.efficiency_optimizer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Resource decay mechanism
        self.resource_decay = nn.Parameter(torch.ones(self.resource_dim) * self.resource_decay_rate)
        self.energy_decay = nn.Parameter(torch.ones(self.energy_dim) * self.resource_decay_rate)
    
    def track_energy_state(
        self, 
        gene_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Track cellular energy state.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
        
        Returns:
            Energy state [batch, energy_dim]
        """
        # Get current energy state
        energy_state = self.energy_tracker(gene_embeddings.mean(dim=1))  # [batch, energy_dim]
        
        return energy_state
    
    def track_resource_state(
        self, 
        gene_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Track cellular resource state.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
        
        Returns:
            Resource state [batch, resource_dim]
        """
        # Get current resource state
        resource_state = self.resource_tracker(gene_embeddings.mean(dim=1))  # [batch, resource_dim]
        
        return resource_state
    
    def predict_energy_cost(
        self, 
        gene_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict energy cost for gene expression.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
        
        Returns:
            Energy costs [batch, seq_len, 1]
        """
        # Predict energy cost for each gene
        energy_costs = self.energy_cost_predictor(gene_embeddings)
        
        return energy_costs
    
    def allocate_resources(
        self, 
        gene_embeddings: torch.Tensor,
        energy_state: torch.Tensor,
        resource_state: torch.Tensor,
        energy_costs: torch.Tensor
    ) -> torch.Tensor:
        """
        Allocate resources based on energy and resource constraints.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
            energy_state: Energy state [batch, energy_dim]
            resource_state: Resource state [batch, resource_dim]
            energy_costs: Energy costs [batch, seq_len, 1]
        
        Returns:
            Resource-allocated embeddings
        """
        batch_size, seq_len, d_model = gene_embeddings.shape
        
        # Expand energy and resource states
        energy_expanded = energy_state.unsqueeze(1).expand(-1, seq_len, -1)
        resource_expanded = resource_state.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine with gene embeddings
        combined_input = torch.cat([
            gene_embeddings, 
            resource_expanded, 
            energy_expanded
        ], dim=-1)
        
        # Allocate resources
        resource_allocated = self.resource_allocator(combined_input)
        
        # Apply resource competition
        resource_competition_scores = self.resource_competition(
            torch.cat([gene_embeddings, resource_allocated], dim=-1)
        )
        
        # Apply resource allocation
        final_embeddings = gene_embeddings + resource_allocated * resource_competition_scores
        
        return final_embeddings
    
    def apply_resource_aware_attention(
        self, 
        gene_embeddings: torch.Tensor,
        energy_state: torch.Tensor,
        resource_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply resource-aware attention mechanism.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
            energy_state: Energy state [batch, energy_dim]
            resource_state: Resource state [batch, resource_dim]
        
        Returns:
            Tuple of (attended_embeddings, attention_weights)
        """
        # Encode metabolic state
        metabolic_state = self.metabolic_encoder(
            torch.cat([energy_state, resource_state], dim=-1)
        )
        
        # Use metabolic state as query for attention
        attended_output, attention_weights = self.resource_attention(
            gene_embeddings,  # Query
            gene_embeddings,  # Key
            gene_embeddings,  # Value
            key_padding_mask=None
        )
        
        # Apply metabolic state influence
        metabolic_influence = metabolic_state.unsqueeze(1).expand(-1, gene_embeddings.size(1), -1)
        attended_output = attended_output + metabolic_influence
        
        return attended_output, attention_weights
    
    def optimize_efficiency(
        self, 
        gene_embeddings: torch.Tensor,
        energy_costs: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimize gene expression efficiency.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
            energy_costs: Energy costs [batch, seq_len, 1]
        
        Returns:
            Efficiency-optimized embeddings
        """
        # Apply efficiency optimization
        optimized_embeddings = self.efficiency_optimizer(gene_embeddings)
        
        # Weight by energy efficiency
        efficiency_weights = torch.exp(-energy_costs * self.energy_efficiency)
        optimized_embeddings = optimized_embeddings * efficiency_weights
        
        return optimized_embeddings
    
    def apply_resource_decay(
        self, 
        energy_state: torch.Tensor,
        resource_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply resource decay over time.
        
        Args:
            energy_state: Energy state [batch, energy_dim]
            resource_state: Resource state [batch, resource_dim]
        
        Returns:
            Tuple of (decayed_energy, decayed_resources)
        """
        # Apply decay
        decayed_energy = energy_state * self.energy_decay.unsqueeze(0)
        decayed_resources = resource_state * self.resource_decay.unsqueeze(0)
        
        return decayed_energy, decayed_resources
    
    def forward_resource_constrained(
        self, 
        src: torch.Tensor, 
        mask: torch.Tensor,
        previous_energy_state: Optional[torch.Tensor] = None,
        previous_resource_state: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Resource-constrained forward pass.
        
        Args:
            src: Input tensor [batch, seq_len, token_dim]
            mask: Attention mask [batch, seq_len]
            previous_energy_state: Previous energy state
            previous_resource_state: Previous resource state
        
        Returns:
            Tuple of (gene_output, embedding, dataset_emb, resource_info)
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
        
        # Track energy and resource states
        energy_state = self.track_energy_state(src)
        resource_state = self.track_resource_state(src)
        
        # Apply resource decay if previous states provided
        if previous_energy_state is not None:
            energy_state = energy_state * self.energy_decay.unsqueeze(0) + previous_energy_state * 0.1
        if previous_resource_state is not None:
            resource_state = resource_state * self.resource_decay.unsqueeze(0) + previous_resource_state * 0.1
        
        # Predict energy costs
        energy_costs = self.predict_energy_cost(src)
        
        # Allocate resources
        resource_allocated = self.allocate_resources(
            src, energy_state, resource_state, energy_costs
        )
        
        # Apply resource-aware attention
        attended_output, attention_weights = self.apply_resource_aware_attention(
            resource_allocated, energy_state, resource_state
        )
        
        # Optimize efficiency
        efficiency_optimized = self.optimize_efficiency(attended_output, energy_costs)
        
        # Apply transformer encoder
        output = self.transformer_encoder(efficiency_optimized, src_key_padding_mask=None)
        
        # Decode with resource information
        gene_output = self.decoder(output)
        
        # Extract embeddings
        embedding = gene_output[:, 0, :]  # CLS token
        embedding = F.normalize(embedding, dim=1)
        
        # Dataset embedding (if available)
        dataset_emb = None
        if self.dataset_token is not None:
            dataset_emb = gene_output[:, -1, :]
        
        # Prepare resource information
        resource_info = {
            'energy_state': energy_state,
            'resource_state': resource_state,
            'energy_costs': energy_costs,
            'attention_weights': attention_weights,
            'resource_allocated': resource_allocated,
            'efficiency_optimized': efficiency_optimized
        }
        
        return gene_output, embedding, dataset_emb, resource_info
    
    def forward(self, src: torch.Tensor, mask: torch.Tensor, **kwargs):
        """Override forward to use resource-constrained processing."""
        previous_energy_state = kwargs.get('previous_energy_state', None)
        previous_resource_state = kwargs.get('previous_resource_state', None)
        
        return self.forward_resource_constrained(
            src, mask, previous_energy_state, previous_resource_state, **kwargs
        )
    
    def compute_resource_loss(
        self, 
        predictions: torch.Tensor,
        targets: torch.Tensor,
        resource_info: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute resource constraint loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            resource_info: Resource information from forward pass
        
        Returns:
            Resource loss
        """
        # Base prediction loss
        base_loss = F.mse_loss(predictions, targets)
        
        # Energy efficiency loss
        energy_costs = resource_info['energy_costs']
        energy_efficiency_loss = energy_costs.mean()
        
        # Resource utilization loss (encourage efficient resource use)
        resource_state = resource_info['resource_state']
        resource_utilization_loss = (1 - resource_state).mean()
        
        # Total loss
        total_loss = base_loss + 0.1 * energy_efficiency_loss + 0.05 * resource_utilization_loss
        
        return total_loss
    
    def get_resource_statistics(
        self, 
        resource_info: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Get statistics about resource usage.
        
        Args:
            resource_info: Resource information from forward pass
        
        Returns:
            Dictionary containing resource statistics
        """
        energy_state = resource_info['energy_state']
        resource_state = resource_info['resource_state']
        energy_costs = resource_info['energy_costs']
        
        return {
            'energy_level': energy_state.mean(),
            'resource_level': resource_state.mean(),
            'energy_costs': energy_costs.mean(),
            'energy_efficiency': self.energy_efficiency,
            'resource_decay_rates': self.resource_decay,
            'energy_decay_rates': self.energy_decay
        }
    
    def update_resource_constraints(
        self, 
        new_energy_efficiency: float,
        new_resource_decay_rate: float
    ):
        """
        Update resource constraints.
        
        Args:
            new_energy_efficiency: New energy efficiency value
            new_resource_decay_rate: New resource decay rate
        """
        self.energy_efficiency = new_energy_efficiency
        self.resource_decay_rate = new_resource_decay_rate
        self.resource_decay.data.fill_(new_resource_decay_rate)
        self.energy_decay.data.fill_(new_resource_decay_rate)
