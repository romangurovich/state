"""
Cumulative Combination Models

Implements combinations of multiple biological enhancements to create
increasingly sophisticated virtual cell models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
from .base_enhanced import BaseEnhancedModel
from .hierarchical import HierarchicalGeneModel
from .temporal import TemporalDynamicsModel
from .regulatory import RegulatoryConstrainedModel
from .memory import CellularMemoryModel
from .stochastic import StochasticCellularModel
from .resource import ResourceConstrainedModel
from .multiscale import MultiScaleModel


class HierarchicalTemporalModel(HierarchicalGeneModel, TemporalDynamicsModel):
    """
    Combines hierarchical gene organization with temporal dynamics.
    Processes genes hierarchically while capturing temporal response kinetics.
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize both parent classes
        HierarchicalGeneModel.__init__(self, *args, **kwargs)
        TemporalDynamicsModel.__init__(self, *args, **kwargs)
        
        # Override conflicting components
        self._init_combined_components()
    
    def _init_combined_components(self):
        """Initialize combined components."""
        # Combined pathway-temporal processing
        self.pathway_temporal_processor = nn.Sequential(
            nn.Linear(self.d_model + self.pathway_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Temporal pathway attention
        self.temporal_pathway_attention = nn.MultiheadAttention(
            self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
    
    def forward_combined(
        self, 
        src: torch.Tensor, 
        mask: torch.Tensor,
        pathway_ids: torch.Tensor,
        compartment_ids: torch.Tensor,
        time_steps: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """Combined hierarchical-temporal forward pass."""
        # Get hierarchical processing
        gene_output, embedding, dataset_emb, hierarchical_info = self.forward_hierarchical(
            src, mask, pathway_ids, compartment_ids, **kwargs
        )
        
        # Get temporal processing
        temporal_output, temporal_embedding, temporal_dataset_emb, temporal_info = self.forward_temporal(
            src, mask, time_steps, **kwargs
        )
        
        # Combine hierarchical and temporal information
        combined_embedding = (embedding + temporal_embedding) / 2
        
        # Prepare combined information
        combined_info = {
            'hierarchical_info': hierarchical_info,
            'temporal_info': temporal_info,
            'combined_embedding': combined_embedding
        }
        
        return gene_output, combined_embedding, dataset_emb, combined_info


class HierarchicalRegulatoryModel(HierarchicalGeneModel, RegulatoryConstrainedModel):
    """
    Combines hierarchical gene organization with regulatory constraints.
    Processes genes hierarchically while enforcing biological constraints.
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize both parent classes
        HierarchicalGeneModel.__init__(self, *args, **kwargs)
        RegulatoryConstrainedModel.__init__(self, *args, **kwargs)
        
        # Override conflicting components
        self._init_combined_components()
    
    def _init_combined_components(self):
        """Initialize combined components."""
        # Combined pathway-regulatory processing
        self.pathway_regulatory_processor = nn.Sequential(
            nn.Linear(self.d_model + self.pathway_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Regulatory pathway attention
        self.regulatory_pathway_attention = nn.MultiheadAttention(
            self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
    
    def forward_combined(
        self, 
        src: torch.Tensor, 
        mask: torch.Tensor,
        pathway_ids: torch.Tensor,
        compartment_ids: torch.Tensor,
        gene_types: Optional[torch.Tensor] = None,
        tf_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """Combined hierarchical-regulatory forward pass."""
        # Get hierarchical processing
        gene_output, embedding, dataset_emb, hierarchical_info = self.forward_hierarchical(
            src, mask, pathway_ids, compartment_ids, **kwargs
        )
        
        # Get regulatory processing
        regulatory_output, regulatory_embedding, regulatory_dataset_emb, regulatory_info = self.forward_regulatory(
            src, mask, gene_types, tf_mask, target_mask, **kwargs
        )
        
        # Combine hierarchical and regulatory information
        combined_embedding = (embedding + regulatory_embedding) / 2
        
        # Prepare combined information
        combined_info = {
            'hierarchical_info': hierarchical_info,
            'regulatory_info': regulatory_info,
            'combined_embedding': combined_embedding
        }
        
        return gene_output, combined_embedding, dataset_emb, combined_info


class TemporalMemoryModel(TemporalDynamicsModel, CellularMemoryModel):
    """
    Combines temporal dynamics with cellular memory.
    Captures temporal responses while maintaining memory of previous states.
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize both parent classes
        TemporalDynamicsModel.__init__(self, *args, **kwargs)
        CellularMemoryModel.__init__(self, *args, **kwargs)
        
        # Override conflicting components
        self._init_combined_components()
    
    def _init_combined_components(self):
        """Initialize combined components."""
        # Combined temporal-memory processing
        self.temporal_memory_processor = nn.Sequential(
            nn.Linear(self.d_model + self.memory_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Memory-aware temporal attention
        self.memory_temporal_attention = nn.MultiheadAttention(
            self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
    
    def forward_combined(
        self, 
        src: torch.Tensor, 
        mask: torch.Tensor,
        time_steps: Optional[torch.Tensor] = None,
        previous_states: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """Combined temporal-memory forward pass."""
        # Get temporal processing
        temporal_output, temporal_embedding, temporal_dataset_emb, temporal_info = self.forward_temporal(
            src, mask, time_steps, previous_states, **kwargs
        )
        
        # Get memory processing
        memory_output, memory_embedding, memory_dataset_emb, memory_info = self.forward_memory(
            src, mask, previous_states, **kwargs
        )
        
        # Combine temporal and memory information
        combined_embedding = (temporal_embedding + memory_embedding) / 2
        
        # Prepare combined information
        combined_info = {
            'temporal_info': temporal_info,
            'memory_info': memory_info,
            'combined_embedding': combined_embedding
        }
        
        return temporal_output, combined_embedding, temporal_dataset_emb, combined_info


class StochasticResourceModel(StochasticCellularModel, ResourceConstrainedModel):
    """
    Combines stochastic behavior with resource constraints.
    Models cellular variability while respecting resource limitations.
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize both parent classes
        StochasticCellularModel.__init__(self, *args, **kwargs)
        ResourceConstrainedModel.__init__(self, *args, **kwargs)
        
        # Override conflicting components
        self._init_combined_components()
    
    def _init_combined_components(self):
        """Initialize combined components."""
        # Combined stochastic-resource processing
        self.stochastic_resource_processor = nn.Sequential(
            nn.Linear(self.d_model + self.noise_dim + self.resource_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Resource-aware stochastic attention
        self.resource_stochastic_attention = nn.MultiheadAttention(
            self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
    
    def forward_combined(
        self, 
        src: torch.Tensor, 
        mask: torch.Tensor,
        previous_energy_state: Optional[torch.Tensor] = None,
        previous_resource_state: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """Combined stochastic-resource forward pass."""
        # Get stochastic processing
        stochastic_output, stochastic_embedding, stochastic_dataset_emb, stochastic_info = self.forward_stochastic(
            src, mask, **kwargs
        )
        
        # Get resource processing
        resource_output, resource_embedding, resource_dataset_emb, resource_info = self.forward_resource_constrained(
            src, mask, previous_energy_state, previous_resource_state, **kwargs
        )
        
        # Combine stochastic and resource information
        combined_embedding = (stochastic_embedding + resource_embedding) / 2
        
        # Prepare combined information
        combined_info = {
            'stochastic_info': stochastic_info,
            'resource_info': resource_info,
            'combined_embedding': combined_embedding
        }
        
        return stochastic_output, combined_embedding, stochastic_dataset_emb, combined_info


class FullVirtualCellModel(
    HierarchicalGeneModel,
    TemporalDynamicsModel,
    RegulatoryConstrainedModel,
    CellularMemoryModel,
    StochasticCellularModel,
    ResourceConstrainedModel,
    MultiScaleModel
):
    """
    Full virtual cell model combining all biological enhancements.
    This is the most comprehensive model that incorporates:
    - Hierarchical gene organization
    - Temporal dynamics
    - Regulatory constraints
    - Cellular memory
    - Stochastic behavior
    - Resource constraints
    - Multi-scale processing
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize all parent classes
        HierarchicalGeneModel.__init__(self, *args, **kwargs)
        TemporalDynamicsModel.__init__(self, *args, **kwargs)
        RegulatoryConstrainedModel.__init__(self, *args, **kwargs)
        CellularMemoryModel.__init__(self, *args, **kwargs)
        StochasticCellularModel.__init__(self, *args, **kwargs)
        ResourceConstrainedModel.__init__(self, *args, **kwargs)
        MultiScaleModel.__init__(self, *args, **kwargs)
        
        # Override conflicting components
        self._init_full_combined_components()
    
    def _init_full_combined_components(self):
        """Initialize full combined components."""
        # Master integration network
        self.master_integration = nn.Sequential(
            nn.Linear(
                self.d_model + 
                self.pathway_dim + 
                self.memory_dim + 
                self.noise_dim + 
                self.resource_dim + 
                self.molecular_scale_dim + 
                self.pathway_scale_dim + 
                self.cellular_scale_dim,
                self.d_model
            ),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Cross-enhancement attention
        self.cross_enhancement_attention = nn.MultiheadAttention(
            self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Enhancement fusion weights
        self.enhancement_fusion_weights = nn.Parameter(torch.ones(7) / 7)
    
    def forward_full(
        self, 
        src: torch.Tensor, 
        mask: torch.Tensor,
        pathway_ids: Optional[torch.Tensor] = None,
        compartment_ids: Optional[torch.Tensor] = None,
        time_steps: Optional[torch.Tensor] = None,
        previous_states: Optional[List[torch.Tensor]] = None,
        gene_types: Optional[torch.Tensor] = None,
        tf_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        previous_energy_state: Optional[torch.Tensor] = None,
        previous_resource_state: Optional[torch.Tensor] = None,
        pathway_groups: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """Full virtual cell forward pass."""
        # Get all enhancement outputs
        enhancement_outputs = {}
        
        # Hierarchical processing
        if pathway_ids is not None and compartment_ids is not None:
            _, hierarchical_embedding, _, hierarchical_info = self.forward_hierarchical(
                src, mask, pathway_ids, compartment_ids, **kwargs
            )
            enhancement_outputs['hierarchical'] = hierarchical_embedding
        
        # Temporal processing
        if time_steps is not None:
            _, temporal_embedding, _, temporal_info = self.forward_temporal(
                src, mask, time_steps, previous_states, **kwargs
            )
            enhancement_outputs['temporal'] = temporal_embedding
        
        # Regulatory processing
        if gene_types is not None:
            _, regulatory_embedding, _, regulatory_info = self.forward_regulatory(
                src, mask, gene_types, tf_mask, target_mask, **kwargs
            )
            enhancement_outputs['regulatory'] = regulatory_embedding
        
        # Memory processing
        if previous_states is not None:
            _, memory_embedding, _, memory_info = self.forward_memory(
                src, mask, previous_states, **kwargs
            )
            enhancement_outputs['memory'] = memory_embedding
        
        # Stochastic processing
        _, stochastic_embedding, _, stochastic_info = self.forward_stochastic(
            src, mask, **kwargs
        )
        enhancement_outputs['stochastic'] = stochastic_embedding
        
        # Resource processing
        _, resource_embedding, _, resource_info = self.forward_resource_constrained(
            src, mask, previous_energy_state, previous_resource_state, **kwargs
        )
        enhancement_outputs['resource'] = resource_embedding
        
        # Multi-scale processing
        if pathway_groups is not None:
            _, multiscale_embedding, _, multiscale_info = self.forward_multiscale(
                src, mask, pathway_groups, **kwargs
            )
            enhancement_outputs['multiscale'] = multiscale_embedding
        
        # Combine all enhancements
        if enhancement_outputs:
            # Weighted combination of all enhancements
            combined_embedding = torch.zeros_like(stochastic_embedding)
            weight_sum = 0
            
            for i, (enhancement_name, embedding) in enumerate(enhancement_outputs.items()):
                weight = self.enhancement_fusion_weights[i]
                combined_embedding += weight * embedding
                weight_sum += weight
            
            combined_embedding = combined_embedding / weight_sum
        else:
            combined_embedding = stochastic_embedding
        
        # Apply master integration
        integrated_embedding = self.master_integration(
            torch.cat([src.mean(dim=1), combined_embedding], dim=-1)
        )
        
        # Apply transformer encoder
        output = self.transformer_encoder(integrated_embedding.unsqueeze(1), src_key_padding_mask=None)
        
        # Decode with full integration
        gene_output = self.decoder(output)
        
        # Extract embeddings
        embedding = gene_output[:, 0, :]  # CLS token
        embedding = F.normalize(embedding, dim=1)
        
        # Dataset embedding (if available)
        dataset_emb = None
        if self.dataset_token is not None:
            dataset_emb = gene_output[:, -1, :]
        
        # Prepare full information
        full_info = {
            'enhancement_outputs': enhancement_outputs,
            'combined_embedding': combined_embedding,
            'integrated_embedding': integrated_embedding,
            'hierarchical_info': hierarchical_info if 'hierarchical' in enhancement_outputs else None,
            'temporal_info': temporal_info if 'temporal' in enhancement_outputs else None,
            'regulatory_info': regulatory_info if 'regulatory' in enhancement_outputs else None,
            'memory_info': memory_info if 'memory' in enhancement_outputs else None,
            'stochastic_info': stochastic_info,
            'resource_info': resource_info,
            'multiscale_info': multiscale_info if 'multiscale' in enhancement_outputs else None
        }
        
        return gene_output, embedding, dataset_emb, full_info
    
    def forward(self, src: torch.Tensor, mask: torch.Tensor, **kwargs):
        """Override forward to use full virtual cell processing."""
        return self.forward_full(src, mask, **kwargs)
    
    def compute_full_loss(
        self, 
        predictions: torch.Tensor,
        targets: torch.Tensor,
        full_info: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute full virtual cell loss."""
        # Base prediction loss
        base_loss = F.mse_loss(predictions, targets)
        
        # Enhancement-specific losses
        enhancement_losses = []
        
        if full_info['hierarchical_info'] is not None:
            hierarchical_loss = self.compute_pathway_consistency_loss(
                predictions, targets, 
                full_info['hierarchical_info'].get('pathway_ids', None)
            )
            enhancement_losses.append(hierarchical_loss)
        
        if full_info['temporal_info'] is not None:
            temporal_loss = self.compute_temporal_consistency_loss(
                [predictions], [targets]
            )
            enhancement_losses.append(temporal_loss)
        
        if full_info['regulatory_info'] is not None:
            regulatory_loss = self.compute_regulatory_loss(
                predictions, targets, full_info['regulatory_info']
            )
            enhancement_losses.append(regulatory_loss)
        
        if full_info['memory_info'] is not None:
            memory_loss = torch.tensor(0.0, device=predictions.device)  # Placeholder
            enhancement_losses.append(memory_loss)
        
        if full_info['stochastic_info'] is not None:
            stochastic_loss = self.compute_stochastic_loss(
                predictions, targets, 
                full_info['stochastic_info']['uncertainty'],
                full_info['stochastic_info']
            )
            enhancement_losses.append(stochastic_loss)
        
        if full_info['resource_info'] is not None:
            resource_loss = self.compute_resource_loss(
                predictions, targets, full_info['resource_info']
            )
            enhancement_losses.append(resource_loss)
        
        if full_info['multiscale_info'] is not None:
            multiscale_loss = self.compute_multiscale_loss(
                predictions, targets, full_info['multiscale_info']
            )
            enhancement_losses.append(multiscale_loss)
        
        # Weighted combination of all losses
        total_loss = base_loss
        for i, loss in enumerate(enhancement_losses):
            weight = self.enhancement_fusion_weights[i] * 0.1
            total_loss += weight * loss
        
        return total_loss
