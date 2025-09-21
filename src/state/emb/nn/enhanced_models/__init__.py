"""
Enhanced Virtual Cell Models

This package contains biologically-inspired enhancements to the base StateEmbeddingModel.
Each model implements specific biological principles to improve the realism of the virtual cell.

Models:
- HierarchicalGeneModel: Hierarchical gene organization by pathways and compartments
- TemporalDynamicsModel: Temporal dynamics with fast/slow responses
- RegulatoryConstrainedModel: Biologically constrained attention mechanisms
- CellularMemoryModel: Epigenetic memory and cellular state memory
- StochasticCellularModel: Stochastic cellular behavior and noise modeling
- ResourceConstrainedModel: Energy and resource constraint modeling
- MultiScaleModel: Multi-scale biological processing
- Cumulative models: Combinations of multiple enhancements
"""

from .base_enhanced import BaseEnhancedModel
from .hierarchical import HierarchicalGeneModel
from .temporal import TemporalDynamicsModel
from .regulatory import RegulatoryConstrainedModel
from .memory import CellularMemoryModel
from .stochastic import StochasticCellularModel
from .resource import ResourceConstrainedModel
from .multiscale import MultiScaleModel
from .cumulative import (
    HierarchicalTemporalModel,
    HierarchicalRegulatoryModel,
    TemporalMemoryModel,
    StochasticResourceModel,
    FullVirtualCellModel
)

__all__ = [
    "BaseEnhancedModel",
    "HierarchicalGeneModel",
    "TemporalDynamicsModel", 
    "RegulatoryConstrainedModel",
    "CellularMemoryModel",
    "StochasticCellularModel",
    "ResourceConstrainedModel",
    "MultiScaleModel",
    "HierarchicalTemporalModel",
    "HierarchicalRegulatoryModel",
    "TemporalMemoryModel",
    "StochasticResourceModel",
    "FullVirtualCellModel"
]
