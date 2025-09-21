# Enhanced Virtual Cell Models

This package contains biologically-inspired enhancements to the base StateEmbeddingModel, designed to create more realistic "virtual cells" that better capture the complexity and constraints of real cellular systems.

## Overview

The enhanced models implement various biological principles to improve the realism and accuracy of cell perturbation prediction:

1. **Hierarchical Gene Organization** - Groups genes by pathways and cellular compartments
2. **Temporal Dynamics** - Captures fast/slow response kinetics and feedback loops
3. **Regulatory Network Constraints** - Enforces known gene-gene interactions
4. **Cellular Memory System** - Remembers previous perturbations and states
5. **Stochastic Cellular Behavior** - Models cellular noise and variability
6. **Resource Constraints** - Considers energy and resource limitations
7. **Multi-Scale Processing** - Handles molecular, pathway, and cellular scales

## Models

### Individual Enhancement Models

#### `HierarchicalGeneModel`
- **Purpose**: Organizes genes hierarchically by biological pathways and cellular compartments
- **Key Features**:
  - Pathway-level attention mechanisms
  - Compartment-aware processing
  - Cross-pathway interactions
- **Use Case**: When you have pathway annotations and want to leverage biological structure

#### `TemporalDynamicsModel`
- **Purpose**: Captures temporal dynamics of cellular responses
- **Key Features**:
  - Fast/slow response kinetics
  - Feedback loop modeling
  - Time-dependent attention
- **Use Case**: When you need to model time-dependent responses to perturbations

#### `RegulatoryConstrainedModel`
- **Purpose**: Enforces biological constraints on gene-gene interactions
- **Key Features**:
  - Known interaction matrix integration
  - Directional regulatory relationships
  - Sparse network enforcement
- **Use Case**: When you have prior knowledge about gene regulatory networks

#### `CellularMemoryModel`
- **Purpose**: Implements cellular memory and epigenetic mechanisms
- **Key Features**:
  - Epigenetic memory bank
  - Memory retrieval and update
  - Cellular state memory
- **Use Case**: When you need the model to remember previous perturbations

#### `StochasticCellularModel`
- **Purpose**: Models stochastic cellular behavior and noise
- **Key Features**:
  - Intrinsic and extrinsic noise modeling
  - Heterogeneous response generation
  - Uncertainty quantification
- **Use Case**: When you need to capture cellular variability and noise

#### `ResourceConstrainedModel`
- **Purpose**: Models energy and resource constraints
- **Key Features**:
  - Energy state tracking
  - Resource allocation mechanisms
  - Metabolic constraint enforcement
- **Use Case**: When you need to model cellular resource limitations

#### `MultiScaleModel`
- **Purpose**: Processes information at multiple biological scales
- **Key Features**:
  - Molecular, pathway, and cellular scale processing
  - Cross-scale attention mechanisms
  - Scale integration
- **Use Case**: When you need comprehensive multi-scale modeling

### Cumulative Combination Models

#### `HierarchicalTemporalModel`
- **Combines**: Hierarchical organization + Temporal dynamics
- **Use Case**: When you need both pathway structure and temporal responses

#### `HierarchicalRegulatoryModel`
- **Combines**: Hierarchical organization + Regulatory constraints
- **Use Case**: When you have pathway structure and regulatory knowledge

#### `TemporalMemoryModel`
- **Combines**: Temporal dynamics + Cellular memory
- **Use Case**: When you need temporal responses with memory

#### `StochasticResourceModel`
- **Combines**: Stochastic behavior + Resource constraints
- **Use Case**: When you need variability modeling with resource limits

#### `FullVirtualCellModel`
- **Combines**: All enhancements
- **Use Case**: When you want the most comprehensive and realistic model

## Usage

### Basic Usage

```python
from src.state.emb.nn.enhanced_models import HierarchicalGeneModel

# Initialize model
model = HierarchicalGeneModel(
    token_dim=5120,
    d_model=512,
    nhead=8,
    d_hid=2048,
    nlayers=6,
    output_dim=128,
    num_pathways=1000,
    num_compartments=5
)

# Forward pass with pathway information
pathway_ids = torch.randint(0, 1000, (batch_size, seq_len))
compartment_ids = torch.randint(0, 5, (batch_size, seq_len))

gene_output, embedding, dataset_emb, info = model(
    src, mask, 
    pathway_ids=pathway_ids,
    compartment_ids=compartment_ids
)
```

### Advanced Usage with Multiple Enhancements

```python
from src.state.emb.nn.enhanced_models import FullVirtualCellModel

# Initialize full virtual cell model
model = FullVirtualCellModel(
    token_dim=5120,
    d_model=512,
    nhead=8,
    d_hid=2048,
    nlayers=6,
    output_dim=128,
    # Hierarchical parameters
    num_pathways=1000,
    num_compartments=5,
    # Temporal parameters
    time_steps=5,
    # Memory parameters
    memory_size=1000,
    # Stochastic parameters
    noise_dim=64,
    # Resource parameters
    resource_dim=32,
    # Multi-scale parameters
    molecular_scale_dim=256,
    pathway_scale_dim=512,
    cellular_scale_dim=1024
)

# Forward pass with all enhancements
gene_output, embedding, dataset_emb, full_info = model(
    src, mask,
    pathway_ids=pathway_ids,
    compartment_ids=compartment_ids,
    time_steps=time_steps,
    previous_states=previous_states,
    gene_types=gene_types,
    tf_mask=tf_mask,
    target_mask=target_mask,
    previous_energy_state=energy_state,
    previous_resource_state=resource_state,
    pathway_groups=pathway_groups
)
```

## Model Selection Guide

### For Basic Enhancement
- **Start with**: `HierarchicalGeneModel` or `TemporalDynamicsModel`
- **When to use**: When you have specific biological annotations (pathways, time series)

### For Moderate Enhancement
- **Use**: `HierarchicalTemporalModel` or `StochasticResourceModel`
- **When to use**: When you need multiple complementary enhancements

### For Maximum Realism
- **Use**: `FullVirtualCellModel`
- **When to use**: When you want the most biologically realistic model possible

## Performance Considerations

### Memory Usage
- **Individual models**: 2-4x base model memory
- **Cumulative models**: 4-8x base model memory
- **Full model**: 8-16x base model memory

### Training Time
- **Individual models**: 1.5-3x base model training time
- **Cumulative models**: 3-6x base model training time
- **Full model**: 6-12x base model training time

### Recommended Hardware
- **Individual models**: 16GB+ GPU memory
- **Cumulative models**: 32GB+ GPU memory
- **Full model**: 64GB+ GPU memory

## Biological Interpretability

Each model provides interpretability features:

- **Attention weights**: Show which genes/pathways are most important
- **Memory statistics**: Track cellular memory usage
- **Resource utilization**: Monitor energy and resource consumption
- **Stochastic analysis**: Quantify cellular variability
- **Scale-specific outputs**: Understand different biological scales

## Example Workflows

### 1. Pathway-Aware Perturbation Prediction
```python
model = HierarchicalGeneModel(...)
# Train with pathway annotations
# Analyze pathway-level attention weights
# Predict perturbation effects within pathways
```

### 2. Temporal Response Modeling
```python
model = TemporalDynamicsModel(...)
# Train with time series data
# Predict fast vs slow responses
# Model feedback loops
```

### 3. Memory-Enhanced Prediction
```python
model = CellularMemoryModel(...)
# Train with sequential perturbations
# Leverage memory for better predictions
# Analyze memory utilization
```

### 4. Full Virtual Cell
```python
model = FullVirtualCellModel(...)
# Train with all available biological data
# Get comprehensive cellular modeling
# Analyze all biological aspects
```

## Contributing

To add new enhancements:

1. Create a new model class inheriting from `BaseEnhancedModel`
2. Implement the enhancement in the `_init_*_components()` method
3. Override the `forward_*()` method for the enhancement
4. Add the model to the `__init__.py` file
5. Update this README with documentation

## References

- Base model: StateEmbeddingModel
- Biological inspiration: Cellular biology, systems biology, computational biology
- Implementation: PyTorch, Lightning
