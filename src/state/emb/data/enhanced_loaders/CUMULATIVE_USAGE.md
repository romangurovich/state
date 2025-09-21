# Cumulative Data Loaders Usage Guide

This guide explains how to use the custom data loaders for each cumulative combination model.

## Overview

The cumulative loaders are specifically designed for models that combine multiple biological enhancements. Each loader provides the appropriate normalization strategies for its corresponding cumulative model.

## Available Cumulative Loaders

### 1. HierarchicalTemporalLoader

**For**: `HierarchicalTemporalModel`
**Combines**: Hierarchical gene organization + Temporal dynamics

```python
from src.state.emb.data.enhanced_loaders import HierarchicalTemporalLoader
from src.state.emb.nn.enhanced_models import HierarchicalTemporalModel

# Create loader
loader = HierarchicalTemporalLoader(
    cfg=cfg,
    num_pathways=1000,
    num_compartments=5,
    time_steps=5,
    pathway_annotation_file="pathways.txt",
    compartment_annotation_file="compartments.txt",
    temporal_sampling_strategy="uniform"
)

# Create model
model = HierarchicalTemporalModel(
    token_dim=512,
    d_model=512,
    nhead=8,
    d_hid=2048,
    nlayers=6,
    output_dim=128,
    pathway_info=pathway_info,
    compartment_info=compartment_info,
    time_steps=5
)

# Use in training
dataloader = DataLoader(
    dataset,
    batch_size=cfg.model.batch_size,
    collate_fn=loader,
    num_workers=4
)
```

**Key Features**:
- Pathway and compartment awareness
- Temporal dynamics processing
- Combined hierarchical-temporal statistics
- Cross-correlation analysis

### 2. HierarchicalRegulatoryLoader

**For**: `HierarchicalRegulatoryModel`
**Combines**: Hierarchical gene organization + Regulatory constraints

```python
from src.state.emb.data.enhanced_loaders import HierarchicalRegulatoryLoader
from src.state.emb.nn.enhanced_models import HierarchicalRegulatoryModel

# Create loader
loader = HierarchicalRegulatoryLoader(
    cfg=cfg,
    num_pathways=1000,
    num_compartments=5,
    interaction_matrix_file="interactions.pt",
    gene_type_annotation_file="gene_types.txt",
    regulatory_strength_threshold=0.5
)

# Create model
model = HierarchicalRegulatoryModel(
    token_dim=512,
    d_model=512,
    nhead=8,
    d_hid=2048,
    nlayers=6,
    output_dim=128,
    pathway_info=pathway_info,
    compartment_info=compartment_info,
    gene_interaction_matrix=interaction_matrix
)
```

**Key Features**:
- Pathway and compartment awareness
- Gene interaction constraints
- Regulatory network processing
- TF/Target gene classification

### 3. TemporalMemoryLoader

**For**: `TemporalMemoryModel`
**Combines**: Temporal dynamics + Cellular memory

```python
from src.state.emb.data.enhanced_loaders import TemporalMemoryLoader
from src.state.emb.nn.enhanced_models import TemporalMemoryModel

# Create loader
loader = TemporalMemoryLoader(
    cfg=cfg,
    time_steps=5,
    memory_dim=512,
    memory_size=1000,
    num_memory_types=5,
    num_epigenetic_states=3,
    memory_decay_rate=0.9
)

# Create model
model = TemporalMemoryModel(
    token_dim=512,
    d_model=512,
    nhead=8,
    d_hid=2048,
    nlayers=6,
    output_dim=128,
    time_steps=5,
    memory_size=1000
)
```

**Key Features**:
- Temporal response kinetics
- Epigenetic memory tracking
- Memory decay modeling
- Temporal-memory correlations

### 4. StochasticResourceLoader

**For**: `StochasticResourceModel`
**Combines**: Stochastic behavior + Resource constraints

```python
from src.state.emb.data.enhanced_loaders import StochasticResourceLoader
from src.state.emb.nn.enhanced_models import StochasticResourceModel

# Create loader
loader = StochasticResourceLoader(
    cfg=cfg,
    noise_dim=64,
    num_noise_types=3,
    resource_dim=32,
    num_resource_types=5,
    energy_dim=16,
    nutrient_dim=16,
    noise_strength_range=(0.1, 0.9),
    resource_efficiency_range=(0.1, 1.0)
)

# Create model
model = StochasticResourceModel(
    token_dim=512,
    d_model=512,
    nhead=8,
    d_hid=2048,
    nlayers=6,
    output_dim=128,
    noise_dim=64,
    resource_dim=32
)
```

**Key Features**:
- Intrinsic and extrinsic noise modeling
- Resource allocation mechanisms
- Energy and nutrient constraints
- Stochastic-resource correlations

### 5. FullVirtualCellLoader (Cumulative)

**For**: `FullVirtualCellModel`
**Combines**: All biological enhancements

```python
from src.state.emb.data.enhanced_loaders import CumulativeFullVirtualCellLoader
from src.state.emb.nn.enhanced_models import FullVirtualCellModel

# Create loader
loader = CumulativeFullVirtualCellLoader(
    cfg=cfg,
    num_pathways=1000,
    num_compartments=5,
    time_steps=5,
    memory_dim=512,
    noise_dim=64,
    resource_dim=32,
    num_scales=3,
    pathway_annotation_file="pathways.txt",
    compartment_annotation_file="compartments.txt",
    interaction_matrix_file="interactions.pt",
    gene_type_annotation_file="gene_types.txt"
)

# Create model
model = FullVirtualCellModel(
    token_dim=512,
    d_model=512,
    nhead=8,
    d_hid=2048,
    nlayers=6,
    output_dim=128,
    pathway_info=pathway_info,
    compartment_info=compartment_info,
    time_steps=5,
    memory_size=1000,
    noise_dim=64,
    resource_dim=32
)
```

**Key Features**:
- All normalization strategies combined
- Comprehensive biological context
- Cross-component correlation analysis
- Virtual cell metrics

## Usage Patterns

### Basic Usage

```python
# 1. Create the appropriate loader for your model
loader = HierarchicalTemporalLoader(cfg=cfg, ...)

# 2. Create your cumulative model
model = HierarchicalTemporalModel(...)

# 3. Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=cfg.model.batch_size,
    collate_fn=loader,
    num_workers=4
)

# 4. Use in training loop
for batch in dataloader:
    # The loader provides all necessary context
    batch_sentences, Xs, Ys, idxs, batch_weights, masks, total_counts_all, batch_sentences_counts, dataset_nums, *context = batch
    
    # Your model processes the enhanced data
    outputs = model(batch_sentences, masks, **context)
```

### Advanced Usage with Statistics

```python
# Monitor normalization effectiveness
for epoch in range(num_epochs):
    for batch in dataloader:
        # Training step
        ...
    
    # Get comprehensive statistics
    stats = loader.get_combined_stats()
    print(f"Epoch {epoch} stats: {stats}")
    
    # Reset statistics for next epoch
    loader.reset_combined_stats()
```

### Custom Processing

```python
# Custom count processing
def custom_processing(loader, counts, gene_names):
    # Use the loader's specialized processing
    processed_counts, context = loader.hierarchical_temporal_count_processing(
        counts, gene_names
    )
    
    # Add custom processing
    custom_processed = your_custom_function(processed_counts)
    
    return custom_processed, context

# Use in your pipeline
processed_counts, context = custom_processing(loader, counts, gene_names)
```

## Statistics and Monitoring

Each cumulative loader provides comprehensive statistics:

```python
# Get combined statistics
stats = loader.get_combined_stats()

# Examples of available statistics:
print(f"Pathway-temporal correlation: {stats['avg_pathway_temporal_correlation']}")
print(f"Compartment-regulatory interaction: {stats['avg_compartment_regulatory_interaction']}")
print(f"Temporal-memory correlation: {stats['avg_temporal_memory_correlation']}")
print(f"Stochastic-resource efficiency: {stats['avg_stochastic_resource_efficiency']}")
print(f"Virtual cell metric: {stats['avg_virtual_cell_metric']}")
```

## Best Practices

### 1. Choose the Right Loader
- Match the loader to your model exactly
- Use `FullVirtualCellLoader` for the most comprehensive model
- Use specific cumulative loaders for targeted combinations

### 2. Monitor Statistics
- Track normalization effectiveness
- Monitor cross-component correlations
- Watch for biological plausibility

### 3. Tune Parameters
- Adjust normalization parameters based on your data
- Use appropriate annotation files when available
- Consider computational constraints

### 4. Handle Context
- Ensure your model can handle the additional context
- Use the context information effectively
- Maintain biological interpretability

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or use gradient checkpointing
2. **Context Mismatch**: Ensure model expects the context provided by loader
3. **Statistics Overflow**: Reset statistics regularly
4. **Performance Issues**: Use compiled models and mixed precision

### Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check loader configuration
print(f"Loader type: {type(loader)}")
print(f"Combined normalization: {loader.combined_normalization}")

# Monitor statistics
stats = loader.get_combined_stats()
print(f"Statistics: {stats}")
```

## Performance Considerations

- **Memory Usage**: Cumulative loaders use more memory due to multiple normalization strategies
- **Computational Overhead**: Additional processing adds computational cost
- **Training Stability**: Enhanced normalization can improve training stability
- **Biological Realism**: More accurate representation of cellular processes

## Future Enhancements

- **Adaptive Normalization**: Automatically adjust based on data characteristics
- **Cross-Modal Support**: Handle multiple data modalities
- **Dynamic Weighting**: Adjust normalization weights during training
- **Efficient Implementations**: Optimized versions for large-scale training

## Contributing

When adding new cumulative loaders:

1. Inherit from appropriate parent loaders
2. Implement combined processing methods
3. Add comprehensive statistics tracking
4. Include proper documentation
5. Add unit tests
6. Update this usage guide
