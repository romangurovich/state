# Enhanced Data Loaders

This package contains enhanced data loaders with various normalization strategies designed for different biological models and use cases. Each loader implements specific normalization techniques tailored to the biological principles underlying different model architectures.

## Overview

The enhanced data loaders provide:

- **Biological Context-Aware Normalization**: Normalization strategies that respect biological principles
- **Model-Specific Processing**: Tailored data processing for different enhanced model architectures
- **Comprehensive Statistics Tracking**: Detailed statistics for monitoring normalization effectiveness
- **Modular Design**: Easy integration with existing training pipelines

## Available Loaders

### 1. BaseEnhancedLoader

**Purpose**: Base class with common normalization functionality

**Key Features**:
- Gene-wise and cell-wise normalization
- Robust statistics (median, MAD)
- Quantile normalization
- Batch-aware normalization
- Biological context normalization

**Usage**:
```python
from src.state.emb.data.enhanced_loaders import BaseEnhancedLoader

loader = BaseEnhancedLoader(
    cfg=cfg,
    normalization_type="z_score",
    per_gene_normalization=True,
    per_cell_normalization=True,
    robust_normalization=False,
    quantile_normalization=False,
    batch_aware_normalization=True,
    biological_normalization=True
)
```

### 2. HierarchicalDataLoader

**Purpose**: Normalization for hierarchical gene organization models

**Key Features**:
- Pathway-aware normalization
- Compartment-specific processing
- Hierarchical attention mechanisms
- Cross-pathway interaction modeling

**Usage**:
```python
from src.state.emb.data.enhanced_loaders import HierarchicalDataLoader

loader = HierarchicalDataLoader(
    cfg=cfg,
    num_pathways=1000,
    num_compartments=5,
    pathway_annotation_file="pathways.txt",
    compartment_annotation_file="compartments.txt"
)
```

### 3. TemporalDataLoader

**Purpose**: Normalization for temporal dynamics models

**Key Features**:
- Time-step specific normalization
- Fast/slow response kinetics
- Feedback loop modeling
- Temporal attention mechanisms

**Usage**:
```python
from src.state.emb.data.enhanced_loaders import TemporalDataLoader

loader = TemporalDataLoader(
    cfg=cfg,
    time_steps=5,
    max_sequence_length=100,
    temporal_sampling_strategy="uniform"
)
```

### 4. RegulatoryDataLoader

**Purpose**: Normalization for regulatory network models

**Key Features**:
- Interaction-aware normalization
- Gene type-specific processing (TF vs Target)
- Regulatory strength modeling
- Interaction matrix constraints

**Usage**:
```python
from src.state.emb.data.enhanced_loaders import RegulatoryDataLoader

loader = RegulatoryDataLoader(
    cfg=cfg,
    interaction_matrix_file="interactions.pt",
    gene_type_annotation_file="gene_types.txt",
    regulatory_strength_threshold=0.5
)
```

### 5. MemoryDataLoader

**Purpose**: Normalization for cellular memory models

**Key Features**:
- Memory-aware normalization
- Epigenetic state tracking
- Memory retrieval mechanisms
- Memory decay modeling

**Usage**:
```python
from src.state.emb.data.enhanced_loaders import MemoryDataLoader

loader = MemoryDataLoader(
    cfg=cfg,
    memory_dim=512,
    memory_size=1000,
    num_memory_types=5,
    num_epigenetic_states=3,
    memory_decay_rate=0.9
)
```

### 6. StochasticDataLoader

**Purpose**: Normalization for stochastic cellular behavior models

**Key Features**:
- Noise-aware normalization
- Intrinsic and extrinsic noise modeling
- Variability prediction
- Stochastic sampling

**Usage**:
```python
from src.state.emb.data.enhanced_loaders import StochasticDataLoader

loader = StochasticDataLoader(
    cfg=cfg,
    noise_dim=64,
    num_noise_types=3,
    intrinsic_noise_dim=32,
    extrinsic_noise_dim=32,
    noise_strength_range=(0.1, 0.9)
)
```

### 7. ResourceDataLoader

**Purpose**: Normalization for resource-constrained models

**Key Features**:
- Resource-aware normalization
- Energy constraint modeling
- Nutrient constraint processing
- Resource allocation mechanisms

**Usage**:
```python
from src.state.emb.data.enhanced_loaders import ResourceDataLoader

loader = ResourceDataLoader(
    cfg=cfg,
    resource_dim=32,
    num_resource_types=5,
    energy_dim=16,
    nutrient_dim=16,
    resource_efficiency_range=(0.1, 1.0)
)
```

### 8. MultiScaleDataLoader

**Purpose**: Normalization for multi-scale models

**Key Features**:
- Scale-aware normalization
- Hierarchical processing
- Cross-scale interactions
- Molecular, pathway, and cellular scales

**Usage**:
```python
from src.state.emb.data.enhanced_loaders import MultiScaleDataLoader

loader = MultiScaleDataLoader(
    cfg=cfg,
    molecular_scale_dim=128,
    pathway_scale_dim=256,
    cellular_scale_dim=512,
    num_hierarchy_levels=3,
    num_scales=3
)
```

### 9. FullVirtualCellLoader

**Purpose**: Comprehensive normalization for the full virtual cell model

**Key Features**:
- Combines all normalization strategies
- Comprehensive biological context
- Adaptive weighting
- Full integration

**Usage**:
```python
from src.state.emb.data.enhanced_loaders import FullVirtualCellLoader

loader = FullVirtualCellLoader(
    cfg=cfg,
    num_pathways=1000,
    num_compartments=5,
    time_steps=5,
    memory_dim=512,
    noise_dim=64,
    resource_dim=32,
    pathway_annotation_file="pathways.txt",
    compartment_annotation_file="compartments.txt"
)
```

## Normalization Strategies

### 1. Gene-wise Normalization
- **Z-score normalization**: Standardize gene expression across cells
- **Robust normalization**: Use median and MAD for outlier resistance
- **Quantile normalization**: Align gene expression distributions

### 2. Cell-wise Normalization
- **Total count normalization**: Normalize by total UMI count
- **Cell type normalization**: Normalize within cell types
- **Batch normalization**: Handle batch effects

### 3. Biological Context Normalization
- **Pathway normalization**: Normalize within biological pathways
- **Compartment normalization**: Normalize within cellular compartments
- **Temporal normalization**: Normalize across time steps
- **Regulatory normalization**: Normalize based on regulatory relationships

### 4. Advanced Normalization
- **Memory normalization**: Incorporate cellular memory states
- **Stochastic normalization**: Model cellular variability
- **Resource normalization**: Account for resource constraints
- **Multi-scale normalization**: Process at different biological scales

## Statistics and Monitoring

Each loader provides comprehensive statistics tracking:

```python
# Get normalization statistics
stats = loader.get_normalization_stats()

# Get model-specific statistics
hierarchical_stats = loader.get_pathway_stats()
temporal_stats = loader.get_temporal_stats()
regulatory_stats = loader.get_regulatory_stats()
memory_stats = loader.get_memory_stats()
stochastic_stats = loader.get_stochastic_stats()
resource_stats = loader.get_resource_stats()
multiscale_stats = loader.get_multiscale_stats()
comprehensive_stats = loader.get_comprehensive_stats()

# Reset statistics
loader.reset_normalization_stats()
```

## Integration with Enhanced Models

The enhanced loaders are designed to work seamlessly with the enhanced models:

```python
# Example integration
from src.state.emb.nn.enhanced_models import HierarchicalGeneModel
from src.state.emb.data.enhanced_loaders import HierarchicalDataLoader

# Create enhanced model
model = HierarchicalGeneModel(
    token_dim=512,
    d_model=512,
    nhead=8,
    d_hid=2048,
    nlayers=6,
    output_dim=128,
    pathway_info=pathway_info,
    compartment_info=compartment_info
)

# Create enhanced loader
loader = HierarchicalDataLoader(
    cfg=cfg,
    num_pathways=1000,
    num_compartments=5,
    pathway_annotation_file="pathways.txt",
    compartment_annotation_file="compartments.txt"
)

# Use in training
dataloader = DataLoader(
    dataset,
    batch_size=cfg.model.batch_size,
    collate_fn=loader,
    num_workers=4
)
```

## Performance Considerations

### Memory Usage
- Enhanced loaders use more memory due to additional normalization modules
- Consider reducing batch size if memory is limited
- Use gradient checkpointing for large models

### Computational Overhead
- Normalization adds computational overhead
- Use compiled models when possible
- Consider using mixed precision training

### Training Stability
- Enhanced normalization can improve training stability
- Monitor statistics to ensure normalization is working correctly
- Adjust learning rates if needed

## Best Practices

1. **Start Simple**: Begin with BaseEnhancedLoader and add complexity gradually
2. **Monitor Statistics**: Track normalization statistics to ensure effectiveness
3. **Validate Biologically**: Ensure normalization strategies make biological sense
4. **Tune Parameters**: Adjust normalization parameters based on your data
5. **Use Appropriate Loader**: Choose the loader that matches your model architecture

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or use gradient checkpointing
2. **Training Instability**: Check normalization statistics and adjust parameters
3. **Poor Performance**: Ensure normalization strategies are appropriate for your data
4. **Slow Training**: Use compiled models and mixed precision training

### Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check normalization statistics
stats = loader.get_normalization_stats()
print(f"Normalization stats: {stats}")

# Reset statistics and monitor
loader.reset_normalization_stats()
# ... training ...
stats = loader.get_normalization_stats()
print(f"Updated stats: {stats}")
```

## Future Enhancements

- **Adaptive Normalization**: Automatically adjust normalization based on data characteristics
- **Cross-Modal Normalization**: Handle multiple data modalities (RNA, protein, etc.)
- **Dynamic Normalization**: Adjust normalization during training
- **Efficient Implementations**: Optimized versions for large-scale training

## Contributing

When adding new normalization strategies:

1. Inherit from BaseEnhancedLoader
2. Implement model-specific normalization modules
3. Add comprehensive statistics tracking
4. Include proper documentation
5. Add unit tests
6. Update this README

## License

This package is part of the state embedding project and follows the same license terms.
