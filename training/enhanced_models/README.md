# Enhanced Models Training Notebooks

This directory contains training notebooks for all enhanced models that implement biological realism improvements to the base State Embedding Model.

## 📚 Available Notebooks

### Individual Enhanced Models

| Notebook | Model | Description | Key Features |
|----------|-------|-------------|--------------|
| `HierarchicalGeneModel_Training.ipynb` | HierarchicalGeneModel | Pathway and compartment organization | Pathway-aware processing, compartment-specific normalization |
| `TemporalDynamicsModel_Training.ipynb` | TemporalDynamicsModel | Cellular response kinetics | Temporal response modeling, feedback loops |
| `RegulatoryConstrainedModel_Training.ipynb` | RegulatoryConstrainedModel | Gene regulatory networks | Interaction modeling, regulatory constraints |
| `CellularMemoryModel_Training.ipynb` | CellularMemoryModel | Epigenetic memory | Memory persistence, epigenetic states |
| `StochasticCellularModel_Training.ipynb` | StochasticCellularModel | Cellular variability | Noise modeling, stochastic behavior |
| `ResourceConstrainedModel_Training.ipynb` | ResourceConstrainedModel | Resource limitations | Energy constraints, resource allocation |
| `MultiScaleModel_Training.ipynb` | MultiScaleModel | Multi-scale integration | Molecular to cellular processing |

### Cumulative Models

| Notebook | Model | Description | Combined Features |
|----------|-------|-------------|-------------------|
| `HierarchicalTemporalModel_Training.ipynb` | HierarchicalTemporalModel | Hierarchical + Temporal | Pathway organization + temporal kinetics |
| `HierarchicalRegulatoryModel_Training.ipynb` | HierarchicalRegulatoryModel | Hierarchical + Regulatory | Pathway organization + regulatory networks |
| `TemporalMemoryModel_Training.ipynb` | TemporalMemoryModel | Temporal + Memory | Temporal kinetics + cellular memory |
| `StochasticResourceModel_Training.ipynb` | StochasticResourceModel | Stochastic + Resource | Variability + resource constraints |
| `FullVirtualCellModel_Training.ipynb` | FullVirtualCellModel | All enhancements | Complete virtual cell simulation |

## 🚀 Quick Start

### 1. Choose Your Model
Select the notebook that matches your biological modeling needs:

- **For pathway analysis**: `HierarchicalGeneModel_Training.ipynb`
- **For temporal studies**: `TemporalDynamicsModel_Training.ipynb`
- **For regulatory networks**: `RegulatoryConstrainedModel_Training.ipynb`
- **For comprehensive modeling**: `FullVirtualCellModel_Training.ipynb`

### 2. Run the Notebook
```bash
# Navigate to the training directory
cd training/enhanced_models

# Launch Jupyter
jupyter notebook

# Open your chosen notebook
```

### 3. Configure Your Model
Each notebook includes configuration sections where you can:
- Set model parameters
- Configure data loaders
- Adjust training settings
- Specify biological annotations

## 📋 Notebook Structure

Each training notebook follows this structure:

### 1. **Setup and Imports**
- Import required libraries
- Set up logging and device configuration
- Import enhanced models and loaders

### 2. **Configuration Setup**
- Model hyperparameters
- Training settings
- Biological annotation files
- Data loader parameters

### 3. **Model Initialization**
- Create the enhanced model
- Initialize with biological parameters
- Display model architecture

### 4. **Data Loader Setup**
- Configure enhanced data loader
- Set up normalization strategies
- Create dummy dataset for demonstration

### 5. **Model Testing**
- Test forward pass
- Verify model functionality
- Check output shapes and types

### 6. **Training Setup**
- Configure Lightning trainer
- Set up optimization parameters
- Enable logging and checkpointing

### 7. **Training Execution**
- Run model training
- Monitor training progress
- Handle training errors

### 8. **Model Evaluation**
- Test trained model
- Generate embeddings
- Analyze model performance

### 9. **Visualization**
- Plot embedding visualizations
- Analyze model-specific statistics
- Display biological insights

### 10. **Model Saving**
- Save trained model
- Export embeddings
- Store configuration

## 🔧 Configuration Examples

### Hierarchical Gene Model
```python
config = {
    'hierarchical': {
        'num_pathways': 1000,
        'num_compartments': 5,
        'pathway_annotation_file': 'data/pathways.txt',
        'compartment_annotation_file': 'data/compartments.txt'
    }
}
```

### Temporal Dynamics Model
```python
config = {
    'temporal': {
        'time_steps': 5,
        'max_sequence_length': 100,
        'temporal_sampling_strategy': 'uniform'
    }
}
```

### Full Virtual Cell Model
```python
config = {
    'full_virtual_cell': {
        'num_pathways': 1000,
        'num_compartments': 5,
        'time_steps': 5,
        'memory_dim': 512,
        'noise_dim': 64,
        'resource_dim': 32,
        'num_scales': 3
    }
}
```

## 📊 Expected Outputs

### Model Training
- Training loss curves
- Validation metrics
- Model checkpoint files

### Embeddings
- Cell state embeddings
- Gene expression representations
- Biological context information

### Statistics
- Model-specific metrics
- Biological validation statistics
- Performance comparisons

### Visualizations
- t-SNE embedding plots
- PCA projections
- Biological pathway analysis
- Temporal dynamics plots

## 🧬 Biological Annotations

### Required Files
- **Pathway annotations**: Gene-to-pathway mappings
- **Compartment annotations**: Cellular localization data
- **Interaction matrices**: Gene regulatory networks
- **Gene type annotations**: Functional classifications

### Example Format
```python
# Pathway annotations
pathway_annotations = {
    'GENE_001': 0,  # Pathway ID
    'GENE_002': 0,
    'GENE_003': 1,
    # ...
}

# Compartment annotations
compartment_annotations = {
    'GENE_001': 0,  # Compartment ID (nucleus, cytoplasm, etc.)
    'GENE_002': 1,
    'GENE_003': 0,
    # ...
}
```

## 🔍 Model Comparison

### Individual Models
- **Focused**: Each model addresses specific biological aspects
- **Modular**: Easy to understand and modify
- **Efficient**: Lower computational requirements
- **Specialized**: Optimized for specific biological questions

### Cumulative Models
- **Comprehensive**: Combine multiple biological aspects
- **Realistic**: More biologically accurate
- **Complex**: Higher computational requirements
- **Integrated**: Holistic cellular modeling

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all enhanced models are installed
   - Check Python path configuration
   - Verify model dependencies

2. **Configuration Errors**
   - Validate configuration parameters
   - Check file paths for annotations
   - Verify data loader settings

3. **Training Issues**
   - Monitor GPU memory usage
   - Adjust batch size if needed
   - Check learning rate settings

4. **Data Issues**
   - Verify dataset format
   - Check annotation file formats
   - Ensure data loader compatibility

### Getting Help

1. **Check Logs**: Review training logs for error messages
2. **Validate Inputs**: Ensure data and configuration are correct
3. **Test Components**: Run individual model components
4. **Debug Mode**: Use smaller datasets for testing

## 📈 Performance Tips

### Training Optimization
- Use mixed precision training
- Enable gradient accumulation
- Monitor memory usage
- Use appropriate batch sizes

### Model Selection
- Start with individual models
- Progress to cumulative models
- Choose based on biological needs
- Consider computational resources

### Data Preparation
- Preprocess annotations
- Validate data formats
- Use appropriate normalization
- Check data quality

## 🔬 Research Applications

### Single-Cell Analysis
- Cell type identification
- Developmental trajectory analysis
- Disease state classification
- Drug response prediction

### Gene Expression Modeling
- Expression prediction
- Perturbation response
- Regulatory network inference
- Pathway activity analysis

### Virtual Cell Simulation
- Cellular behavior modeling
- Drug screening
- Disease mechanism studies
- Therapeutic target identification

## 📚 References

- Enhanced Models: `src/state/emb/nn/enhanced_models/`
- Enhanced Loaders: `src/state/emb/data/enhanced_loaders/`
- Base Model: `src/state/emb/nn/model.py`
- Data Loaders: `src/state/emb/data/loader.py`

## 🤝 Contributing

To add new training notebooks:

1. Follow the existing notebook structure
2. Include comprehensive documentation
3. Add configuration examples
4. Test with dummy data
5. Update this README

## 📄 License

These notebooks are part of the State Embedding Model project and follow the same licensing terms.
