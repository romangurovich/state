# State Embedding Model Architecture

```mermaid
graph TB
    %% Input Processing
    subgraph Input["Input Processing"]
        PE[Protein Embeddings<br/>E in R N x token_dim]
        CLS[CLS Token<br/>theta_cls in R token_dim]
        DS[Dataset Token<br/>theta_ds in R token_dim]
        CNT[Count Values<br/>c in R N]
    end
    
    %% Encoder Stack
    subgraph Encoder["Encoder Stack"]
        LIN1[Linear<br/>R token_dim to R d_model]
        LN1[LayerNorm<br/>R d_model]
        SILU[SiLU Activation]
        SCALE[Scale by sqrt d_model]
    end
    
    %% Count Processing
    subgraph CountProc["Count Processing (Optional)"]
        CNT_ENC[Count Encoder<br/>R 1 to R 512 to R 10]
        SOFT[Softmax<br/>Bin Weights]
        BIN_EMB[Bin Embeddings<br/>E_bin in R 10 x d_model]
        WEIGHT[Weighted Sum<br/>sum w_i * E_bin_i]
    end
    
    %% Transformer Stack
    subgraph Transformer["Flash Transformer Stack"]
        direction TB
        T1[FlashTransformerEncoderLayer 1]
        T2[FlashTransformerEncoderLayer 2]
        TDOTS[...]
        TN[FlashTransformerEncoderLayer nlayers]
        
        T1 --> T2
        T2 --> TDOTS
        TDOTS --> TN
    end
    
    %% SkipBlock Definition
    subgraph SkipBlock["SkipBlock Architecture"]
        direction TB
        SB_IN[Input x in R d]
        SB_LIN1[Linear: d to 2d]
        SB_RELU[ReLU]
        SB_LIN2[Linear: 2d to d]
        SB_ADD[Add Residual]
        SB_LN[LayerNorm]
        SB_OUT[Output]
        
        SB_IN --> SB_LIN1
        SB_LIN1 --> SB_RELU
        SB_RELU --> SB_LIN2
        SB_ADD --> SB_LN
        SB_LN --> SB_OUT
        SB_IN -.-> SB_ADD
        SB_LIN2 --> SB_ADD
    end
    
    %% Decoder Stack
    subgraph Decoder["Decoder Stack"]
        SB1[SkipBlock<br/>R d_model to R d_model]
        LIN2[Linear<br/>R d_model to R output_dim]
    end
    
    %% Binary Decoder
    subgraph BinaryDecoder["Binary Decoder"]
        CONCAT[Concatenate<br/>gene_emb, cls_emb, counts, ds_emb]
        SB2[SkipBlock<br/>R output_dim+d_model+z_dim]
        SB3[SkipBlock<br/>R output_dim+d_model+z_dim]
        LIN3[Linear<br/>R output_dim+d_model+z_dim to R 1]
    end
    
    %% Dataset Classification
    subgraph DatasetClass["Dataset Classification (Optional)"]
        DS_ENC[Dataset Encoder<br/>R output_dim to R d_model to R num_datasets]
        CE_LOSS[CrossEntropy Loss]
    end
    
    %% Main Flow
    PE --> LIN1
    CLS --> LIN1
    DS --> LIN1
    LIN1 --> LN1
    LN1 --> SILU
    SILU --> SCALE
    
    CNT --> CNT_ENC
    CNT_ENC --> SOFT
    SOFT --> WEIGHT
    BIN_EMB --> WEIGHT
    WEIGHT --> SCALE
    
    SCALE --> T1
    T1 --> T2
    T2 --> TDOTS
    TDOTS --> TN
    TN --> SB1
    SB1 --> LIN2
    
    %% Binary Decoder Flow
    LIN2 --> CONCAT
    CONCAT --> SB2
    SB2 --> SB3
    SB3 --> LIN3
    
    %% Dataset Classification Flow
    LIN2 --> DS_ENC
    DS_ENC --> CE_LOSS
    
    %% Outputs
    LIN2 --> EMB[State Embedding<br/>e in R output_dim]
    LIN3 --> PRED[Binary Prediction<br/>p in R 1]
    DS_ENC --> DS_PRED[Dataset Prediction<br/>d in R num_datasets]
    
    %% Styling
    classDef input fill:#e3f2fd
    classDef encoder fill:#f3e5f5
    classDef transformer fill:#e8f5e8
    classDef decoder fill:#fff3e0
    classDef output fill:#fce4ec
    classDef optional fill:#f5f5f5,stroke-dasharray: 5 5
    
    class PE,CLS,DS,CNT input
    class LIN1,LN1,SILU,SCALE encoder
    class T1,T2,TDOTS,TN transformer
    class SB1,LIN2,SB2,SB3,LIN3 decoder
    class EMB,PRED,DS_PRED output
    class CountProc,DatasetClass optional
```

## Architecture Components

### **Input Processing**
- **Protein Embeddings**: Pre-trained embeddings E ∈ R^(N×token_dim)
- **CLS Token**: Learnable parameter θ_cls ∈ R^token_dim
- **Dataset Token**: Optional learnable parameter θ_ds ∈ R^token_dim
- **Count Values**: Optional count data c ∈ R^N

### **Encoder Stack**
- **Linear Projection**: R^token_dim → R^d_model
- **Layer Normalization**: Pre-activation normalization
- **SiLU Activation**: Smooth ReLU variant
- **Scaling**: Multiply by √d_model (transformer standard)

### **Count Processing** (Optional)
- **Count Encoder**: 1D → 512D → 10D (soft binning)
- **Softmax**: Convert to bin probabilities
- **Bin Embeddings**: Learnable embeddings E_bin ∈ R^(10×d_model)
- **Weighted Sum**: Σ w_i * E_bin_i

### **Flash Transformer Stack**
- **nlayers** FlashTransformerEncoderLayer instances
- Each layer: Multi-head attention + FFN + residual connections
- Optimized for memory efficiency

### **SkipBlock Architecture**
- **Residual Connection**: x + f(x)
- **MLP**: d → 2d → d with ReLU
- **Layer Normalization**: Post-residual normalization

### **Decoder Stack**
- **SkipBlock**: R^d_model → R^d_model
- **Linear Projection**: R^d_model → R^output_dim

### **Binary Decoder**
- **Concatenation**: [gene_emb, cls_emb, counts, ds_emb]
- **Two SkipBlocks**: Processing concatenated features
- **Final Linear**: R^(output_dim+d_model+z_dim) → R^1

### **Dataset Classification** (Optional)
- **Dataset Encoder**: R^output_dim → R^d_model → R^num_datasets
- **CrossEntropy Loss**: Multi-class classification

## Key Design Decisions

1. **Pre-activation LayerNorm**: Applied before SiLU in encoder
2. **SiLU Activation**: Smooth ReLU for better gradients
3. **SkipBlock Pattern**: Residual connections with MLP expansion
4. **Flash Transformer**: Memory-efficient attention implementation
5. **Multi-task Learning**: Joint embedding + binary prediction + dataset classification
6. **Soft Binning**: Count data processed as probability distributions over bins
