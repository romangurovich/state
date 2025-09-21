"""
Temporal Dynamics Model

Implements temporal dynamics for cellular responses to perturbations,
capturing fast and slow response kinetics and feedback loops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
from .base_enhanced import BaseEnhancedModel


class TemporalDynamicsModel(BaseEnhancedModel):
    """
    Temporal dynamics model that captures:
    1. Fast response kinetics (immediate gene responses)
    2. Slow response kinetics (delayed gene responses)
    3. Feedback loops and regulatory mechanisms
    4. Time-dependent attention mechanisms
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
        # Temporal-specific parameters
        time_steps: int = 5,
        fast_response_layers: int = 2,
        slow_response_layers: int = 3,
        feedback_layers: int = 2,
        temporal_attention_heads: int = 8,
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
            time_steps=time_steps,
            **kwargs
        )
        
        self.fast_response_layers = fast_response_layers
        self.slow_response_layers = slow_response_layers
        self.feedback_layers = feedback_layers
        self.temporal_attention_heads = temporal_attention_heads
        
        self._init_temporal_components()
    
    def _init_temporal_components(self):
        """Initialize temporal processing components."""
        # Fast response LSTM (immediate responses)
        self.fast_response_lstm = nn.LSTM(
            self.d_model,
            self.d_model,
            num_layers=self.fast_response_layers,
            batch_first=True,
            dropout=self.dropout if self.fast_response_layers > 1 else 0
        )
        
        # Slow response LSTM (delayed responses)
        self.slow_response_lstm = nn.LSTM(
            self.d_model,
            self.d_model,
            num_layers=self.slow_response_layers,
            batch_first=True,
            dropout=self.dropout if self.slow_response_layers > 1 else 0
        )
        
        # Feedback LSTM (regulatory feedback)
        self.feedback_lstm = nn.LSTM(
            self.d_model * 2,  # Input + previous output
            self.d_model,
            num_layers=self.feedback_layers,
            batch_first=True,
            dropout=self.dropout if self.feedback_layers > 1 else 0
        )
        
        # Temporal attention mechanism
        self.temporal_attention = nn.MultiheadAttention(
            self.d_model,
            num_heads=self.temporal_attention_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Time-dependent processing
        self.time_encoder = nn.Sequential(
            nn.Linear(1, self.d_model // 4),
            nn.SiLU(),
            nn.Linear(self.d_model // 4, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Response type classifier (fast vs slow)
        self.response_classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, 2)  # Fast (0) or Slow (1)
        )
        
        # Temporal fusion
        self.temporal_fusion = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),  # Fast + Slow + Feedback
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Time-dependent decoders
        self.temporal_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.SiLU(),
                nn.Linear(self.d_model, self.output_dim)
            ) for _ in range(self.time_steps)
        ])
        
        # Feedback gates
        self.feedback_gate = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Sigmoid()
        )
    
    def process_fast_response(
        self, 
        gene_embeddings: torch.Tensor,
        time_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        Process fast response kinetics.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
            time_encoding: Time encoding [batch, seq_len, d_model]
        
        Returns:
            Fast response embeddings
        """
        # Combine gene embeddings with time encoding
        fast_input = gene_embeddings + time_encoding
        
        # Process through fast response LSTM
        fast_output, _ = self.fast_response_lstm(fast_input)
        
        return fast_output
    
    def process_slow_response(
        self, 
        gene_embeddings: torch.Tensor,
        time_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        Process slow response kinetics.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
            time_encoding: Time encoding [batch, seq_len, d_model]
        
        Returns:
            Slow response embeddings
        """
        # Combine gene embeddings with time encoding
        slow_input = gene_embeddings + time_encoding
        
        # Process through slow response LSTM
        slow_output, _ = self.slow_response_lstm(slow_input)
        
        return slow_output
    
    def process_feedback_loops(
        self, 
        fast_response: torch.Tensor,
        slow_response: torch.Tensor,
        previous_output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process feedback loops between fast and slow responses.
        
        Args:
            fast_response: Fast response embeddings [batch, seq_len, d_model]
            slow_response: Slow response embeddings [batch, seq_len, d_model]
            previous_output: Previous time step output [batch, seq_len, d_model]
        
        Returns:
            Feedback-processed embeddings
        """
        # Combine fast and slow responses
        combined_response = torch.cat([fast_response, slow_response], dim=-1)
        
        # Add previous output if available
        if previous_output is not None:
            feedback_input = torch.cat([combined_response, previous_output], dim=-1)
        else:
            feedback_input = combined_response
        
        # Process through feedback LSTM
        feedback_output, _ = self.feedback_lstm(feedback_input)
        
        return feedback_output
    
    def apply_temporal_attention(
        self, 
        temporal_embeddings: torch.Tensor,
        time_steps: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply temporal attention across time steps.
        
        Args:
            temporal_embeddings: Temporal embeddings [batch, time_steps, d_model]
            time_steps: Time step indices [batch, time_steps]
        
        Returns:
            Temporally attended embeddings
        """
        # Apply temporal attention
        attended_output, attention_weights = self.temporal_attention(
            temporal_embeddings, temporal_embeddings, temporal_embeddings
        )
        
        return attended_output, attention_weights
    
    def classify_response_type(self, gene_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Classify genes as fast or slow responders.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
        
        Returns:
            Response type predictions [batch, seq_len, 2]
        """
        return self.response_classifier(gene_embeddings)
    
    def forward_temporal(
        self, 
        src: torch.Tensor, 
        mask: torch.Tensor,
        time_steps: Optional[torch.Tensor] = None,
        previous_states: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Temporal forward pass.
        
        Args:
            src: Input tensor [batch, seq_len, token_dim]
            mask: Attention mask [batch, seq_len]
            time_steps: Time step indices [batch, seq_len]
            previous_states: Previous time step states
        
        Returns:
            Tuple of (gene_output, embedding, dataset_emb, temporal_info)
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
        
        # Generate time encoding
        if time_steps is None:
            time_steps = torch.zeros(src.size(0), src.size(1), device=src.device)
        
        time_encoding = self.time_encoder(time_steps.unsqueeze(-1).float())
        
        # Process fast response
        fast_response = self.process_fast_response(src, time_encoding)
        
        # Process slow response
        slow_response = self.process_slow_response(src, time_encoding)
        
        # Process feedback loops
        previous_output = previous_states[-1] if previous_states else None
        feedback_response = self.process_feedback_loops(
            fast_response, slow_response, previous_output
        )
        
        # Fuse temporal responses
        temporal_fused = self.temporal_fusion(
            torch.cat([fast_response, slow_response, feedback_response], dim=-1)
        )
        
        # Apply temporal attention
        temporal_attended, temporal_attention_weights = self.apply_temporal_attention(
            temporal_fused.unsqueeze(1), time_steps.unsqueeze(-1)
        )
        
        # Apply transformer encoder
        output = self.transformer_encoder(temporal_attended.squeeze(1), src_key_padding_mask=None)
        
        # Decode with temporal information
        gene_output = self.decoder(output)
        
        # Extract embeddings
        embedding = gene_output[:, 0, :]  # CLS token
        embedding = F.normalize(embedding, dim=1)
        
        # Dataset embedding (if available)
        dataset_emb = None
        if self.dataset_token is not None:
            dataset_emb = gene_output[:, -1, :]
        
        # Classify response types
        response_types = self.classify_response_type(src)
        
        # Prepare temporal information
        temporal_info = {
            'fast_response': fast_response,
            'slow_response': slow_response,
            'feedback_response': feedback_response,
            'temporal_attention_weights': temporal_attention_weights,
            'response_types': response_types,
            'time_encoding': time_encoding
        }
        
        return gene_output, embedding, dataset_emb, temporal_info
    
    def forward(self, src: torch.Tensor, mask: torch.Tensor, **kwargs):
        """Override forward to use temporal processing."""
        time_steps = kwargs.get('time_steps', None)
        previous_states = kwargs.get('previous_states', None)
        
        return self.forward_temporal(src, mask, time_steps, previous_states, **kwargs)
    
    def predict_temporal_sequence(
        self, 
        initial_state: torch.Tensor,
        num_steps: int = 5
    ) -> List[torch.Tensor]:
        """
        Predict temporal sequence of cellular responses.
        
        Args:
            initial_state: Initial cellular state [batch, d_model]
            num_steps: Number of time steps to predict
        
        Returns:
            List of predicted states for each time step
        """
        predictions = []
        current_state = initial_state
        
        for t in range(num_steps):
            # Create time encoding
            time_encoding = self.time_encoder(torch.tensor([[t]], device=initial_state.device).float())
            
            # Process through temporal components
            fast_resp = self.process_fast_response(current_state.unsqueeze(1), time_encoding.unsqueeze(0))
            slow_resp = self.process_slow_response(current_state.unsqueeze(1), time_encoding.unsqueeze(0))
            
            # Apply feedback
            feedback_resp = self.process_feedback_loops(fast_resp, slow_resp)
            
            # Fuse responses
            fused = self.temporal_fusion(
                torch.cat([fast_resp, slow_resp, feedback_resp], dim=-1)
            )
            
            # Update state
            current_state = fused.squeeze(1)
            predictions.append(current_state)
        
        return predictions
    
    def compute_temporal_consistency_loss(
        self, 
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            predictions: List of predicted states
            targets: List of target states
        
        Returns:
            Temporal consistency loss
        """
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have the same length")
        
        losses = []
        for pred, target in zip(predictions, targets):
            loss = F.mse_loss(pred, target)
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
    def get_response_kinetics(
        self, 
        gene_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get response kinetics for each gene.
        
        Args:
            gene_embeddings: Gene embeddings [batch, seq_len, d_model]
        
        Returns:
            Dictionary containing response kinetics
        """
        # Classify response types
        response_types = self.classify_response_type(gene_embeddings)
        
        # Get fast and slow response probabilities
        fast_prob = F.softmax(response_types, dim=-1)[:, :, 0]
        slow_prob = F.softmax(response_types, dim=-1)[:, :, 1]
        
        return {
            'fast_response_prob': fast_prob,
            'slow_response_prob': slow_prob,
            'response_types': response_types
        }
