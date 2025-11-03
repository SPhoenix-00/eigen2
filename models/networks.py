"""
Neural Network architectures for Project Eigen 2
Actor-Critic networks for stock trading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from torch.utils.checkpoint import checkpoint

from utils.config import Config


class FeatureExtractor(nn.Module):
    """
    Extracts features from multi-column time-series data.
    Uses 1D CNN across features, then LSTM across time.
    """
    
    def __init__(self, num_columns: int = 669, num_features: int = 9):
        super().__init__()
        
        self.num_columns = num_columns
        self.num_features = num_features
        
        # 1D CNN to extract features from the 9 elements per cell
        # Input: [batch, num_columns, context_days, 9]
        # Process each column's time series independently
        self.conv1 = nn.Conv1d(num_features, Config.CNN_FILTERS, 
                               kernel_size=Config.CNN_KERNEL_SIZE, 
                               padding=Config.CNN_KERNEL_SIZE // 2)
        self.bn1 = nn.BatchNorm1d(Config.CNN_FILTERS)
        
        # LSTM to capture temporal dependencies
        # Input: [batch, num_columns, context_days, CNN_FILTERS]
        self.lstm_input_size = Config.CNN_FILTERS
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=Config.LSTM_HIDDEN,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=Config.LSTM_BIDIRECTIONAL,
            dropout=0.1 if Config.LSTM_LAYERS > 1 else 0.0
        )
        
        # Output size after LSTM
        self.lstm_output_size = Config.LSTM_HIDDEN * (2 if Config.LSTM_BIDIRECTIONAL else 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extractor.
        
        Args:
            x: Input tensor [batch, context_days, num_columns, 9]
            
        Returns:
            Features tensor [batch, num_columns, lstm_output_size]
        """
        batch_size = x.shape[0]
        context_days = x.shape[1]
        
        # Reshape to process each column independently
        # [batch, context_days, num_columns, 9] -> [batch * num_columns, 9, context_days]
        x = x.permute(0, 2, 3, 1)  # [batch, num_columns, 9, context_days]
        x = x.reshape(batch_size * self.num_columns, self.num_features, context_days)
        
        # CNN across features
        x = self.conv1(x)  # [batch * num_columns, CNN_FILTERS, context_days]
        x = self.bn1(x)
        x = F.relu(x)
        
        # Reshape for LSTM: [batch * num_columns, context_days, CNN_FILTERS]
        x = x.permute(0, 2, 1)
        
        # LSTM across time
        # Handle NaN values by replacing with zeros (LSTM will learn to ignore)
        x = torch.nan_to_num(x, nan=0.0)
        
        lstm_out, _ = self.lstm(x)  # [batch * num_columns, context_days, lstm_output_size]
        
        # Take last time step output
        x = lstm_out[:, -1, :]  # [batch * num_columns, lstm_output_size]
        
        # Reshape back to separate columns
        x = x.reshape(batch_size, self.num_columns, self.lstm_output_size)
        
        return x


class AttentionModule(nn.Module):
    """
    Multi-head attention to learn which stocks/indicators matter.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_columns, embed_dim]
            
        Returns:
            Attended features [batch, num_columns, embed_dim]
        """
        # Self-attention across columns
        attn_out, _ = self.attention(x, x, x)
        
        # Residual connection + normalization
        x = self.norm(x + attn_out)
        
        return x


class Actor(nn.Module):
    """
    Actor network: outputs actions [coefficient, sale_target] for each stock.
    """
    
    def __init__(self):
        super().__init__()
        
        # Feature extraction
        self.feature_extractor = FeatureExtractor()
        
        # Optional attention
        if Config.USE_ATTENTION:
            self.attention = AttentionModule(
                embed_dim=self.feature_extractor.lstm_output_size,
                num_heads=Config.ATTENTION_HEADS
            )
        else:
            self.attention = None
        
        # Separate processing for investable stocks vs context features
        # Investable stocks (columns 10-117 = 108 stocks)
        investable_input_dim = self.feature_extractor.lstm_output_size
        
        # Context features (remaining columns pooled)
        context_input_dim = self.feature_extractor.lstm_output_size
        
        # Process investable stocks individually
        self.investable_fc = nn.Sequential(
            nn.Linear(investable_input_dim, Config.ACTOR_HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(Config.ACTOR_HIDDEN_DIMS[0], Config.ACTOR_HIDDEN_DIMS[1]),
            nn.ReLU(),
        )
        
        # Process context features
        self.context_fc = nn.Sequential(
            nn.Linear(context_input_dim, Config.ACTOR_HIDDEN_DIMS[1]),
            nn.ReLU(),
        )
        
        # Combined processing for each stock
        combined_dim = Config.ACTOR_HIDDEN_DIMS[1] * 2  # Investable + context
        
        # Output heads for each stock
        self.coefficient_head = nn.Sequential(
            nn.Linear(combined_dim, Config.ACTOR_HIDDEN_DIMS[2]),
            nn.ReLU(),
            nn.Linear(Config.ACTOR_HIDDEN_DIMS[2], 1),
        )
        
        self.sale_target_head = nn.Sequential(
            nn.Linear(combined_dim, Config.ACTOR_HIDDEN_DIMS[2]),
            nn.ReLU(),
            nn.Linear(Config.ACTOR_HIDDEN_DIMS[2], 1),
            nn.Sigmoid()  # Output in [0, 1], will scale to [MIN, MAX] sale target
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: [batch, context_days, num_columns, 9]
            
        Returns:
            actions: [batch, 108, 2] with [coefficient, sale_target] per stock
        """
        batch_size = state.shape[0]
        
        # Extract features
        features = self.feature_extractor(state)
        
        # Apply attention if enabled
        if self.attention is not None:
            features = self.attention(features)
        
        # Split investable stocks from context
        investable_features = features[:, Config.INVESTABLE_START_COL:Config.INVESTABLE_END_COL+1, :]
        # [batch, 108, lstm_output_size]
        
        context_features = torch.cat([
            features[:, :Config.INVESTABLE_START_COL, :],
            features[:, Config.INVESTABLE_END_COL+1:, :]
        ], dim=1)  # [batch, remaining_columns, lstm_output_size]
        
        # Process investable stocks
        investable_processed = self.investable_fc(investable_features)
        # [batch, 108, ACTOR_HIDDEN_DIMS[1]]
        
        # Process and pool context features
        context_processed = self.context_fc(context_features)
        # [batch, remaining_columns, ACTOR_HIDDEN_DIMS[1]]
        context_pooled = torch.mean(context_processed, dim=1, keepdim=True)
        # [batch, 1, ACTOR_HIDDEN_DIMS[1]]
        context_pooled = context_pooled.expand(-1, Config.NUM_INVESTABLE_STOCKS, -1)
        # [batch, 108, ACTOR_HIDDEN_DIMS[1]]
        
        # Combine
        combined = torch.cat([investable_processed, context_pooled], dim=-1)
        # [batch, 108, combined_dim]
        
        # Output heads
        raw_coefficients = self.coefficient_head(combined).squeeze(-1)
        # [batch, 108]
        
        raw_sale_targets = self.sale_target_head(combined).squeeze(-1)
        # [batch, 108]
        
        # Apply activations
        # Coefficient: >= 1 or 0 (using threshold)
        coefficients = self._apply_coefficient_activation(raw_coefficients)
        
        # Sale target: scale from [0, 1] to [MIN_SALE_TARGET, MAX_SALE_TARGET]
        sale_targets = (Config.MIN_SALE_TARGET + 
                       raw_sale_targets * (Config.MAX_SALE_TARGET - Config.MIN_SALE_TARGET))
        
        # Stack into action tensor
        actions = torch.stack([coefficients, sale_targets], dim=-1)
        # [batch, 108, 2]
        
        return actions
    
    def _apply_coefficient_activation(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Apply activation to ensure coefficient >= 1 or ~0.
        Uses smooth activation for better gradient flow during training.
        
        Args:
            raw: Raw output from network
            
        Returns:
            Activated coefficients
        """
        # Smooth activation for better training
        # Use exponential to map wider range of raw values to [1, inf)
        # Negative values -> ~0, Positive values -> >= 1
        coefficients = torch.where(
            raw > 0,
            torch.exp(raw * 0.5) + 0.5,  # Scale down raw, then exp ensures >= 1 for positive
            torch.sigmoid(raw * 2.0) * 0.1  # Nearly 0 for negative
        )
        
        return coefficients


class Critic(nn.Module):
    """
    Critic network: estimates Q-value for state-action pair.
    """
    
    def __init__(self):
        super().__init__()
        
        # Feature extraction (shared with actor)
        self.feature_extractor = FeatureExtractor()
        
        # Optional attention
        if Config.USE_ATTENTION:
            self.attention = AttentionModule(
                embed_dim=self.feature_extractor.lstm_output_size,
                num_heads=Config.ATTENTION_HEADS
            )
        else:
            self.attention = None
        
        # Action encoding
        # Actions are [108, 2], flatten to 216
        action_dim = Config.NUM_INVESTABLE_STOCKS * Config.ACTION_DIM
        
        # State features: pool all columns
        state_dim = self.feature_extractor.lstm_output_size
        
        # Critic network
        self.critic_fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, Config.CRITIC_HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(Config.CRITIC_HIDDEN_DIMS[0], Config.CRITIC_HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Linear(Config.CRITIC_HIDDEN_DIMS[1], 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: [batch, context_days, num_columns, 9]
            action: [batch, 108, 2]
            
        Returns:
            Q-values: [batch, 1]
        """
        # Extract features
        features = self.feature_extractor(state)
        
        # Apply attention if enabled
        if self.attention is not None:
            features = self.attention(features)
        
        # Pool features across all columns
        state_features = torch.mean(features, dim=1)  # [batch, lstm_output_size]
        
        # Flatten action
        action_flat = action.reshape(action.shape[0], -1)  # [batch, 216]
        
        # Concatenate state and action
        x = torch.cat([state_features, action_flat], dim=-1)
        
        # Critic network
        q_value = self.critic_fc(x)  # [batch, 1]
        
        return q_value


# Test networks
if __name__ == "__main__":
    print("Testing Neural Networks...\n")
    
    # Set device
    device = Config.DEVICE
    print(f"Using device: {device}\n")
    
    # Create dummy input
    batch_size = 4
    dummy_state = torch.randn(batch_size, Config.CONTEXT_WINDOW_DAYS, 669, Config.FEATURES_PER_CELL).to(device)
    
    print(f"Input state shape: {dummy_state.shape}")
    
    # Test Actor
    print("\n--- Testing Actor ---")
    actor = Actor().to(device)
    print(f"Actor parameters: {sum(p.numel() for p in actor.parameters()):,}")
    
    with torch.no_grad():
        actions = actor(dummy_state)
    
    print(f"Output actions shape: {actions.shape}")
    print(f"Coefficients range: [{actions[:, :, 0].min():.2f}, {actions[:, :, 0].max():.2f}]")
    print(f"Sale targets range: [{actions[:, :, 1].min():.2f}, {actions[:, :, 1].max():.2f}]")
    print(f"Number of non-zero coefficients: {(actions[:, :, 0] > 0).sum().item()} / {actions[:, :, 0].numel()}")
    print(f"Number with coef > 1.0: {(actions[:, :, 0] > 1.0).sum().item()}")
    
    # Test Critic
    print("\n--- Testing Critic ---")
    critic = Critic().to(device)
    print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")
    
    with torch.no_grad():
        q_values = critic(dummy_state, actions)
    
    print(f"Output Q-values shape: {q_values.shape}")
    print(f"Q-values range: [{q_values.min():.2f}, {q_values.max():.2f}]")
    
    # Test gradient flow
    print("\n--- Testing Gradient Flow ---")
    actor.train()
    critic.train()
    
    actions = actor(dummy_state)
    q_values = critic(dummy_state, actions)
    loss = q_values.mean()
    loss.backward()
    
    print("✓ Gradients computed successfully")
    
    # Check for NaN in gradients
    has_nan = False
    for name, param in list(actor.named_parameters()) + list(critic.named_parameters()):
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"  WARNING: NaN gradient in {name}")
            has_nan = True
    
    if not has_nan:
        print("✓ No NaN gradients detected")
    
    print("\n--- Memory Usage ---")
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    print("\n✓ Neural network tests complete!")