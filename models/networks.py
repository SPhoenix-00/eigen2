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
    Gradient checkpointing enabled to reduce memory usage.
    """

    def __init__(self, num_columns: int = Config.TOTAL_COLUMNS, num_features: int = Config.FEATURES_PER_CELL):
        super().__init__()

        self.num_columns = num_columns
        self.num_features = num_features

        # 1D CNN to extract features from the selected feature elements per cell
        # Input: [batch, num_columns, context_days, num_features]
        # Process each column's time series independently
        self.conv1 = nn.Conv1d(num_features, Config.CNN_FILTERS,
                               kernel_size=Config.CNN_KERNEL_SIZE,
                               padding=Config.CNN_KERNEL_SIZE // 2)
        self.bn1 = nn.BatchNorm1d(Config.CNN_FILTERS)
        self.dropout_conv = nn.Dropout(Config.DROPOUT_2D_RATE)

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

        # Enable gradient checkpointing
        self.use_gradient_checkpointing = True

    def _cnn_block(self, x: torch.Tensor) -> torch.Tensor:
        """CNN processing block for gradient checkpointing."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout_conv(x)
        return x

    def _lstm_block(self, x: torch.Tensor) -> torch.Tensor:
        """LSTM processing block for gradient checkpointing."""
        x = torch.nan_to_num(x, nan=0.0)
        lstm_out, _ = self.lstm(x)
        return lstm_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extractor with gradient checkpointing.

        Args:
            x: Input tensor [batch, context_days, num_columns, num_features]

        Returns:
            Features tensor [batch, num_columns, lstm_output_size]
        """
        batch_size = x.shape[0]
        context_days = x.shape[1]
        num_columns = x.shape[2]

        # Reshape for CNN: [batch * num_columns, num_features, context_days]
        x = x.permute(0, 2, 3, 1)  # [batch, num_columns, num_features, context_days]
        x = x.reshape(batch_size * num_columns, self.num_features, context_days)

        # CNN across features (with gradient checkpointing during training)
        if self.training and self.use_gradient_checkpointing:
            x = checkpoint(self._cnn_block, x, use_reentrant=False)
        else:
            x = self._cnn_block(x)

        # Reshape for LSTM: [batch * num_columns, context_days, CNN_FILTERS]
        x = x.permute(0, 2, 1)

        # LSTM across time (with gradient checkpointing during training)
        if self.training and self.use_gradient_checkpointing:
            lstm_out = checkpoint(self._lstm_block, x, use_reentrant=False)
        else:
            lstm_out = self._lstm_block(x)

        # Take the AVERAGE of the last 3 time steps
        x = torch.mean(lstm_out[:, -3:, :], dim=1)  # [batch * num_columns, lstm_output_size]

        # Reshape back to separate batch and columns: [batch, num_columns, lstm_output_size]
        output = x.reshape(batch_size, num_columns, self.lstm_output_size)

        return output


class AttentionModule(nn.Module):
    """
    Multi-head attention to learn which stocks/indicators matter.
    Supports both self-attention and cross-attention modes.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, use_cross_attention: bool = False,
                 attention_dropout: float = 0.1):
        super().__init__()

        self.use_cross_attention = use_cross_attention
        self.attention_dropout = attention_dropout

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        self.norm = nn.LayerNorm(embed_dim)

        # For cross-attention: learnable query vector (the Actor's "brain")
        if use_cross_attention:
            self.query = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x: torch.Tensor, return_attention_weights: bool = False):
        """
        Args:
            x: [batch, num_columns, embed_dim]
            return_attention_weights: Whether to return attention weights

        Returns:
            If return_attention_weights=True: (output, attention_weights)
            If return_attention_weights=False: output only

            output: [batch, num_columns, embed_dim] or [batch, 1, embed_dim] for cross-attention
            attention_weights: [batch, num_columns] (only if return_attention_weights=True)
        """
        if self.use_cross_attention:
            # Cross-attention: single query attends to all columns
            batch_size = x.shape[0]

            # Expand query for batch
            query = self.query.expand(batch_size, -1, -1)  # [batch, 1, embed_dim]

            # Cross-attention: Q from query, K and V from input features
            attn_out, attn_weights = self.attention(query, x, x, need_weights=True, average_attn_weights=True)
            # attn_out: [batch, 1, embed_dim]
            # attn_weights: [batch, 1, num_columns] or [batch, num_heads, 1, num_columns]

            # Average across heads if needed and squeeze to [batch, num_columns]
            if attn_weights.dim() == 4:
                attn_weights = attn_weights.mean(dim=1)  # Average across heads
            attn_weights = attn_weights.squeeze(1)  # [batch, num_columns]

            # Apply attention dropout during training
            if self.training and self.attention_dropout > 0:
                attn_weights = self._apply_attention_dropout(attn_weights)

            # No residual for cross-attention (query is different from input)
            output = self.norm(attn_out)

            if return_attention_weights:
                return output, attn_weights
            else:
                return output
        else:
            # Self-attention across columns (original behavior)
            attn_out, attn_weights = self.attention(x, x, x, need_weights=return_attention_weights,
                                                     average_attn_weights=True if return_attention_weights else False)

            # Residual connection + normalization
            output = self.norm(x + attn_out)

            if return_attention_weights:
                # Average across heads and reshape to [batch, num_columns]
                if attn_weights.dim() == 4:
                    attn_weights = attn_weights.mean(dim=1)
                attn_weights = attn_weights.mean(dim=1)  # Average across query positions for self-attention
                return output, attn_weights
            else:
                return output

    def _apply_attention_dropout(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Apply attention dropout: randomly zero out some attention weights and renormalize.
        This prevents the model from getting stuck in local optima.

        Args:
            attn_weights: [batch, num_columns] - softmax weights that sum to 1.0

        Returns:
            Dropped and renormalized weights [batch, num_columns]
        """
        # Create dropout mask (1.0 = keep, 0.0 = drop)
        dropout_mask = torch.bernoulli(torch.full_like(attn_weights, 1.0 - self.attention_dropout))

        # Apply mask
        masked_weights = attn_weights * dropout_mask

        # Renormalize so they sum to 1.0
        # Add small epsilon to avoid division by zero
        weight_sum = masked_weights.sum(dim=1, keepdim=True) + 1e-8
        renormalized_weights = masked_weights / weight_sum

        return renormalized_weights


class Actor(nn.Module):
    """
    Actor network: outputs actions [coefficient, sale_target] for each stock.
    Uses cross-attention to determine feature importance across all columns.
    Gradient checkpointing enabled to reduce memory usage.
    """

    def __init__(self):
        super().__init__()

        # Feature extraction
        self.feature_extractor = FeatureExtractor()

        # Cross-attention for feature importance
        if Config.USE_ATTENTION:
            self.attention = AttentionModule(
                embed_dim=self.feature_extractor.lstm_output_size,
                num_heads=Config.ATTENTION_HEADS,
                use_cross_attention=True,  # Use cross-attention
                attention_dropout=0.1
            )
        else:
            self.attention = None

        # Store last attention weights for logging
        self.last_attention_weights = None

        # Separate processing for investable stocks vs context features
        # Investable stocks (columns 10-117 = 108 stocks)
        investable_input_dim = self.feature_extractor.lstm_output_size

        # Context features (remaining columns pooled)
        context_input_dim = self.feature_extractor.lstm_output_size

        # Process investable stocks individually
        self.investable_fc = nn.Sequential(
            nn.Linear(investable_input_dim, Config.ACTOR_HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.ACTOR_HIDDEN_DIMS[0], Config.ACTOR_HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
        )

        # Process context features
        self.context_fc = nn.Sequential(
            nn.Linear(context_input_dim, Config.ACTOR_HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
        )

        # Combined processing for each stock
        combined_dim = Config.ACTOR_HIDDEN_DIMS[1] * 2  # Investable + context

        # Output heads for each stock
        self.coefficient_head = nn.Sequential(
            nn.Linear(combined_dim, Config.ACTOR_HIDDEN_DIMS[2]),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE_HEADS),
            nn.Linear(Config.ACTOR_HIDDEN_DIMS[2], 1),
        )

        self.sale_target_head = nn.Sequential(
            nn.Linear(combined_dim, Config.ACTOR_HIDDEN_DIMS[2]),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE_HEADS),
            nn.Linear(Config.ACTOR_HIDDEN_DIMS[2], 1),
            nn.Sigmoid()  # Output in [0, 1], will scale to [MIN, MAX] sale target
        )

        # Enable gradient checkpointing
        self.use_gradient_checkpointing = True
        
    def _process_investable(self, investable_features: torch.Tensor) -> torch.Tensor:
        """Process investable stocks (for gradient checkpointing)."""
        return self.investable_fc(investable_features)

    def _process_context(self, global_context: torch.Tensor) -> torch.Tensor:
        """Process context features (for gradient checkpointing)."""
        return self.context_fc(global_context)

    def _process_coefficient_head(self, combined: torch.Tensor) -> torch.Tensor:
        """Process coefficient head (for gradient checkpointing)."""
        return self.coefficient_head(combined)

    def _process_sale_target_head(self, combined: torch.Tensor) -> torch.Tensor:
        """Process sale target head (for gradient checkpointing)."""
        return self.sale_target_head(combined)

    def forward(self, state: torch.Tensor, return_attention_weights: bool = False) -> torch.Tensor:
        """
        Forward pass with gradient checkpointing.

        Args:
            state: [batch, context_days, num_columns, 9]
            return_attention_weights: Whether to return attention weights for logging

        Returns:
            actions: [batch, 108, 2] with [coefficient, sale_target] per stock
            (optional) attention_weights: [batch, num_columns] if return_attention_weights=True
        """
        batch_size = state.shape[0]

        # Extract features from all columns
        features = self.feature_extractor(state)
        # [batch, num_columns, lstm_output_size]

        # Apply cross-attention if enabled
        attention_weights = None
        if self.attention is not None:
            # Cross-attention: single query attends to all features
            global_context, attention_weights = self.attention(features, return_attention_weights=True)
            # global_context: [batch, 1, lstm_output_size]
            # attention_weights: [batch, num_columns]

            # Store attention weights for logging
            self.last_attention_weights = attention_weights.detach()

            # Process global context through context FC (with checkpointing)
            global_context = global_context.squeeze(1)  # [batch, lstm_output_size]
            if self.training and self.use_gradient_checkpointing:
                context_processed = checkpoint(self._process_context, global_context, use_reentrant=False)
            else:
                context_processed = self._process_context(global_context)
            context_processed = context_processed.unsqueeze(1)  # [batch, 1, ACTOR_HIDDEN_DIMS[1]]
        else:
            # Fallback: pool all features if no attention
            context_features = torch.mean(features, dim=1)  # [batch, lstm_output_size]
            if self.training and self.use_gradient_checkpointing:
                context_processed = checkpoint(self._process_context, context_features, use_reentrant=False)
            else:
                context_processed = self._process_context(context_features)
            context_processed = context_processed.unsqueeze(1)  # [batch, 1, ACTOR_HIDDEN_DIMS[1]]

        # Extract investable stock features (still use raw features for stock-specific processing)
        investable_features = features[:, Config.INVESTABLE_START_COL:Config.INVESTABLE_END_COL+1, :]
        # [batch, 108, lstm_output_size]

        # Process investable stocks (with checkpointing)
        if self.training and self.use_gradient_checkpointing:
            investable_processed = checkpoint(self._process_investable, investable_features, use_reentrant=False)
        else:
            investable_processed = self._process_investable(investable_features)
        # [batch, 108, ACTOR_HIDDEN_DIMS[1]]

        # Expand context to all stocks
        context_expanded = context_processed.expand(-1, Config.NUM_INVESTABLE_STOCKS, -1)
        # [batch, 108, ACTOR_HIDDEN_DIMS[1]]

        # Combine
        combined = torch.cat([investable_processed, context_expanded], dim=-1)
        # [batch, 108, combined_dim]

        # Output heads (with checkpointing)
        if self.training and self.use_gradient_checkpointing:
            raw_coefficients = checkpoint(self._process_coefficient_head, combined, use_reentrant=False).squeeze(-1)
            raw_sale_targets = checkpoint(self._process_sale_target_head, combined, use_reentrant=False).squeeze(-1)
        else:
            raw_coefficients = self._process_coefficient_head(combined).squeeze(-1)
            raw_sale_targets = self._process_sale_target_head(combined).squeeze(-1)

        # Apply activations
        # Coefficient: >= 1 or 0 (using threshold)
        coefficients = self._apply_coefficient_activation(raw_coefficients)

        # Sale target: scale from [0, 1] to [MIN_SALE_TARGET, MAX_SALE_TARGET]
        sale_targets = (Config.MIN_SALE_TARGET +
                       raw_sale_targets * (Config.MAX_SALE_TARGET - Config.MIN_SALE_TARGET))

        # Stack into action tensor
        actions = torch.stack([coefficients, sale_targets], dim=-1)
        # [batch, 108, 2]

        if return_attention_weights:
            return actions, attention_weights
        else:
            return actions
    
    def _apply_coefficient_activation(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Apply activation to normalize coefficient to [0, 1] range.
        Uses sigmoid for bounded output, allowing agent to learn when NOT to trade.

        During training: outputs continuous values for gradient flow
        During inference: will be compared against threshold in trading logic

        Args:
            raw: Raw output from network

        Returns:
            Activated coefficients normalized to [0, 1] range
        """
        # Sigmoid: maps raw values to [0, 1] smoothly
        # This allows the agent to learn that values below COEFFICIENT_THRESHOLD
        # should not trigger trades, giving it the option to not trade
        coefficients = torch.sigmoid(raw)

        return coefficients

    def get_attention_weights(self) -> torch.Tensor:
        """
        Get the last computed attention weights (feature importance).

        Returns:
            Attention weights [batch, num_columns] or None if not available
        """
        return self.last_attention_weights


class Critic(nn.Module):
    """
    Critic network: estimates Q-value for state-action pair.
    Gradient checkpointing enabled to reduce memory usage.
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
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.CRITIC_HIDDEN_DIMS[0], Config.CRITIC_HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.CRITIC_HIDDEN_DIMS[1], 1)
        )

        # Enable gradient checkpointing
        self.use_gradient_checkpointing = True
        
    def _process_critic_fc(self, x: torch.Tensor) -> torch.Tensor:
        """Process critic FC layers (for gradient checkpointing)."""
        return self.critic_fc(x)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gradient checkpointing.

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

        # Critic network (with checkpointing)
        if self.training and self.use_gradient_checkpointing:
            q_value = checkpoint(self._process_critic_fc, x, use_reentrant=False)
        else:
            q_value = self._process_critic_fc(x)

        return q_value


# Test networks
if __name__ == "__main__":
    print("Testing Neural Networks...\n")
    
    # Set device
    device = Config.DEVICE
    print(f"Using device: {device}\n")
    
    # Create dummy input
    batch_size = 4
    dummy_state = torch.randn(batch_size, Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL).to(device)

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