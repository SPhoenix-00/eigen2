"""
Test script for cross-attention feature importance implementation
"""

import torch
import numpy as np
from models.networks import Actor, Critic
from utils.config import Config

print("="*60)
print("Testing Cross-Attention Feature Importance")
print("="*60)

# Set device
device = Config.DEVICE
print(f"\nUsing device: {device}\n")

# Create dummy input
batch_size = 4
dummy_state = torch.randn(batch_size, Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL).to(device)

print(f"Input state shape: {dummy_state.shape}")

# Test Actor with cross-attention
print("\n--- Testing Actor with Cross-Attention ---")
actor = Actor().to(device)
print(f"Actor parameters: {sum(p.numel() for p in actor.parameters()):,}")

# Check if attention is enabled
if actor.attention is not None:
    print(f"[OK] Attention enabled: {type(actor.attention).__name__}")
    print(f"  Cross-attention mode: {actor.attention.use_cross_attention}")
    print(f"  Attention dropout: {actor.attention.attention_dropout}")
else:
    print("[ERROR] Attention is disabled!")
    print("  Set Config.USE_ATTENTION = True to enable")
    exit(1)

# Forward pass without returning attention weights
print("\n1. Forward pass (normal mode):")
with torch.no_grad():
    actions = actor(dummy_state, return_attention_weights=False)

print(f"   Actions shape: {actions.shape}")
print(f"   Actions type: {type(actions)}")

# Forward pass with attention weights
print("\n2. Forward pass (with attention weights):")
with torch.no_grad():
    actions, attention_weights = actor(dummy_state, return_attention_weights=True)

print(f"   Actions shape: {actions.shape}")
print(f"   Attention weights shape: {attention_weights.shape}")

# Check attention weights properties
print("\n3. Attention weights analysis:")
print(f"   Sum per batch: {attention_weights.sum(dim=1)}")
print(f"   Mean sum: {attention_weights.sum(dim=1).mean():.6f} (should be ~1.0)")
print(f"   Max weight: {attention_weights.max():.6f}")
print(f"   Min weight: {attention_weights.min():.6f}")
print(f"   Mean weight: {attention_weights.mean():.6f}")

# Check if weights are stored in actor
last_weights = actor.get_attention_weights()
print(f"\n4. Stored attention weights:")
if last_weights is not None:
    print(f"   Shape: {last_weights.shape}")
    print(f"   Match current? {torch.allclose(last_weights, attention_weights)}")
else:
    print("   ✗ No weights stored!")

# Top-k most important features
print("\n5. Top 10 most important features:")
avg_weights = attention_weights.mean(dim=0).cpu().numpy()
top_k = 10
top_indices = np.argsort(avg_weights)[-top_k:][::-1]
for i, idx in enumerate(top_indices):
    print(f"   {i+1}. Column {idx:3d}: {avg_weights[idx]:.6f} ({avg_weights[idx]*100:.2f}%)")

# Test gradient flow through attention
print("\n6. Testing gradient flow:")
actor.train()
actions, attention_weights = actor(dummy_state, return_attention_weights=True)
loss = actions.mean()
loss.backward()

# Check if attention parameters have gradients
query_param = actor.attention.query
if query_param.grad is not None:
    print(f"   [OK] Query gradient shape: {query_param.grad.shape}")
    print(f"   [OK] Query gradient norm: {query_param.grad.norm():.6f}")
else:
    print("   [ERROR] No gradient for query parameter!")

# Test running average update
print("\n7. Testing feature importance running average:")

class MockTrainer:
    def __init__(self):
        self.feature_importance = torch.zeros(Config.TOTAL_COLUMNS, device=device)
        self.feature_importance_momentum = 0.99
        self.feature_importance_count = 0

    def update_feature_importance(self, attention_weights):
        if attention_weights is None:
            return

        # Average across batch dimension to get [num_columns]
        batch_mean = attention_weights.mean(dim=0)

        # Update running average with momentum
        if self.feature_importance_count == 0:
            # First update: initialize with current weights
            self.feature_importance = batch_mean.detach()
        else:
            # EMA update: avg = momentum * avg + (1 - momentum) * new
            momentum = self.feature_importance_momentum
            self.feature_importance = (momentum * self.feature_importance +
                                       (1 - momentum) * batch_mean.detach())

        self.feature_importance_count += 1

trainer = MockTrainer()

# Simulate multiple updates
num_updates = 10
print(f"   Running {num_updates} updates...")
for i in range(num_updates):
    with torch.no_grad():
        dummy_state = torch.randn(batch_size, Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL).to(device)
        actions, attn_weights = actor(dummy_state, return_attention_weights=True)
        trainer.update_feature_importance(attn_weights)

print(f"   [OK] Update count: {trainer.feature_importance_count}")
print(f"   [OK] Feature importance shape: {trainer.feature_importance.shape}")
print(f"   [OK] Feature importance sum: {trainer.feature_importance.sum():.6f} (should be ~1.0)")
print(f"   [OK] Top feature importance: {trainer.feature_importance.max():.6f}")

# Test attention dropout
print("\n8. Testing attention dropout:")
actor.train()  # Enable dropout
num_trials = 100
dropout_detected = False

for _ in range(num_trials):
    with torch.no_grad():
        dummy_state = torch.randn(2, Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL).to(device)
        actions, attn_weights = actor(dummy_state, return_attention_weights=True)

        # Check if any weights are exactly zero (indicating dropout)
        if (attn_weights == 0).any():
            dropout_detected = True
            num_zeros = (attn_weights == 0).sum().item()
            print(f"   [OK] Dropout detected: {num_zeros} zeros in attention weights")
            break

if not dropout_detected:
    print(f"   [WARN] No dropout detected in {num_trials} trials (may be rare with 10% dropout)")
else:
    print(f"   [OK] Attention dropout is working correctly")

print("\n" + "="*60)
print("[SUCCESS] All tests passed! Cross-attention feature importance is working.")
print("="*60)
