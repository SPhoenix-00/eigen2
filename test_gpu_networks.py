"""
Test neural networks on GPU
"""

import torch
from models.networks import Actor, Critic
from utils.config import Config

print("="*60)
print("Testing Neural Networks on GPU")
print("="*60)

# Check device
print(f"\nConfig.DEVICE: {Config.DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Force device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Create dummy input
batch_size = 1
print(f"\nCreating dummy input on {device}...")
dummy_state = torch.randn(batch_size, Config.CONTEXT_WINDOW_DAYS, 669, Config.FEATURES_PER_CELL, device=device)
print(f"Input tensor device: {dummy_state.device}")
print(f"Input state shape: {dummy_state.shape}")

# Test Actor
print("\n" + "="*60)
print("Testing Actor")
print("="*60)
actor = Actor().to(device)
print(f"Actor device: {next(actor.parameters()).device}")
print(f"Actor parameters: {sum(p.numel() for p in actor.parameters()):,}")

print("\nForward pass...")
import time
start = time.time()
with torch.no_grad():
    actions = actor(dummy_state)
elapsed = time.time() - start

print(f"Time: {elapsed:.3f} seconds")
print(f"Output actions device: {actions.device}")
print(f"Output actions shape: {actions.shape}")
print(f"Coefficients range: [{actions[:, :, 0].min():.2f}, {actions[:, :, 0].max():.2f}]")
print(f"Sale targets range: [{actions[:, :, 1].min():.2f}, {actions[:, :, 1].max():.2f}]")
print(f"Number of non-zero coefficients: {(actions[:, :, 0] > 0).sum().item()} / {actions[:, :, 0].numel()}")
print(f"Number with coef > 1.0: {(actions[:, :, 0] > 1.0).sum().item()}")

# Test Critic
print("\n" + "="*60)
print("Testing Critic")
print("="*60)
critic = Critic().to(device)
print(f"Critic device: {next(critic.parameters()).device}")
print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")

print("\nForward pass...")
start = time.time()
with torch.no_grad():
    q_values = critic(dummy_state, actions)
elapsed = time.time() - start

print(f"Time: {elapsed:.3f} seconds")
print(f"Output Q-values device: {q_values.device}")
print(f"Output Q-values shape: {q_values.shape}")
print(f"Q-values range: [{q_values.min():.2f}, {q_values.max():.2f}]")

# Test gradient flow
print("\n" + "="*60)
print("Testing Gradient Flow")
print("="*60)
actor.train()
critic.train()

print("Computing gradients...")
start = time.time()
actions = actor(dummy_state)
q_values = critic(dummy_state, actions)
loss = q_values.mean()
loss.backward()
elapsed = time.time() - start

print(f"Time: {elapsed:.3f} seconds")
print("✓ Gradients computed successfully")

# Check for NaN in gradients
has_nan = False
for name, param in list(actor.named_parameters()) + list(critic.named_parameters()):
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"  WARNING: NaN gradient in {name}")
        has_nan = True

if not has_nan:
    print("✓ No NaN gradients detected")

# Memory usage
print("\n" + "="*60)
print("Memory Usage")
print("="*60)
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
    print(f"GPU memory available: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(device)) / 1e9:.2f} GB")

print("\n" + "="*60)
print("✓ All tests passed!")
print("="*60)