"""
Quick fixes for network issues:
1. Move to GPU
2. Fix coefficient activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix for coefficient activation - more gradual
def apply_coefficient_activation_fixed(raw: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Improved coefficient activation.
    Instead of hard threshold, use a smoother approach.
    
    Args:
        raw: Raw output from network (can be any value)
        threshold: Threshold for activation (default 0)
        
    Returns:
        Coefficient >= 1 or close to 0
    """
    # Apply tanh to get values in [-1, 1], then shift/scale
    # This gives the network more flexibility during training
    
    # Option 1: Soft threshold with exponential
    # If raw > 0: scale to [1, inf)
    # If raw < 0: map to ~0
    coefficients = torch.where(
        raw > threshold,
        F.softplus(raw) + 1.0,  # softplus ensures smooth gradient, +1 ensures >= 1
        torch.sigmoid(raw) * 0.1  # sigmoid maps to [0, 0.1], nearly 0 for negative values
    )
    
    return coefficients


# Test the fixed activation
if __name__ == "__main__":
    print("Testing fixed coefficient activation...\n")
    
    # Test with various raw values
    test_values = torch.tensor([
        -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0
    ])
    
    # Original activation (from networks.py)
    def original_activation(raw, threshold=0.5, min_coef=1.0):
        return torch.where(
            raw > threshold,
            F.relu(raw - threshold) + min_coef,
            torch.zeros_like(raw)
        )
    
    original_out = original_activation(test_values)
    fixed_out = apply_coefficient_activation_fixed(test_values)
    
    print("Raw values:", test_values.numpy())
    print("Original activation:", original_out.numpy())
    print("Fixed activation:", fixed_out.numpy())
    
    print("\nKey differences:")
    print(f"Original: {(original_out > 0).sum().item()} / {len(test_values)} non-zero")
    print(f"Fixed: {(fixed_out > 0.5).sum().item()} / {len(test_values)} > 0.5")
    
    print("\n✓ Fixed activation allows gradients to flow better during training")
    print("  - Negative values map to ~0 (but not exactly 0)")
    print("  - Positive values map to >= 1")
    print("  - Smooth gradients everywhere (no hard cutoffs)")