"""
Diagnostic script to identify why the model outputs identical values
"""

import torch
import numpy as np
from models.ddpg_agent import DDPGAgent
from utils.config import Config
from data.loader import StockDataLoader
from pathlib import Path
import json

def check_weight_updates():
    """Check if weights in output heads are actually changing during training"""
    print("="*60)
    print("DIAGNOSTIC 1: Weight Update Check")
    print("="*60)

    # Create agent
    agent = DDPGAgent(agent_id=0)

    # Get initial weights from output heads
    initial_coef_weights = agent.actor.coefficient_head[-1].weight.clone().detach()
    initial_sale_weights = agent.actor.sale_target_head[-2].weight.clone().detach()

    print(f"\nInitial coefficient head weights (sample):")
    print(f"  Mean: {initial_coef_weights.mean().item():.6f}")
    print(f"  Std: {initial_coef_weights.std().item():.6f}")
    print(f"  Min: {initial_coef_weights.min().item():.6f}")
    print(f"  Max: {initial_coef_weights.max().item():.6f}")

    print(f"\nInitial sale_target head weights (sample):")
    print(f"  Mean: {initial_sale_weights.mean().item():.6f}")
    print(f"  Std: {initial_sale_weights.std().item():.6f}")
    print(f"  Min: {initial_sale_weights.min().item():.6f}")
    print(f"  Max: {initial_sale_weights.max().item():.6f}")

    # Create dummy batch and perform 10 updates
    print(f"\nPerforming 10 training updates...")
    for i in range(10):
        dummy_batch = {
            'states': torch.randn(4, Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL),
            'actions': torch.randn(4, Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM),
            'rewards': torch.randn(4, 1),
            'next_states': torch.randn(4, Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL),
            'dones': torch.zeros(4, 1),
        }
        critic_loss, actor_loss = agent.update(dummy_batch, accumulate=False)
        print(f"  Update {i+1}: Actor Loss={actor_loss:.4f}, Critic Loss={critic_loss:.4f}")

    # Get final weights
    final_coef_weights = agent.actor.coefficient_head[-1].weight.clone().detach()
    final_sale_weights = agent.actor.sale_target_head[-2].weight.clone().detach()

    # Calculate change
    coef_change = (final_coef_weights - initial_coef_weights).abs()
    sale_change = (final_sale_weights - initial_sale_weights).abs()

    print(f"\nCoefficient head weight changes:")
    print(f"  Mean change: {coef_change.mean().item():.8f}")
    print(f"  Max change: {coef_change.max().item():.8f}")
    print(f"  % changed (>1e-6): {(coef_change > 1e-6).float().mean().item() * 100:.1f}%")

    print(f"\nSale target head weight changes:")
    print(f"  Mean change: {sale_change.mean().item():.8f}")
    print(f"  Max change: {sale_change.max().item():.8f}")
    print(f"  % changed (>1e-6): {(sale_change > 1e-6).float().mean().item() * 100:.1f}%")

    if coef_change.max().item() < 1e-6 and sale_change.max().item() < 1e-6:
        print("\nWARNING: Weights are NOT updating! This is the problem.")
        return False
    else:
        print("\nWeights are updating correctly.")
        return True


def check_gradient_flow():
    """Check if gradients are flowing to output heads"""
    print("\n" + "="*60)
    print("DIAGNOSTIC 2: Gradient Flow Check")
    print("="*60)

    agent = DDPGAgent(agent_id=0)
    agent.actor.train()
    agent.critic.train()

    # Create dummy batch
    dummy_batch = {
        'states': torch.randn(4, Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL).to(Config.DEVICE),
        'actions': torch.randn(4, Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM).to(Config.DEVICE),
        'rewards': torch.randn(4, 1).to(Config.DEVICE),
        'next_states': torch.randn(4, Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL).to(Config.DEVICE),
        'dones': torch.zeros(4, 1).to(Config.DEVICE),
    }

    # Zero gradients
    agent.actor_optimizer.zero_grad()

    # Forward pass
    actions = agent.actor(dummy_batch['states'])
    q_values = agent.critic(dummy_batch['states'], actions)
    actor_loss = -q_values.mean()

    # Backward pass
    actor_loss.backward()

    # Check gradients in output heads
    print("\nChecking gradients in output heads:")

    # Coefficient head
    coef_head_grad = agent.actor.coefficient_head[-1].weight.grad
    if coef_head_grad is not None:
        print(f"\nCoefficient head gradients:")
        print(f"  Mean: {coef_head_grad.mean().item():.8f}")
        print(f"  Std: {coef_head_grad.std().item():.8f}")
        print(f"  Max: {coef_head_grad.max().item():.8f}")
        print(f"  % non-zero: {(coef_head_grad.abs() > 1e-8).float().mean().item() * 100:.1f}%")

        if coef_head_grad.abs().max().item() < 1e-8:
            print("  ⚠️  WARNING: Gradients are vanishing!")
    else:
        print(f"  ❌ ERROR: No gradients computed for coefficient head!")
        return False

    # Sale target head
    sale_head_grad = agent.actor.sale_target_head[-2].weight.grad
    if sale_head_grad is not None:
        print(f"\nSale target head gradients:")
        print(f"  Mean: {sale_head_grad.mean().item():.8f}")
        print(f"  Std: {sale_head_grad.std().item():.8f}")
        print(f"  Max: {sale_head_grad.max().item():.8f}")
        print(f"  % non-zero: {(sale_head_grad.abs() > 1e-8).float().mean().item() * 100:.1f}%")

        if sale_head_grad.abs().max().item() < 1e-8:
            print("  ⚠️  WARNING: Gradients are vanishing!")
    else:
        print(f"  ❌ ERROR: No gradients computed for sale target head!")
        return False

    print("\n✓ Gradients are flowing to output heads.")
    return True


def check_output_diversity():
    """Check if the model produces diverse outputs for different inputs"""
    print("\n" + "="*60)
    print("DIAGNOSTIC 3: Output Diversity Check")
    print("="*60)

    agent = DDPGAgent(agent_id=0)
    agent.actor.eval()

    # Generate 10 different random states
    print("\nGenerating 10 different random states and checking outputs...\n")

    all_coefficients = []
    all_sale_targets = []

    for i in range(10):
        state = np.random.randn(Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL)
        action = agent.select_action(state, add_noise=False)

        # Extract unique values
        unique_coefs = np.unique(action[:, 0])
        unique_sales = np.unique(action[:, 1])

        print(f"State {i+1}:")
        print(f"  Unique coefficients: {len(unique_coefs)}")
        print(f"  Coefficient range: [{action[:, 0].min():.4f}, {action[:, 0].max():.4f}]")
        print(f"  Unique sale targets: {len(unique_sales)}")
        print(f"  Sale target range: [{action[:, 1].min():.2f}%, {action[:, 1].max():.2f}%]")

        all_coefficients.append(action[:, 0])
        all_sale_targets.append(action[:, 1])

    # Check if outputs are identical across different inputs
    all_coefficients = np.array(all_coefficients)
    all_sale_targets = np.array(all_sale_targets)

    coef_variance = all_coefficients.var(axis=0).mean()
    sale_variance = all_sale_targets.var(axis=0).mean()

    print(f"\n--- Cross-Input Analysis ---")
    print(f"Mean variance in coefficients across inputs: {coef_variance:.8f}")
    print(f"Mean variance in sale targets across inputs: {sale_variance:.8f}")

    if coef_variance < 1e-6:
        print("\n⚠️  WARNING: Coefficients are identical across different inputs!")
        print("   This suggests the model is not responsive to input state.")
        return False

    if sale_variance < 1e-6:
        print("\n⚠️  WARNING: Sale targets are identical across different inputs!")
        print("   This suggests the model is not responsive to input state.")
        return False

    # Check if outputs are identical within a single state
    print(f"\n--- Within-State Analysis (first state only) ---")
    first_state_coefs = all_coefficients[0]
    first_state_sales = all_sale_targets[0]

    unique_coefs_single = len(np.unique(first_state_coefs))
    unique_sales_single = len(np.unique(first_state_sales))

    print(f"Unique coefficients for 108 stocks: {unique_coefs_single}")
    print(f"Unique sale targets for 108 stocks: {unique_sales_single}")

    if unique_coefs_single == 1:
        print(f"\n⚠️  PROBLEM FOUND: All 108 stocks have IDENTICAL coefficient: {first_state_coefs[0]:.4f}")
        print("   The model is outputting the same coefficient for every stock!")

    if unique_sales_single == 1:
        print(f"\n⚠️  PROBLEM FOUND: All 108 stocks have IDENTICAL sale_target_pct: {first_state_sales[0]:.2f}%")
        print("   The model is outputting the same sale target for every stock!")

    if unique_coefs_single == 1 or unique_sales_single == 1:
        return False

    print("\n✓ Outputs are diverse.")
    return True


def check_loaded_model():
    """Check if a loaded model from checkpoint has the same issue"""
    print("\n" + "="*60)
    print("DIAGNOSTIC 4: Loaded Model Check")
    print("="*60)

    # Try to find the latest checkpoint
    checkpoint_dir = Path("checkpoints")

    # Look for last_run.json
    last_run_file = Path("last_run.json")
    if last_run_file.exists():
        with open(last_run_file, 'r') as f:
            last_run_info = json.load(f)
        run_name = last_run_info['run_name']
        checkpoint_path = checkpoint_dir / run_name / "best_agent.pth"

        if checkpoint_path.exists():
            print(f"\nLoading checkpoint: {checkpoint_path}")
            agent = DDPGAgent(agent_id=0)
            agent.load(str(checkpoint_path))

            print("\nTesting loaded model outputs...")

            # Test with a random state
            state = np.random.randn(Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL)
            action = agent.select_action(state, add_noise=False)

            unique_coefs = len(np.unique(action[:, 0]))
            unique_sales = len(np.unique(action[:, 1]))

            print(f"  Unique coefficients: {unique_coefs} / 108")
            print(f"  Coefficient range: [{action[:, 0].min():.4f}, {action[:, 0].max():.4f}]")
            print(f"  Unique sale targets: {unique_sales} / 108")
            print(f"  Sale target range: [{action[:, 1].min():.2f}%, {action[:, 1].max():.2f}%]")

            if unique_coefs == 1:
                print(f"\n⚠️  LOADED MODEL HAS THE PROBLEM!")
                print(f"   All stocks have identical coefficient: {action[0, 0]:.4f}")

            if unique_sales == 1:
                print(f"\n⚠️  LOADED MODEL HAS THE PROBLEM!")
                print(f"   All stocks have identical sale_target_pct: {action[0, 1]:.2f}%")

            return unique_coefs > 1 and unique_sales > 1
        else:
            print(f"\n! Checkpoint not found: {checkpoint_path}")
            print("  Skipping loaded model check.")
            return None
    else:
        print("\n! No last_run.json found. Skipping loaded model check.")
        return None


def main():
    print("\n" + "="*60)
    print("DIAGNOSING MODEL OUTPUT ISSUE")
    print("="*60)
    print("\nThis script will run 4 diagnostic tests to identify why")
    print("the model outputs identical coefficient and sale_target_pct")
    print("for all trades.\n")

    results = {}

    # Test 1: Weight updates
    results['weight_updates'] = check_weight_updates()

    # Test 2: Gradient flow
    results['gradient_flow'] = check_gradient_flow()

    # Test 3: Output diversity
    results['output_diversity'] = check_output_diversity()

    # Test 4: Loaded model
    results['loaded_model'] = check_loaded_model()

    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)

    print("\nTest Results:")
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⊘ SKIPPED"
        print(f"  {test_name}: {status}")

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    if not results['weight_updates']:
        print("\n1. Weights are not updating during training.")
        print("   Possible causes:")
        print("   - Learning rate is too small")
        print("   - Gradients are vanishing")
        print("   - Optimizer is not stepping correctly")

    if not results['gradient_flow']:
        print("\n2. Gradients are not flowing to output heads.")
        print("   Possible causes:")
        print("   - Dead ReLU neurons in earlier layers")
        print("   - Gradient checkpointing issue")
        print("   - Frozen layers somewhere in the network")

    if not results['output_diversity']:
        print("\n3. Model outputs are not diverse.")
        print("   Possible causes:")
        print("   - Weights are stuck at initialization values")
        print("   - Batch normalization running stats are frozen")
        print("   - Dropout is interfering during eval mode")
        print("   - Self-attention is collapsing to uniform weights")

    if results['loaded_model'] is False:
        print("\n4. Loaded checkpoint has the same issue.")
        print("   This suggests the model never learned properly during training.")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
