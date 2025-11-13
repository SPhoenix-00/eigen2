"""
Test observation noise regularization feature.

This test verifies that:
1. Training mode adds noise to observations
2. Evaluation mode does not add noise
3. Mode can be switched dynamically
"""

import numpy as np
from environment.trading_env import TradingEnvironment
from utils.config import Config


def test_observation_noise():
    """Test that observation noise is applied correctly in training mode."""

    print("Testing observation noise regularization...\n")

    # Create synthetic data for testing
    num_days = Config.CONTEXT_WINDOW_DAYS + 100
    num_stocks = Config.NUM_INVESTABLE_STOCKS + Config.INVESTABLE_START_COL
    num_features = Config.FEATURES_PER_CELL

    # Create random data
    np.random.seed(42)
    data_array = np.random.randn(num_days, num_stocks, num_features)
    dates = np.array([f"2024-{i//252+1:02d}-{i%252+1:02d}" for i in range(num_days)])

    # Create normalization stats
    norm_stats = {
        'mean': np.zeros((Config.CONTEXT_WINDOW_DAYS, num_stocks, num_features)),
        'std': np.ones((Config.CONTEXT_WINDOW_DAYS, num_stocks, num_features))
    }

    # Create environments
    start_idx = Config.CONTEXT_WINDOW_DAYS
    end_idx = start_idx + 10

    env_train = TradingEnvironment(
        data_array=data_array,
        dates=dates,
        normalization_stats=norm_stats,
        start_idx=start_idx,
        end_idx=end_idx,
        is_training=True
    )

    env_eval = TradingEnvironment(
        data_array=data_array,
        dates=dates,
        normalization_stats=norm_stats,
        start_idx=start_idx,
        end_idx=end_idx,
        is_training=False
    )

    print("Test 1: Training mode adds noise to observations")
    print("-" * 50)

    # Get two observations from training mode
    obs_train_1, _ = env_train.reset()
    obs_train_2, _ = env_train.reset()

    # They should be different due to random noise
    train_diff = np.abs(obs_train_1 - obs_train_2).mean()
    print(f"  Mean difference between two training observations: {train_diff:.6f}")

    if train_diff > 0:
        print("  ✓ PASS: Training observations differ (noise applied)")
    else:
        print("  ✗ FAIL: Training observations are identical (no noise)")
        return False

    print("\nTest 2: Evaluation mode does not add noise")
    print("-" * 50)

    # Get two observations from eval mode - should be identical
    obs_eval_1, _ = env_eval.reset()
    obs_eval_2, _ = env_eval.reset()

    eval_diff = np.abs(obs_eval_1 - obs_eval_2).mean()
    print(f"  Mean difference between two eval observations: {eval_diff:.6f}")

    if eval_diff == 0:
        print("  ✓ PASS: Eval observations are identical (no noise)")
    else:
        print("  ✗ FAIL: Eval observations differ (unexpected noise)")
        return False

    print("\nTest 3: Dynamic mode switching works")
    print("-" * 50)

    # Switch training env to eval mode
    env_train.set_training_mode(False)
    obs_switched_1, _ = env_train.reset()
    obs_switched_2, _ = env_train.reset()

    switched_diff = np.abs(obs_switched_1 - obs_switched_2).mean()
    print(f"  Mean difference after switching to eval: {switched_diff:.6f}")

    if switched_diff == 0:
        print("  ✓ PASS: Switched environment behaves like eval (no noise)")
    else:
        print("  ✗ FAIL: Switched environment still has noise")
        return False

    # Switch back to training
    env_train.set_training_mode(True)
    obs_back_1, _ = env_train.reset()
    obs_back_2, _ = env_train.reset()

    back_diff = np.abs(obs_back_1 - obs_back_2).mean()
    print(f"  Mean difference after switching back to training: {back_diff:.6f}")

    if back_diff > 0:
        print("  ✓ PASS: Switched back to training (noise applied)")
    else:
        print("  ✗ FAIL: Switched back but no noise")
        return False

    print("\n" + "=" * 50)
    print("✓ ALL TESTS PASSED!")
    print("=" * 50)
    print(f"\nConfiguration:")
    print(f"  - Observation noise std: {Config.OBSERVATION_NOISE_STD}")
    print(f"  - Training mode: adds Gaussian noise to observations")
    print(f"  - Evaluation mode: returns clean observations")
    print(f"\nWhy this matters:")
    print(f"  Adding noise during training prevents overfitting by forcing")
    print(f"  the agent to learn robust, generalized patterns rather than")
    print(f"  memorizing exact values.")

    return True


if __name__ == "__main__":
    success = test_observation_noise()
    exit(0 if success else 1)
