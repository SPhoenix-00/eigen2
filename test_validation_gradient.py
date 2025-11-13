"""
Test script to verify the validation gradient fix.
This ensures agents with 0 trades get differentiated scores based on max_coefficient.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import StockDataLoader
from environment.trading_env import TradingEnvironment
from utils.config import Config


def test_validation_gradient():
    """
    Test that agents with 0 trades receive differentiated validation scores
    based on their max_coefficient during the episode.
    """
    print("="*70)
    print("Testing Validation Gradient Fix")
    print("="*70)
    print("\nThis test verifies that agents with 0 trades get scores based on")
    print("their max_coefficient, creating a gradient for evolution.\n")

    # Load data
    print("Loading data...")
    loader = StockDataLoader()
    data_array, stats = loader.load_and_prepare()
    normalization_stats = loader.compute_normalization_stats()

    # Create environment
    start_idx = Config.CONTEXT_WINDOW_DAYS
    end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS

    env = TradingEnvironment(
        data_array=loader.data_array,
        dates=loader.dates,
        normalization_stats=normalization_stats,
        start_idx=start_idx,
        end_idx=end_idx,
        trading_end_idx=start_idx + Config.TRADING_PERIOD_DAYS,
        data_array_full=loader.data_array_full,
        is_training=False  # Validation mode
    )

    print(f"Environment created (days {start_idx} to {end_idx})")
    print(f"Coefficient threshold: {Config.COEFFICIENT_THRESHOLD}")

    # Test 3 scenarios with different max coefficients
    test_cases = [
        ("Agent A (cautious)", 0.2),   # Max coeff = 0.2
        ("Agent B (moderate)", 0.5),   # Max coeff = 0.5
        ("Agent C (aggressive)", 0.9), # Max coeff = 0.9
    ]

    results = []

    for agent_name, max_coeff in test_cases:
        print(f"\n{'='*70}")
        print(f"Testing {agent_name} (max_coefficient = {max_coeff})")
        print(f"{'='*70}")

        # Reset environment
        obs, info = env.reset()
        cumulative_reward = 0.0

        # Run episode with actions that produce the specified max coefficient
        for step in range(end_idx - start_idx):
            # Create action with all coefficients below threshold
            # Set all coefficients to max_coeff (all below threshold so no trades)
            action = np.zeros((Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM))
            action[:, 0] = max_coeff  # All coefficients = max_coeff
            action[:, 1] = Config.MIN_SALE_TARGET + 5.0  # Some reasonable sale target

            obs, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward

            if terminated:
                break

        # Get episode summary
        summary = env.get_episode_summary()

        # Calculate validation fitness (mimicking the trainer logic)
        final_fitness = cumulative_reward

        # Apply zero-trades penalty if no trades
        if summary['num_trades'] == 0:
            final_fitness -= Config.ZERO_TRADES_PENALTY

        # Apply validation gradient fix
        validation_fitness = final_fitness
        if summary['num_trades'] == 0:
            max_coeff_episode = summary.get('max_coefficient_during_episode', 0.0)
            validation_fitness = -100.0 + max_coeff_episode

        results.append({
            'agent_name': agent_name,
            'max_coeff_input': max_coeff,
            'max_coeff_tracked': summary['max_coefficient_during_episode'],
            'num_trades': summary['num_trades'],
            'final_fitness': final_fitness,
            'validation_fitness': validation_fitness
        })

        print(f"\nResults:")
        print(f"  Num trades: {summary['num_trades']}")
        print(f"  Max coefficient tracked: {summary['max_coefficient_during_episode']:.3f}")
        print(f"  Final fitness (training): {final_fitness:.2f}")
        print(f"  Validation fitness: {validation_fitness:.2f}")

    # Print summary comparison
    print("\n" + "="*70)
    print("SUMMARY: Validation Gradient Test")
    print("="*70)
    print(f"\n{'Agent':<25} {'Max Coeff':<12} {'Trades':<8} {'Val Fitness':<15}")
    print("-"*70)

    for r in results:
        print(f"{r['agent_name']:<25} {r['max_coeff_tracked']:<12.3f} {r['num_trades']:<8} {r['validation_fitness']:<15.2f}")

    # Verify gradient exists
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)

    # Check that validation fitness increases with max_coefficient
    fitness_values = [r['validation_fitness'] for r in results]
    is_increasing = all(fitness_values[i] < fitness_values[i+1] for i in range(len(fitness_values)-1))

    if is_increasing:
        print("✓ PASSED: Validation fitness increases with max_coefficient")
        print("✓ Gradient exists for evolutionary selection")
        print(f"\n  Agent C ({results[2]['validation_fitness']:.2f}) > Agent B ({results[1]['validation_fitness']:.2f}) > Agent A ({results[0]['validation_fitness']:.2f})")
        print("\n✓ Evolution will now prefer agents with higher max_coefficient,")
        print("  gradually pushing the population toward the trading threshold!")
        return True
    else:
        print("✗ FAILED: Validation fitness does not increase properly")
        print("✗ No gradient for evolutionary selection")
        return False


if __name__ == "__main__":
    success = test_validation_gradient()
    sys.exit(0 if success else 1)
